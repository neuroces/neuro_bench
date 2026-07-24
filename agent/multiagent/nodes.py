"""Node functions for the NeuroBench multi-agent graph.

Each node is an ``async`` function taking :class:`MultiAgentState` and returning
a partial state update, mirroring ``lit_review_agent/graph.py``. Every LLM call
goes through ``inspect_ai.model.get_model()`` (the eval-configured model), so the
whole system runs on a single model chosen at eval time (e.g. Claude).

Integrity boundary: **only** :func:`data_scientist_node` passes ``tools=`` to the
model and can touch the recording. The orchestrator, domain expert and critic
call the model with no tools and see only the Data Scientist's textual report.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Type, TypeVar

from agent.inspect_tools import tools_for
from agent.multiagent.state import (
    AnalysisPlan,
    CriticDecision,
    ExpertInterpretation,
    MultiAgentState,
)
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    execute_tools,
    get_model,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_T = TypeVar("_T", bound=BaseModel)

_ANSWER_RE = re.compile(r"^\s*answer:\s*", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Structured-output helper
# ---------------------------------------------------------------------------


def _parse_model(text: str, model_cls: Type[_T]) -> _T | None:
    """Parse a model's text into ``model_cls``, tolerant of fences and prose.

    Returns ``None`` on any failure so callers can apply a safe fallback (as in
    ``lit_review_agent/critic.py``).
    """
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return model_cls(**json.loads(raw[start:end]))
    except Exception:  # noqa: BLE001 - any parse issue → fallback
        return None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_ORCH_SYSTEM = (
    "You are the orchestrator of a computational-neurophysiology analysis team. "
    "You do NOT have data access. Given a question about a real electrophysiology "
    "recording, write a short analysis plan telling the Data Scientist which "
    "signal-analysis steps to run (the ground-truth label is withheld and must be "
    "derived from the signal). Return ONLY a JSON object: "
    '{"steps": ["..."], "rationale": "..."}'
)

_DS_SYSTEM = (
    "You are the Data Scientist. You are the only agent with data-analysis tools "
    "for this recording. Follow the orchestrator's plan, call the tools to analyze "
    "the signal, and then write a concise, numbers-first report of what you found "
    "(band powers, dominant band, AP width, rheobase, firing pattern, ISI stats, "
    "etc. as relevant). Do NOT state a final answer or label — report only the "
    "measurements and derived features. The raw file and label tables are withheld; "
    "infer nothing you did not measure."
)

_EXPERT_SYSTEM = (
    "You are a neurophysiologist. You have NO data access — you see only the Data "
    "Scientist's report of measurements. Interpret those numbers to infer the "
    "answer to the question (e.g. brain state from the EEG spectrum, or interneuron "
    "Cre line from current-clamp features). Return ONLY a JSON object: "
    '{"interpretation": "your neurophysiological reasoning", '
    '"draft_answer": "the answer in the exact format the question requests"}'
)

_CRITIC_SYSTEM = (
    "You are a verification critic. You have NO data access. Decide whether the "
    "draft answer is (a) grounded in the Data Scientist's reported measurements "
    "(not invented) and (b) valid for the requested answer format. Return ONLY a "
    'JSON object: {"decision": "approve" or "revise", "feedback": "what to fix if '
    'revising", "reasoning": "brief justification"}. Approve if the answer is '
    "supported and well-formed; request revision only for concrete, fixable "
    "problems (ungrounded claim, wrong format, missing multi-part key)."
)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def orchestrator_plan_node(state: MultiAgentState) -> dict:
    """Plan the analysis (no tools) → ``analysis_plan``."""
    model = get_model()
    user = ChatMessageUser(content=f"Question:\n{state['question']}")
    output = await model.generate(input=[ChatMessageSystem(content=_ORCH_SYSTEM), user])
    plan = _parse_model(output.completion, AnalysisPlan)
    if plan is None:
        analysis_plan = output.completion.strip()
    else:
        steps = "\n".join(f"- {s}" for s in plan.steps)
        analysis_plan = f"{plan.rationale}\n{steps}".strip()
    logger.info("Orchestrator plan produced (%d chars)", len(analysis_plan))
    return {"analysis_plan": analysis_plan}


async def data_scientist_node(state: MultiAgentState) -> dict:
    """The only node with tools: run the bounded analysis loop → ``ds_report``."""
    model = get_model()
    tools = tools_for(state.get("category"))
    max_steps = state.get("ds_max_steps", 6)

    feedback = state.get("critic_feedback", [])
    revise_note = ""
    if feedback and "[revise]" in feedback[-1].lower():
        revise_note = f"\n\nThe critic requested revision — address this:\n{feedback[-1]}"

    user = ChatMessageUser(
        content=(
            f"Question:\n{state['question']}\n\n"
            f"Analysis plan from the orchestrator:\n{state.get('analysis_plan', '')}"
            f"{revise_note}"
        )
    )
    msgs = [ChatMessageSystem(content=_DS_SYSTEM), user]

    output = None
    for step in range(max_steps):
        output = await model.generate(input=msgs, tools=tools)
        msgs.append(output.message)
        if not output.message.tool_calls:
            break
        result = await execute_tools(msgs, tools)
        msgs.extend(result.messages)
        logger.info("DS step %d: executed %d tool call(s)", step, len(result.messages))
    else:
        # Hit the step budget with tool calls pending — force a text summary.
        msgs.append(
            ChatMessageUser(
                content="Stop calling tools now and summarize your measurements."
            )
        )
        output = await model.generate(input=msgs, tools=[])

    report = output.completion if output else ""
    logger.info("DS report produced (%d chars)", len(report))
    return {"ds_report": report}


async def domain_expert_node(state: MultiAgentState) -> dict:
    """Interpret the DS report (no tools) → ``expert_interpretation`` + ``draft_answer``."""
    model = get_model()
    user = ChatMessageUser(
        content=(
            f"Question:\n{state['question']}\n\n"
            f"Data Scientist's report:\n{state.get('ds_report', '')}"
        )
    )
    output = await model.generate(
        input=[ChatMessageSystem(content=_EXPERT_SYSTEM), user]
    )
    interp = _parse_model(output.completion, ExpertInterpretation)
    if interp is None:
        text = output.completion.strip()
        return {"expert_interpretation": text, "draft_answer": text}
    return {
        "expert_interpretation": interp.interpretation,
        "draft_answer": interp.draft_answer,
    }


async def critic_node(state: MultiAgentState) -> dict:
    """Verify the draft answer (no tools) → append feedback, bump iteration."""
    model = get_model()
    iteration = state.get("iteration", 0)
    user = ChatMessageUser(
        content=(
            f"Question:\n{state['question']}\n\n"
            f"Data Scientist's report:\n{state.get('ds_report', '')}\n\n"
            f"Draft answer:\n{state.get('draft_answer', '')}"
        )
    )
    output = await model.generate(
        input=[ChatMessageSystem(content=_CRITIC_SYSTEM), user]
    )
    decision = _parse_model(output.completion, CriticDecision)
    if decision is None:
        decision = CriticDecision(
            decision="approve",
            reasoning="Critic parsing failed; defaulting to approve to avoid a loop.",
        )
    note = decision.feedback or decision.reasoning
    entry = f"Iteration {iteration} [{decision.decision}]: {note}"
    logger.info("Critic (iter %d): %s", iteration, decision.decision)
    return {
        "critic_feedback": state.get("critic_feedback", []) + [entry],
        "iteration": iteration + 1,
    }


async def finalize_node(state: MultiAgentState) -> dict:
    """Compose the final completion ending in a correctly-formatted ANSWER: line."""
    interp = (state.get("expert_interpretation") or "").strip()
    draft = (state.get("draft_answer") or "").strip()
    # Strip any ANSWER: prefix the expert may have added; we re-add it canonically.
    draft = _ANSWER_RE.sub("", draft).strip()
    body = interp or "Based on the Data Scientist's measurements."
    final = f"{body}\n\nANSWER: {draft}".strip()
    return {"final_answer": final}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def critic_router(state: MultiAgentState) -> str:
    """Route after the critic: revise (back to DS) or finalize."""
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 2)
    if iteration >= max_iter:
        logger.info("Critic router: max iterations (%d) → finalize", max_iter)
        return "finalize"
    feedback = state.get("critic_feedback", [])
    last = feedback[-1] if feedback else ""
    if "[revise]" in last.lower():
        logger.info("Critic router: revision requested → data_scientist")
        return "data_scientist"
    logger.info("Critic router: approved → finalize")
    return "finalize"
