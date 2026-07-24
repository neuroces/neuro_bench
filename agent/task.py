"""The NeuroBench Inspect task.

Composes: a setup solver that publishes each sample's asset/redaction into the
tool context, the built-in ``react`` agent equipped with our EEG tools, and the
partial-credit scorer. Run with::

    inspect eval agent/task.py --model anthropic/claude-... --limit 3
    inspect eval agent/task.py --model mockllm/model   # offline smoke
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.util import store_as

from agent.context import QuestionContext
from agent.dataset import neurobench_dataset
from agent.inspect_tools import tools_for
from agent.scorers import neurobench_scorer

SYSTEM_PROMPT = (
    "You are a computational neurophysiologist. You are given questions about real "
    "electrophysiology recordings and a set of analysis tools. The ground-truth "
    "state labels are withheld: you must derive each answer by analyzing the signal "
    "(e.g. spectral analysis of the EEG) using the tools, not by looking it up. "
    "Use the tools as needed, reason about the electrophysiology, then give your "
    "final answer on an 'ANSWER:' line in the exact format requested."
)

CAT2_PROMPT = (
    "You are a computational neurophysiologist analyzing patch-clamp recordings of "
    "single cortical interneurons. The cell-type label (Cre line) is withheld: you "
    "must infer it from the current-clamp electrophysiology (AP width, firing "
    "pattern, rheobase, F-I curve) using the tools. Fast-spiking, non-adapting cells "
    "with narrow spikes and high rheobase are typically Pvalb; adapting cells with "
    "broader spikes and lower rheobase are typically Sst. Reason about the "
    "electrophysiology, then give your final answer on an 'ANSWER:' line."
)


def _prompt_for(category: int | None) -> str:
    if category == 2:
        return CAT2_PROMPT
    if category == 3:
        return SYSTEM_PROMPT
    return GENERAL_PROMPT


GENERAL_PROMPT = (
    "You are a computational neurophysiologist. Each question concerns a real "
    "electrophysiology recording (an EEG epoch, or a single-cell patch-clamp "
    "recording) with the ground-truth label withheld. Use the analysis tools to "
    "examine the signal (spectral analysis for EEG; AP width, firing pattern, "
    "rheobase and the F-I curve for patch-clamp), reason about the "
    "electrophysiology, and infer the answer from the data rather than looking it "
    "up. Give your final answer on an 'ANSWER:' line in the requested format."
)


@solver
def setup_context():
    """Publish the sample's source + redaction into the sample-scoped tool context."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        src = state.metadata["source"]
        ctx = store_as(QuestionContext)
        ctx.dandi_id = src["dandi_id"]
        ctx.version = src["version"]
        ctx.asset_path = src["asset_path"]
        ctx.redaction = list(state.metadata.get("redaction", []))
        ctx.channel = src.get("channel") or 0
        ctx.window = src.get("window")
        return state

    return solve


@task
def neurobench(category: int | None = 3) -> Task:
    return Task(
        dataset=neurobench_dataset(category),
        solver=[
            setup_context(),
            react(prompt=_prompt_for(category), tools=tools_for(category), attempts=1),
        ],
        scorer=neurobench_scorer(),
    )


@task
def neurobench_multiagent(category: int | None = 3) -> Task:
    """PHA-style multi-agent arm: same dataset + scorer, role-separated solver.

    Uses the LangGraph app (orchestrator + data scientist + domain expert +
    critic) as the solver. Every sub-agent runs on the eval-configured model, so
    ``--model X`` makes this directly comparable to the ``react`` arm.
    """
    from agent.multiagent.solver import multiagent_solver

    return Task(
        dataset=neurobench_dataset(category),
        solver=[setup_context(), multiagent_solver(category)],
        scorer=neurobench_scorer(),
    )
