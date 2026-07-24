"""Inspect solver that runs the multi-agent LangGraph app.

Seeds :class:`MultiAgentState` from the Inspect sample (input text + metadata),
invokes the compiled graph, and writes the graph's final completion back onto
``state.output`` so the shared :func:`neurobench_scorer` can grade it exactly as
it grades the ``react`` arm.
"""

from __future__ import annotations

from agent.multiagent.graph import compile_graph
from agent.multiagent.state import MultiAgentState
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.solver import Generate, TaskState, solver


@solver
def multiagent_solver(
    category: int | None = None,
    max_iterations: int = 2,
    ds_max_steps: int = 6,
):
    """Run the PHA-style multi-agent graph as an Inspect solver."""

    app = compile_graph(category)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        meta = state.metadata
        initial: MultiAgentState = {
            "question": state.input_text,
            "answer_type": meta["answer_type"],
            "category": meta.get("category", category),
            "choices": meta.get("choices"),
            "tolerance": meta.get("tolerance"),
            "parts": meta.get("parts"),
            "analysis_plan": "",
            "ds_report": "",
            "expert_interpretation": "",
            "draft_answer": "",
            "critic_feedback": [],
            "iteration": 0,
            "max_iterations": max_iterations,
            "ds_max_steps": ds_max_steps,
            "final_answer": None,
        }

        final = await app.ainvoke(initial)
        completion = final.get("final_answer") or ""
        model_name = get_model().name
        state.output = ModelOutput.from_content(model_name, completion)
        return state

    return solve
