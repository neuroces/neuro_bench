"""State schema for the NeuroBench multi-agent (PHA-style) graph.

Mirrors ``lit_review_agent/state.py``: a ``TypedDict`` carrying the shared state
between LangGraph nodes, plus Pydantic models for the structured outputs each
non-data node emits.

The integrity model lives in *which* fields each node reads: only the Data
Scientist produces ``ds_report`` (from tool calls); every downstream node reads
that textual report and never the raw file. ``QuestionContext`` (asset id /
version / redaction) is not carried here — it stays in the Inspect store and is
consumed only inside the tool wrappers, so the orchestrator, domain expert and
critic literally cannot open the NWB file.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel


class AnalysisPlan(BaseModel):
    """Orchestrator's plan: what the Data Scientist should compute."""

    steps: list[str] = []
    rationale: str = ""


class ExpertInterpretation(BaseModel):
    """Domain expert's reading of the Data Scientist's numbers."""

    interpretation: str = ""
    draft_answer: str = ""


class CriticDecision(BaseModel):
    """Critic's verdict on whether the draft answer is grounded and well-formed."""

    decision: Literal["approve", "revise"]
    feedback: str = ""
    reasoning: str = ""


class MultiAgentState(TypedDict):
    """Shared state flowing through the multi-agent graph."""

    # Question spec (seeded from the Inspect sample)
    question: str
    answer_type: str
    category: int | None
    choices: dict[str, str] | None
    tolerance: float | None
    parts: list[dict[str, Any]] | None

    # Working fields filled by the nodes
    analysis_plan: str
    ds_report: str
    expert_interpretation: str
    draft_answer: str

    # Critic feedback loop
    critic_feedback: list[str]
    iteration: int
    max_iterations: int

    # Data Scientist bounded tool loop
    ds_max_steps: int

    # Final composed completion (ends in an ANSWER: line)
    final_answer: str | None
