"""PHA-style multi-agent solver for NeuroBench.

Exposes the LangGraph-based multi-agent solver as an Inspect solver so it can be
compared to the single-agent ``react`` arm under the same dataset and scorer.
"""

from __future__ import annotations

from agent.multiagent.graph import build_graph, compile_graph
from agent.multiagent.solver import multiagent_solver
from agent.multiagent.state import (
    AnalysisPlan,
    CriticDecision,
    ExpertInterpretation,
    MultiAgentState,
)

__all__ = [
    "build_graph",
    "compile_graph",
    "multiagent_solver",
    "AnalysisPlan",
    "CriticDecision",
    "ExpertInterpretation",
    "MultiAgentState",
]
