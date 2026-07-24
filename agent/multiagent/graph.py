"""LangGraph wiring for the NeuroBench multi-agent solver.

    START → orchestrator_plan → data_scientist → domain_expert → critic ─┐
                                     ↑                                     │
                                     └──── (revise: loop back to DS) ──────┘
                                                                          │
                                            (approve / max-iter) → finalize → END

Mirrors ``lit_review_agent/graph.py`` (linear edges + a single conditional
router with a max-iteration guard).
"""

from __future__ import annotations

from agent.multiagent.nodes import (
    critic_node,
    critic_router,
    data_scientist_node,
    domain_expert_node,
    finalize_node,
    orchestrator_plan_node,
)
from agent.multiagent.state import MultiAgentState
from langgraph.graph import END, START, StateGraph


def build_graph() -> StateGraph:
    """Build and return the (uncompiled) multi-agent StateGraph."""
    graph = StateGraph(MultiAgentState)

    graph.add_node("orchestrator_plan", orchestrator_plan_node)
    graph.add_node("data_scientist", data_scientist_node)
    graph.add_node("domain_expert", domain_expert_node)
    graph.add_node("critic", critic_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "orchestrator_plan")
    graph.add_edge("orchestrator_plan", "data_scientist")
    graph.add_edge("data_scientist", "domain_expert")
    graph.add_edge("domain_expert", "critic")

    graph.add_conditional_edges(
        "critic",
        critic_router,
        {"data_scientist": "data_scientist", "finalize": "finalize"},
    )

    graph.add_edge("finalize", END)
    return graph


def compile_graph(category: int | None = None):
    """Build and compile the graph, ready to ``ainvoke``.

    ``category`` is accepted for API symmetry with the task/solver wiring; the
    Data Scientist node reads the category from the graph state, so the compiled
    graph itself is category-agnostic.
    """
    return build_graph().compile()
