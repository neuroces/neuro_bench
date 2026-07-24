"""Unit tests for the NeuroBench multi-agent graph.

Graph construction, node functions (with ``get_model`` patched), and the critic
router. All offline — no network, no API keys. Async nodes are driven with
``asyncio.run`` since pytest-asyncio is not a dependency.
"""

from __future__ import annotations

import asyncio
from collections import namedtuple
from unittest.mock import AsyncMock, patch

from agent.multiagent.graph import build_graph, compile_graph
from agent.multiagent.nodes import (
    critic_node,
    critic_router,
    data_scientist_node,
    domain_expert_node,
    finalize_node,
    orchestrator_plan_node,
)
from agent.multiagent.state import MultiAgentState
from inspect_ai.model import ModelOutput


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------

_ExecResult = namedtuple("_ExecResult", ["messages", "output"])


class _FakeModel:
    """A model whose ``generate`` yields scripted outputs in order (last repeats)."""

    def __init__(self, outputs, name: str = "mockllm/model"):
        self._outputs = list(outputs)
        self._i = 0
        self.name = name

    async def generate(self, input, tools=(), **kwargs):
        out = self._outputs[min(self._i, len(self._outputs) - 1)]
        self._i += 1
        return out


def _content(text: str) -> ModelOutput:
    return ModelOutput.from_content("mockllm/model", text)


def _base_state(**overrides) -> MultiAgentState:
    defaults: MultiAgentState = {
        "question": "Classify the brain state.\n\nEnd your reply with 'ANSWER: <letter>'.",
        "answer_type": "mcq",
        "category": 3,
        "choices": {"A": "awake", "B": "anesthetized"},
        "tolerance": None,
        "parts": None,
        "analysis_plan": "",
        "ds_report": "",
        "expert_interpretation": "",
        "draft_answer": "",
        "critic_feedback": [],
        "iteration": 0,
        "max_iterations": 2,
        "ds_max_steps": 6,
        "final_answer": None,
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


class TestGraphConstruction:
    def test_builds_with_all_nodes(self):
        graph = build_graph()
        for name in (
            "orchestrator_plan",
            "data_scientist",
            "domain_expert",
            "critic",
            "finalize",
        ):
            assert name in graph.nodes

    def test_compiles_successfully(self):
        app = compile_graph(3)
        assert app is not None
        assert type(app).__name__ == "CompiledStateGraph"


# ---------------------------------------------------------------------------
# orchestrator_plan_node
# ---------------------------------------------------------------------------


class TestOrchestratorNode:
    def test_parses_structured_plan(self):
        model = _FakeModel(
            [_content('{"steps": ["compute PSD"], "rationale": "spectrum tells state"}')]
        )
        with patch("agent.multiagent.nodes.get_model", return_value=model):
            result = asyncio.run(orchestrator_plan_node(_base_state()))
        assert "compute PSD" in result["analysis_plan"]
        assert "spectrum tells state" in result["analysis_plan"]

    def test_falls_back_to_raw_text(self):
        model = _FakeModel([_content("just prose, not json")])
        with patch("agent.multiagent.nodes.get_model", return_value=model):
            result = asyncio.run(orchestrator_plan_node(_base_state()))
        assert result["analysis_plan"] == "just prose, not json"


# ---------------------------------------------------------------------------
# data_scientist_node
# ---------------------------------------------------------------------------


class TestDataScientistNode:
    def test_reports_without_tool_calls(self):
        model = _FakeModel([_content("Band powers: theta dominant, peak 6 Hz.")])
        with patch("agent.multiagent.nodes.get_model", return_value=model):
            result = asyncio.run(data_scientist_node(_base_state()))
        assert "theta dominant" in result["ds_report"]

    def test_runs_tool_loop_then_summarizes(self):
        outputs = [
            ModelOutput.for_tool_call(
                "mockllm/model",
                tool_name="compute_psd",
                tool_arguments={"start_s": 0.0, "duration_s": 10.0},
            ),
            _content("After analysis: delta dominant, peak 2 Hz."),
        ]
        model = _FakeModel(outputs)
        exec_mock = AsyncMock(return_value=_ExecResult(messages=[], output=None))
        with (
            patch("agent.multiagent.nodes.get_model", return_value=model),
            patch("agent.multiagent.nodes.execute_tools", exec_mock),
        ):
            result = asyncio.run(data_scientist_node(_base_state()))
        exec_mock.assert_awaited_once()
        assert "delta dominant" in result["ds_report"]


# ---------------------------------------------------------------------------
# domain_expert_node
# ---------------------------------------------------------------------------


class TestDomainExpertNode:
    def test_parses_interpretation_and_draft(self):
        model = _FakeModel(
            [_content('{"interpretation": "theta implies awake", "draft_answer": "A"}')]
        )
        state = _base_state(ds_report="theta dominant")
        with patch("agent.multiagent.nodes.get_model", return_value=model):
            result = asyncio.run(domain_expert_node(state))
        assert result["expert_interpretation"] == "theta implies awake"
        assert result["draft_answer"] == "A"

    def test_falls_back_on_unparseable(self):
        model = _FakeModel([_content("prose answer B")])
        with patch("agent.multiagent.nodes.get_model", return_value=model):
            result = asyncio.run(domain_expert_node(_base_state()))
        assert result["draft_answer"] == "prose answer B"


# ---------------------------------------------------------------------------
# critic_node
# ---------------------------------------------------------------------------


class TestCriticNode:
    def test_approve_increments_iteration_and_logs(self):
        model = _FakeModel(
            [_content('{"decision": "approve", "feedback": "", "reasoning": "grounded"}')]
        )
        state = _base_state(iteration=0)
        with patch("agent.multiagent.nodes.get_model", return_value=model):
            result = asyncio.run(critic_node(state))
        assert result["iteration"] == 1
        assert "[approve]" in result["critic_feedback"][-1]

    def test_revise_records_feedback(self):
        model = _FakeModel(
            [_content('{"decision": "revise", "feedback": "answer not grounded"}')]
        )
        with patch("agent.multiagent.nodes.get_model", return_value=model):
            result = asyncio.run(critic_node(_base_state()))
        assert "[revise]" in result["critic_feedback"][-1]
        assert "not grounded" in result["critic_feedback"][-1]

    def test_parse_failure_defaults_to_approve(self):
        model = _FakeModel([_content("not json at all")])
        with patch("agent.multiagent.nodes.get_model", return_value=model):
            result = asyncio.run(critic_node(_base_state()))
        assert "[approve]" in result["critic_feedback"][-1]


# ---------------------------------------------------------------------------
# finalize_node
# ---------------------------------------------------------------------------


class TestFinalizeNode:
    def test_composes_answer_line(self):
        state = _base_state(
            expert_interpretation="theta implies awake", draft_answer="A"
        )
        result = asyncio.run(finalize_node(state))
        assert result["final_answer"].strip().endswith("ANSWER: A")

    def test_strips_duplicate_answer_prefix(self):
        state = _base_state(expert_interpretation="reasoning", draft_answer="ANSWER: B")
        result = asyncio.run(finalize_node(state))
        assert result["final_answer"].count("ANSWER:") == 1
        assert result["final_answer"].strip().endswith("ANSWER: B")


# ---------------------------------------------------------------------------
# critic_router
# ---------------------------------------------------------------------------


class TestCriticRouter:
    def test_routes_to_finalize_on_approve(self):
        state = _base_state(
            iteration=1, critic_feedback=["Iteration 0 [approve]: grounded"]
        )
        assert critic_router(state) == "finalize"

    def test_routes_to_ds_on_revise(self):
        state = _base_state(
            iteration=1,
            max_iterations=2,
            critic_feedback=["Iteration 0 [revise]: fix it"],
        )
        assert critic_router(state) == "data_scientist"

    def test_routes_to_finalize_at_max_iterations(self):
        state = _base_state(
            iteration=2,
            max_iterations=2,
            critic_feedback=["Iteration 1 [revise]: still wrong"],
        )
        assert critic_router(state) == "finalize"

    def test_routes_to_finalize_with_no_feedback(self):
        state = _base_state(iteration=1)
        assert critic_router(state) == "finalize"
