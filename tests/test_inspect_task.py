"""Tests for the Inspect integration (offline wiring + gated live run)."""

from __future__ import annotations

import os

import pytest
from agent.dataset import build_prompt, neurobench_dataset
from agent.scorers import extract_answer


# --- dataset (offline) -------------------------------------------------------


def test_dataset_builds_category_3():
    ds = neurobench_dataset(category=3)
    assert len(ds) == 10
    ids = {s.id for s in ds}
    assert "C3-Q01" in ids
    s = next(s for s in ds if s.id == "C3-Q01")
    assert s.metadata["answer_type"] == "mcq"
    assert s.metadata["truth"] == "awake"
    assert s.metadata["redaction"] == ["intervals"]
    assert "ANSWER:" in s.input


def test_build_prompt_includes_format_hint():
    q = {"prompt": "Classify the state.", "answer_type": "numeric"}
    assert "ANSWER:" in build_prompt(q)


# --- answer extraction (offline) --------------------------------------------


def test_extract_scalar_and_multi():
    assert extract_answer("blah\nANSWER: B", "mcq") == "B"
    assert extract_answer("onset\nANSWER: 1410 s", "numeric") == "1410 s"
    assert extract_answer("x\nANSWER: state=B; band=A", "multi") == {
        "state": "B",
        "band": "A",
    }


def test_extract_falls_back_to_last_line():
    assert extract_answer("no marker here\njust text", "exact") == "just text"


def test_extract_verbose_mcq_answers():
    # Real react submissions are prose; we still recover the choice letter.
    assert extract_answer("A) Awake \u2014 the PSD is theta-dominant, not anesthesia.", "mcq") == "A"
    assert extract_answer("B (isoflurane anesthesia) \u2014 delta dominates.", "mcq") == "B"
    # A value-only answer (no standalone letter) is returned for value matching.
    assert extract_answer("awake", "mcq") == "awake"


def test_extract_verbose_multi_answer():
    parts = [{"key": "state"}, {"key": "band"}]
    out = extract_answer("state: B (anesthetized), band: A (delta)", "multi", parts)
    assert out == {"state": "B", "band": "A"}


def test_extract_tolerates_answer_tags_and_stray_xml():
    # Real Claude output: answer at the START, trailing tool-call/answer tags.
    txt = "A) Awake \u2014 theta-dominant spectrum, not anesthesia.</answer>\n</invoke>"
    assert extract_answer(txt, "mcq") == "A"
    assert extract_answer("<answer>B (anesthetized)</answer>", "mcq") == "B"
    assert extract_answer("The rheobase is 100 pA.</answer>", "numeric") == "The rheobase is 100 pA."


# --- end-to-end wiring with a scripted mock (offline) -----------------------


def test_offline_eval_scores_correct_answer():
    from agent.scorers import neurobench_scorer
    from agent.task import setup_context
    from inspect_ai import eval, Task
    from inspect_ai.model import get_model, ModelOutput
    from inspect_ai.solver import generate

    ds = neurobench_dataset(category=3)[:1]  # C3-Q01, truth "awake"
    task = Task(
        dataset=ds, solver=[setup_context(), generate()], scorer=neurobench_scorer()
    )
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.from_content("mockllm/model", "reasoning\nANSWER: awake")
        ],
    )
    log = eval(task, model=model, display="none")[0]
    assert log.status == "success"
    score = list(log.samples[0].scores.values())[0]
    assert score.value == 1.0


def test_offline_eval_scores_wrong_answer_zero():
    from agent.scorers import neurobench_scorer
    from agent.task import setup_context
    from inspect_ai import eval, Task
    from inspect_ai.model import get_model, ModelOutput
    from inspect_ai.solver import generate

    ds = neurobench_dataset(category=3)[:1]  # truth "awake"
    task = Task(
        dataset=ds, solver=[setup_context(), generate()], scorer=neurobench_scorer()
    )
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.from_content("mockllm/model", "ANSWER: anesthetized")
        ],
    )
    log = eval(task, model=model, display="none")[0]
    assert list(log.samples[0].scores.values())[0].value == 0.0


# --- full agent loop with tools (needs API key + network) -------------------


@pytest.mark.network
def test_react_tool_loop_with_scripted_mock():
    """Full react + tools + redaction + scorer loop, driven by a scripted mock."""
    from agent.inspect_tools import all_tools
    from agent.scorers import neurobench_scorer
    from agent.task import setup_context, SYSTEM_PROMPT
    from inspect_ai import eval, Task
    from inspect_ai.model import get_model, ModelOutput

    try:
        from inspect_ai.agent import react
    except ImportError:
        from inspect_ai.solver import react  # older layout

    ds = [
        s for s in neurobench_dataset(category=3) if s.id == "C3-Q02"
    ]  # truth anesthetized
    task = Task(
        dataset=ds,
        solver=[
            setup_context(),
            react(prompt=SYSTEM_PROMPT, tools=all_tools(), attempts=1),
        ],
        scorer=neurobench_scorer(),
    )
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.for_tool_call(
                "mockllm/model",
                tool_name="compute_psd",
                tool_arguments={"start_s": 1900.0, "duration_s": 30.0, "channel": 0},
            ),
            ModelOutput.for_tool_call(
                "mockllm/model",
                tool_name="submit",
                tool_arguments={"answer": "ANSWER: anesthetized"},
            ),
        ],
    )
    log = eval(task, model=model, display="none")[0]
    assert log.status == "success"
    assert list(log.samples[0].scores.values())[0].value == 1.0


@pytest.mark.network
def test_react_cat2_tool_loop_with_scripted_mock():
    """Full react + icephys tools + subject redaction + scorer loop for Category 2."""
    from inspect_ai import Task, eval
    from inspect_ai.model import ModelOutput, get_model

    from agent.inspect_tools import tools_for
    from agent.scorers import neurobench_scorer
    from agent.task import CAT2_PROMPT, setup_context

    try:
        from inspect_ai.agent import react
    except ImportError:
        from inspect_ai.solver import react

    ds = [s for s in neurobench_dataset(category=2) if s.id == "C2-Q01"]  # truth Pvalb
    task = Task(
        dataset=ds,
        solver=[setup_context(), react(prompt=CAT2_PROMPT, tools=tools_for(2), attempts=1)],
        scorer=neurobench_scorer(),
    )
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.for_tool_call(
                "mockllm/model", tool_name="list_current_clamp_sweeps", tool_arguments={}
            ),
            ModelOutput.for_tool_call(
                "mockllm/model", tool_name="submit", tool_arguments={"answer": "ANSWER: Pvalb"}
            ),
        ],
    )
    log = eval(task, model=model, display="none")[0]
    assert log.status == "success"
    assert list(log.samples[0].scores.values())[0].value == 1.0


@pytest.mark.network
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY for live agent run"
)
def test_live_react_agent_smoke():
    from inspect_ai import eval

    from agent.task import neurobench

    log = eval(
        neurobench(category=3),
        model="anthropic/claude-3-5-sonnet-latest",
        limit=1,
        display="none",
    )[0]
    assert log.status == "success"
