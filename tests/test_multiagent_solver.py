"""Offline smoke test for the multi-agent Inspect solver.

Runs the full graph end-to-end through ``inspect eval`` with ``mockllm/model``
(no network, no API keys) and asserts a scorable completion — one ending in an
``ANSWER:`` line — is produced and graded.
"""

from __future__ import annotations


def test_multiagent_solver_produces_scorable_completion():
    from agent.dataset import neurobench_dataset
    from agent.multiagent.solver import multiagent_solver
    from agent.scorers import neurobench_scorer
    from agent.task import setup_context
    from inspect_ai import Task, eval

    ds = neurobench_dataset(category=3)[:1]  # C3-Q01
    task = Task(
        dataset=ds,
        solver=[setup_context(), multiagent_solver(3)],
        scorer=neurobench_scorer(),
    )
    log = eval(task, model="mockllm/model", display="none")[0]

    assert log.status == "success"
    sample = log.samples[0]
    # The graph must have written a completion ending in an ANSWER: line.
    assert "ANSWER:" in sample.output.completion
    # And the shared scorer must have produced a numeric score for it.
    score = list(sample.scores.values())[0]
    assert isinstance(score.value, (int, float))


def test_multiagent_task_registers_and_runs():
    from agent.task import neurobench_multiagent
    from inspect_ai import eval

    log = eval(
        neurobench_multiagent(category=3),
        model="mockllm/model",
        limit=1,
        display="none",
    )[0]
    assert log.status == "success"
    assert "ANSWER:" in log.samples[0].output.completion
