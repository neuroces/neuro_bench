"""Build an Inspect dataset from the benchmark questions.

Joins ``questions.json`` with the ``answers.json`` key (ground truth) and emits
one Inspect ``Sample`` per question. The full question spec + truth travel in
``metadata`` so the scorer can grade without re-reading files, and the source +
redaction are used by a setup solver to populate the tool context.
"""

from __future__ import annotations

import json
from typing import Any

from benchmark import verify_ground_truth
from inspect_ai.dataset import Sample

_ANSWER_FORMAT = {
    "mcq": "End your reply with a line 'ANSWER: <letter>' (e.g. ANSWER: B).",
    "exact": "End your reply with a line 'ANSWER: <value>'.",
    "numeric": "End your reply with a line 'ANSWER: <number>' (digits only).",
}


def build_prompt(question: dict[str, Any]) -> str:
    if question["answer_type"] == "multi":
        keys = [p["key"] for p in question["parts"]]
        pairs = "; ".join(f"{k}=<letter>" for k in keys)
        fmt = f"End your reply with a line 'ANSWER: {pairs}'."
    else:
        fmt = _ANSWER_FORMAT[question["answer_type"]]
    return f"{question['prompt']}\n\n{fmt}"


def _target_text(truth: Any) -> str:
    return truth if isinstance(truth, str) else json.dumps(truth, sort_keys=True)


def neurobench_dataset(category: int | None = None) -> list[Sample]:
    """Return Inspect samples, optionally filtered to one category."""
    questions = verify_ground_truth.load_questions()
    answers = verify_ground_truth.load_answers()
    samples: list[Sample] = []
    for q in questions:
        if category is not None and q["category"] != category:
            continue
        truth = answers[q["id"]]["answer"]
        metadata = {
            "id": q["id"],
            "category": q["category"],
            "answer_type": q["answer_type"],
            "choices": q.get("choices"),
            "tolerance": q.get("tolerance"),
            "parts": q.get("parts"),
            "source": q["source"],
            "redaction": q.get("redaction", []),
            "truth": truth,
        }
        samples.append(
            Sample(
                id=q["id"],
                input=build_prompt(q),
                target=_target_text(truth),
                metadata=metadata,
            )
        )
    return samples
