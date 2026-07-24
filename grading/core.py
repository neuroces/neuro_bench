"""Harness-independent grading engine.

Pure functions that grade a model answer against ground truth using a question's
spec (``answer_type``, ``choices``, ``tolerance``, ``parts``). The Inspect scorer
is a thin wrapper around :func:`grade_answer`; keeping the logic here means it is
fully unit-testable offline and reusable outside any harness.
"""

from __future__ import annotations

from typing import Any

from grading.numeric_tolerance import within_tolerance
from grading.partial_credit import set_score


def _norm(value: object) -> str:
    return str(value).strip().lower()


def mcq_to_value(answer: object, choices: dict[str, str] | None) -> str:
    """Translate a choice key (e.g. 'B') to its value; pass values through."""
    a = str(answer).strip()
    if choices:
        for key, val in choices.items():
            if a.lower() == key.lower():
                return val
    return a


def grade_mcq(model: object, truth: object, choices: dict[str, str] | None) -> bool:
    return _norm(mcq_to_value(model, choices)) == _norm(truth)


def grade_exact(model: object, truth: object) -> bool:
    return _norm(model) == _norm(truth)


def grade_scalar(
    answer_type: str,
    model: object,
    truth: object,
    *,
    choices: dict[str, str] | None = None,
    tolerance: float | None = None,
) -> float:
    """Grade a single (non-multi) answer, returning 1.0 or 0.0."""
    if answer_type == "mcq":
        return 1.0 if grade_mcq(model, truth, choices) else 0.0
    if answer_type == "exact":
        return 1.0 if grade_exact(model, truth) else 0.0
    if answer_type == "numeric":
        if tolerance is None:
            raise ValueError("numeric answer_type requires a tolerance")
        return 1.0 if within_tolerance(model, truth, tolerance) else 0.0
    if answer_type == "set":
        return set_score(model or [], truth or [])
    raise ValueError(f"unknown answer_type {answer_type!r}")


def grade_answer(
    question: dict[str, Any], model_answer: Any, truth: Any
) -> dict[str, Any]:
    """Grade a model answer against ``truth`` for a question spec.

    Returns ``{"score": float in [0,1], "passed": bool, "parts": {...}?}``.
    Multi-part questions score the mean of their part scores.
    """
    if question["answer_type"] != "multi":
        score = grade_scalar(
            question["answer_type"],
            model_answer,
            truth,
            choices=question.get("choices"),
            tolerance=question.get("tolerance"),
        )
        return {"score": score, "passed": score >= 1.0}

    model_answer = model_answer if isinstance(model_answer, dict) else {}
    truth = truth if isinstance(truth, dict) else {}
    parts: dict[str, float] = {}
    for part in question["parts"]:
        key = part["key"]
        parts[key] = grade_scalar(
            part["answer_type"],
            model_answer.get(key),
            truth.get(key),
            choices=part.get("choices"),
            tolerance=part.get("tolerance"),
        )
    score = sum(parts.values()) / len(parts) if parts else 0.0
    return {"score": score, "passed": score >= 1.0, "parts": parts}
