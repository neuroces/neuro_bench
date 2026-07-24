"""Inspect scorer wrapping the harness-independent grading engine.

Extracts the model's final answer (an ``ANSWER:`` line) from its output, then
grades it with :func:`grading.core.grade_answer`. The score value is the
question's fractional score (1.0, 0.5 for half a multi-part, or 0.0), so the
mean metric is the benchmark accuracy with partial credit.
"""

from __future__ import annotations

import re
from typing import Any

from grading.core import grade_answer
from inspect_ai.scorer import mean, Score, scorer, stderr, Target
from inspect_ai.solver import TaskState

_ANSWER_RE = re.compile(r"ANSWER:\s*(.+)", re.IGNORECASE)
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_CHOICE_RE = re.compile(r"\b([A-Da-d])\b")


def _mcq_token(text: str) -> str:
    """Reduce a (possibly verbose) MCQ answer to its choice letter, else return as-is.

    Handles real model submissions like "A) Awake — ..." or "B (anesthesia)". A
    standalone letter A-D is treated as the choice; otherwise the text is returned
    so value-style answers ("awake") can still be matched by the grader.
    """
    match = _CHOICE_RE.search(text)
    return match.group(1).upper() if match else text


def _raw_answer(text: str) -> str:
    """Best-effort final-answer string, tolerant of <answer> tags and stray XML.

    Priority: an <answer>...</answer> block, then an 'ANSWER:' line, then the last
    non-empty line of the tag-stripped text. Residual tags (e.g. a leaked
    '</invoke>') are removed.
    """
    tag = _ANSWER_TAG_RE.search(text)
    if tag:
        raw = tag.group(1)
    else:
        marks = _ANSWER_RE.findall(text)
        if marks:
            raw = marks[-1]
        else:
            cleaned = _TAG_RE.sub("", text).strip()
            lines = [ln for ln in cleaned.splitlines() if ln.strip()]
            raw = lines[-1] if lines else ""
    return _TAG_RE.sub("", raw).strip()


def extract_answer(text: str, answer_type: str, parts: list[dict] | None = None) -> Any:
    """Parse the model's final answer from its output text.

    Scalar types return a string; ``multi`` returns a ``{part_key: value}`` dict
    parsed from ``key=value`` pairs (with a per-key letter fallback).
    """
    text = text or ""
    raw = _raw_answer(text)

    if answer_type == "mcq":
        return _mcq_token(raw)
    if answer_type != "multi":
        return raw

    result: dict[str, str] = {}
    for token in re.split(r"[;,\n]", raw):
        if "=" in token:
            key, val = token.split("=", 1)
            result[key.strip()] = _mcq_token(val.strip())
    # Fallback: search the whole answer for "<key> ... <letter>" per part.
    if parts:
        for part in parts:
            key = part["key"]
            if key not in result:
                m = re.search(rf"{re.escape(key)}\W+([A-Da-d])\b", raw, re.IGNORECASE)
                if m:
                    result[key] = m.group(1).upper()
    return result


@scorer(metrics=[mean(), stderr()])
def neurobench_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata
        answer_type = meta["answer_type"]
        extracted = extract_answer(
            state.output.completion, answer_type, meta.get("parts")
        )
        question = {
            "answer_type": answer_type,
            "choices": meta.get("choices"),
            "tolerance": meta.get("tolerance"),
            "parts": meta.get("parts"),
        }
        result = grade_answer(question, extracted, meta["truth"])
        return Score(
            value=result["score"],
            answer=str(extracted),
            explanation=f"expected={meta['truth']!r} parts={result.get('parts')}",
            metadata={"passed": result["passed"], "parts": result.get("parts")},
        )

    return score
