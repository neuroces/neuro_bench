"""Tests for the harness-independent grading engine (all offline)."""

from __future__ import annotations

import pytest
from grading import core
from grading.numeric_tolerance import parse_number, within_tolerance
from grading.partial_credit import jaccard, set_score


# --- numeric tolerance -------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (3, 3.0),
        (3.5, 3.5),
        ("1414.51", 1414.51),
        ("about 1400 seconds", 1400.0),
        ("-2.5e1", -25.0),
    ],
)
def test_parse_number(value, expected):
    assert parse_number(value) == expected


def test_parse_number_rejects_non_numbers():
    for bad in (None, "no digits", True):
        with pytest.raises(ValueError):
            parse_number(bad)


def test_within_tolerance():
    assert within_tolerance("1400", 1414.51, 45.0)
    assert not within_tolerance("1300", 1414.51, 45.0)
    assert not within_tolerance("garbage", 1414.51, 45.0)


# --- jaccard / set score -----------------------------------------------------


def test_jaccard():
    assert jaccard([1, 2, 3], [1, 2, 3]) == 1.0
    assert jaccard([], []) == 1.0
    assert jaccard([1, 2], [3, 4]) == 0.0
    assert jaccard([1, 2, 3, 4], [3, 4]) == 0.5


def test_set_score_thresholds():
    assert set_score([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert set_score([1, 2, 3, 4, 5, 6], [1, 2, 3, 4]) == 1.0  # jaccard 0.66 >= 0.6
    assert set_score([1], [1, 2, 3, 4]) == 0.5  # partial overlap
    assert set_score([9], [1, 2, 3, 4]) == 0.0


# --- mcq / exact -------------------------------------------------------------


def test_grade_mcq_accepts_key_or_value():
    choices = {"A": "awake", "B": "anesthetized"}
    assert core.grade_mcq("B", "anesthetized", choices)
    assert core.grade_mcq("anesthetized", "anesthetized", choices)
    assert core.grade_mcq("ANESTHETIZED", "anesthetized", choices)
    assert not core.grade_mcq("A", "anesthetized", choices)


def test_grade_exact_case_insensitive():
    assert core.grade_exact("Awake ", "awake")
    assert not core.grade_exact("asleep", "awake")


# --- scalar dispatch ---------------------------------------------------------


def test_grade_scalar_numeric_requires_tolerance():
    with pytest.raises(ValueError):
        core.grade_scalar("numeric", 1400, 1414.51)


def test_grade_scalar_unknown_type():
    with pytest.raises(ValueError):
        core.grade_scalar("mystery", "x", "x")


# --- full answers ------------------------------------------------------------


def test_grade_answer_mcq():
    q = {"answer_type": "mcq", "choices": {"A": "awake", "B": "anesthetized"}}
    assert core.grade_answer(q, "B", "anesthetized")["passed"]
    assert not core.grade_answer(q, "A", "anesthetized")["passed"]


def test_grade_answer_numeric():
    q = {"answer_type": "numeric", "tolerance": 45.0}
    assert core.grade_answer(q, "1400", 1414.51)["passed"]
    assert core.grade_answer(q, "1000", 1414.51)["score"] == 0.0


def test_grade_answer_multi_partial_credit():
    q = {
        "answer_type": "multi",
        "parts": [
            {
                "key": "state",
                "answer_type": "mcq",
                "choices": {"A": "awake", "B": "anesthetized"},
            },
            {
                "key": "band",
                "answer_type": "mcq",
                "choices": {"A": "delta", "B": "theta"},
            },
        ],
    }
    truth = {"state": "anesthetized", "band": "delta"}
    assert core.grade_answer(q, {"state": "B", "band": "A"}, truth)["score"] == 1.0
    half = core.grade_answer(q, {"state": "B", "band": "B"}, truth)
    assert half["score"] == 0.5
    assert not half["passed"]
    assert core.grade_answer(q, {}, truth)["score"] == 0.0
