"""Tests for the benchmark question set and ground-truth harness."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from benchmark import ground_truth, verify_ground_truth

BENCH = Path(__file__).parent.parent / "benchmark"


@pytest.fixture(scope="module")
def questions():
    return verify_ground_truth.load_questions()


@pytest.fixture(scope="module")
def answers():
    return verify_ground_truth.load_answers()


# --- schema & structure (offline) -------------------------------------------


def test_questions_validate_against_schema(questions):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((BENCH / "questions.schema.json").read_text())
    jsonschema.validate(instance=questions, schema=schema)


def test_question_ids_unique_and_well_formed(questions):
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"C\d+-Q\d+", qid) for qid in ids)


def test_ground_truth_methods_are_registered(questions):
    for q in questions:
        specs = (
            [p["ground_truth"] for p in q["parts"]]
            if q["answer_type"] == "multi"
            else [q["ground_truth"]]
        )
        for spec in specs:
            assert spec["method"] in ground_truth.RESOLVERS


def test_mcq_and_numeric_have_required_fields(questions):
    for q in questions:
        if q["answer_type"] == "mcq":
            assert q.get("choices")
        if q["answer_type"] == "numeric":
            assert q.get("tolerance") is not None


# --- answer-key integrity (offline) -----------------------------------------


def test_every_question_has_an_answer(questions, answers):
    assert {q["id"] for q in questions} <= set(answers)


def test_answer_hashes_match_stored_values(answers):
    for qid, rec in answers.items():
        assert ground_truth.answer_hash(rec["answer"]) == rec["sha256"], qid


def test_state_questions_are_balanced(answers):
    states = [
        rec["answer"]
        for qid, rec in answers.items()
        if rec["answer"] in ("awake", "anesthetized")
    ]
    assert states.count("awake") == states.count("anesthetized")


# --- ground truth re-derivation (network) -----------------------------------


@pytest.mark.network
def test_recomputed_answers_match_key(questions, answers):
    computed = verify_ground_truth.compute_all(questions)
    for qid, got in computed.items():
        assert got == answers[qid]["answer"], qid
