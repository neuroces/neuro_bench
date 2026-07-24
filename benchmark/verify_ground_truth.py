"""Verify (or emit) benchmark ground-truth answers.

Two modes:

* ``--emit``  — stream each question's NWB file, recompute its answer with the
  registered resolver, and write ``answers.json`` (canonical answers + SHA-256).
  This is how the answer key is *generated*; answers are never hand-typed.
* ``--check`` (default) — recompute every answer and confirm it still matches
  ``answers.json`` (value + hash), and validate ``questions.json`` against the
  JSON schema. This is the integrity gate that guarantees every authored answer
  remains derivable from the raw data.

Both modes require network access (they stream from DANDI).

Usage::

    python -m benchmark.verify_ground_truth --emit
    python -m benchmark.verify_ground_truth            # check
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark.ground_truth import answer_hash, resolve
from data.nwb_access import open_nwb

_HERE = Path(__file__).parent
QUESTIONS_PATH = _HERE / "questions.json"
SCHEMA_PATH = _HERE / "questions.schema.json"
ANSWERS_PATH = _HERE / "answers.json"


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def load_answers(path: Path = ANSWERS_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate_schema(questions: list[dict[str, Any]]) -> None:
    """Validate questions against the JSON schema (no-op if jsonschema missing)."""
    try:
        import jsonschema
    except ImportError:
        print("  (jsonschema not installed; skipping schema validation)")
        return
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(instance=questions, schema=schema)
    print(f"  schema OK ({len(questions)} questions)")


def compute_answer(nwb, question: dict[str, Any]) -> Any:
    """Recompute a question's answer from an open NWB object."""
    if question["answer_type"] == "multi":
        return {
            part["key"]: resolve(nwb, part["ground_truth"])
            for part in question["parts"]
        }
    return resolve(nwb, question["ground_truth"])


def _group_by_asset(questions: list[dict[str, Any]]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for q in questions:
        s = q["source"]
        groups[(s["dandi_id"], s["version"], s["asset_path"])].append(q)
    return groups


def compute_all(questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve every question, opening each shared asset only once."""
    results: dict[str, Any] = {}
    for (dandi_id, version, asset_path), group in _group_by_asset(questions).items():
        with open_nwb(dandi_id, version, asset_path) as nwb:
            for q in group:
                results[q["id"]] = compute_answer(nwb, q)
    return results


def emit(questions: list[dict[str, Any]]) -> None:
    validate_schema(questions)
    computed = compute_all(questions)
    answers = {
        qid: {"answer": ans, "sha256": answer_hash(ans)}
        for qid, ans in sorted(computed.items())
    }
    ANSWERS_PATH.write_text(json.dumps(answers, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(answers)} answers to {ANSWERS_PATH.name}")
    for qid, rec in answers.items():
        print(f"  {qid}: {rec['answer']}")


def check(questions: list[dict[str, Any]]) -> int:
    validate_schema(questions)
    stored = load_answers()
    computed = compute_all(questions)

    failures = 0
    for q in questions:
        qid = q["id"]
        got = computed[qid]
        exp = stored.get(qid, {}).get("answer")
        exp_hash = stored.get(qid, {}).get("sha256")
        ok = got == exp and answer_hash(got) == exp_hash
        failures += 0 if ok else 1
        status = "OK  " if ok else "FAIL"
        detail = f"{got!r}" if ok else f"got={got!r} expected={exp!r}"
        print(f"[{status}] {qid:<8} {detail}")

    missing = set(stored) - {q["id"] for q in questions}
    for qid in sorted(missing):
        print(f"[WARN] {qid} in answers.json but not in questions.json")

    print(f"\n{len(questions) - failures}/{len(questions)} answers verified.")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify or emit benchmark ground truth."
    )
    parser.add_argument(
        "--emit", action="store_true", help="Generate answers.json from NWB."
    )
    args = parser.parse_args()

    questions = load_questions()
    if args.emit:
        emit(questions)
        return 0
    return check(questions)


if __name__ == "__main__":
    raise SystemExit(main())
