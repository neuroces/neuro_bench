"""Run NeuroBench on a model and write a results summary.

Loads API keys from ``.env``, runs the Inspect task for the chosen model /
category / limit, and writes ``results/<model>.json`` with per-question scores
and the overall accuracy. The full Inspect ``.eval`` trajectory log (reviewable
with ``inspect view``) is written under ``logs/`` and its path is recorded.

Examples::

    python -m agent.run_benchmark --model anthropic/claude-3-5-sonnet-latest --limit 3
    python -m agent.run_benchmark --model openai/gpt-4o --category 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_RESULTS_DIR = Path(__file__).parent.parent / "results"


def _safe_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def run(
    model: str,
    category: int | None,
    limit: int | None,
    max_samples: int = 4,
    message_limit: int = 40,
    arch: str = "react",
) -> dict:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from inspect_ai import eval as inspect_eval

    if arch == "multiagent":
        from agent.task import neurobench_multiagent as _task
    else:
        from agent.task import neurobench as _task

    log = inspect_eval(
        _task(category=category),
        model=model,
        limit=limit,
        max_samples=max_samples,
        message_limit=message_limit,
        fail_on_error=False,
        display="plain",
    )[0]

    results = []
    for sample in log.samples or []:
        score = next(iter(sample.scores.values())) if sample.scores else None
        results.append(
            {
                "id": sample.id,
                "category": (sample.metadata or {}).get("category"),
                "answer_type": (sample.metadata or {}).get("answer_type"),
                "score": None if score is None else score.value,
                "answer": None if score is None else score.answer,
                "truth": (sample.metadata or {}).get("truth"),
            }
        )
    scored = [r["score"] for r in results if r["score"] is not None]
    summary = {
        "model": model,
        "arch": arch,
        "category": category,
        "n_questions": len(results),
        "mean_score": round(sum(scored) / len(scored), 4) if scored else None,
        "status": log.status,
        "log_location": getattr(log, "location", None),
        "results": results,
    }

    _RESULTS_DIR.mkdir(exist_ok=True)
    suffix = "__multiagent" if arch == "multiagent" else ""
    out = _RESULTS_DIR / f"{_safe_name(model)}{suffix}.json"
    out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"\nWrote {out}")
    print(
        f"model={model}  arch={arch}  n={summary['n_questions']}  "
        f"mean_score={summary['mean_score']}"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NeuroBench on a model.")
    parser.add_argument(
        "--model", required=True, help="Inspect model id, e.g. anthropic/claude-3-5-sonnet-latest"
    )
    parser.add_argument(
        "--category", type=int, default=None, help="Restrict to one category (e.g. 3)."
    )
    parser.add_argument("--limit", type=int, default=None, help="Max number of questions to run.")
    parser.add_argument("--max-samples", type=int, default=4, help="Max concurrent samples.")
    parser.add_argument("--message-limit", type=int, default=40, help="Max messages per sample.")
    parser.add_argument(
        "--arch",
        choices=["react", "multiagent"],
        default="react",
        help="Harness architecture: single-agent react or PHA-style multiagent.",
    )
    args = parser.parse_args()
    run(
        args.model,
        args.category,
        args.limit,
        args.max_samples,
        args.message_limit,
        args.arch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
