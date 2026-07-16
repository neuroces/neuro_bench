# NeuroMysteryBench

A BioMysteryBench-style evaluation benchmark for neuroscience AI, built on the [DANDI Archive](https://dandiarchive.org/).

NeuroMysteryBench evaluates whether AI models can answer neurophysiology questions whose answers are anchored to **objective, externally-validated ground truth** (histology, transcriptomic cell types, expert sleep scoring, birthdate labeling) rather than to subjective interpretation. Questions use **real, messy** recordings streamed directly from DANDI, and are graded on the final answer, not the analytical path.

See [`project3_plan.md`](project3_plan.md) for the design rationale and [`project3_implementation_plan.md`](project3_implementation_plan.md) for the staged build plan.

## Status

Stage 0 (scaffolding + environment) complete. Under active development.

## Repository layout

```
neuro_bench/
├── benchmark/      # questions.json, schema, ground-truth verification
├── data/           # NWB streaming layer + asset_manifest.csv
├── grading/        # grader + numeric tolerance + partial credit
├── agent/          # tools, model clients, prompts, run_benchmark
├── notebooks/      # demo notebooks (explorer, sample questions, grading)
├── results/        # model + human-baseline run outputs
├── analysis/       # comparison_report.ipynb
├── paper/          # arXiv-style write-up
└── tests/          # unit + smoke tests
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
make install-dev          # editable install + dev tools
cp .env.example .env      # add API keys for the backends you plan to run
```

## Testing

```bash
make test-offline   # CI-safe: skips network/DANDI-dependent tests (uses MockModel)
make test           # full suite (may stream from DANDI)
```

## Design principles

1. **Method-agnostic, ground-truth-anchored** — graded on the final answer, derivable from an objective property of the data.
2. **Real, messy data** — actual published DANDI recordings, streamed lazily (no bulk downloads).
3. **Superhuman questions permitted** — some questions require computational analysis a human could not do quickly.
