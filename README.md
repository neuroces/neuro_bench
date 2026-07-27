# NeuroBench

A BioMysteryBench-style evaluation benchmark for neuroscience AI, built on the [DANDI Archive](https://dandiarchive.org/).

NeuroBench evaluates whether AI models can answer neurophysiology questions whose answers are anchored to **objective, externally-validated ground truth** (anatomical localization, transcriptomic cell types, expert anesthesia/sleep scoring) rather than to subjective interpretation. Questions use **real, messy** recordings streamed directly from DANDI, and are graded on the final answer, not the analytical path.

NeuroBench also compares two **harness architectures** on the same questions under the same scorer: a single-agent `react` loop and a role-separated, PHA-style **multi-agent** system (see [Harness architectures](#harness-architectures)).

See [`project3_plan.md`](project3_plan.md) for the design rationale and [`project3_implementation_plan.md`](project3_implementation_plan.md) for the staged build plan.

## Status

Under active development. Both harness arms — single-agent `react` and the
PHA-style `multiagent` solver — are implemented and run against the currently
authored questions: **30 each in Categories 1, 2 and 3 (90 total)**. Categories 4
and 5 remain documented future work (see [`project3_plan.md`](project3_plan.md)).

## Question categories

Each category anchors its answer to an objective property of a real DANDI
recording, with the ground-truth label **withheld** from the agent so the model
must derive the answer from the signal (see [`benchmark/schema.md`](benchmark/schema.md)).

| Cat | Modality (DANDI) | Withheld label | The model must infer… | From… |
|---|---|---|---|---|
| **1** | Neuropixels extracellular spike-sorted units — IBL Brain Wide Map (`000409`) | `electrodes.location` (anatomical region) | the unit's **gross brain region** (Cortex / Hippocampus / Thalamus / Striatum / Midbrain) | peak-channel waveform trough-to-peak width, mean firing rate, and ISI statistics (firing regularity / bursting) |
| **2** | Whole-cell patch-clamp of single cortical interneurons — Patch-seq (`000035`) | `subject.genotype` (Cre driver line) | the **interneuron subtype** (Sst vs Pvalb), or the **rheobase** | current-clamp features: AP width, firing-pattern adaptation, rheobase, F-I curve |
| **3** | Cortical EEG under isoflurane — simultaneous EEG/ephys (`000458`) | `intervals` (sleep/anesthesia epoch labels) | the **brain state** (awake vs anesthetized), dominant band, or anesthesia onset | spectral analysis (band powers, dominant band) of the EEG epoch |

Ground truth for every question is *recomputed* from the full NWB file by a
resolver in [`benchmark/ground_truth.py`](benchmark/ground_truth.py) and the answer
key ([`benchmark/answers.json`](benchmark/answers.json)) is generated from those
resolvers, never hand-typed. Run one category with `--category {1,2,3}`.

## Repository layout

```
neuro_bench/
├── benchmark/      # questions.json, schema, ground-truth verification
├── data/           # NWB streaming layer + asset_manifest.csv
├── grading/        # grader + numeric tolerance + partial credit
├── agent/          # tools, model clients, prompts, run_benchmark
│   └── multiagent/ # PHA-style multi-agent solver (LangGraph state/nodes/graph)
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

## Running the benchmark on a model

Add an API key to `.env` (only the backend you run is needed):

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
```

Then run (start with a tiny sample to sanity-check cost and the agent loop):

```bash
# tiny sample (3 questions)
python -m agent.run_benchmark --model anthropic/claude-3-5-sonnet-latest --limit 3

# one category, or the full set
python -m agent.run_benchmark --model anthropic/claude-3-5-sonnet-latest --category 3
python -m agent.run_benchmark --model openai/gpt-4o

# pick the harness architecture (default: react)
python -m agent.run_benchmark --model anthropic/claude-3-5-sonnet-latest --category 3 --arch multiagent

inspect view   # review agent trajectories from the .eval logs
```

Results summaries are written to `results/<model>.json` (react) or
`results/<model>__multiagent.json` (multiagent); full trajectory logs go to `logs/`.

## Harness architectures

Both arms share the same dataset (`neurobench_dataset`) and scorer
(`neurobench_scorer`), so their scores are directly comparable. Select an arm with
`--arch {react,multiagent}` (or the `neurobench` / `neurobench_multiagent` Inspect
tasks in `agent/task.py`).

- **`react` (single-agent).** Inspect's built-in ReAct loop: one model, given the
  full tool set, alternates reasoning and tool calls until it submits an answer.

- **`multiagent` (PHA-style).** A [LangGraph](https://langchain-ai.github.io/langgraph/)
  app adapted from Google's *Personal Health Agent* (arXiv 2508.20148), living in
  `agent/multiagent/`:

  `orchestrator → data scientist → domain expert → critic → (revise ↺ | finalize)`

  Every sub-agent runs on the eval-configured model, so `--model X` uses model X
  throughout. **Integrity boundary:** only the Data Scientist node is given tools
  and can touch the recording; the orchestrator, domain expert and critic call the
  model with no tools and see only the Data Scientist's textual report. Question
  context (asset / version / redaction) stays in the Inspect store and is consumed
  only inside the tool wrappers, so no other node can open the NWB file.

## Design principles

1. **Method-agnostic, ground-truth-anchored** — graded on the final answer, derivable from an objective property of the data.
2. **Real, messy data** — actual published DANDI recordings, streamed lazily (no bulk downloads).
3. **Superhuman questions permitted** — some questions require computational analysis a human could not do quickly.
