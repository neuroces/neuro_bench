# NeuroMysteryBench — Implementation Plan

Staged development plan for building the benchmark described in `project3_plan.md`.

## Guiding decisions (locked)

- **Sequencing:** *Vertical slice first.* Build **one category end-to-end** (data access → question → tools → grader → agent run → demo notebook) to de-risk the full pipeline, then replicate the proven pattern to the other four categories.
- **Pilot category:** **Category 3 — Brain State Classification from EEG (DANDI:000458)**. Chosen because it is small (24 files), exercises the signal-processing toolchain (`get_lfp_epoch`/EEG + `compute_psd`), and uses **both** grading modes (exact-match state label + numeric peak-frequency within tolerance). It is the most representative single slice of the whole system. *(Swappable — Category 2 patch-seq is the fallback if 000458 streaming proves unreliable.)*

  > **Stage-1 audit correction (decided):** 000458's actual ground truth is **awake vs. isoflurane anesthesia** (`epochs.tags` / `trials.behavioral_epoch`), *not* the Awake/NREM/REM sleep stages the original plan assumed. The sampled session (`sub-521885`) contains a 30-channel `ElectricalSeriesEEG` but no LFP series and no spike units. **Category 3 is reframed to awake-vs-anesthetized brain-state classification from EEG spectral features.** Ground truth remains real and objective.
- **Model backends:** External API keys (`anthropic`, `openai`, `google-generativeai`). Harness is written model-agnostic behind a `ModelClient` interface with a `MockModel` for offline testing so no key is required to develop/CI the grader.
- **Demo notebooks (all three):** (1) Dataset-contents explorer, (2) Sample-questions walkthrough with reference analysis, (3) Grading & evaluation demo.

---

## Stage 0 — Environment & scaffolding

**Goal:** Reproducible env and empty repo skeleton that imports cleanly.

- Create repo structure from `project3_plan.md` (benchmark/, data/, grading/, agent/, results/, analysis/, paper/, notebooks/).
- `pyproject.toml` / `requirements.txt` pinning the Key Dependencies (dandi, pynwb, remfile, mne, scipy, numpy, anthropic, openai, google-generativeai, matplotlib, pandas, jupyter, pytest).
- `.env.example` for API keys; `.gitignore` (exclude `.env`, downloaded NWB caches, results secrets).
- `README.md` stub + repo layout diagram.
- Smoke test: `pytest` collects, all packages import, `DandiAPIClient` instantiates.

**Deliverable:** Installable skeleton; CI-friendly `make test` / `make lint`.

---

## Stage 1 — Data access layer (shared foundation)

**Goal:** One reliable, cached, lazy NWB streaming utility used by every category.

- `data/nwb_access.py`: `load_nwb(dandi_id, version, asset_path)` using `remfile` + `h5py` + `pynwb` (lazy, no full download). Add on-disk LRU cache of resolved asset URLs.
- `data/asset_manifest.csv`: schema `question_id, dandi_id, version, asset_path, time_window, unit_subset, notes`.
- `data/download_assets.py`: optional pre-fetch/verify for offline runs.
- Robustness: retry/backoff on redirects, version pinning per dataset, integrity check (asset size/etag).

**Deliverable:** `load_nwb(...)` returns a usable NWB object for one 000458 asset; manifest has ≥1 real row.

---

## Stage 2 — Demo Notebook #1 (Dataset-contents explorer) — pilot dataset

**Goal:** Prove we can see and understand the data before authoring questions.

- `notebooks/01_dataset_explorer.ipynb` (starts with 000458, section-per-dataset scaffold):
  - Stream an NWB file, print metadata tree, subjects/sessions.
  - Show the **units table**, **LFP series**, and the **ground-truth `TimeIntervals`** (sleep scoring) with plots (LFP trace, spectrogram, state hypnogram).
  - Narrative: where the ground-truth label lives and why it is independent of the LFP.

**Deliverable:** Executed notebook demonstrating dataset contents + ground-truth location for the pilot.

---

## Stage 3 — Question schema + pilot questions (Category 3)

**Goal:** Formal, gradeable question format and the first 8–10 real questions.

- `benchmark/schema.md` + `questions.schema.json`: fields `id, category, dandi_id, version, asset_path, prompt, answer_type (mcq|exact|numeric|multi), choices, tolerance, ground_truth_ref, points`.
- `benchmark/questions.json`: 8–10 Category-3 questions.
- `benchmark/answers.json`: ground-truth answers, hashed for integrity; each answer traceable to a reference computation.
- **Ground-truth verification script** (`benchmark/verify_ground_truth.py`): recomputes each answer from the NWB file so authored answers are provably derivable (this is the scientific rigor gate).

**Deliverable:** 8–10 verified Category-3 questions with reproducible ground truth.

---

## Stage 4 — Agent tool set (shared)

**Goal:** The minimal controlled toolbox from the plan, method-agnostic.

- `agent/tools.py`: `load_nwb_file`, `get_spike_waveforms`, `get_lfp_epoch`, `compute_psd`, `compute_isi`, `compute_cross_correlogram`, and the `run_python` escape hatch (sandboxed subprocess, numpy/scipy/mne allowed, time/mem limits, no network).
- Tool schemas exported in Anthropic/OpenAI/Gemini tool-use formats from a single source of truth.
- Unit tests per tool against a cached 000458 asset.

**Deliverable:** Tool suite passing tests; each tool callable standalone.

---

## Stage 5 — Grading harness (shared)

**Goal:** Deterministic, well-tested grader independent of any model.

- `grading/grader.py`: dispatch by `answer_type`; Pass/Fail + partial-credit (0/0.5/1.0); emits per-question and aggregate scores + trajectory log.
- `grading/numeric_tolerance.py`: ± tolerance handling.
- `grading/partial_credit.py`: Jaccard similarity for assembly questions (used later by Cat 5).
- `pytest` suite covering exact/mcq/numeric/multi-part with fixture answers (no model or network needed → runs in CI).

**Deliverable:** Grader green on fixtures; grades the Stage-3 questions given a fixture answer set.

---

## Stage 6 — Agent runner + model clients

**Goal:** Run a model (or mock) over questions with tool use, capture trajectories.

- `agent/prompts.py`: system prompt + question template (BioMysteryBench-style, method-agnostic).
- `agent/model_clients.py`: `ModelClient` interface + `AnthropicClient`, `OpenAIClient`, `GeminiClient`, `MockModel` (scripted answers for tests).
- `agent/run_benchmark.py`: loops questions, drives tool-use loop, writes `results/<model>_results.json` (answers + full trajectory).
- End-to-end pilot with `MockModel` (offline), then one real model on the 8–10 Category-3 questions.

**Deliverable:** `results/` JSON for the pilot category from at least the mock + one real model; **vertical slice complete.**

---

## Stage 7 — Demo Notebooks #2 and #3

**Goal:** Communicate the concepts the user asked to showcase.

- `notebooks/02_sample_questions.ipynb`: per-category example question → the **reference analysis** that derives the ground-truth answer (spectral analysis for Cat 3, etc.). Shows *why the question is answerable and non-circular*.
- `notebooks/03_grading_and_eval.ipynb`: demonstrates each grading mode (exact, numeric tolerance, Jaccard), runs the grader on pilot results, and shows a mock **model-vs-human** comparison table + plots. Serves as the template for the final report.

**Deliverable:** Two executed notebooks demonstrating questions/reference analysis and grading/evaluation.

---

## Stage 8 — Replicate to remaining categories

**Goal:** Apply the proven pattern to Categories 1, 2, 4, 5.

Per category, repeat Stages 1–3 patterns (manifest rows, dataset-explorer section, questions + verified ground truth) and reuse Stages 4–6 infra. Order by ascending difficulty/data-access risk:

1. **Cat 2 — Patch-seq 000035** (single-neuron files; cell-type + F-I numeric).
2. **Cat 1 — IBL 000409** (region MCQ from waveforms; large files → tight unit-subset windows).
3. **Cat 4 — Senzai/Buzsaki 000003** (GC vs mossy; isolation-quality metrics).
4. **Cat 5 — 000552** (hard/superhuman assembly detection; exercises Jaccard partial credit).

**Deliverable:** `questions.json` complete (40–50 questions, all ground truth verified); explorer notebook covers all 5 datasets.

---

## Stage 9 — Multi-model runs + human baseline

**Goal:** The comparison the project exists to produce.

- Run Claude Opus, GPT-4o, Gemini 1.5 Pro over the full set → `results/*_results.json`.
- Human-baseline protocol + capture form (15 questions, 5 each from Cat 1–3, excluding Cat 5) → `results/human_baseline.json`.
- Cost/latency/tool-call telemetry captured per run.

**Deliverable:** Complete raw results for all models + human baseline.

---

## Stage 10 — Analysis, write-up, publication

**Goal:** Ship the public artifact.

- `analysis/comparison_report.ipynb`: accuracy by category/model, partial-credit, trajectory qualitative analysis, model-vs-human plots.
- `paper/neuromysterybench.md`: arXiv-style write-up.
- `README.md`: usage, reproduction steps, leaderboard.
- License, dataset attribution, GitHub publication checklist.

**Deliverable:** Public repo + report + paper draft.

---

## Cross-cutting

- **Testing:** grader + tools fully unit-tested with cached fixtures (CI runs offline via `MockModel`); ground-truth verification script gates every authored answer.
- **Reproducibility:** pinned DANDI dataset versions, hashed answers, deterministic tolerances, versioned results.
- **Integrity:** `answers.json` hashed; verification recomputes from raw NWB; no answer without a reproducible derivation.
- **Dependency-pin correction (Stage 1):** the plan's `remfile>=0.5.0` does not exist on PyPI; corrected to `remfile>=0.1.14` (latest) in `pyproject.toml` / `requirements.txt`.

## Milestone mapping (vs. plan's Week table)

| Plan Week | Covered by |
|---|---|
| 1 (audit/scoping) | Stages 0–2 |
| 2–3 (question writing) | Stages 3, 8 |
| 4 (tools/harness/grader) | Stages 4–6 |
| 5 (runs + human baseline) | Stage 9 |
| 6 (analysis/publish) | Stage 10 |
| (added) demo notebooks | Stages 2, 7 |

## Open items to confirm before Stage 8+

- Sandbox strategy for `run_python` (subprocess vs container) given untrusted model-generated code.
- Human-baseline recruitment feasibility (3–5 neuro PhDs) — affects Stage 9 timeline.
- Exact pinned versions for 000409 / 000552 (largest / hardest datasets).
