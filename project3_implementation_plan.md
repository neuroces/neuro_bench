# NeuroBench — Implementation Plan

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
- `paper/neurobench.md`: arXiv-style write-up.
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

## Progress log

- **Stage 0–4 complete** (data layer, question set + verification, tool engine).
- **Harness decision:** adopted **Inspect AI** for the eval framework (see
  `docs/harness_evaluation.md`).
- **Stage 5 (grading) complete:** pure engine in `grading/` (numeric tolerance,
  Jaccard partial credit, mcq/exact/numeric/multi dispatch) wrapped by an Inspect
  `scorer` (`agent/scorers.py`). Fully offline-tested.
- **Stage 6 (agent runner) largely delivered via Inspect:** `agent/dataset.py`
  (questions → Samples), `agent/task.py` (setup solver + `react` agent + scorer),
  `agent/inspect_tools.py` (@tool wrappers over the tested `ToolSession`),
  `agent/context.py` (sample-scoped redaction context). Validated end-to-end
  offline (scripted mock) and with the full react+tools loop over real data.
- **Remaining for Stage 6/9:** real-model runs (needs API keys) + `inspect view`
  trajectory review; optional `human_cli` baseline.
- **Stage 7 (demo notebooks) complete:** `01_dataset_explorer`, `02_sample_questions`
  (reference analysis + verification, incl. anesthesia-onset detection), and
  `03_grading_and_eval` (grading modes, simulated-model report, Inspect-harness
  scoring, model-vs-human template). All executed end-to-end, 0 errors.
- **Question refinement (Stage 7):** Q09 reframed from induction-onset (gas-on,
  not EEG-resolvable within tight tolerance) to **anesthesia onset** (isoflurane
  _anesthesia epoch start, 1684.09 s, ±120 s); added `anesthesia_onset_time`
  resolver. Reference detector uses the delta-power surge (detected 1710 s ✓).
- **Stage 8 (Category 2) complete:** 10 patch-seq questions on DANDI:000035.
  Audit correction — ground truth is the **Cre genotype** (not an embedded
  t-type), and the dataset's cells are **Sst vs Pvalb** interneurons, so Cat 2 is
  fast-spiking (Pvalb) vs adapting (Sst) classification + rheobase. Added
  resolvers `cell_subtype`/`cell_class`/`rheobase`/`firing_rate_at_current`,
  icephys `ToolSession` tools (`list_current_clamp_sweeps`,
  `get_current_clamp_sweep`), `subject` redaction, and category-based tool
  selection. 20/20 answers verified; Cat-2 react loop validated.
- **Stage 9 (first real run):** ran Claude `claude-sonnet-5` on a 3-question Cat-3
  sample via `agent/run_benchmark.py`. Found + fixed a grading bug: the `react`
  agent submits prose, so verbose answers ("A) Awake — ...") were mis-scored 0;
  `extract_answer` now recovers the choice letter. Sample result: **2/3**
  (Q01✓ Q02✓ Q03✗). Refinement flags: Q03 is a genuine model miss (awake-like
  spectrum); **Q04 is spectrally atypical** for anesthesia (alpha-dominant,
  delta/total 0.17) — review/swap before scaling up. Model access: valid ids are
  e.g. `claude-sonnet-5` (not `claude-3-5-sonnet-latest`, which 404s).
- **Anesthesia-window refinement:** audited all Cat-3 anesthesia windows by
  delta/total. sub-521886 (Q04) was alpha/spindle-dominated throughout (max 0.25)
  so Q04 moved to **sub-551399 @ 2210 s (0.99)**; Q08 moved to **sub-551397 @
  4335 s (0.59)**; Q06 → **1581 s (0.76)**; Q10 → **3502 s (0.89)**; Q02 kept
  (0.78). All still verify as anesthetized (Q10 delta). 64 offline tests pass.
- **Full 20-question run (claude-sonnet-5):** overall **0.70** — Cat-3 **0.90**
  (9/10; only Q07 missed), Cat-2 **0.50** (5/10). Notable finding: the model has
  a **Pvalb bias** — it labeled all 4 Sst cells as Pvalb (0/4 Sst, 3/4 Pvalb);
  rheobase + multi-part correct. Two more harness bugs found & fixed: (1) multi
  answer hint now lists the real part keys (was "key1/key2" → Q10 mis-scored);
  (2) extraction tolerates `<answer>` tags / stray `</invoke>` (was scoring a
  correct C3-Q01 as `</invoke>`). Also: `@tool` wrappers now return errors
  gracefully and the runner uses `fail_on_error=False` + `max_samples` (20
  concurrent samples thrashed DANDI streaming). 65 offline tests pass.

## Open items to confirm before Stage 8+

- Swap `run_python`'s subprocess sandbox for Inspect's **Docker** sandbox (hardening).
- Human-baseline recruitment feasibility (3–5 neuro PhDs) — affects Stage 9 timeline.
- Exact pinned versions for 000409 / 000552 (largest / hardest datasets).
