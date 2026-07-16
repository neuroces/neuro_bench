# Project 3: NeuroMysteryBench
## A BioMysteryBench-Style Evaluation Dataset for Neuroscience AI, Built on DANDI

**Estimated timeline:** 6–8 weeks (solo, part-time)  
**Primary output:** A public GitHub repository containing a benchmark dataset, grading harness, and a model comparison report  
**Demonstrates:** Benchmark design, scientific evaluation methodology, neuroscience domain depth, Anthropic-aligned evaluation philosophy

---

## Design Philosophy

This project directly replicates the core design principles of Anthropic's BioMysteryBench, applied to neurophysiology data from the DANDI Archive. The three principles are:

1. **Method-agnostic, ground-truth-anchored questions.** Questions are graded on the final answer, not the analytical path. The answer must be derivable from a controllable, objective property of the data — not from a scientist's subjective interpretation.
2. **Real, messy data.** All questions use actual experimental recordings from published DANDI datasets, not simulated or cleaned data. This preserves the noise and ambiguity of real neuroscience.
3. **Superhuman question generation is permitted.** Some questions may be difficult or impossible for a human to answer without computational tools. This is a feature, not a bug — it tests the model's ability to use code and analysis pipelines, not just recall.

---

## Dataset Selection from DANDI

The following four DANDI datasets are selected as the primary data sources. Each was chosen because it has (a) a published paper providing external validation of the ground truth, (b) structured metadata that can serve as objective labels, and (c) enough sessions/subjects to generate multiple independent questions.

| DANDI ID | Dataset Name | Modality | Ground Truth Source | Files |
|---|---|---|---|---|
| **000409** | IBL Brain Wide Map | Neuropixels extracellular ephys | Brain region labels from histology + Allen CCF atlas | 2,048 |
| **000035** | Patch-seq mouse motor cortex (Gouwens et al.) | Patch-seq (ephys + transcriptomics) | Cell type labels from transcriptomic clustering | 185 |
| **000003** | Hippocampal granule cells and mossy cells (Senzai & Buzsaki) | Tetrode ephys + LFP | Cell type labels (GC vs. mossy cell) from waveform + anatomy | 101 |
| **000458** | Simultaneous EEG + extracellular ephys + cortical imaging | Multi-modal (EEG, LFP, spikes) | Brain state labels (awake, NREM, REM) from EEG scoring | 24 |

A fifth dataset is used for a harder, "superhuman" question category:

| DANDI ID | Dataset Name | Modality | Ground Truth Source | Files |
|---|---|---|---|---|
| **000552** | Hippocampus embryonic birthdate (Bhaskaran et al.) | Tetrode ephys | Birthdate-ordered cell assembly identity from metadata | 117 |

---

## Question Design: Five Categories

The benchmark is organized into five question categories, each drawing from one primary dataset. Each category contains 8–10 questions, for a total of **40–50 questions**. Questions are written so that the answer is a short string, number, or multiple-choice selection that can be automatically graded.

---

### Category 1: Brain Region Identification from Spike Waveforms
**Dataset:** DANDI:000409 — IBL Brain Wide Map  
**Ground truth:** Brain region labels assigned by histological verification against the Allen Common Coordinate Framework (CCF). These labels are stored in the NWB file metadata and were validated by the IBL's own quality control pipeline.

**Question type:** Given a set of spike waveforms and firing rate statistics extracted from a single recording session, identify the most likely brain region from which the unit was recorded.

**Example question:**
> *The attached NWB file contains extracellular recordings from a single probe insertion. The file includes spike times, mean waveforms, and inter-spike interval distributions for 47 isolated units. Based on the electrophysiological properties of these units, which of the following brain regions does this probe most likely traverse? (A) Primary visual cortex (VISp), (B) Dorsal striatum (CP), (C) Hippocampus CA1, (D) Anterior cingulate cortex (ACA)*

**Why this works:** The IBL used histological verification as the ground truth for brain region assignment. The model cannot simply look up the answer — it must analyze the waveform shapes, firing rates, and ISI distributions computationally and reason about which brain region they are consistent with.

**Grading:** Exact match against the histologically verified label stored in `electrodes.location` in the NWB file.

**Access:**
```python
import dandi.dandiapi as da
client = da.DandiAPIClient()
dandiset = client.get_dandiset("000409", "0.260309.1324")
# Stream individual NWB files without full download
asset = next(dandiset.get_assets_by_glob("*probe*"))
```

---

### Category 2: Cell Type Classification from Patch-seq Electrophysiology
**Dataset:** DANDI:000035 — Temperature-controlled Patch-seq recordings (Gouwens et al., *Nature Neuroscience* 2020)  
**Ground truth:** Cell type labels derived from transcriptomic clustering (t-types) published in the accompanying paper. Each NWB file contains a single neuron's electrophysiological features (action potential shape, firing pattern, input resistance) alongside its assigned t-type from RNA sequencing.

**Question type:** Given the electrophysiological features of a single neuron (action potential waveform, F-I curve, subthreshold properties), classify it as excitatory or inhibitory, and identify its most likely subtype.

**Example question:**
> *The attached NWB file contains a patch-clamp recording from a single neuron in mouse motor cortex. The recording includes responses to 15 current injection steps from -100 pA to +500 pA. Based solely on the electrophysiological response properties, is this neuron (A) an excitatory pyramidal cell, (B) a fast-spiking parvalbumin interneuron, (C) a somatostatin interneuron, or (D) a VIP interneuron? Report the firing rate at +300 pA injection.*

**Why this works:** The t-type labels were derived from independent transcriptomic data, not from the electrophysiology itself. This means the electrophysiology is a genuine predictor of the ground truth, but the ground truth is not circular. The model must perform actual analysis of the waveform and firing pattern.

**Grading:** Two-part: (1) exact match on excitatory/inhibitory classification, (2) exact match on subtype (from a controlled vocabulary of 5 subtypes), (3) numeric answer for firing rate within ±5 Hz.

**Access:**
```python
dandiset = client.get_dandiset("000035", "0.211014.0808")
# Each file is a single neuron; metadata includes t-type label
for asset in dandiset.get_assets_by_glob("*.nwb"):
    # asset.get_metadata() contains cell_type field
    pass
```

---

### Category 3: Brain State Classification from LFP
**Dataset:** DANDI:000458 — Simultaneous EEG, extracellular ephys, and cortical imaging  
**Ground truth:** Sleep/wake state labels (Awake, NREM, REM) scored from EEG by a human expert and stored in the NWB file as `TimeIntervals`. This is the gold standard for sleep scoring in rodents.

**Question type:** Given a 30-second epoch of local field potential (LFP) data from hippocampus or cortex, classify the brain state.

**Example question:**
> *The attached file contains a 30-second LFP recording from mouse hippocampus (CA1). Compute the power spectral density of this epoch. Based on the dominant frequency band and the theta/delta power ratio, classify this epoch as: (A) Active waking, (B) Quiet waking, (C) NREM sleep, or (D) REM sleep. Report the peak frequency in Hz.*

**Why this works:** Sleep state labels are assigned by human experts from EEG, independently of the LFP. The LFP contains highly characteristic signatures (theta oscillations during REM/active waking, delta waves during NREM, sharp-wave ripples during quiet waking) that can be detected computationally. The model must run spectral analysis to answer correctly.

**Grading:** Exact match on state classification; numeric answer for peak frequency within ±1 Hz.

**Access:**
```python
dandiset = client.get_dandiset("000458", "0.230317.0039")
# TimeIntervals table contains sleep scoring ground truth
```

---

### Category 4: Unit Quality and Spike Sorting Validation
**Dataset:** DANDI:000003 — Hippocampal granule cells and mossy cells (Senzai & Buzsaki, *Neuron* 2017)  
**Ground truth:** Cell type labels (granule cell vs. mossy cell) assigned by the authors using a combination of waveform shape, firing rate, and anatomical location. These labels are stored in the NWB file and were validated by juxtacellular labeling in a subset of cells.

**Question type:** Given spike waveform features and firing statistics for a set of units recorded in the dentate gyrus, distinguish granule cells from mossy cells and identify units that likely represent multi-unit activity (poor isolation quality).

**Example question:**
> *The attached NWB file contains 23 isolated units from the dentate gyrus of a mouse. For each unit, you have access to the mean spike waveform, peak-to-trough ratio, half-width, and inter-spike interval distribution. (1) How many of these units are classified as mossy cells? (2) Which unit ID has the highest isolation distance score? (3) What is the mean firing rate of the granule cell population during the theta phase?*

**Why this works:** The granule cell vs. mossy cell distinction is validated by juxtacellular labeling — an independent anatomical ground truth. Isolation quality metrics (L-ratio, isolation distance) are computed from the spike waveform PCA and are objective numerical values. The model must run actual spike sorting quality analysis.

**Grading:** Exact integer match for cell counts; unit ID match; numeric answer for firing rate within ±0.5 Hz.

---

### Category 5: Temporal Structure and Assembly Detection (Hard / Potentially Superhuman)
**Dataset:** DANDI:000552 — Preconfigured dynamics in the hippocampus guided by embryonic birthdate (Bhaskaran et al., *Science* 2023)  
**Ground truth:** Cell assembly membership derived from the embryonic birthdate of each neuron, stored as metadata. Neurons born on the same embryonic day tend to fire together in coordinated assemblies — a ground truth that was established by BrdU birthdating, an independent experimental method.

**Question type:** Given spike trains from a population of hippocampal neurons, identify which neurons belong to the same coordinated assembly, and determine the embryonic birthdate group they most likely correspond to.

**Example question:**
> *The attached NWB file contains spike trains from 68 CA1 pyramidal neurons recorded during a 20-minute open-field session. Using the spike train cross-correlations and pairwise co-firing rates, identify which neurons form the most coherent assembly (highest pairwise correlation). The metadata contains embryonic birthdate labels for each neuron. What is the embryonic day (E14, E15, E16, or E17) of the majority of neurons in the assembly you identified?*

**Why this works:** The embryonic birthdate labels were assigned by BrdU injection on specific embryonic days — a completely independent experimental procedure from the electrophysiology. The model must perform assembly detection (a non-trivial computational task) and then match its result against the metadata ground truth. This is a "superhuman" question in the BioMysteryBench sense: the pattern is real and verifiable, but detecting it requires computational analysis that most humans could not perform quickly.

**Grading:** Exact match on embryonic day label; partial credit for identifying the correct assembly members (Jaccard similarity ≥ 0.6).

---

## Technical Architecture

### Repository Structure

```
neuromysterybench/
├── README.md
├── benchmark/
│   ├── questions.json          # All 45 questions with metadata
│   ├── answers.json            # Ground truth answers (hashed for integrity)
│   └── category_descriptions.md
├── data/
│   ├── download_assets.py      # Downloads NWB files from DANDI
│   └── asset_manifest.csv      # Maps question_id → DANDI asset path
├── grading/
│   ├── grader.py               # Automated grading harness
│   ├── numeric_tolerance.py    # Handles ±tolerance for numeric answers
│   └── partial_credit.py       # Jaccard similarity for assembly questions
├── agent/
│   ├── run_benchmark.py        # Runs Claude on all questions
│   ├── tools.py                # NWB reading, spike analysis, LFP tools
│   └── prompts.py              # System prompt and question templates
├── results/
│   ├── claude_opus_results.json
│   ├── gpt4o_results.json
│   └── human_baseline.json
├── analysis/
│   └── comparison_report.ipynb # Jupyter notebook with full analysis
└── paper/
    └── neuromysterybench.md    # Write-up suitable for arXiv submission
```

### Data Access Strategy

All data is streamed directly from DANDI using the `dandi` Python client and `remfile` for lazy NWB loading — no full dataset downloads are required. Each question references a specific asset by its DANDI path and a specific time window or unit subset, keeping the data footprint minimal.

```python
# Example: lazy NWB streaming (no full download)
import remfile
import h5py
import pynwb

asset_url = asset.get_content_url(follow_redirects=1, strip_query=True)
rem = remfile.File(asset_url)
h5_file = h5py.File(rem)
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()
```

### Agent Tool Set

The Claude agent is given a minimal, controlled tool set — mirroring BioMysteryBench's approach of providing "canonical tools" without over-specifying the method:

| Tool | Purpose |
|---|---|
| `load_nwb_file(dandi_id, asset_path)` | Streams and returns NWB file metadata + unit table |
| `get_spike_waveforms(nwb, unit_ids)` | Returns mean waveforms for specified units |
| `get_lfp_epoch(nwb, start_time, duration, channel)` | Returns LFP time series |
| `compute_psd(signal, fs)` | Computes power spectral density |
| `compute_isi(spike_times)` | Computes inter-spike interval distribution |
| `compute_cross_correlogram(spike_times_a, spike_times_b)` | Pairwise cross-correlation |
| `run_python(code)` | Executes arbitrary Python (numpy, scipy, MNE) for custom analysis |

The `run_python` tool is the key escape hatch — it allows the model to perform any analysis not covered by the named tools, preserving method-agnosticism.

---

## Evaluation Protocol

### Grading

Each question has a primary answer (exact match or within numeric tolerance) and optionally a secondary answer for partial credit. The grading script outputs:

- **Pass/Fail** per question
- **Partial credit score** (0, 0.5, or 1.0) for multi-part questions
- **Trajectory log** (all tool calls and reasoning steps) for qualitative analysis

### Human Baseline

Recruit 3–5 neuroscience PhD students or postdocs to attempt a random subset of 15 questions (5 per category, excluding Category 5) using only Python and the DANDI client — no LLM assistance. Record time-to-answer and accuracy. This provides the human baseline for comparison.

### Models to Evaluate

Run the benchmark on at least three models to make the comparison meaningful:

| Model | Rationale |
|---|---|
| Claude Opus (latest) | Primary target; Anthropic's flagship model |
| GPT-4o | Strong baseline; widely used in science |
| Gemini 1.5 Pro | Google's science-focused model; used in Co-Scientist |

---

## Milestone Timeline

| Week | Milestone | Deliverable |
|---|---|---|
| 1 | Dataset audit and question scoping | `asset_manifest.csv` + draft question list |
| 2 | Question writing (Categories 1–3) | 25 questions with ground truth verified |
| 3 | Question writing (Categories 4–5) | 20 more questions; `questions.json` complete |
| 4 | Tool implementation and agent harness | `tools.py`, `run_benchmark.py`, `grader.py` |
| 5 | Run Claude and GPT-4o; collect human baseline | Raw results JSON files |
| 6 | Analysis, write-up, and GitHub publication | `comparison_report.ipynb`, `neuromysterybench.md` |

---

## Why This Project Stands Out

This project is specifically calibrated to demonstrate the skills Anthropic's Beneficial Deployments team values most. It directly mirrors the methodology of BioMysteryBench — the benchmark Anthropic built internally to evaluate Claude on bioinformatics — but applies it to neurophysiology, your domain of expertise. The combination of (1) deep neuroscience knowledge required to design valid questions, (2) engineering skill required to build the grading harness and agent tools, and (3) evaluation methodology aligned with Anthropic's own approach makes this a uniquely compelling portfolio artifact.

Publishing the benchmark publicly on GitHub and arXiv also creates a reusable community resource, which aligns with the Beneficial Deployments team's mission of building "ecosystem-level tooling" that scales beyond a single institution.

---

## Key Dependencies

```
dandi>=0.62.0        # DANDI client for streaming NWB files
pynwb>=2.7.0         # NWB file reading
remfile>=0.5.0       # Lazy remote file streaming (no full download)
mne>=1.7.0           # EEG/LFP analysis (PSD, filtering)
scipy>=1.13.0        # Signal processing
numpy>=1.26.0        # Numerical computation
anthropic>=0.30.0    # Claude API with tool use
matplotlib>=3.9.0    # Visualization
pandas>=2.2.0        # Data management
```

---

## References

- Senzai, Y. & Buzsáki, G. (2017). Physiological properties and behavioral correlates of hippocampal granule cells and mossy cells. *Neuron*, 93(3), 691–704. [DANDI:000003]
- Gouwens, N.W. et al. (2020). Integrated morphoelectric and transcriptomic classification of cortical GABAergic cells. *Cell*, 183(4), 935–953. [DANDI:000035]
- International Brain Laboratory (2023). A brain-wide map of neural activity during complex behaviour. *Nature*, 623, 745–754. [DANDI:000409]
- Bhaskaran, S. et al. (2023). Preconfigured dynamics in the hippocampus are guided by embryonic birthdate and rate of neurogenesis. *Science*, 381(6660). [DANDI:000552]
- Senzai, Y. et al. (2019). Layer-specific physiological features and interlaminar interactions in the primary visual cortex of the mouse. *Neuron*, 101(3), 500–513. [DANDI:000458]
- Anthropic. (2026). Evaluating Claude's bioinformatics research capabilities with BioMysteryBench. https://www.anthropic.com/research/Evaluating-Claude-For-Bioinformatics-With-BioMysteryBench
