# Self-improving agents (prompt optimization) — side-branch sketch

**Status:** Sketch / parked. Not started. Return to this if hand-written prompt
refinement proves insufficient.

**Question:** Can we automatically refine the agents' prompts for each category
(e.g. reduce the Cat-1 "default to Thalamus" bias) by letting each role iterate
against a held-out set of questions — without compromising the benchmark's
integrity?

## Motivation

NeuroBench compares two harnesses (`react`, `multiagent`) under one scorer. Prompt
quality strongly affects both — the Cat-1 pilot showed a systematic bias (models
label most extracellular units "Thalamus"). We currently fix this by hand-editing
`CAT1_PROMPT` (react) and `_DOMAIN_HINTS` (multiagent). A self-improving branch
would instead *learn* the prompts from data, and let us ask a research question:
**does automated prompt optimization close (or widen) the react-vs-multiagent gap?**

## Hard constraint: integrity first

The entire value of NeuroBench is that scores are honest and un-gamed. Automated
prompt optimization is a form of training, so it must obey a strict split:

- **The optimizer must never see the eval questions.** Freeze a `train` split it
  iterates on and a **disjoint** `eval` split it never touches; report only
  eval-split scores.
- **Blocker: question volume.** With ~10 questions/category, a train/eval split
  leaves too few to evaluate on. **Step one is authoring more questions**
  (target ~20–30/category), not building the optimizer.
- **Fairness across arms.** Apply the *same* optimizer and *same* budget to both
  `react` and `multiagent`. If one arm gets tuned prompts and the other hand-written
  ones, the architecture comparison is confounded — the very thing we measure.
- **Overfitting watch.** Our chosen units skew high-firing; an optimizer may learn
  dataset-specific thresholds rather than physiology. The held-out split is the
  guard; also sanity-check learned prompts read as real neuroscience.

## Design (incremental, simplest first)

Keep the scored benchmark on fixed, human-written prompts. Do optimization on a
**branch/flag**, reusing the existing scorer as the reward signal.

### Phase A — Splits + harness plumbing
- Add a `split` field (or a deterministic hash-based partition) to
  `benchmark/questions.json`; expose `neurobench_dataset(category, split=...)`.
- A small `optimize/` module: given a prompt candidate, run the arm on the *train*
  split via the existing `run_benchmark` path and return `mean_score` as reward.

### Phase B — Candidate-based optimization (APE/OPRO-style, no new deps)
- For each *role* prompt (react system prompt; multiagent per-node prompts), use
  the model itself to propose N candidate rewrites given the current prompt +
  train-split failures ("here are the misclassifications; propose a better prompt").
- Score each candidate on the train split; keep the best; repeat for a few rounds.
- This is a shallow search — cheap, transparent, and easy to audit.

### Phase C — Per-role credit assignment (the hard part)
- Only the final answer is scored, so improving one multiagent role in isolation is
  noisy. Options: (1) optimize one role at a time holding others fixed
  (coordinate ascent); (2) hold the other roles on a strong fixed model and vary
  only the target role; (3) later, adopt a framework (e.g. DSPy) that handles
  multi-module optimization — only if Phases B/C justify the dependency.

### Phase D — Report
- Evaluate the frozen best prompts on the **eval** split for both arms; compare to
  the hand-written baselines. Emit a short report (train vs eval, per-category,
  per-arm) so gains are attributable and overfitting is visible.

## Out of scope / non-goals
- No optimizer output enters the scored benchmark by default; the main results
  stay on human-written prompts unless we deliberately adopt an optimized set and
  document it.
- No RL / weight training — prompts only.
- Not a replacement for authoring good questions; it depends on that first.

## Open questions
- How many questions/category make a train/eval split statistically meaningful?
- Do we optimize a single shared prompt per role, or category-specific prompts
  (current design is category-specific via `_prompt_for` / `_DOMAIN_HINTS`)?
- Is the goal better *absolute* accuracy, or a *fairer/cleaner* architecture
  comparison? These can pull the design in different directions.

## First concrete step (when we return)
Author ~20–30 Cat-1 questions spanning more regions and firing regimes (including
low-rate striatal/cortical units, not just high-spike-count ones), then revisit
Phase A. Until then, the hand-written `CAT1_PROMPT` + `_DOMAIN_HINTS` are the
baseline to beat.
