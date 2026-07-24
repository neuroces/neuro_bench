# Question schema

Every benchmark question is a JSON object validated by
[`questions.schema.json`](questions.schema.json). This document explains the
fields and the integrity model.

## Design guarantees

1. **Answers are derivable, not looked up.** Each question carries a
   `ground_truth` spec naming a *resolver* (see [`ground_truth.py`](ground_truth.py))
   that recomputes the answer from the raw NWB file. `answers.json` is *generated*
   by running these resolvers, never hand-typed.
2. **The model cannot read the label.** The `redaction` list names parts of the
   NWB file that the agent harness must withhold at run time (e.g. `intervals`,
   which holds the sleep/anesthesia label tables). Ground-truth verification uses
   the *full* file; the model sees only the redacted signal.
3. **Integrity.** `answers.json` stores a SHA-256 of each canonical answer so a
   tampered answer key is detectable.

## Fields

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Stable id, `C<category>-Q<n>` (e.g. `C3-Q01`). |
| `category` | yes | Category number (1–5). |
| `title` | no | Short human label. |
| `prompt` | yes | Question text shown to the model. |
| `source` | yes | Which asset/window/channel the question is about (below). |
| `redaction` | no | NWB parts to withhold from the model (`intervals`/`units`/`processing`). |
| `answer_type` | yes | `mcq`, `exact`, `numeric`, or `multi`. |
| `choices` | mcq | Map of choice key → label. |
| `tolerance` | numeric | Absolute tolerance for numeric grading. |
| `points` | no | Weight (default 1). |
| `ground_truth` | non-multi | `{ "method": <resolver>, "params": {...} }`. |
| `parts` | multi | List of sub-questions, each with its own `answer_type`/`ground_truth`. |

### `source`
```json
{ "dandi_id": "000458", "version": "0.230317.0039",
  "asset_path": "sub-.../file.nwb", "channel": 0,
  "window": { "start_s": 600.0, "duration_s": 30.0 } }
```

### `ground_truth` methods (Category 3)
| Method | Returns | Params |
|---|---|---|
| `state_at_window` | `"awake"` / `"anesthetized"` | `start_s`, `duration_s` |
| `induction_onset_time` | float seconds | — |
| `eeg_channel_count` | int | — |
| `dominant_band` | `"delta"`/`"theta"`/`"alpha"`/`"beta"` | `start_s`, `duration_s`, `channel` |

## Answer types & grading

- **mcq** — graded on the choice *value* (e.g. `"anesthetized"`); the grader also
  accepts the matching choice key (e.g. `"B"`).
- **exact** — case-insensitive string match.
- **numeric** — pass if `abs(model - truth) <= tolerance`.
- **multi** — per-part scoring; question score is the mean of part scores.
