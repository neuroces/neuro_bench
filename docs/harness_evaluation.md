# Harness evaluation: Inspect AI vs. custom (decision record)

**Question:** Should NeuroBench use an open-source evaluation harness
(Inspect AI) instead of hand-rolled grading + agent + sandbox code?

**Method:** Grounded in live sources (fetched 2026-07-16): Inspect AI PyPI
metadata and the current docs index (`llms.txt`), not memory.

## What Inspect AI actually provides (verified, v0.3.247)

- **Python ≥3.10** (we run 3.12 ✓); ~105 dependencies; core stack httpx +
  pydantic + rich + textual. Provider SDKs (`anthropic`, `openai`,
  `google-genai`) are pluggable — install per backend you use.
- **Tasks = datasets + solvers + scorers.** Clean separation matching our needs.
- **Models:** one uniform API across providers, with caching, concurrency
  control, cost/token/time limits, and fallbacks.
- **Tools:** custom `@tool` functions; **standard code-execution tools**;
  **sandboxing** to isolate model-generated code, with **local or container
  (Docker) runtimes** — directly solves our `run_python` hardening open item.
- **Scorers:** standard text-match + multiple-choice; **custom scorers**
  (`Score`/`Value`/`Target`); model-graded; multiple-scorers; metrics;
  deferred/re-scoring workflow.
- **Agents:** built-in ReAct agent, custom agents, and a **`human_cli()` human
  agent for human baselining** — relevant to our Stage-9 human baseline.
- **Logging/viewer:** `.eval` trajectory logs + `inspect view` UI for free.

## Mapping to our stages

| Our component | Inspect equivalent | Verdict |
|---|---|---|
| `data/` (NWB streaming, registry, manifest) | — (outside Inspect) | **Keep as-is** |
| `benchmark/ground_truth.py` resolvers | — (our science) | **Keep as-is** |
| `benchmark/verify_ground_truth.py` | — (harness-independent) | **Keep as-is** |
| `questions.json` / `answers.json` | `Dataset` of `Sample`s (input=prompt, target=answer, metadata=source/window/redaction) | Thin adapter |
| `agent/tools.py` (`ToolSession`) | `@tool` functions + sandbox provisioning | **Rework** (logic ports; reuses `ground_truth` helpers) |
| `agent/sandbox.py` | Inspect sandbox (Docker/local) | **Delete** (net simpler, gains Docker) |
| `agent/tool_specs.py` (provider exports) | Inspect generates tool schemas | **Delete** |
| Stage 5 grader | 3–4 custom scorers (choice, numeric-tolerance, Jaccard, multi) | **Rework** (small) |
| Stage 6 runner + model clients | `Task` + `react` solver + model providers | **Rework** (net simpler) |
| Stage 9 human baseline | `human_cli()` on the *same* tasks/tools | **Bonus** (fairer baseline) |

### Redaction under Inspect
Our integrity model (the model must not read the label tables) maps cleanly:
we only ever place the **allowed, redacted signal** into the sample/sandbox —
never the raw NWB — so labels are unreachable by construction. This is the same
guarantee our `ToolSession._guard` provides today, realized via data
provisioning instead.

## Cost / benefit

**Keep regardless of decision (the hard, scientific core):** `data/`,
`benchmark/` resolvers + verification, and the explorer notebook. These are
harness-agnostic and already done + tested.

**Benefits of adopting Inspect:**
- Deletes our sandbox + provider-spec code; solves `run_python` hardening
  (Docker) — a flagged open item.
- Multi-provider model layer with caching, concurrency, and cost limits (Stage 9).
- Trajectory logs + viewer + standard metrics (Stage 6/10) for free.
- `human_cli()` gives a same-tools human baseline (Stage 9).
- Aligns with the "Anthropic/AISI evaluation philosophy" framing of the project.

**Costs / risks of adopting:**
- Heavy dependency (~105 pkgs) and **Docker required** for the hardened sandbox
  (a local sandbox exists but is less isolated).
- Reworks the Stage-4 tool layer we just built (sunk cost — but resolvers/verify
  are preserved).
- Couples the benchmark to Inspect's task/sample shape (portability tradeoff).
- Some non-standard glue (redaction-by-provisioning) still custom.

## Recommendation

**Adopt Inspect AI for the agent / model / scorer / sandbox layer; keep
`data/` and `benchmark/` (resolvers + verification) as the harness-agnostic
scientific core.** Grading logic is trivial either way, so the deciding factor
is the *agent + sandbox + multi-provider* layer — exactly where Inspect removes
the most custom, error-prone code and hands us Docker isolation, cost controls,
trajectory logging, and a same-tools human baseline.

**Choose custom-only instead if** we want to avoid the Docker requirement and
the large dependency, and prefer full control over a small, already-working
tool+grader path.

## Decision (2026-07-16)

**Adopted Inspect AI** for the eval framework, via an **incremental** path:

- **Use Inspect now for:** dataset/`Sample` loading, the `react` agent + solver,
  custom `scorer`s (wrapping our pure `grading/` engine), the multi-provider
  model layer (`mockllm` for offline tests; anthropic/openai/google-genai for
  real runs), and `.eval` trajectory logs + `inspect view`.
- **Reuse (tested) for the tool compute engine:** `agent/tools.py::ToolSession`
  and `agent/sandbox.py` (subprocess isolation for `run_python`), wrapped as
  Inspect `@tool`s. This keeps the redaction guarantee we already tested.
- **Deferred:** swapping `run_python`'s subprocess sandbox for Inspect's
  **Docker** sandbox (hardening) — a follow-up once the offline slice is green.
- **Kept harness-agnostic (unchanged):** `data/` and `benchmark/` (resolvers +
  `verify_ground_truth.py`).

Rationale: delivers the main Inspect benefits (multi-provider, agent loop,
scorers, trajectory viewer, `human_cli` baseline) immediately and offline-
testable, without blocking on Docker, while not discarding tested compute code.
