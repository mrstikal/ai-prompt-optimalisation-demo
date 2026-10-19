# DSPy Prompt Optimization Lab

Standalone lab for reviewing Czech precious-metals articles with one DSPy
predictor and comparing a frozen zero-shot baseline with a genuine GEPA
instruction-optimized program.

**Status: S1 only.** This stage establishes the isolated Python environment,
dependency lock, package skeleton, and configuration contract. It contains no
UI, dataset, predictor, metrics, provider calls, or experiment results. No paid
API call has been made.

## Requirements

- Linux or WSL2
- uv 0.13.1
- Project Python 3.14.8, pinned by `.python-version`

The host may have CPython 3.15.0 installed. DSPy 3.4.1 declares
`requires-python >=3.10,<3.16`, but its `litellm` and `gepa[dspy]==0.1.4`
dependencies are marked `python_version < "3.15"`. On Python 3.15 `dspy.GEPA`
is an ImportError stand-in, so this project deliberately rejects 3.15 instead
of silently substituting the host interpreter.

## Setup

```bash
uv python install 3.14.8
uv sync --frozen
uv run python --version
uv run pytest
```

`uv run python --version` must report `Python 3.14.8`. `uv.lock` is part of
the deliverable and must be committed; `uv sync --frozen` must not re-resolve
dependencies. Dependency changes require a deliberate `uv lock` and review.

## Configuration

Live work reads server-side environment variables. Replay and offline
inspection need no key and make no provider, catalog, or billing request.

```bash
cp .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env` before running a Live command. An empty
value is rejected with `MissingConfigurationError`. Then verify the loaded
configuration without making a request:

```bash
uv run --env-file .env python -c \
  "from dspy_lab.config import load_settings; print(load_settings('live').redacted())"
```

`load_settings("replay")` is the safe default. It never requires
`OPENROUTER_API_KEY`. `load_settings("live")` validates the key, bare
OpenRouter model IDs, provider pins, numeric limits, and the relationship
between the GEPA metric-call target and the total optimization call cap. It
performs no network request; provider capability, strict-schema, and pricing
preflight belongs to S8 and requires separate authorization.

Frozen initial defaults are defined in `src/dspy_lab/config.py`:

| Setting | Default |
|---|---|
| Task model | `google/gemini-2.5-flash-lite` |
| Task provider pin | `google-vertex/eu` |
| Reflection model | `deepseek/deepseek-chat-v3.1` |
| Reflection provider pin | `deepinfra/fp4` |
| Task temperature / output cap | 0.0 / 2048 tokens |
| Reflection temperature / output cap | 1.0 / 4096 tokens |
| GEPA metric-call target | 120 |
| Optimization call / USD cap | 160 / 2.00 USD |
| Baseline development call / USD cap | 10 / 0.50 USD |
| Benchmark call / USD cap | 20 / 0.50 USD |
| Playground call / USD cap | 2 / 0.10 USD |
| Cache / threads / automatic retries | off / 1 / 0 |
| GEPA seed / reflection minibatch | 42 / 3 |

`MAX_OPTIMIZATION_CALLS` counts every admitted optimization model-dispatch
attempt, not only reflection proposals. GEPA's `max_metric_calls` is a separate
limit. Budget enforcement itself is implemented in S7.

## Frozen API contracts

These are the version-specific contracts S1 is pinned against. Later stages
must test them against the installed packages rather than assuming they hold.

- `dspy==3.4.1`, `gepa[dspy]==0.1.4`, `litellm==1.106.0`,
  `pydantic==2.14.1`, and `streamlit==1.66.0` are the frozen contract versions.
  `verify_runtime_contract()` rejects a missing or stubbed `dspy.GEPA`, an
  unexpected module, a missing `gepa.optimize`, or a missing LiteLLM
  completion backend.
- `dspy.GEPA` accepts exactly one of `auto`, `max_full_evals`, or
  `max_metric_calls` and compiles with explicit training and validation sets.
  The GEPA metric signature is
  `metric(gold, pred, trace=None, pred_name=None, pred_trace=None)` and returns
  `dspy.Prediction(score=..., feedback=...)`.
- `dspy.Evaluate` is keyword-only. Its `.score` is a rounded 0-100 percentage
  mean of per-case scores, and `.results` contains per-case triples. Neither is
  corpus micro-F1.
- Stock `JSONAdapter` repairs JSON, extracts objects with a regex, and filters
  unknown root fields. The planned task parser is a small strict subclass that
  rejects those behaviors; it is implemented in S6.
- Task and reflection LMs use `engine="litellm"`, `num_retries=0`,
  `cache=False`, `n=1`, and one thread. Callbacks are observational because
  their exceptions are swallowed; budget enforcement happens before dispatch
  in S7.
- LiteLLM 1.106.0 tries to refresh its remote model-cost map during import.
  `dspy_lab/__init__.py` sets `LITELLM_LOCAL_MODEL_COST_MAP=True` before any
  DSPy/LiteLLM import, so package import, Replay, and offline tests never fetch
  it. LiteLLM's local map is not the authoritative experiment price; Live
  provider prices are snapshotted from OpenRouter in S8.
- Program persistence uses `.save(path, save_program=False)` plus a fresh
  instance and `.load(path)`. State is not a self-contained safe class
  implementation, and no untrusted pickle artifact is loaded.

## Testing

```bash
uv run pytest
```

The default suite is offline: it requires no API key, makes no paid call, and
excludes tests marked `live`. A later provider preflight is selected explicitly
with `uv run pytest -m live` only after a separate budget confirmation. Tests
verify the actual Python version, the committed lockfile, genuine GEPA/LiteLLM
availability, Python 3.15 rejection, configuration validation, secret
redaction, and no network access during package import.

## Scope and limitations

S1 does not implement the Streamlit screens, dataset, scoring, DSPy program,
budget guard, provider preflight, optimization, benchmark, or Replay archive.
Those stages remain in `dspy_plan.md` sections 12-15. The host Python 3.15.0
installation is untouched; only this project's isolated environment uses
3.14.8.
