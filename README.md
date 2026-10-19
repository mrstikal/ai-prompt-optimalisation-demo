# DSPy Prompt Optimization Lab

Standalone lab for reviewing Czech precious-metals articles with one DSPy
predictor and comparing a frozen zero-shot baseline with a genuine GEPA
instruction-optimized program.

**Status: S1-S10 implemented.** S1 establishes the isolated Python
environment, dependency lock, package skeleton, and configuration contract. S2
adds the frozen six-code editorial rules, strict typed schemas,
canonicalization/offset validation, and the first five training specimens. S3
completes the 30-case training set with review audit, family descriptions, and a
coverage report. S4 adds 10 validation and 10 held-out test cases, a frozen split
manifest, family/near-duplicate checks, and a 50-entry review audit. S5 adds the
deterministic evidence matcher, corpus and per-category metrics, clean-case
reward, and the scalar/GEPA feedback interfaces, backed by hand-computed unit
fixtures and dataset-level self-match checks. S6 adds the frozen one-predictor
DSPy baseline, the strict JSON adapter, the canonical four-input whitelist, the
baseline fingerprint/manifest, and offline scripted-transport tests. S7 adds the
strict-budget guarded LM, one shared atomic request ledger, conservative Decimal
reservations, reconciliation for unknown costs, cancellation/budget-stop
semantics, request telemetry, and secret redaction, all tested against a mocked
LiteLLM transport. S8 adds frozen OpenRouter route profiles, strict
single-provider routing, public endpoint-catalog snapshots, an explicitly
confirmed three-call compatibility preflight, and local JSON preflight records.
S9 adds atomic state-only program artifacts, linked development/optimization/
benchmark run manifests with exact case/rule snapshots, a strictly offline
Replay reader, and shared sent-versus-preview trace formatting. S10 adds the
separately confirmed baseline validation evaluation and genuine GEPA compile
paths, complete native candidate statistics, request/response/event
persistence, linked development/optimization run records, and selected-program
serialization. The Streamlit UI and benchmark engine are not implemented. No
paid API call has been made.

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
performs no network request. Provider capability, strict-schema, and pricing
checks happen only in the separately authorized S8 preflight.

Frozen initial defaults are defined in `src/dspy_lab/config.py`:

| Setting | Default |
|---|---|
| Task model | `deepseek/deepseek-v4.1-flash-1017` |
| Task provider pin | `deepinfra/fp8` |
| Reflection model | `openai/gpt-6-sol` |
| Reflection provider pin | `openai` |
| Cheaper reflection alternative | `deepseek/deepseek-v4.1-flash-1017` on `deepinfra/fp8` |
| Task temperature / output cap | 0.0 / 2048 tokens |
| Reflection temperature / output cap | 1.0 / 4096 tokens |
| GEPA metric-call target | 120 |
| Optimization call / USD cap | 160 / 5.00 USD |
| Baseline development call / USD cap | 10 / 0.50 USD |
| Benchmark call / USD cap | 20 / 0.50 USD |
| Playground call / USD cap | 2 / 0.15 USD |
| Preflight calls / USD cap | 3 / 4.50 USD |
| Cache / threads / automatic retries | off / 1 / 0 |
| GEPA seed / reflection minibatch | 42 / 3 |

`MAX_OPTIMIZATION_CALLS` counts every admitted optimization model-dispatch
attempt, not only reflection proposals. GEPA's `max_metric_calls` is a separate
limit. Budget enforcement itself is implemented in S7.

## Dataset contract (S2-S4)

- `data/editorial_rules_v1.md` is the frozen six-code policy: definitions,
  exclusions, precedence, positive/negative examples, numerical/hypothetical
  boundaries, negation, quotations, citation rules, and evidence limits.
- `src/dspy_lab/schemas.py` defines strict Pydantic models for model findings,
  cases, product facts, sources, gold annotations, match options, review
  records, and review-audit entries. Unknown codes, extra fields, wrong types,
  whitespace-only text, and overlong evidence/explanation values are rejected
  without coercion.
- `src/dspy_lab/data.py` applies NFC, LF, and NBSP/NNBSP normalization, builds
  the canonical title + blank line + body text, validates unique literal
  evidence, half-open code-point anchors/context windows, the title/body
  separator, split composition, code coverage, hard negatives, same-code
  multi-issue cases, reference consistency, family disjointness, near-duplicate
  flags, review audit, and dataset hashes.
- `data/v1/train.jsonl` contains 30 training cases: 6 clean, 12 one-issue, and
  12 two-issue articles with 36 gold findings, exactly six occurrences of each
  code, four hard negatives, and one same-code multi-issue case.
- `data/v1/validation.jsonl` and `data/v1/test.jsonl` contain 10 cases each:
  2 clean, 4 one-issue, and 4 two-issue articles with 12 gold findings and
  exactly two occurrences of each code per split. The test split is sealed for
  optimization.
- `data/v1/split_manifest.json` freezes the 30/10/10 counts, family
  assignments, per-file and combined hashes, and the near-duplicate report.
  `data/v1/coverage_report.json` records the training-only coverage report.
  `data/v1/reviews.jsonl` contains one approved review-audit entry per case.
- The labels have an independent automated review by `decision-reviewer`.
  Training was reviewed in three batches; two unannotated title guarantees were
  removed and verified, and two related descriptive titles were neutralized.
  The validation/test split review is recorded in the review audit, and the
  final manifest audit was completed before freezing. Human sign-off is still
  required before final gold and before the held-out run.

```bash
uv run python -c \
  "from pathlib import Path; from dspy_lab.data import load_all_splits; print({k: len(v) for k, v in load_all_splits(Path('data/v1')).items()})"
```

## Baseline program and scoring (S5-S6)

- `src/dspy_lab/metrics.py` implements deterministic literal evidence matching,
  anchor/context containment, maximum-cardinality one-to-one assignment,
  TP/FP/FN accounting, per-category/micro/macro F1, clean-case reward,
  invalid-output accounting, and deterministic GEPA feedback. `scalar_metric`
  and `feedback_metric` share the same `score_case` core.
- `src/dspy_lab/program.py` defines exactly one `EditorialQASignature`, one
  `dspy.Predict` inside `EditorialQA`, and the `StrictJSONAdapter` subclass. The
  adapter reuses DSPy formatting/native-schema generation but rejects repairs,
  duplicate keys, nonfinite values, unknown fields, defaults, and coercion. The
  module has no demonstrations and persists no LM.
- `prompts/baseline.txt` is the frozen manual baseline instruction text.
  `baseline_manifest()` records the instructions, schema, prompt/rules/source
  hashes, zero demos, one predictor, no LM, and a stable fingerprint.
- `tests/test_program.py` drives the real DSPy predictor through a scripted
  offline transport, checks native-schema and JSON-mode requests, proves parse
  failures make exactly one request, verifies the four-input whitelist and no
  gold leakage, and integrates the resulting `dspy.Prediction` with the S5
  metric.

## Budget guard and request ledger (S7)

- `src/dspy_lab/llm.py` defines `GuardedLM`, a genuine `dspy.LM` with
  `engine="litellm"` that reserves one attempt and a conservative Decimal USD
  amount before delegating. It enforces `num_retries=0`, `cache=False`, `n=1`,
  the configured output cap, and the expected schema mode.
- One `BudgetLedger` is shared by task and reflection LMs and by
  `.copy()`/`copy.copy`/`copy.deepcopy`, so a copy cannot reset a budget. Its
  lock makes concurrent admission atomic. The call cap and USD cap are
  inclusive at the boundary.
- By default the input reservation uses the full admissible context bound; an
  optional verified token counter plus formatting overhead can reduce it.
  `ModelPricing` keeps Decimal USD-per-million rates, converts from catalog
  USD-per-token rates, and includes request/cache-write charges.
- `BudgetExceeded`/`BudgetCancelled` are execution-control `BudgetStop`
  exceptions, not quality failures. A stop persists: later calls are rejected
  without dispatch even if a library catches the exception.
- Every request record preserves phase/model/provider, messages, sanitized
  request parameters, reservation, generation ID, token usage, finish reason,
  estimated/actual cost status, and errors with secrets redacted. Unknown-cost
  and timeout reservations remain unresolved until `reconcile()`.
- `Settings.optimization_budget()`, `baseline_dev_budget()`,
  `benchmark_budget()`, and `playground_budget()` expose the named
  `BudgetLimits` phases.

## Provider preflight (S8)

- `src/dspy_lab/config.py` freezes three known OpenRouter routes: the task
  route `deepseek/deepseek-v4.1-flash-1017` on `deepinfra/fp8` (native strict
  schema, 0.098/0.294 USD per million), the strong default reflection route
  `openai/gpt-6-sol` on `openai` (native schema, 4.00/15.00 USD per million
  long-context ceiling), and the cheaper alternative reflection route
  `deepseek/deepseek-v4.1-flash-1017` on `deepinfra/fp8` (explicit JSON-object
  mode). It also freezes context/output bounds, cache-write ceilings, and a
  3-call/4.50 USD preflight budget. The cheaper reflection route can be
  selected with `REFLECTION_MODEL`/`REFLECTION_PROVIDER` and needs about
  0.35 USD for the same three-call preflight.
- `src/dspy_lab/llm.py` pins every real LM to one provider with
  `only`/`order`, `allow_fallbacks=false`, `require_parameters=true`,
  `data_collection=deny`, and reasoning disabled. Provider routing, schema
  mode, `drop_params`, streaming, cache controls, and retries are rejected
  before dispatch if they deviate from the contract.
- `src/dspy_lab/preflight.py` is the only S8 network entry point. It fetches
  the public endpoint catalog for both routes, rejects missing tags, prices
  above the frozen ceilings, service tiers, and schema drift, then makes
  exactly two task calls (one clean and one multi-issue training case) and one
  256-token reflection call. There are no retries or model swaps.
- Each attempt writes `runs/preflight/<run_id>.json` atomically with mode 0600.
  The record captures the public endpoint snapshot, request messages, routing,
  request parameters (including the strict wire `response_format`), generation
  IDs, usage, estimated cost, finish reasons, schema modes, checks, warnings,
  and the shared ledger. The API key is
  redacted and never written.
- The default 4.50 USD cap covers the conservative full-context reservation
  of two DeepSeek task calls plus one GPT-6 Sol reflection call (about 4.47
  USD). The cheaper DeepSeek reflection alternative reserves about 0.31 USD;
  lower `MAX_PREFLIGHT_COST_USD` deliberately when selecting it. The raised
  optimization cap (5.00 USD) is required because one full-context GPT-6 Sol
  reflection reservation is about 4.26 USD; actual token spend is normally
  much lower.

```bash
# Explicit authorization; this spends real money (up to the confirmed cap).
uv run --env-file .env python -m dspy_lab.preflight \
  --confirm-preflight-budget
```

The preflight fails before any model dispatch when the conservative
reservation does not fit the cap. Catalog metadata requests are public and
cost no tokens, but they do require network access. Do not run the command
until the separate preflight budget has been approved.

## Baseline Development and GEPA (S10)

- `run_baseline_development()` evaluates the frozen zero-demo baseline once on
  the supplied validation cases under a separate explicit confirmation. The
  default budget is 10 admitted calls and 0.50 USD. Every case receives a
  recorded outcome, including parse/request errors and cases not reached after a
  budget stop; per-case cost and request IDs stay visible. The summary also
  labels the always-empty reward as a calculated reference, not a model run.
- `run_optimization()` requires a separate confirmation and a completed linked
  baseline development run with the same frozen baseline program, rules,
  validation snapshot, metric version, and task model/request protocol. It
  reserves one-third of the metric target as admitted-call headroom, capped at
  40 slots (the default 120/160 configuration gives exactly 40), while the
  ledger cap remains authoritative for any final-batch overhang.
- The compile uses the real `dspy.GEPA` with the shared feedback metric, seed
  42, one thread, zero demos, no merges, and a unique native log directory. The
  baseline object and artifact are checked for immutability. A fresh copy is
  optimized; task and reflection calls share one ledger.
- Requests and raw responses are persisted as they complete. `gepa_stats.json`
  keeps the complete native `DspyGEPAResult.to_dict()` payload, candidate
  instructions, lineage, validation scores, discovery counts, and the observed
  metric/task/reflection counts. Successful runs save the selected state-only
  program with baseline lineage and round-trip it before the manifest is
  written. An unchanged baseline may win; it is linked without a duplicate
  artifact. Perfect-baseline skips and all-proposal failures are labeled
  explicitly, and budget-stopped or failed compiles are never marked completed.

## Artifacts and Replay (S9)

- Programs are saved under `artifacts/programs/<program_id>/` as
  `program.json` (trusted DSPy state only), `manifest.json`, and
  `instructions.txt`. The manifest records the content-derived program ID,
  parent/creation IDs, origin (`live` or `mock_test`), zero demos, `lm: null`,
  full signature schema, rules/prompt/source hashes, dependency versions,
  state hash/size, and checksums. No benchmark score or credential is stored.
- Saving creates the artifact in a temporary sibling directory, scans for
  configured secret values (including `OPENROUTER_API_KEY` by default) and
  credential-bearing keys, then atomically renames the directory. File writes
  use a temporary file, `fsync`, mode 0600, and `os.replace`; an interrupted
  write cannot appear complete. Run files are scanned for the same configured
  secrets without a size-based skip.
- Live loading requires `origin=live` unless `allow_mock=True` is explicit, and
  checks the manifest/schema/state hashes, rules/prompt/source identity,
  adapter version, and exact Python/application/DSPy/GEPA/LiteLLM/Pydantic
  versions before calling the trusted DSPy `.load()`. A mismatch is a clear
  compatibility error, not a warning or best-effort import. The predictor keeps
  `lm=None`; Live inference attaches the guarded LM through runtime context.
- Run records live under
  `runs/development|optimization|benchmarks|playground/<run_id>/`. A manifest
  links program IDs and hashes, exact `inputs_snapshot.jsonl` cases, the rules
  snapshot, outcomes, requests, and optional native GEPA statistics. Development snapshots are validation-only, optimization
  snapshots are train+validation, and benchmarks are test-only; linked runs
  must share rules and dataset versions.
- `open_replay_archive()` validates the benchmark's file hashes, linked
  development/optimization run IDs, program hashes, exact rule snapshots, and
  every outcome's case, candidate, arm/phase, program, split, and canonical
  input hash. It reads only recorded JSON/JSONL/text data: no current-dataset
  lookup, LM construction, provider/catalog request, GEPA compile, `Predict`,
  `Evaluate`, or native program `.load()`.
- Replay exposes `additional_api_calls=0` and `additional_api_cost_usd=0.00`.
  A completed, released `origin=live` benchmark is the only archive reported as
  measured. Mock/test archives remain visibly labeled and untrusted for
  results; missing, incomplete, corrupt, or mismatched records fail explicitly.
- `format_sent_trace()` preserves the exact recorded messages and provider
  metadata, including generation ID, usage, and estimated cost.
  `format_preview_trace()` is a separate type with `sent=false`, no generation
  ID/usage/cost, and an explicit notice. S8 preflight call records use the same
  trace shape.
- No released live archive exists yet. S9 tests build temporary mock archives
  and never present them as measured results.

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
excludes tests marked `live`. The S8 live test is selected explicitly only
after a separate budget confirmation:

```bash
RUN_LIVE_PREFLIGHT=1 uv run --env-file .env pytest -m live \
  tests/test_preflight.py
```

Tests
verify the actual Python version, the committed lockfile, genuine GEPA/LiteLLM
availability, Python 3.15 rejection, configuration validation, secret
redaction, no network access during package import, strict schema behavior,
canonicalization, exact evidence offsets, all six codes, precedence and
non-violations, 30/10/10 split composition and code coverage, hard negatives,
the same-code multi-issue case, reference consistency, family and hash
disjointness, near-duplicate checks, the 50-entry review audit, the frozen split
manifest, deterministic scoring and maximum-cardinality matching, duplicate and
evidence-failure accounting, undefined-ratio behavior, scalar/GEPA wrapper
agreement, one-predictor DSPy baseline behavior, strict JSON/native-schema
requests, zero-demo fingerprint stability, no gold leakage, exact budget
reservation/call caps, shared-ledger copies, concurrent admission, unknown-cost
retention, cancellation, real adapter fallback accounting, price-unit
conversion, secret redaction, rejection of the sealed test path by the
optimization loader, frozen route profiles, single-provider routing, schema
mode determinism, catalog drift rejection, the exact two-task/one-reflection
preflight plan, failure records without retries, preflight budget enforcement,
state-only program identity/load compatibility, runtime LM reattachment,
atomic/secret-safe artifacts, linked run and snapshot consistency, offline
Replay without LM/native loading or current-dataset substitution, mock archive
labeling, sent-versus-preview trace separation, ten-case baseline outcome and
budget retention, separate phase confirmations, fixed task-protocol linkage,
baseline immutability, zero demos, genuine offline GEPA candidate discovery,
native statistics and proposal/call capture, selected-program round-trip,
unchanged-baseline and perfect-skip labeling, and stopped/failed compile
records.

## Scope and limitations

S1-S10 complete the 50-case dataset, seal the held-out test split, provide
deterministic scoring and GEPA feedback, freeze the one-predictor DSPy baseline
with a strict adapter, add the strict-budget guarded LM/ledger, implement the
explicitly authorized provider preflight with local records, add safe
program/run persistence with a read-only Replay reader, and implement the
separately confirmed baseline validation and genuine GEPA compile paths. They
do not implement the Streamlit screens, benchmark engine, or Replay
presentation. Those stages remain in `dspy_plan.md` sections 12-15. No live
preflight, optimization, benchmark, or released archive exists, and no paid API
call has been made. The host
Python 3.15.0 installation is untouched; only this project's isolated
environment uses 3.14.8.
