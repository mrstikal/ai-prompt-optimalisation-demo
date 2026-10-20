# DSPy Prompt Optimization Lab

Explore whether an automatically optimized prompt can review Czech articles
more accurately than a manually written prompt, without changing the language
model. This interactive demo checks precious-metals articles for editorial
problems, explains its findings, and compares the original and optimized
instructions on the same articles.

Use it to:

- Find unsupported numbers, misleading investment claims, unfinished text,
  incorrect citations, and contradictions with supplied product facts.
- Optimize review instructions with DSPy and GEPA instead of editing prompts
  through trial and error.
- Compare detection quality, missed issues, false alarms, response times, token
  usage, and costs for both prompts.
- Inspect the instructions, optimization candidates, and actual messages sent
  to the models.
- Try both prompts on your own article in the Live Playground.
- Present a recorded experiment repeatedly in Replay without an API key,
  internet access, or additional model charges.

**Results are not bundled with the source code.** A fresh checkout opens the
interface but has no recorded experiment to display. Prepare a real experiment
with the Live runbook below, or obtain a complete trusted archive from its
operator. Optimization can improve, preserve, or reduce quality; the demo does
not promise an improvement or substitute sample results for measurements.

## Contents

- [How the demo works](#how-the-demo-works)
- [Using the interface](#using-the-interface)
- [Understanding the results](#understanding-the-results)
- [Technologies and their benefits](#technologies-and-their-benefits)
- [Runbook: first installation](#runbook-first-installation)
- [Configuration and budgets](#configuration-and-budgets)
- [Runbook: prepare a Live experiment](#runbook-prepare-a-live-experiment)
- [Runbook: daily operation](#runbook-daily-operation)
- [Troubleshooting](#troubleshooting)
- [Scope and limitations](#scope-and-limitations)

## How the demo works

### Article review

The reviewer receives an article title and body, editorial rules, and optional
product facts and source excerpts. It checks only this supplied information:
it does not search the web or independently verify cited URLs.

Each finding contains a category code, an exact quote from the article, and a
short explanation in Czech. A valid review may contain no findings. The demo
reviews text; it does not rewrite the article or provide investment advice.

| Category | What the reviewer looks for |
|---|---|
| `UNSUPPORTED_NUMERIC_CLAIM` | A numerical product, market, historical-price, supply, or performance claim unsupported by the supplied references. |
| `INVESTMENT_ADVICE_AS_FACT` | An unqualified investment recommendation presented as an objective fact. |
| `UNWARRANTED_CERTAINTY` | An unconditional promise of future returns, price growth, or absence of investment risk. |
| `UNRESOLVED_PLACEHOLDER` | Unfinished content such as `TODO`, `{{...}}`, or an editorial marker. |
| `INVALID_CITATION` | A citation to a missing local source, or to an excerpt that does not support the attributed claim. |
| `PRODUCT_FACT_MISMATCH` | A product attribute that contradicts the supplied reference data. |

The full definitions, exceptions, and examples are in the
[editorial rules](data/editorial_rules_v1.md).

### From baseline to comparison

1. **Evaluate the baseline.** Run the original, manually written instructions
   against the validation articles to establish a starting point.
2. **Optimize the instructions.** GEPA examines training results and feedback,
   asks a reflection model to propose better instructions, and evaluates
   candidates using training and validation data.
3. **Select and freeze a prompt.** Keep the selected instructions for the final
   comparison. The original prompt may remain the best candidate.
4. **Run a paired benchmark.** Review each held-out article, kept out of prompt
   tuning, with both frozen prompts. The task model, rules, reference
   information, and inference settings stay the same; only the instructions
   differ.
5. **Inspect or replay the experiment.** Browse the recorded findings,
   prompts, metrics, costs, and timing without rerunning inference.

The included dataset has **50 Czech articles**: 30 for training, 10 for
validation, and 10 for the held-out benchmark. It includes clean articles and
articles with one or more annotated issues. These reference annotations, also
called *gold labels*, provide the expected findings for scoring. Test articles
are excluded from optimization and candidate selection.

### Live and Replay

| | Live Experiment | Replay Experiment |
|---|---|---|
| Purpose | Prepare experiments and review new articles. | Inspect an existing recorded experiment. |
| API key and connectivity | Required for model calls. | Not required after local setup and archive transfer. |
| Model calls | Only after a separate budget confirmation for each action. | None. |
| Additional API cost | Paid usage, subject to the confirmed limits. | Zero; original experiment costs remain visible. |
| New optimization or benchmark | Available with the required prior runs. | Unavailable. |
| New Playground review | Available after optimization. | Disabled. |

The application always starts in **Replay Experiment**. Starting the server,
changing screens, or inspecting records does not authorize a paid operation.

## Using the interface

Choose **Experiment mode**, then a page under **Screen**. In Replay, select a
**Recorded benchmark archive**; in Live, select a **Recorded baseline
development run** to work with its linked experiments.

| Screen | What you can do |
|---|---|
| **Overview** | Check experiment identity, models, data versions, warnings, and whether the archive is a measured Live experiment. |
| **Baseline** | Review the original instructions and validation outcomes, including errors and the starting quality metrics. |
| **Optimization** | Browse candidates and optimization history. In Live, confirm and start a new optimization using a completed baseline evaluation. |
| **Comparison** | Inspect paired benchmark results, category-level metrics, corrected issues, newly missed issues, costs, and timing. In Live, explicitly start a new benchmark. |
| **Prompt Inspector** | Compare instructions and inspect recorded task/reflection messages. Unsent previews are labeled separately from actual requests. |
| **Playground** | In Live, review your own article with both frozen prompts without a gold-based score, and inspect prior Playground records. New inference is disabled in Replay. |

An empty or damaged archive produces a visible message. The application never
generates replacement results automatically. Only a completed, released Live
benchmark is labeled as a **measured** Replay experiment; mock or unreleased
records are not evidence of measured performance.

## Understanding the results

The comparison checks findings against reference annotations, including whether
the quoted passage actually identifies the issue. Repeated or unsupported
findings are not silently removed, and malformed responses remain errors.

| Measure | How to interpret it |
|---|---|
| Precision | Of the reported issues, how many were correct? Higher means fewer false alarms. |
| Recall | Of the expected issues, how many were found? Higher means fewer missed issues. |
| Micro-F1 | Combines precision and recall across all findings in the benchmark. This is the main overall quality measure. |
| Macro-F1 | Gives each of the six categories equal weight. Useful for spotting uneven performance. |
| Exact match | The share of articles with exactly the expected findings and no extras. |
| Mean article reward | The average per-article scoring reward used during optimization. It is not micro-F1. The displayed DSPy Evaluate percentage is also a mean reward. |
| Parsing success | Whether the model returned a valid structured response, not whether the findings were correct. |
| Cost and latency | Historical request costs and response times for the recorded experiment. Labels distinguish actual, estimated, reserved (`committed`), pending, unavailable, or zero known cost (`none`). |

An official quality-improvement verdict requires a complete, comparable, valid
benchmark with higher micro-F1 and no reduction in exact match. Incomplete or
invalid runs cannot establish an improvement. Cost-only break-even is shown
only when quality does not regress, cost amounts are known (actual, estimated,
or zero), and per-article savings are positive; otherwise there is no finite
supported break-even point. A result based on estimated costs is labeled
conditional, not a final billing calculation.

Validation scores guide optimization, so they are not independent proof of
improvement. The test set is small, and repeated benchmarks on the same test
data carry an exploratory-reuse warning. Reopening Replay is not a new test.

## Technologies and their benefits

### DSPy: measurable prompt optimization

[DSPy](https://dspy.ai/) describes a language-model task through named inputs,
typed outputs, and an evaluation metric. In this demo, that task is editorial
review. Instead of judging a prompt only by a few convincing examples, you
can measure how well it detects known issues across a dataset.

DSPy connects the reviewer, evaluation, and optimization workflow. Its benefit
here is a repeatable comparison: the original and optimized instructions solve
the same task and return the same kind of result.

### GEPA: learning from explanations of mistakes

[GEPA](https://github.com/gepa-ai/gepa) is the optimizer used through DSPy. It
uses results and textual feedback about failures to propose new instructions,
then evaluates candidates and keeps promising alternatives.

This is **prompt optimization, not model fine-tuning**: model weights do not
change. The result is an inspectable instruction text. Candidate history and
instruction differences help explain what changed, while the held-out
benchmark checks whether those changes generalize beyond the search data.

### Task and reflection models: separate roles

The **task model** reviews articles. The **reflection model** proposes prompt
changes during optimization; it is not a second reviewer in the final
benchmark. Separating these roles allows a smaller task model to serve the
repeatable review workload while a stronger model helps improve instructions.

A *route* fixes both the model and the provider serving its requests. The
default task route is DeepSeek V4.1 Flash
(`deepseek/deepseek-v4.1-flash-1017`, provider `deepinfra/fp8`). The default
reflection route is GPT-6 Sol (`openai/gpt-6-sol`, provider `openai`). A
supported lower-cost alternative uses the same DeepSeek route for reflection.
Neither choice guarantees better results; quality and economics must be read
from the experiment.

### OpenRouter and LiteLLM: access and visibility

[OpenRouter](https://openrouter.ai/) provides API access to the configured
models. One server-side API key supplies both model roles. Fixed provider
routes avoid silently switching providers during an experiment, and a
compatibility preflight checks the advertised capabilities and price limits.

[LiteLLM](https://docs.litellm.ai/) connects DSPy to those APIs and exposes
response metadata such as token usage and request identifiers. Together with
the demo's budget controls, this makes model usage visible and attributable to
each phase. Live inference still sends article content to external services;
running the browser locally does not make inference local.

### Streamlit: interactive inspection

[Streamlit](https://streamlit.io/) provides the browser interface for browsing
experiments, inspecting prompts, and starting confirmed Live jobs. The same
interface can present an archived experiment without repeating paid work.
No separate database service or frontend installation is required, and
Streamlit usage telemetry is disabled for this project.

### Pydantic, JSON, and local archives: trustworthy records

[Pydantic](https://docs.pydantic.dev/) validates structured inputs and outputs.
It helps distinguish a valid review from a malformed model response instead
of treating every returned string as a successful answer.

Experiments are stored locally as JSON (structured text), JSONL (one JSON
record per line), and plain text, including prompts, reference snapshots,
outcomes, request metadata, and checksums. This supports offline Replay,
integrity checks, and backups without a database. It also means the archive
directories must be preserved together, not just a screenshot or one summary
file.

### Python, uv, and pytest: reproducible operation

Python runs the application and its AI tooling. [uv](https://docs.astral.sh/uv/)
installs the project's Python interpreter and locked dependencies into an
isolated environment, leaving the system interpreter untouched.
[pytest](https://docs.pytest.org/) provides the offline verification command
used before operation and after environment changes.

The validated environment is Python **3.14.8**, uv **0.13.1**, DSPy **3.4.1**,
GEPA **0.1.4**, LiteLLM **1.106.0**, Pydantic **2.14.1**, and Streamlit
**1.66.0**. Use the supplied lockfile rather than installing current package
versions individually. Python 3.15 is not supported by this project's
DSPy/GEPA dependency combination.

## Runbook: first installation

### 1. Prepare the machine

Use Linux or a Linux distribution inside WSL2 (Windows Subsystem for Linux 2),
a Bash shell, Git, curl, and a modern browser. Installation needs internet
access; Live work also needs an OpenRouter account, API key, and sufficient
credit. Replay needs neither an account nor a key.

On Ubuntu or Ubuntu under WSL2, install the basic tools if missing:

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates
```

Install the validated uv version using its official installer, if needed:

```bash
curl -LsSf https://astral.sh/uv/0.13.1/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Expect `uv 0.13.1`. The installer executes a downloaded script; review the
[official installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
if your environment requires another installation method.

### 2. Obtain and install the project

Skip cloning if you already have the repository. Run all subsequent commands
from the repository root, the directory containing `app.py` and `uv.lock`.

```bash
git clone https://github.com/mrstikal/ai-prompt-optimalisation-demo.git
cd ai-prompt-optimalisation-demo
uv python install 3.14.8
uv sync --frozen
uv run --frozen --offline python --version
uv run --frozen --offline pytest -q
```

Python must report `Python 3.14.8`, and the tests must pass. The default test
suite is offline: it does not need a key or make paid calls. `uv sync --frozen`
creates the local `.venv` from `uv.lock`; no manual activation is needed.

### 3. Start the interface safely

```bash
uv run --frozen --offline streamlit run app.py \
  --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

Open **http://127.0.0.1:8501**. Expect **Replay Experiment** and the six-screen
navigation. With no archive installed, expect **No recorded experiment
available**, not example measurements. Stop the server with `Ctrl+C` before
continuing with Live setup.

`--server.headless true` skips Streamlit's interactive first-run email prompt
and automatic browser opening. Open the URL yourself after the server starts.

To use an existing archive instead, restore the complete `runs/` and
`artifacts/` directories from a trusted experiment prepared with a compatible
project version using [Preserve and restore experiments](#preserve-and-restore-experiments).
Start the same command and select **Recorded benchmark archive**. For a new
experiment, continue below.

## Configuration and budgets

Create `.env` once, without overwriting an existing configuration:

```bash
cp -n .env.example .env
chmod 600 .env
```

`chmod 600` allows only your user account to read or write `.env`. Edit the
file locally and set `OPENROUTER_API_KEY` to your own key. Keep the remaining
defaults for the first run. The file is ignored by Git. Do not put the key in
source code, a URL, a screenshot, or a shared archive.

Commands load `.env` only when it is explicitly passed with `--env-file .env`.
Validate the configuration without contacting a provider:

```bash
uv run --frozen --offline --env-file .env python -c \
  "from dspy_lab.config import load_settings; print(load_settings('live').redacted())"
```

Expect a redacted settings summary, not the secret itself. This validates
local configuration, not key validity, available credit, or model availability.

### Spending limits

Every paid phase needs its own approval. These are maximum budgets, not price
quotes or promised costs:

| Phase | Maximum request attempts | Maximum USD | Environment settings |
|---|---:|---:|---|
| Compatibility preflight | 3 | 4.50 | `MAX_PREFLIGHT_COST_USD` (call count is fixed) |
| Baseline validation | 10 | 0.50 | `MAX_BASELINE_DEV_CALLS`, `MAX_BASELINE_DEV_COST_USD` |
| Optimization, including task and reflection calls | 160 | 5.00 | `MAX_OPTIMIZATION_CALLS`, `MAX_OPTIMIZATION_COST_USD` |
| Paired held-out benchmark | 20 | 0.50 | `MAX_BENCHMARK_CALLS`, `MAX_BENCHMARK_COST_USD` |
| One Playground comparison | 2 | 0.15 | `MAX_PLAYGROUND_CALLS`, `MAX_PLAYGROUND_COST_USD` |

The four preparation phases have combined default caps of **10.50 USD**;
one optional Playground review adds **0.15 USD**. These are not a shared
lifetime allowance: each new run authorizes a new phase budget.
Call limits count admitted request attempts, not only successful answers.

Before sending a request, the application reserves a conservative cost based
on the allowed context and output size. A run can therefore stop even when
the displayed spend is below its cap. The default preflight cap accounts for
this reservation, not just the short sample responses. Pending or unknown
costs are not treated as free, and cancellation cannot refund sent requests.

### Other settings

| Setting | Default and purpose |
|---|---|
| `TASK_MODEL`, `TASK_PROVIDER` | DeepSeek route listed above; used for both comparison arms. |
| `REFLECTION_MODEL`, `REFLECTION_PROVIDER` | GPT-6 Sol route listed above; used during optimization. |
| `TEMPERATURE`, `MAX_OUTPUT_TOKENS` | `0.0`, `2048` for article review. |
| `REFLECTION_TEMPERATURE`, `REFLECTION_MAX_OUTPUT_TOKENS` | `1.0`, `4096` for prompt proposals. |
| `MAX_OPTIMIZATION_METRIC_CALLS` | `120`; GEPA's evaluation target, distinct from the total 160-call budget. Keep the default 40-call headroom for the first run. |
| `REQUEST_TIMEOUT_SECONDS` | `120` per request. |

Automatic retries and inference caching are disabled. Only the supported
model/provider combinations are accepted; model names are bare OpenRouter
IDs, without an extra `openrouter/` prefix. Do not change task settings halfway
through a linked experiment.

For the supported cheaper reflection alternative, set both
`REFLECTION_MODEL=deepseek/deepseek-v4.1-flash-1017` and
`REFLECTION_PROVIDER=deepinfra/fp8` before preparation. A preflight cap of
`MAX_PREFLIGHT_COST_USD=0.35` covers its current conservative reservation;
rerun preflight to check compatibility. Do not assume arbitrary models or
providers can be substituted.

## Runbook: prepare a Live experiment

Complete installation and configuration first. Review each budget separately
before executing the corresponding command or submitting a confirmation form.
These steps make real, paid API calls; they are not setup checks.

Before the first held-out benchmark, arrange independent human approval of the
reference annotations. Automated dataset checks are not human sign-off. Keep
test articles and labels out of prompt tuning, including manual tuning.

The bundled review audit in `data/v1/reviews.jsonl` contains automated reviews,
not human approvals. Human sign-off is a separate dataset-review process;
the demo provides no CLI command to record it. Until human approvals are
recorded in that audit, expect a `gold_approval` warning and partial dataset
acceptance even when archive validation passes. Do not edit frozen dataset
or audit files merely to silence the warning.

### 1. Check provider compatibility

```bash
uv run --frozen --env-file .env python -m dspy_lab.preflight \
  --confirm-preflight-budget
```

This checks the public provider catalog and makes two task-model calls plus
one reflection-model call. Expect a JSON summary with `status: "passed"`,
`admitted_calls: 3`, and a `record_path` under `runs/preflight/`. If it fails,
resolve the cause before continuing. There is no automatic model fallback.

### 2. Evaluate the original prompt

```bash
uv run --frozen --env-file .env python -m dspy_lab.release \
  baseline-development --confirm-budget
```

This evaluates the baseline on the 10 validation articles. Expect a JSON
summary with `status: "completed"`, `origin: "live"`, a `run_id`, and a
`run_dir` under `runs/development/`. Note the run ID. Baseline preparation is
a CLI operation; the UI does not create this prerequisite for you.

Do not proceed with a failed or stopped run. Do not repeat a successful
baseline evaluation for every presentation; the recorded run can be reused.

### 3. Optimize the instructions

Start the server with the Live configuration:

```bash
uv run --frozen --env-file .env streamlit run app.py \
  --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

1. Open **http://127.0.0.1:8501** and choose **Live Experiment**.
2. In **Recorded baseline development run**, select the completed run from
   the previous step. Open **Baseline** to inspect its validation results.
3. Open **Optimization**. Review the selected run, task/reflection routes,
   training and validation counts, and the optimization budget. The earlier
   baseline evaluation has its own budget; it is not rerun by this action.
4. Check the optimization-budget authorization box and click **Start confirmed
   optimization**.
5. Keep the server running. Review progress, candidate history, and costs.
   Wait for `completed`, then note the optimization run ID and selected prompt.

`stopped`, `failed`, and `interrupted` are not successful optimizations. An
unchanged baseline can legitimately be selected; inspect the warnings and
candidate history rather than assuming optimization must improve the prompt.

### 4. Run and release the paired benchmark

1. Open **Comparison** in Live mode.
2. In **Completed optimization run for the paired benchmark**, select the
   completed optimization linked to your baseline run.
3. Review the held-out-test warning and the separate benchmark budget.
4. Check **Release the completed benchmark as a measured Replay archive**.
5. Check the benchmark-budget and sealed-split confirmation: permission to
   open the test articles excluded from optimization. Click **Start confirmed
   benchmark**.
6. Wait for `completed`. Note **Recorded benchmark run**, check **Released as
   measured: yes**, and review the verdict, validity, and warnings about prior
   uses of the test data.

This reviews all 10 test articles with both prompts, for 20 planned calls. It
does not optimize again. Repeating it exposes the same test set again and is
recorded as exploratory reevaluation, not a fresh blind result. A stopped or
failed benchmark cannot be released as a measured archive; a completed run
with invalid responses still cannot support an official improvement verdict.

### 5. Validate the archive offline

Replace the value below with the recorded benchmark run ID. No key or network
access is required for this validation:

```bash
BENCHMARK_RUN_ID='replace-with-recorded-benchmark-run-id'
uv run --frozen --offline python -m dspy_lab.release validate \
  --benchmark-run-id "$BENCHMARK_RUN_ID" \
  --json "runs/release_validation/$BENCHMARK_RUN_ID-acceptance.json"
```

Expect `Release validation: PASS` and exit code 0. Validation checks archive
integrity, linked records, paired inputs, metrics, costs, timing, and repeated
read-only access. It rejects mock, unreleased, incomplete, or corrupt archives.
Read warnings and acceptance items too: a pass does not replace human sign-off
or certify unperformed operator checks. Playground behavior and the offline
test suite are not proven by a benchmark archive alone.

Keep the resulting acceptance report with the experiment. Do not write reports
inside an individual run archive or `artifacts/programs/`.

### 6. Optionally review your own article

1. Open **Playground** in Live mode and select the completed run under
   **Optimization run supplying both artifacts**.
2. Fill in **Article title** and **Article body** with a Czech article. Both
   are required; limits are 300 and 20,000 characters respectively.
3. Leave **Optional product facts (JSON object)** as `{}` and **Optional
   sources (JSON array)** as `[]`, or supply references in the formats below.
4. Leave **Store this article and request text locally for audit** unchecked
   unless local content retention is intended.
5. Confirm the separate Playground budget and external transmission of the
   article, then click **Run unscored playground**.
6. Compare both sets of findings, errors, messages, usage, costs, and timing.

Example product facts, not a required template:

```json
{"metal": "silver", "mass_g": 100, "purity": "999/1000"}
```

Example local source excerpt:

```json
[
  {
    "source_id": "S1",
    "title": "Product specification",
    "excerpt": "The silver bar has a mass of 100 g and a purity of 999/1000.",
    "provenance": "Operator-supplied product sheet"
  }
]
```

`mass_g` must be a JSON number, not a quoted string. Each source requires
`source_id`, `excerpt`, and `provenance`; source IDs must be unique. Citations
such as `[S1]` refer to these supplied IDs. Excerpts are supplied as text, not
downloaded from `provenance`. Unknown fields are rejected.

Playground is **unscored**: there are no gold labels, it does not affect
candidate selection, and its results are not benchmark evidence. With content
retention off, only hashes, status, cost, and timing are saved; findings and
messages remain in the current app session. Opting in also saves content for
later inspection under **Recorded playground runs** in Live Playground.

## Runbook: daily operation

### Present a recorded experiment

After installation and archive preparation, use:

```bash
uv run --frozen --offline streamlit run app.py \
  --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

Select **Replay Experiment**, choose **Recorded benchmark archive**, and check
its measured status and warnings in **Overview**. A useful presentation order
is **Baseline**, **Optimization**, **Prompt Inspector**, then **Comparison**.
The original costs remain visible; additional API calls and API cost are zero.
No new Playground review or experiment can run in this mode.

For Live operation, use the same startup command with `--env-file .env` and
without `--offline`, as shown in the preparation runbook. Selecting Live alone
does not start a job. Review the selected run and budget before every action.

### Stop, cancel, and restart

- Let active work finish, or use **Cancel optimization**, **Cancel benchmark**,
  or **Cancel playground** on its job panel. Cancellation prevents subsequent
  dispatches, but an already-sent request may finish and incur a charge.
- Wait for a terminal status before pressing `Ctrl+C` in the server terminal.
  Closing the browser tab is not cancellation of server-side work.
- Restart with the appropriate startup command above. Completed records remain
  available. A job interrupted by a server restart is recorded explicitly and
  is never automatically resumed; any new attempt needs a fresh confirmation
  and budget review.
- Run only one application instance per working directory. The UI permits one
  paid job at a time, and mode switching is blocked while it is active. Do not
  run the paid CLI preparation commands concurrently with UI jobs.

If port 8501 is occupied, use `--server.port 8502` and open
**http://127.0.0.1:8502**. Keep `--server.address 127.0.0.1`: the demo is intended
for local operation, not an unauthenticated public service with spending
controls exposed to other users.

### Preserve and restore experiments

| Path | What to preserve |
|---|---|
| `runs/` | Preflight records, baseline evaluations, optimizations, benchmarks, Playground records, job history, and acceptance reports. |
| `artifacts/` | Saved original/selected prompts and their compatibility and integrity metadata. |
| `.env` | Local credentials and settings; keep separately in protected storage, never in a shared demo archive. |
| `uv.lock`, `.python-version`, project revision | The environment and application version needed to operate the saved experiment reliably. |

`runs/` and `artifacts/` are ignored by Git. Cloning or pulling the repository
does not transfer experiment results. Preserve both directories together;
copying only a benchmark directory omits its linked baseline, optimization,
and prompt records.

For example, after a completed experiment, stop the server and create a local
backup from the repository root:

```bash
umask 077
tar -czf "../dspy-lab-state-$(date -u +%Y%m%dT%H%M%SZ).tar.gz" \
  runs artifacts uv.lock .python-version
git rev-parse HEAD
```

`umask 077` restricts newly created files in this shell to your user account,
including the backup. Record the printed project revision with the backup and
preserve any local source changes separately. The archive excludes `.env`,
but may still contain article text, model messages, reference annotations,
and outputs. Treat it as sensitive, check retained content before sharing,
and restrict backup access.

To restore, install the matching project revision in a separate clean
directory, extract the trusted backup there with `tar -xzf /path/to/backup.tar.gz`,
run `uv python install 3.14.8` and `uv sync --frozen`, then rerun offline tests
and release validation with the original benchmark ID. Start Replay and verify
the same IDs, warnings, and results. Configure a new local `.env` only if Live
operation is needed. Do not overlay a backup onto active or different runs.

### Configuration changes and maintenance

Stop the server before changing `.env`, then restart it with `--env-file .env`.
Keep the task model, provider, rules, and inference settings consistent across
linked phases; incompatible changes are rejected rather than silently mixed.
Review a new configuration and preflight before preparing a new experiment.

After restoring or changing the environment, rerun:

```bash
uv sync --frozen
uv run --frozen --offline python --version
uv run --frozen --offline pytest -q
```

Do not use `uv lock`, unpinned package upgrades, or manual archive edits as
routine fixes. Dependency changes need a deliberate compatibility check, and
editing archived files invalidates their integrity checks. Historical Replay
results should remain unchanged.

## Troubleshooting

| Symptom | Action |
|---|---|
| `uv: command not found` | Open a new shell after installation, or add the installer directory to `PATH`; for the default install, use `export PATH="$HOME/.local/bin:$PATH"`. |
| Wrong Python version, or Python 3.15 rejected | Run `uv python install 3.14.8`, `uv sync --frozen`, and use `uv run` from the project root instead of the system `python`. |
| Offline startup cannot find a package or interpreter | Complete the online installation first. `--offline` does not download a missing environment. |
| No recorded experiment available | Install a complete trusted archive, or finish the Live workflow through benchmark release. Installation alone creates no measurements. |
| Live configuration is incomplete or the key is missing | Set `OPENROUTER_API_KEY` in `.env` and restart using `uv run --frozen --env-file .env streamlit run ...`. A plain startup does not load the file. |
| Preflight fails on model, provider, schema, or price checks | Read its error/record, verify supported route IDs and connectivity, and check provider availability. Do not bypass it by silently switching routes. |
| Budget stop despite low displayed spend | Review unresolved reservations and both call/USD caps. Conservative reservations may exceed the remaining budget. Do not repeatedly restart or raise caps without reviewing new spending. |
| Optimization has no usable baseline run | Complete the CLI baseline-development step and select its `completed`, `live` record. Stopped, failed, and mock runs are not usable prerequisites. |
| Linked settings or artifact versions differ | Restore the matching settings and project environment, or prepare a new consistently configured experiment. Do not edit manifests to bypass checks. |
| Another job or active-job lock blocks a start | Check all server processes and **Recent Live job records**. Let the real job finish or cancel it. After a crash, restart once for recovery; inspect an unreadable lock rather than deleting it while work may be active. |
| Multiple released archives make validation ambiguous | Pass the intended `--benchmark-run-id` explicitly, as in the runbook. |
| Archive integrity or release validation fails | Preserve the error and original records, verify all linked directories are present, and restore a known-good backup if needed. Do not regenerate or relabel data as a fix. |
| Playground rejects references | Use one JSON object for facts and an array for sources; use numeric `mass_g`, unique source IDs, and all required source fields. Unknown fields are not accepted. |
| Playground findings disappear after a session ends | This is expected with content retention off. Only explicitly retained article/request content is available after restart. |

## Scope and limitations

This is a local, focused demonstration of prompt optimization, not a production
publishing gate, general fact-checking service, or investment adviser. Reviews
cover only the six editorial categories and supplied references. Human review
remains necessary before publishing, and a clean model response is not proof
that an article is correct.

The 50-case dataset is small and domain-specific. Automated checks and reviews
do not replace human approval of reference annotations. A valid outcome can be
unchanged, worse, mixed, or inconclusive. Preserve the original warnings,
failed outcomes, and test-reuse history when presenting results.
