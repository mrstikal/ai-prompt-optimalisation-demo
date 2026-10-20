from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import urllib.request

import dspy
import litellm
import pytest
from streamlit.testing.v1 import AppTest

import dspy_lab.data as data_module
import dspy_lab.llm as llm_module
import dspy_lab.preflight as preflight_module
from dspy_lab.artifacts import (
    ArtifactSchemaError,
    atomic_write_json,
    open_replay_archive,
    read_run_manifest,
    read_run_summary,
    write_run_manifest,
)
from dspy_lab.config import PROJECT_ROOT, load_settings
from dspy_lab.llm import BudgetLedger

sys.path.insert(0, str(PROJECT_ROOT))
import app as app_module
import ui_jobs

from ui_fixture import build_ui_archive

MODE_REPLAY = "Replay Experiment"
MODE_LIVE = "Live Experiment"


def _forbidden(*_args, **_kwargs):
    raise AssertionError("Replay/Live read-only screens must not call this")


def _wrapper(
    tmp_path: Path,
    fixture: Path | None,
    *,
    real_worker: bool = False,
    runner_expr: str = "ui_jobs.make_runner()",
) -> Path:
    runs = fixture / "runs" if fixture is not None else tmp_path / "empty-runs"
    programs = (
        fixture / "programs" if fixture is not None else tmp_path / "empty-programs"
    )
    manager_lines = [
        "if app._JOB_MANAGER_OVERRIDE is None:",
        "    app._JOB_MANAGER_OVERRIDE = app.JobManager(",
        f"        Path({str(runs)!r}),",
        f"        Path({str(programs)!r}),",
    ]
    if not real_worker:
        manager_lines.append(f"        runner={runner_expr},")
    manager_lines.append("    )")
    wrapper = tmp_path / "ui_wrapper.py"
    wrapper.write_text(
        "\n".join(
            [
                "import sys",
                "from pathlib import Path",
                f"sys.path.insert(0, {str(PROJECT_ROOT)!r})",
                f"sys.path.insert(0, {str(PROJECT_ROOT / 'tests')!r})",
                "import app",
                "import ui_jobs",
                f"app.RUNS_ROOT = Path({str(runs)!r})",
                f"app.PROGRAMS_ROOT = Path({str(programs)!r})",
                *manager_lines,
                "app.main()",
            ]
        ),
        encoding="utf-8",
    )
    return wrapper


def _run_app(
    tmp_path: Path,
    *,
    build: bool = True,
    changed: bool = True,
    with_requests: bool = True,
    with_stats: bool = True,
    measured: bool = False,
    real_worker: bool = False,
    runner_expr: str = "ui_jobs.make_runner()",
) -> AppTest:
    fixture = None
    if build:
        fixture = tmp_path / "fixture"
        build_ui_archive(
            fixture,
            changed=changed,
            with_requests=with_requests,
            with_stats=with_stats,
            measured=measured,
        )
    app_module._JOB_MANAGER_OVERRIDE = None
    app = AppTest.from_file(
        str(
            _wrapper(
                tmp_path,
                fixture,
                real_worker=real_worker,
                runner_expr=runner_expr,
            )
        ),
        default_timeout=60,
    )
    app.run()
    assert not app.exception, [str(item.value) for item in app.exception]
    return app


def _select_screen(app: AppTest, name: str) -> AppTest:
    app.radio[1].set_value(name).run()
    assert not app.exception, [str(item.value) for item in app.exception]
    return app


def _set_mode(app: AppTest, mode: str) -> AppTest:
    app.radio[0].set_value(mode).run()
    assert not app.exception, [str(item.value) for item in app.exception]
    return app


def _click_button(app: AppTest, label: str) -> AppTest:
    for button in app.button:
        if button.label == label:
            button.click().run()
            assert not app.exception, [
                str(item.value) for item in app.exception
            ]
            return app
    raise AssertionError(f"button {label!r} not found")


def _job_spec(
    tmp_path: Path,
    *,
    mode: str = MODE_LIVE,
    token: str = "fixture-token",
    run_id: str = "opt-fixture-1",
    epoch: int = 0,
    generation: int = 0,
    settings=None,
):
    return app_module.OptimizationJobSpec(
        mode=mode,
        run_id=run_id,
        token=token,
        mode_epoch=epoch,
        confirmation_generation=generation,
        settings=settings or load_settings("live"),
        baseline_development_run_id="dev-ui",
        train_cases=(),
        validation_cases=(),
        rules_text="fixture rules",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )


def _select(app: AppTest, label: str, value: str) -> AppTest:
    for box in app.selectbox:
        if box.label == label:
            box.select(value).run()
            assert not app.exception, [
                str(item.value) for item in app.exception
            ]
            return app
    raise AssertionError(f"selectbox {label!r} not found")


def _all_text(app: AppTest) -> str:
    parts: list[str] = []
    for collection in (
        app.markdown,
        app.text,
        app.caption,
        app.code,
        app.info,
        app.warning,
        app.error,
        app.success,
        app.subheader,
        app.header,
    ):
        parts.extend(str(item.value) for item in collection)
    parts.extend(str(item.label) for item in app.expander)
    parts.extend(str(item.label) for item in app.button)
    parts.extend(str(item.label) for item in app.checkbox)
    for box in app.selectbox:
        parts.append(str(box.label))
        parts.append(str(box.value))
        parts.extend(str(option) for option in box.options)
    return "\n".join(parts)


def _forbidden_calls(monkeypatch) -> None:
    monkeypatch.setattr(dspy, "LM", _forbidden)
    monkeypatch.setattr(dspy, "Evaluate", _forbidden)
    monkeypatch.setattr(dspy, "GEPA", _forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)
    monkeypatch.setattr(litellm, "completion", _forbidden)
    monkeypatch.setattr(litellm, "acompletion", _forbidden)
    monkeypatch.setattr(llm_module, "build_guarded_lm", _forbidden)
    monkeypatch.setattr(
        preflight_module, "fetch_endpoint_catalog", _forbidden
    )
    monkeypatch.setattr(preflight_module, "run_preflight", _forbidden)


def test_replay_is_default_without_key_or_archive(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _forbidden_calls(monkeypatch)
    app = _run_app(tmp_path, build=False)
    assert app.radio[0].value == MODE_REPLAY
    assert MODE_LIVE in app.radio[0].options
    text = _all_text(app)
    assert "No recorded experiment available" in text
    assert (
        "Switch to **Live Experiment** to prepare an authorized run" in text
    )
    assert "Additional API calls: 0" in text
    assert "additional API cost: 0.00 USD" in text
    assert not app.button
    assert not app.columns
    assert not app.html
    assert not app.table
    assert not app.dataframe


def test_replay_archive_shows_provenance_and_historical_cost(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _forbidden_calls(monkeypatch)
    app = _run_app(tmp_path)
    text = _all_text(app)
    assert any(
        "bench-ui" in option
        and "2026-10-19" in option
        and "completed" in option
        for option in app.selectbox[0].options
    )
    assert "Benchmark run:** `bench-ui`" in text
    assert "Baseline development run:** `dev-ui`" in text
    assert "Optimization run:** `opt-ui`" in text
    assert "Split:** test (sealed benchmark snapshot)" in text
    assert "deepseek/deepseek-v4.1-flash-1017" in text
    assert "deepinfra/fp8" in text
    assert "genuine GEPA; 2 recorded candidate(s)" in text
    assert "Historical original cost (not additional spend)" in text
    assert "0.0123 USD (estimated)" in text
    assert "Additional API calls: 0" in text
    assert "Additional API cost:** 0.00 USD" in text
    assert "Not a measured experiment" in text
    assert "Mock/test archive" in text
    assert "This archive is not released." in text
    assert not app.button


def test_measured_archive_gets_only_measured_labels(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _forbidden_calls(monkeypatch)
    app = _run_app(tmp_path, measured=True)
    text = _all_text(app)
    assert "Measured archive: completed, released Live benchmark" in text
    assert "Measured archive:** yes, released Live benchmark" in text
    assert app.selectbox[0].value.endswith("— measured")
    assert "not measured" not in app.selectbox[0].value
    assert "Not a measured experiment" not in text
    assert "This archive is not released." not in text
    _select_screen(app, "Prompt Inspector")
    text = _all_text(app)
    assert "Release status:** released" in text
    assert "non-measured archive" not in text


def test_replay_baseline_labels_split_and_shows_case_details(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _forbidden_calls(monkeypatch)
    app = _run_app(tmp_path)
    _select_screen(app, "Baseline")
    text = _all_text(app)
    assert "Validation (adaptive development split) — not held-out" in text
    assert "sealed and not displayed on this screen" in text
    assert "Demos:** 0 (zero)" in text
    assert "val-001" in text
    assert "val-003" in text
    assert "val-005" in text
    assert "Fixture false positive on a clean article." in text
    assert "Missed gold issues" in text
    assert "Matched gold issues" in text
    assert "No parsed findings: this is an invalid output." in text
    assert "Error kind:** parse_error" in text
    assert "Outcome error: parse_error: fixture invalid output" in text
    assert "Request IDs:** req-dev-0" in text
    assert "Latency:** 120.0 ms" in text
    assert "0.0001 USD (estimated)" in text
    assert "test-001" not in text
    assert not app.button


def test_replay_inspector_distinguishes_sent_from_preview_and_shows_diff(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _forbidden_calls(monkeypatch)
    app = _run_app(tmp_path)
    _select_screen(app, "Prompt Inspector")
    text = _all_text(app)
    assert (
        "Articles inspected:** held-out test (recorded benchmark snapshot)"
        in text
    )
    assert "Release status:** not released" in text
    assert "non-measured archive" in text
    assert "Unified diff" in text
    assert "Demos:** 0 (zero)" in text
    assert "SENT REQUEST — captured at the LM boundary" in text
    assert "Source:** sent" in text
    assert "Generation ID:** gen-req-baseline-0" in text
    assert "Preview (not sent)" in text
    assert "PREVIEW ONLY" in text
    assert "has no provider usage, generation ID, or cost" in text
    assert "Recorded optimization requests" in text
    assert "#1 — reflection — succeeded" in text
    assert "Input/output signature schema" in text
    assert '"field_type"' in text
    assert "Candidate 0" in text
    assert "Candidate 1" in text
    preview_text = "\n".join(
        str(item.value)
        for item in app.text
        if str(item.value).startswith("Document:")
    )
    assert preview_text
    assert "gen-" not in preview_text
    assert "estimated_cost" not in preview_text

    _select(app, "Instructions to inspect", "Candidate 0 — validation score 0.5")
    text = _all_text(app)
    assert "No instruction change selected" in text
    assert "Unified diff" not in text

    _select(
        app,
        "Instructions to inspect",
        "Candidate 1 — validation score 0.75 — recorded selection",
    )
    text = _all_text(app)
    assert "Unified diff" in text
    assert "Optimized rule: quote the narrowest exact evidence span." in text
    assert not app.button


def test_replay_inspector_reports_missing_request_traces(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _forbidden_calls(monkeypatch)
    app = _run_app(tmp_path, with_requests=False)
    _select_screen(app, "Prompt Inspector")
    text = _all_text(app)
    assert "No sent request trace is archived for this case and arm" in text
    assert "No optimization request trace is archived" in text
    assert "SENT REQUEST" not in text
    assert "PREVIEW ONLY" in text
    assert "no provider usage, generation ID, or cost" in text


def test_replay_unchanged_baseline_has_no_diff(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _forbidden_calls(monkeypatch)
    app = _run_app(tmp_path, changed=False)
    _select_screen(app, "Prompt Inspector")
    text = _all_text(app)
    assert "No instruction change selected" in text
    assert "Demos:** 0 (zero)" in text
    assert all(code.language != "diff" for code in app.code)


def test_replay_corrupt_archive_is_explicit_and_offline(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _forbidden_calls(monkeypatch)
    fixture = tmp_path / "fixture"
    build_ui_archive(fixture)
    (
        fixture
        / "runs"
        / "benchmarks"
        / "bench-ui"
        / "outcomes.jsonl"
    ).unlink()
    app_module._JOB_MANAGER_OVERRIDE = None
    app = AppTest.from_file(
        str(_wrapper(tmp_path, fixture)),
        default_timeout=60,
    )
    app.run()
    assert not app.exception, [str(item.value) for item in app.exception]
    text = _all_text(app)
    assert "No recorded experiment available" in text
    assert "Recorded archive unavailable" in text
    assert "No data will be regenerated and no provider request will be made" in text
    assert "Additional API calls: 0" in text
    assert not app.button


def test_live_reads_recorded_development_run_without_paid_actions(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-ui-read-only")
    _forbidden_calls(monkeypatch)
    app = _run_app(tmp_path)
    app = _set_mode(app, MODE_LIVE)
    assert app.radio[0].value == MODE_LIVE
    assert "Live Experiment — optimization, paired benchmark, and unscored" in _all_text(app)
    assert "mock/test record" in _all_text(app)
    assert "completed (mock_test)" in _all_text(app)
    assert "Recorded baseline development run" in [
        box.label for box in app.selectbox
    ]
    app = _select_screen(app, "Baseline")
    text = _all_text(app)
    assert "Live workspace: server-local recorded development results" in text
    assert "Validation (adaptive development split) — not held-out" in text
    assert "val-001" in text
    assert "test-001" not in text
    assert not app.button
    app = _select_screen(app, "Prompt Inspector")
    text = _all_text(app)
    assert "Recorded optimization run" in [
        box.label for box in app.selectbox
    ]
    assert (
        "Articles inspected:** validation (adaptive development split) "
        "— not held-out" in text
    )
    assert "Release status:** not a benchmark archive" in text
    assert "Unified diff" in text
    assert "SENT REQUEST — captured at the LM boundary" in text
    assert "test-001" not in text
    assert not app.button


def test_live_without_key_is_read_only_and_baseline_still_reads(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _forbidden_calls(monkeypatch)
    app = _run_app(tmp_path)
    app = _set_mode(app, MODE_LIVE)
    text = _all_text(app)
    assert "OPENROUTER_API_KEY" in text
    assert "Reading local development artifacts still works" in text
    assert not app.button
    app = _select_screen(app, "Baseline")
    assert "val-001" in _all_text(app)
    assert not app.button


def test_live_never_loads_test_split(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-ui-read-only")
    _forbidden_calls(monkeypatch)
    app = _run_app(tmp_path)
    original = data_module.load_cases

    def guarded(path):
        if Path(path).name == "test.jsonl":
            raise AssertionError("Live read-only screens must not load test cases")
        return original(path)

    monkeypatch.setattr(data_module, "load_cases", guarded)
    app = _set_mode(app, MODE_LIVE)
    for screen in ("Overview", "Baseline", "Prompt Inspector"):
        app = _select_screen(app, screen)
        assert "test-001" not in _all_text(app)
        assert not app.button


def test_replay_does_not_use_current_dataset_or_construct_lm(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _forbidden_calls(monkeypatch)
    fixture = tmp_path / "fixture"
    archive_dir = build_ui_archive(fixture)
    archive = open_replay_archive(
        archive_dir,
        runs_root=fixture / "runs",
        programs_root=fixture / "programs",
    )

    def forbidden_dataset(*_args, **_kwargs):
        raise AssertionError("Replay must not load the current dataset")

    monkeypatch.setattr(data_module, "load_cases", forbidden_dataset)
    app_module._JOB_MANAGER_OVERRIDE = None
    app = AppTest.from_file(
        str(_wrapper(tmp_path, fixture)),
        default_timeout=60,
    )
    app.run()
    assert not app.exception, [str(item.value) for item in app.exception]
    text = _all_text(app)
    assert archive.snapshot.cases_sha256 in text
    assert f"Test cases in snapshot:** {archive.snapshot.case_count}" in text
    assert "Additional API calls: 0" in text


def test_read_only_screens_use_stackable_components(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _forbidden_calls(monkeypatch)
    app = _run_app(tmp_path)
    for screen in ("Overview", "Baseline", "Prompt Inspector"):
        app = _select_screen(app, screen)
        assert not app.columns
        assert not app.html
        assert not app.table
        assert not app.dataframe
        assert not app.button


def test_overview_reports_missing_candidate_statistics(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _forbidden_calls(monkeypatch)
    app = _run_app(tmp_path, with_stats=False)
    text = _all_text(app)
    assert "No candidate statistics are archived." in text
    assert "0 recorded candidate(s)" not in text


def test_live_stopped_development_run_is_labeled(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-ui-read-only")
    _forbidden_calls(monkeypatch)
    fixture = tmp_path / "fixture"
    build_ui_archive(fixture)
    development_dir = fixture / "runs" / "development" / "dev-ui"
    manifest = read_run_manifest(development_dir)
    stopped = manifest.model_copy(update={"status": "stopped"})
    write_run_manifest(development_dir, stopped)
    app = AppTest.from_file(
        str(_wrapper(tmp_path, fixture)),
        default_timeout=60,
    ).run()
    assert not app.exception, [str(item.value) for item in app.exception]
    app = _set_mode(app, MODE_LIVE)
    app = _select_screen(app, "Baseline")
    text = _all_text(app)
    assert "Development run status is 'stopped'" in text
    assert "not a completed baseline evaluation" in text


def test_cost_status_prefers_pending_and_settled_states() -> None:
    status = app_module._cost_status_from_payload
    assert (
        status(
            {
                "pending_cost_calls": 1,
                "actual_cost_usd": "1",
                "estimated_cost_usd": "1",
            }
        )
        == "pending"
    )
    assert (
        status(
            {
                "unresolved_cost_usd": "0.5",
                "actual_cost_usd": "1",
            }
        )
        == "pending"
    )
    assert (
        status({"actual_cost_usd": "0.2", "estimated_cost_usd": "0.1"})
        == "actual"
    )
    assert (
        status({"actual_cost_usd": "0", "estimated_cost_usd": "0.1"})
        == "estimated"
    )
    assert (
        status(
            {
                "actual_cost_usd": "0",
                "estimated_cost_usd": "0",
                "committed_cost_usd": "0.1",
            }
        )
        == "committed"
    )
    assert (
        status({"actual_cost_usd": "0", "estimated_cost_usd": "0"})
        == "none"
    )
    assert status({"cost_status": "unavailable"}) == "unavailable"


def test_summary_schema_mismatch_fails_closed(tmp_path) -> None:
    fixture = tmp_path / "fixture"
    build_ui_archive(fixture)
    development_dir = fixture / "runs" / "development" / "dev-ui"
    manifest = read_run_manifest(development_dir)
    summary = json.loads(
        (development_dir / "summary.json").read_text(encoding="utf-8")
    )
    summary["schema_version"] = "wrong_summary_v1"
    atomic_write_json(development_dir / "summary.json", summary)
    with pytest.raises(ArtifactSchemaError):
        read_run_summary(development_dir, manifest)


def test_job_manager_is_live_only_single_worker_and_shares_ledger(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-job-fixture")
    ui_jobs.reset()
    manager = app_module.JobManager(
        tmp_path / "runs",
        tmp_path / "programs",
        runner=ui_jobs.make_runner(),
    )
    manager.note_mode(MODE_LIVE)
    with pytest.raises(app_module.JobGateError):
        manager.start_job(
            _job_spec(
                tmp_path,
                mode=MODE_REPLAY,
                token="replay-token",
                run_id="opt-replay-1",
                epoch=manager.mode_epoch,
            )
        )
    handle = manager.start_job(
        _job_spec(
            tmp_path,
            token="token-one",
            run_id="opt-job-1",
            epoch=manager.mode_epoch,
        )
    )
    assert manager.active_handle is handle
    assert ui_jobs.CALLS == 1
    assert ui_jobs.LAST_LEDGER is handle.ledger
    with pytest.raises(app_module.JobBusyError):
        manager.start_job(
            _job_spec(
                tmp_path,
                token="token-two",
                run_id="opt-job-2",
                epoch=manager.mode_epoch,
            )
        )
    assert handle.request_cancel()
    handle._thread.join(timeout=3)
    assert not handle.active
    assert handle.status == app_module.JOB_STATUS_STOPPED
    assert handle.ledger.cancelled
    assert ui_jobs.CALLS == 1
    with pytest.raises(app_module.JobStaleError):
        manager.start_job(
            _job_spec(
                tmp_path,
                token="token-one",
                run_id="opt-job-3",
                epoch=manager.mode_epoch,
            )
        )


def test_job_manager_invalidates_tokens_on_mode_change_and_restart(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-job-fixture")
    ui_jobs.reset()
    ui_jobs.MODE = "complete"
    manager = app_module.JobManager(
        tmp_path / "runs",
        tmp_path / "programs",
        runner=ui_jobs.make_runner(),
    )
    manager.note_mode(MODE_LIVE)
    stale_epoch = manager.mode_epoch
    manager.note_mode(MODE_REPLAY)
    with pytest.raises(app_module.JobStaleError):
        manager.start_job(
            _job_spec(
                tmp_path,
                token="stale-epoch",
                run_id="opt-stale-epoch",
                epoch=stale_epoch,
            )
        )
    handle = manager.start_job(
        _job_spec(
            tmp_path,
            token="consumed-token",
            run_id="opt-complete-1",
            epoch=manager.mode_epoch,
        )
    )
    handle._thread.join(timeout=3)
    assert handle.status == app_module.JOB_STATUS_COMPLETED
    with pytest.raises(app_module.JobStaleError):
        manager.start_job(
            _job_spec(
                tmp_path,
                token="consumed-token",
                run_id="opt-complete-2",
                epoch=manager.mode_epoch,
            )
        )
    restarted = app_module.JobManager(
        tmp_path / "runs",
        tmp_path / "programs",
        runner=ui_jobs.make_runner(),
    )
    restarted.note_mode(MODE_LIVE)
    with pytest.raises(app_module.JobStaleError):
        restarted.start_job(
            _job_spec(
                tmp_path,
                token="consumed-token",
                run_id="opt-complete-3",
                epoch=restarted.mode_epoch,
            )
        )
    with pytest.raises(app_module.JobStaleError):
        restarted.start_job(
            _job_spec(
                tmp_path,
                token="fresh-after-restart",
                run_id="opt-complete-1",
                epoch=restarted.mode_epoch,
            )
        )


def test_job_progress_polling_does_not_start_another_job(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-job-fixture")
    ui_jobs.reset()
    manager = app_module.JobManager(
        tmp_path / "runs",
        tmp_path / "programs",
        runner=ui_jobs.make_runner(),
    )
    manager.note_mode(MODE_LIVE)
    handle = manager.start_job(
        _job_spec(
            tmp_path,
            token="poll-token",
            run_id="opt-poll-1",
            epoch=manager.mode_epoch,
        )
    )
    for _ in range(100):
        if handle.read_events():
            break
        time.sleep(0.01)
    assert handle.read_events()
    for _ in range(5):
        handle.read_events()
        handle.budget_snapshot()
    assert ui_jobs.CALLS == 1
    assert handle.request_cancel()
    handle._thread.join(timeout=3)
    assert handle.status == app_module.JOB_STATUS_STOPPED


def test_job_manager_marks_interrupted_after_process_restart(
    monkeypatch, tmp_path
) -> None:
    runs = tmp_path / "runs"
    jobs = runs / "jobs"
    jobs.mkdir(parents=True)
    run_dir = runs / "optimization" / "opt-dead"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    atomic_write_json(
        jobs / "opt-dead.json",
        {
            "schema_version": app_module.JOB_SCHEMA_VERSION,
            "run_id": "opt-dead",
            "token_sha256": "deadbeef",
            "mode": MODE_LIVE,
            "status": app_module.JOB_STATUS_RUNNING,
            "pid": 424242,
            "created_at": "2026-10-19T12:00:00Z",
            "started_at": "2026-10-19T12:00:00Z",
            "finished_at": None,
            "error": None,
            "stop_reason": None,
            "baseline_development_run_id": "dev-ui",
            "max_calls": 160,
            "max_cost_usd": "5.00",
        },
    )
    atomic_write_json(
        jobs / app_module.ACTIVE_JOB_LOCK_FILENAME,
        {
            "schema_version": app_module.JOB_SCHEMA_VERSION,
            "run_id": "opt-dead",
            "pid": 424242,
            "created_at": "2026-10-19T12:00:00Z",
        },
    )
    monkeypatch.setattr(app_module, "_pid_alive", lambda _pid: False)
    manager = app_module.JobManager(runs, tmp_path / "programs")
    records = manager.recent_records()
    assert records
    assert records[0]["status"] == app_module.JOB_STATUS_INTERRUPTED
    assert "process restarted" in str(records[0]["error"])
    assert not (jobs / app_module.ACTIVE_JOB_LOCK_FILENAME).exists()
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert "job_interrupted" in events
    assert manager.active_handle is None


def test_run_optimization_rejects_mismatched_injected_ledger(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-job-fixture")
    from dspy_lab.experiments import (
        ExperimentConfigurationError,
        run_optimization,
    )

    settings = load_settings("live")
    mismatched = BudgetLedger(max_calls=1, max_cost_usd="0.01")
    with pytest.raises(
        ExperimentConfigurationError, match="ledger limits"
    ):
        run_optimization(
            settings,
            (),
            (),
            baseline_development_run_id="dev-ui",
            confirm_budget=True,
            runs_root=tmp_path / "runs",
            programs_root=tmp_path / "programs",
            ledger=mismatched,
        )
    matched = BudgetLedger.from_limits(settings.optimization_budget())
    with pytest.raises(ExperimentConfigurationError) as error:
        run_optimization(
            settings,
            (),
            (),
            baseline_development_run_id="dev-ui",
            confirm_budget=True,
            runs_root=tmp_path / "runs",
            programs_root=tmp_path / "programs",
            ledger=matched,
        )
    assert "ledger" not in str(error.value).lower()


def test_live_optimization_requires_fresh_confirmation_and_cancel_is_cooperative(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-job-fixture")
    _forbidden_calls(monkeypatch)
    ui_jobs.reset()
    ui_jobs.MODE = "block"
    app = _run_app(tmp_path, measured=True)
    app = _set_mode(app, MODE_LIVE)
    app = _select_screen(app, "Optimization")
    text = _all_text(app)
    assert "Start confirmed optimization" in text
    assert "I confirm this optimization budget" in text
    assert ui_jobs.CALLS == 0
    app = _click_button(app, "Start confirmed optimization")
    assert "Explicit budget confirmation is required" in _all_text(app)
    assert ui_jobs.CALLS == 0
    app.checkbox[0].check().run()
    assert not app.exception, [str(item.value) for item in app.exception]
    started_run_id = str(app.session_state["opt_confirm_run_id"])
    app = _click_button(app, "Start confirmed optimization")
    assert ui_jobs.CALLS == 1
    assert "running" in _all_text(app).lower()
    app.run()
    assert app.radio[0].disabled
    for _ in range(3):
        app.run()
        assert not app.exception, [str(item.value) for item in app.exception]
    assert ui_jobs.CALLS == 1
    assert "optimization_started" in _all_text(app)
    try:
        app.radio[0].set_value(MODE_REPLAY).run()
    except Exception:  # noqa: BLE001 - disabled widget behavior
        pass
    assert app.radio[0].value == MODE_LIVE
    app.session_state["experiment_mode"] = MODE_REPLAY
    app.run()
    assert app.radio[0].value == MODE_LIVE
    assert "Switching to Replay is blocked" in _all_text(app)
    app = _click_button(app, "Cancel optimization")
    manager = app_module._JOB_MANAGER_OVERRIDE
    assert manager is not None
    for _ in range(150):
        handle = manager.latest_handle
        if handle is None or not handle.active:
            break
        time.sleep(0.01)
    for _ in range(150):
        if app_module.read_active_job_lock(
            tmp_path / "fixture" / "runs"
        ) is None:
            break
        time.sleep(0.01)
    app.run()
    text = _all_text(app)
    assert "stopped" in text.lower()
    assert "incurred are not reversed" in text
    assert ui_jobs.CALLS == 1
    assert not app.radio[0].disabled
    assert "Start confirmed optimization" in _all_text(app)
    assert app.session_state["opt_confirm_run_id"] != started_run_id


def test_live_optimization_without_key_or_development_run_has_no_action(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _forbidden_calls(monkeypatch)
    ui_jobs.reset()
    app = _run_app(tmp_path)
    app = _set_mode(app, MODE_LIVE)
    app = _select_screen(app, "Optimization")
    text = _all_text(app)
    assert "requires a configured OPENROUTER_API_KEY" in text
    assert "Start confirmed optimization" not in text
    assert not app.button
    assert ui_jobs.CALLS == 0

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-job-fixture")
    app = _run_app(tmp_path, build=False)
    app = _set_mode(app, MODE_LIVE)
    app = _select_screen(app, "Optimization")
    text = _all_text(app)
    assert "completed baseline development run is required" in text
    assert "Start confirmed optimization" not in text
    assert not app.button


def test_replay_optimization_has_no_paid_execution_path(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _forbidden_calls(monkeypatch)
    import dspy_lab.experiments as experiments_module

    monkeypatch.setattr(
        experiments_module, "run_optimization", _forbidden
    )
    ui_jobs.reset()
    app = _run_app(tmp_path)
    app = _select_screen(app, "Optimization")
    text = _all_text(app)
    assert "Replay exposes no paid execution path" in text
    assert "Start confirmed optimization" not in text
    assert "Cancel optimization" not in text
    assert not app.button
    assert ui_jobs.CALLS == 0


def test_interrupted_job_record_is_visible_after_restart(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-job-fixture")
    _forbidden_calls(monkeypatch)
    fixture = tmp_path / "fixture"
    build_ui_archive(fixture)
    jobs = fixture / "runs" / "jobs"
    run_dir = fixture / "runs" / "optimization" / "opt-interrupted"
    jobs.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    atomic_write_json(
        jobs / "opt-interrupted.json",
        {
            "schema_version": app_module.JOB_SCHEMA_VERSION,
            "run_id": "opt-interrupted",
            "token_sha256": "interrupted-token",
            "mode": MODE_LIVE,
            "status": app_module.JOB_STATUS_RUNNING,
            "pid": 424242,
            "created_at": "2026-10-19T12:00:00Z",
            "started_at": "2026-10-19T12:00:00Z",
            "finished_at": None,
            "error": None,
            "stop_reason": None,
            "baseline_development_run_id": "dev-ui",
            "max_calls": 160,
            "max_cost_usd": "5.00",
        },
    )
    monkeypatch.setattr(app_module, "_pid_alive", lambda _pid: False)
    app_module._JOB_MANAGER_OVERRIDE = None
    app = AppTest.from_file(
        str(_wrapper(tmp_path, fixture)),
        default_timeout=60,
    ).run()
    assert not app.exception, [str(item.value) for item in app.exception]
    app = _set_mode(app, MODE_LIVE)
    app = _select_screen(app, "Optimization")
    text = _all_text(app)
    assert "Recent Live job records" in text
    assert "opt-interrupted" in text
    assert "interrupted" in text
    assert "Cancel optimization" not in text


def test_job_manager_old_form_generation_is_stale_after_another_job(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-job-fixture")
    ui_jobs.reset()
    ui_jobs.MODE = "complete"
    manager = app_module.JobManager(
        tmp_path / "runs",
        tmp_path / "programs",
        runner=ui_jobs.make_runner(),
    )
    manager.note_mode(MODE_LIVE)
    first = manager.start_job(
        _job_spec(
            tmp_path,
            token="first-token",
            run_id="opt-generation-1",
            epoch=manager.mode_epoch,
            generation=manager.confirmation_generation,
        )
    )
    first._thread.join(timeout=3)
    assert first.status == app_module.JOB_STATUS_COMPLETED
    with pytest.raises(app_module.JobStaleError):
        manager.start_job(
            _job_spec(
                tmp_path,
                token="preopened-second-tab",
                run_id="opt-generation-2",
                epoch=manager.mode_epoch,
                generation=0,
            )
        )
    second = manager.start_job(
        _job_spec(
            tmp_path,
            token="fresh-second-token",
            run_id="opt-generation-3",
            epoch=manager.mode_epoch,
            generation=manager.confirmation_generation,
        )
    )
    second._thread.join(timeout=3)
    assert second.status == app_module.JOB_STATUS_COMPLETED
    assert ui_jobs.CALLS == 2


def test_job_manager_marks_pid_reuse_interrupted(
    monkeypatch, tmp_path
) -> None:
    import os as os_module

    runs = tmp_path / "runs"
    jobs = runs / "jobs"
    jobs.mkdir(parents=True)
    run_dir = runs / "optimization" / "opt-pid-reuse"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    atomic_write_json(
        jobs / "opt-pid-reuse.json",
        {
            "schema_version": app_module.JOB_SCHEMA_VERSION,
            "run_id": "opt-pid-reuse",
            "token_sha256": "pid-reuse-token",
            "mode": MODE_LIVE,
            "status": app_module.JOB_STATUS_RUNNING,
            "pid": os_module.getpid(),
            "process_instance_id": "previous-process-instance",
            "created_at": "2026-10-19T12:00:00Z",
            "started_at": "2026-10-19T12:00:00Z",
            "finished_at": None,
            "error": None,
            "stop_reason": None,
            "baseline_development_run_id": "dev-ui",
            "max_calls": 160,
            "max_cost_usd": "5.00",
        },
    )
    monkeypatch.setattr(app_module, "_pid_alive", lambda _pid: True)
    manager = app_module.JobManager(runs, tmp_path / "programs")
    records = manager.recent_records()
    assert records
    assert records[0]["status"] == app_module.JOB_STATUS_INTERRUPTED
    assert "process restarted" in str(records[0]["error"])


def test_live_optimization_worker_failure_is_visible_and_retry_needs_fresh_form(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-job-fixture")
    _forbidden_calls(monkeypatch)
    ui_jobs.reset()
    ui_jobs.MODE = "fail"
    app = _run_app(tmp_path, measured=True)
    app = _set_mode(app, MODE_LIVE)
    app = _select_screen(app, "Optimization")
    app.checkbox[0].check().run()
    app = _click_button(app, "Start confirmed optimization")
    assert ui_jobs.CALLS == 1
    manager = app_module._JOB_MANAGER_OVERRIDE
    assert manager is not None
    for _ in range(150):
        handle = manager.latest_handle
        if handle is None or not handle.active:
            break
        time.sleep(0.01)
    for _ in range(150):
        if app_module.read_active_job_lock(
            tmp_path / "fixture" / "runs"
        ) is None:
            break
        time.sleep(0.01)
    app.run()
    text = _all_text(app)
    assert "Optimization failed" in text
    assert "fixture worker failure" in text
    assert "Cancel optimization" not in text
    assert "Start confirmed optimization" in text
    assert not app.radio[0].disabled


def test_live_form_worker_uses_one_shared_ledger_and_confirmed_origin(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-job-fixture")
    _forbidden_calls(monkeypatch)
    ui_jobs.reset()
    fixture = tmp_path / "fixture"
    build_ui_archive(fixture, measured=True)
    captured: dict[str, object] = {}

    def fake_run_optimization(settings, train, validation, **kwargs):
        captured.update(kwargs)
        captured["settings"] = settings
        captured["train"] = train
        captured["validation"] = validation
        return SimpleNamespace(
            manifest=SimpleNamespace(
                run_id=kwargs["run_id"],
                status="completed",
                origin="live",
            ),
            selected_program=None,
            warnings=(),
        )

    import dspy_lab.experiments as experiments_module

    monkeypatch.setattr(
        experiments_module, "run_optimization", fake_run_optimization
    )
    app_module._JOB_MANAGER_OVERRIDE = None
    app = AppTest.from_file(
        str(_wrapper(tmp_path, fixture, real_worker=True)),
        default_timeout=60,
    ).run()
    assert not app.exception, [str(item.value) for item in app.exception]
    app = _set_mode(app, MODE_LIVE)
    app = _select_screen(app, "Optimization")
    app.checkbox[0].check().run()
    minted_run_id = str(app.session_state["opt_confirm_run_id"])
    app = _click_button(app, "Start confirmed optimization")
    manager = app_module._JOB_MANAGER_OVERRIDE
    assert manager is not None
    for _ in range(150):
        handle = manager.latest_handle
        if handle is None or not handle.active:
            break
        time.sleep(0.01)
    handle = manager.latest_handle
    assert handle is not None
    assert handle.status == app_module.JOB_STATUS_COMPLETED
    assert captured["run_id"] == minted_run_id
    assert captured["confirm_budget"] is True
    assert captured["origin"] == "live"
    assert captured["ledger"] is handle.ledger
    assert captured["settings"].mode == "live"
    assert len(captured["train"]) == 30
    assert len(captured["validation"]) == 5
    assert ui_jobs.CALLS == 0


def test_mock_development_run_cannot_start_paid_optimization(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-job-fixture")
    _forbidden_calls(monkeypatch)
    ui_jobs.reset()
    app = _run_app(tmp_path)
    app = _set_mode(app, MODE_LIVE)
    app = _select_screen(app, "Optimization")
    text = _all_text(app)
    assert "mock/test record" in text
    assert "paid Live optimization requires a completed Live" in text
    assert "Start confirmed optimization" not in text
    assert not app.button
    assert ui_jobs.CALLS == 0


def test_job_manager_clears_stale_active_lock_before_start(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-job-fixture")
    ui_jobs.reset()
    runs = tmp_path / "runs"
    jobs = runs / "jobs"
    jobs.mkdir(parents=True)
    atomic_write_json(
        jobs / app_module.ACTIVE_JOB_LOCK_FILENAME,
        {
            "schema_version": app_module.JOB_SCHEMA_VERSION,
            "run_id": "opt-stale-lock",
            "pid": 424242,
            "created_at": "2026-10-19T12:00:00Z",
        },
    )
    monkeypatch.setattr(app_module, "_pid_alive", lambda _pid: False)
    manager = app_module.JobManager(
        runs, tmp_path / "programs", runner=ui_jobs.make_runner()
    )
    manager.note_mode(MODE_LIVE)
    handle = manager.start_job(
        _job_spec(
            tmp_path,
            token="stale-lock-token",
            run_id="opt-after-stale-lock",
            epoch=manager.mode_epoch,
            generation=manager.confirmation_generation,
        )
    )
    assert handle.active
    assert handle.request_cancel()
    handle._thread.join(timeout=3)
    assert handle.status == app_module.JOB_STATUS_STOPPED


def test_job_manager_redacts_worker_exception_secrets(
    monkeypatch, tmp_path
) -> None:
    secret = "sk-or-v1-super-secret-fixture"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)

    def leaking_runner(spec, handle):
        raise RuntimeError(f"boom {spec.settings.openrouter_api_key}")

    manager = app_module.JobManager(
        tmp_path / "runs",
        tmp_path / "programs",
        runner=leaking_runner,
    )
    manager.note_mode(MODE_LIVE)
    handle = manager.start_job(
        _job_spec(
            tmp_path,
            token="redaction-token",
            run_id="opt-redaction-1",
            epoch=manager.mode_epoch,
            generation=manager.confirmation_generation,
        )
    )
    handle._thread.join(timeout=3)
    assert handle.status == app_module.JOB_STATUS_FAILED
    assert handle.error is not None
    assert secret not in handle.error
    assert "[REDACTED]" in handle.error


def test_job_manager_process_start_token_mismatch_is_interrupted(
    monkeypatch, tmp_path
) -> None:
    runs = tmp_path / "runs"
    jobs = runs / "jobs"
    jobs.mkdir(parents=True)
    for run_id in ("opt-reused-pid", "opt-matching-pid"):
        (runs / "optimization" / run_id).mkdir(parents=True)
    payload_base = {
        "schema_version": app_module.JOB_SCHEMA_VERSION,
        "token_sha256": "token",
        "mode": MODE_LIVE,
        "status": app_module.JOB_STATUS_RUNNING,
        "created_at": "2026-10-19T12:00:00Z",
        "started_at": "2026-10-19T12:00:00Z",
        "finished_at": None,
        "error": None,
        "stop_reason": None,
        "baseline_development_run_id": "dev-ui",
        "max_calls": 160,
        "max_cost_usd": "5.00",
    }
    atomic_write_json(
        jobs / "opt-reused-pid.json",
        {
            **payload_base,
            "run_id": "opt-reused-pid",
            "pid": 424242,
            "process_start_token": "old-start",
        },
    )
    atomic_write_json(
        jobs / "opt-matching-pid.json",
        {
            **payload_base,
            "run_id": "opt-matching-pid",
            "pid": 424243,
            "process_start_token": "same-start",
        },
    )
    monkeypatch.setattr(app_module, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        app_module,
        "_process_start_token",
        lambda pid: "new-start" if pid == 424242 else "same-start",
    )
    manager = app_module.JobManager(runs, tmp_path / "programs")
    records = {record["run_id"]: record for record in manager.recent_records(10)}
    assert records["opt-reused-pid"]["status"] == app_module.JOB_STATUS_INTERRUPTED
    assert records["opt-matching-pid"]["status"] == app_module.JOB_STATUS_RUNNING


def test_job_manager_exclusive_lock_clears_stale_lock_after_init(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-job-fixture")
    ui_jobs.reset()
    runs = tmp_path / "runs"
    manager = app_module.JobManager(
        runs, tmp_path / "programs", runner=ui_jobs.make_runner()
    )
    manager.note_mode(MODE_LIVE)
    atomic_write_json(
        runs / "jobs" / app_module.ACTIVE_JOB_LOCK_FILENAME,
        {
            "schema_version": app_module.JOB_SCHEMA_VERSION,
            "run_id": "opt-stale-after-init",
            "pid": 424242,
            "created_at": "2026-10-19T12:00:00Z",
        },
    )
    monkeypatch.setattr(app_module, "_pid_alive", lambda _pid: False)
    handle = manager.start_job(
        _job_spec(
            tmp_path,
            token="stale-after-init",
            run_id="opt-after-init-lock",
            epoch=manager.mode_epoch,
            generation=manager.confirmation_generation,
        )
    )
    assert handle.active
    assert handle.request_cancel()
    handle._thread.join(timeout=3)
    assert handle.status == app_module.JOB_STATUS_STOPPED
    record = json.loads(
        (runs / "jobs" / "opt-after-init-lock.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["run_id"] == "opt-after-init-lock"


def test_job_manager_exclusive_lock_rejects_live_holder(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-job-fixture")
    runs = tmp_path / "runs"
    manager = app_module.JobManager(
        runs, tmp_path / "programs", runner=ui_jobs.make_runner()
    )
    manager.note_mode(MODE_LIVE)
    atomic_write_json(
        runs / "jobs" / app_module.ACTIVE_JOB_LOCK_FILENAME,
        {
            "schema_version": app_module.JOB_SCHEMA_VERSION,
            "run_id": "opt-other-process",
            "pid": os.getpid(),
            "process_instance_id": app_module.PROCESS_INSTANCE_ID,
            "process_start_token": app_module._process_start_token(
                os.getpid()
            ),
            "created_at": "2026-10-19T12:00:00Z",
        },
    )
    monkeypatch.setattr(
        app_module, "read_active_job_lock", lambda _root: None
    )
    with pytest.raises(app_module.JobBusyError):
        manager.start_job(
            _job_spec(
                tmp_path,
                token="live-holder-token",
                run_id="opt-live-holder",
                epoch=manager.mode_epoch,
                generation=manager.confirmation_generation,
            )
        )


def test_job_manager_cancel_before_dispatch_is_stopped(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-job-fixture")

    def cancel_then_raise(_spec, handle):
        handle.ledger.cancel("fixture_cancel")
        raise RuntimeError("cancelled before dispatch")

    manager = app_module.JobManager(
        tmp_path / "runs",
        tmp_path / "programs",
        runner=cancel_then_raise,
    )
    manager.note_mode(MODE_LIVE)
    handle = manager.start_job(
        _job_spec(
            tmp_path,
            token="cancel-before-dispatch",
            run_id="opt-cancel-before-dispatch",
            epoch=manager.mode_epoch,
            generation=manager.confirmation_generation,
        )
    )
    handle._thread.join(timeout=3)
    assert handle.status == app_module.JOB_STATUS_STOPPED
    assert handle.ledger.snapshot().stop_reason == "cancelled"


def _tree_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            hashes[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return hashes


def _set_text_input(app: AppTest, label: str, value: str, *, run: bool = True) -> AppTest:
    for item in app.text_input:
        if item.label == label:
            item.set_value(value)
            if run:
                app.run()
                assert not app.exception, [
                    str(error.value) for error in app.exception
                ]
            return app
    raise AssertionError(f"text input {label!r} not found")


def _set_text_area(app: AppTest, label: str, value: str, *, run: bool = True) -> AppTest:
    for item in app.text_area:
        if item.label == label:
            item.set_value(value)
            if run:
                app.run()
                assert not app.exception, [
                    str(error.value) for error in app.exception
                ]
            return app
    raise AssertionError(f"text area {label!r} not found")


def _check_checkbox(
    app: AppTest, label: str, *, run: bool = True
) -> AppTest:
    for item in app.checkbox:
        if item.label == label:
            item.check()
            if run:
                app.run()
                assert not app.exception, [
                    str(error.value) for error in app.exception
                ]
            return app
    raise AssertionError(f"checkbox {label!r} not found")


def test_replay_comparison_displays_stored_metrics_and_case_changes(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _forbidden_calls(monkeypatch)
    app = _run_app(tmp_path, measured=True, changed=True)
    before = _tree_hashes(tmp_path / "fixture")
    app = _select_screen(app, "Comparison")
    text = _all_text(app)
    assert "Paired quality metrics" in text
    assert "Micro-F1" in text
    assert "Macro-F1" in text
    assert "Exposure index" in text
    assert "Corrected gold issues" in text
    assert "Optimization candidate history" in text
    assert "Historical costs and conditional break-even" in text
    assert (
        "Stored aggregate metrics were recomputed from the archived raw "
        "findings and gold and matched." in text
    )
    assert "Stored aggregate metrics disagree" not in text
    assert "Additional API calls: 0" in text
    summary = json.loads(
        (
            tmp_path
            / "fixture"
            / "runs"
            / "benchmarks"
            / "bench-ui"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    for arm in ("baseline", "optimized"):
        micro = summary["metrics"][arm]["micro_f1"]
        if micro is not None:
            assert f"{micro * 100:.1f}%" in text
    candidate_box = next(
        box
        for box in app.selectbox
        if box.label == "Recorded candidate instructions"
    )
    candidate_box.set_value(candidate_box.options[0]).run()
    candidate_text = _all_text(app)
    assert "Candidate index:** 0" in candidate_text
    assert "This candidate is the unchanged baseline instructions" in candidate_text
    assert _tree_hashes(tmp_path / "fixture") == before


def test_repeated_replay_reads_do_not_modify_archive_or_call_provider(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _forbidden_calls(monkeypatch)
    app = _run_app(tmp_path, measured=True)
    fixture = tmp_path / "fixture"
    before = _tree_hashes(fixture)
    manifest_before = (
        fixture / "runs" / "benchmarks" / "bench-ui" / "manifest.json"
    ).read_text(encoding="utf-8")
    for screen in (
        "Overview",
        "Baseline",
        "Optimization",
        "Prompt Inspector",
        "Comparison",
        "Playground",
    ):
        app = _select_screen(app, screen)
    for _ in range(2):
        app = _select_screen(app, "Comparison")
        app = _select_screen(app, "Playground")
    assert _tree_hashes(fixture) == before
    assert (
        fixture / "runs" / "benchmarks" / "bench-ui" / "manifest.json"
    ).read_text(encoding="utf-8") == manifest_before


def test_replay_playground_has_no_inference_form(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _forbidden_calls(monkeypatch)
    app = _run_app(tmp_path, measured=True)
    app = _select_screen(app, "Playground")
    text = _all_text(app)
    assert "new Playground inference is disabled" in text
    assert "Run unscored playground" not in text
    assert all(item.label != "Article title" for item in app.text_input)
    assert all(item.label != "Article body" for item in app.text_area)
    assert not app.button


def test_live_playground_requires_separate_confirmation_and_budget(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-s14-fixture")
    _forbidden_calls(monkeypatch)
    ui_jobs.reset()
    ui_jobs.MODE = "complete"
    app = _run_app(
        tmp_path,
        measured=True,
        runner_expr="ui_jobs.make_playground_runner()",
    )
    app = _set_mode(app, MODE_LIVE)
    app = _select_screen(app, "Playground")
    text = _all_text(app)
    assert "Playground budget:** 2 calls / 0.15 USD" in text
    assert "UNSCORED" not in text
    app = _set_text_input(
        app, "Article title", "Nová zlatá mince", run=False
    )
    app = _set_text_area(
        app,
        "Article body",
        "Mince má zaručený výnos. Podle prodejce je investice jistá.",
        run=False,
    )
    app = _click_button(app, "Run unscored playground")
    assert "Explicit playground budget confirmation is required" in _all_text(app)
    assert ui_jobs.PLAYGROUND_CALLS == 0
    app = _check_checkbox(
        app,
        "I confirm the separate playground budget and that article content "
        "will be sent to the configured provider.",
        run=False,
    )
    app = _click_button(app, "Run unscored playground")
    assert ui_jobs.PLAYGROUND_CALLS == 1
    assert ui_jobs.CALLS == 0
    spec = ui_jobs.LAST_PLAYGROUND_SPEC
    assert spec is not None
    assert spec.title == "Nová zlatá mince"
    assert spec.retain_content is False
    manager = app_module._JOB_MANAGER_OVERRIDE
    assert manager is not None
    handle = manager.latest_handle_for("playground")
    assert handle is not None
    assert handle.ledger.max_calls == 2
    assert str(handle.ledger.max_cost_usd) == "0.15"
    for _ in range(50):
        app.run()
        if "UNSCORED PLAYGROUND" in _all_text(app):
            break
        time.sleep(0.01)
    text = _all_text(app)
    assert "UNSCORED PLAYGROUND" in text
    assert "gold labels or benchmark metric" in text
    assert "Program ID:** `program-fixture-baseline`" in text


def test_live_comparison_benchmark_requires_own_confirmation(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-s14-benchmark")
    _forbidden_calls(monkeypatch)
    ui_jobs.reset()
    ui_jobs.MODE = "complete"
    app = _run_app(
        tmp_path,
        measured=True,
        runner_expr="ui_jobs.make_benchmark_runner()",
    )
    app = _set_mode(app, MODE_LIVE)
    app = _select_screen(app, "Comparison")
    text = _all_text(app)
    assert "Start confirmed benchmark" in text
    assert "opens the sealed held-out split" in text
    assert "Recorded benchmark archive" in text
    app = _click_button(app, "Start confirmed benchmark")
    assert "Explicit benchmark budget confirmation is required" in _all_text(app)
    assert ui_jobs.BENCHMARK_CALLS == 0
    app = _check_checkbox(
        app,
        "I confirm this benchmark budget and that this action opens the "
        "sealed held-out split.",
    )
    app = _click_button(app, "Start confirmed benchmark")
    assert ui_jobs.BENCHMARK_CALLS == 1
    spec = ui_jobs.LAST_BENCHMARK_SPEC
    assert spec is not None
    assert spec.baseline_development_run_id == "dev-ui"
    assert spec.optimization_run_id == "opt-ui"
    assert spec.release_results is False
    assert spec.test_cases is None
    manager = app_module._JOB_MANAGER_OVERRIDE
    assert manager is not None
    handle = manager.latest_handle_for("benchmark")
    assert handle is not None
    assert handle.ledger.max_calls == 20
    assert str(handle.ledger.max_cost_usd) == "0.50"


def test_live_comparison_corrupt_archive_is_explicit(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-s14-corrupt")
    _forbidden_calls(monkeypatch)
    app = _run_app(tmp_path, measured=True)
    outcomes = (
        tmp_path
        / "fixture"
        / "runs"
        / "benchmarks"
        / "bench-ui"
        / "outcomes.jsonl"
    )
    outcomes.unlink()
    app = _set_mode(app, MODE_LIVE)
    app = _select_screen(app, "Comparison")
    text = _all_text(app)
    assert "selected benchmark archive cannot be read" in text
    assert "Nothing is regenerated" in text


def test_playground_job_manager_uses_shared_single_job_gate(
    tmp_path,
) -> None:
    runs = tmp_path / "runs"
    programs = tmp_path / "programs"
    settings = app_module.Settings(
        mode="live", openrouter_api_key="sk-or-v1-s14-gate"
    )
    manager = app_module.JobManager(
        runs,
        programs,
        runner=lambda spec, handle: SimpleNamespace(status="completed"),
    )
    playground_handle = manager.start_playground(
        app_module.PlaygroundJobSpec(
            mode=MODE_LIVE,
            run_id="play-gate",
            token="token-play",
            mode_epoch=manager.mode_epoch,
            confirmation_generation=manager.confirmation_generation,
            settings=settings,
            optimization_run_id="opt-ui",
            title="Title",
            article_text="Body",
            product_facts=app_module.ProductFacts(),
            sources=(),
            retain_content=False,
            runs_root=runs,
            programs_root=programs,
        )
    )
    with pytest.raises(app_module.JobBusyError):
        manager.start_benchmark(
            app_module.BenchmarkJobSpec(
                mode=MODE_LIVE,
                run_id="bench-gate",
                token="token-bench",
                mode_epoch=manager.mode_epoch,
                confirmation_generation=manager.confirmation_generation,
                settings=settings,
                baseline_development_run_id="dev-ui",
                optimization_run_id="opt-ui",
                release_results=False,
                runs_root=runs,
                programs_root=programs,
            )
        )
    playground_handle._thread.join(timeout=5)
    assert playground_handle.status == "completed"
    assert manager.active_handle is None


def test_replay_comparison_flags_tampered_stored_metrics(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _forbidden_calls(monkeypatch)
    app = _run_app(tmp_path, measured=True)
    benchmark_dir = (
        tmp_path / "fixture" / "runs" / "benchmarks" / "bench-ui"
    )
    summary_path = benchmark_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metrics"]["baseline"]["tp"] += 1
    atomic_write_json(summary_path, summary)
    manifest_path = benchmark_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["summary.json"] = hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    atomic_write_json(manifest_path, manifest)
    app = _select_screen(app, "Comparison")
    text = _all_text(app)
    assert "Stored aggregate metrics disagree" in text
    assert "Stored baseline tp" in text
    assert "does not match the value recomputed" in text


def test_benchmark_worker_passes_the_handle_ledger(
    monkeypatch, tmp_path
) -> None:
    import dspy_lab.experiments as experiments

    settings = app_module.Settings(
        mode="live", openrouter_api_key="sk-or-v1-s14-worker"
    )
    ledger = BudgetLedger.from_limits(settings.benchmark_budget())
    handle = app_module.JobHandle(
        run_id="bench-worker",
        token_sha256="a" * 64,
        mode_epoch=1,
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
        settings=settings,
        ledger=ledger,
        events_path=tmp_path / "runs" / "benchmarks" / "bench-worker" / "events.jsonl",
        started_at="2026-10-19T12:00:00Z",
        job_kind="benchmark",
    )
    captured: dict[str, object] = {}

    def fake_benchmark(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(status="completed")

    monkeypatch.setattr(experiments, "run_benchmark", fake_benchmark)
    monkeypatch.setattr(
        experiments, "load_benchmark_cases", lambda _data_dir: ()
    )
    spec = app_module.BenchmarkJobSpec(
        mode=MODE_LIVE,
        run_id="bench-worker",
        token="token",
        mode_epoch=1,
        confirmation_generation=1,
        settings=settings,
        baseline_development_run_id="dev-ui",
        optimization_run_id="opt-ui",
        release_results=False,
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
        test_cases=(),
    )
    app_module._run_benchmark_job(spec, handle)
    assert captured["ledger"] is ledger


def test_comparison_recompute_cost_status_precedence(tmp_path) -> None:
    build_ui_archive(tmp_path, measured=True)
    archive = open_replay_archive(
        tmp_path / "runs" / "benchmarks" / "bench-ui",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )
    mixed = [
        outcome.model_copy(
            update={
                "cost_status": (
                    "estimated"
                    if outcome.phase == "baseline" and index == 0
                    else "actual"
                ),
                "cost_usd": Decimal("0.001"),
            }
        )
        for index, outcome in enumerate(archive.outcomes)
    ]
    metrics = app_module._comparison_recompute(archive.cases, mixed)
    assert metrics["baseline"]["cost_status"] == "estimated"
    assert metrics["optimized"]["cost_status"] == "actual"
    unavailable = [
        outcome.model_copy(
            update={"cost_status": "unavailable", "cost_usd": None}
        )
        if outcome.phase == "baseline"
        else outcome
        for outcome in archive.outcomes
    ]
    metrics = app_module._comparison_recompute(
        archive.cases, unavailable
    )
    assert metrics["baseline"]["cost_status"] == "unavailable"
    assert metrics["baseline"]["cost_per_article_usd"] is None


def test_comparison_rescore_preserves_invalid_outcomes(tmp_path) -> None:
    build_ui_archive(tmp_path, measured=True)
    archive = open_replay_archive(
        tmp_path / "runs" / "benchmarks" / "bench-ui",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )
    case = next(
        case for case in archive.cases if case.gold_findings
    )
    outcome = next(
        item for item in archive.outcomes if item.case_id == case.case_id
    ).model_copy(
        update={
            "status": "error",
            "findings": None,
            "error": "parse_error: fixture invalid output",
            "metric": {"error_kind": "parse_error"},
        }
    )
    score = app_module._rescore_outcome(case, outcome)
    assert score.valid is False
    assert score.error_kind == "parse_error"
    assert score.tp == 0
    assert score.fp == 0
    assert score.fn == len(case.gold_findings)
