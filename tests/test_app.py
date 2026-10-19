from __future__ import annotations

import json
from pathlib import Path
import sys
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
from dspy_lab.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
import app as app_module

from ui_fixture import build_ui_archive

MODE_REPLAY = "Replay Experiment"
MODE_LIVE = "Live Experiment"


def _forbidden(*_args, **_kwargs):
    raise AssertionError("Replay/Live read-only screens must not call this")


def _wrapper(tmp_path: Path, fixture: Path | None) -> Path:
    runs = fixture / "runs" if fixture is not None else tmp_path / "empty-runs"
    programs = (
        fixture / "programs" if fixture is not None else tmp_path / "empty-programs"
    )
    wrapper = tmp_path / "ui_wrapper.py"
    wrapper.write_text(
        "\n".join(
            [
                "import sys",
                "from pathlib import Path",
                f"sys.path.insert(0, {str(PROJECT_ROOT)!r})",
                "import app",
                f"app.RUNS_ROOT = Path({str(runs)!r})",
                f"app.PROGRAMS_ROOT = Path({str(programs)!r})",
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
    app = AppTest.from_file(
        str(_wrapper(tmp_path, fixture)), default_timeout=60
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
    for box in app.selectbox:
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
    app = AppTest.from_file(
        str(_wrapper(tmp_path, fixture)), default_timeout=60
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
    assert "Live Experiment — read-only in this stage" in _all_text(app)
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
    app = AppTest.from_file(
        str(_wrapper(tmp_path, fixture)), default_timeout=60
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
        str(_wrapper(tmp_path, fixture)), default_timeout=60
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
