from __future__ import annotations

import copy

import json
import shutil
from decimal import Decimal
from pathlib import Path

import dspy
import dspy.clients.lm as dspy_lm
import pytest

import dspy_lab.release as release
from dspy_lab.config import Settings
from dspy_lab.data import load_cases
from dspy_lab.experiments import (
    run_baseline_development,
    run_benchmark,
    run_optimization,
)

from test_benchmark import (
    benchmark_handler,
    install_transport,
    optimization_handler,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data/v1"
SECRET = "sk-or-v1-s15-release-sentinel"


def make_settings() -> Settings:
    return Settings(
        mode="live",
        openrouter_api_key=SECRET,
        max_baseline_dev_calls=10,
        max_baseline_dev_cost_usd=Decimal("0.50"),
        max_optimization_calls=400,
        max_optimization_metric_calls=20,
        max_optimization_cost_usd=Decimal("5.00"),
        max_benchmark_calls=20,
        max_benchmark_cost_usd=Decimal("0.50"),
        request_timeout_seconds=5.0,
    )


@pytest.fixture(scope="module")
def release_workspace(tmp_path_factory):
    """Temporary mocked-transport archive for validator tests only.

    It exercises the real S10/S11 writers and is never presented as an
    experiment result outside this test module.
    """
    tmp = tmp_path_factory.mktemp("release-workspace")
    runs = tmp / "runs"
    programs = tmp / "programs"
    settings = make_settings()
    train = load_cases(DATA_DIR / "train.jsonl")[:2]
    development = load_cases(DATA_DIR / "validation.jsonl")
    test = load_cases(DATA_DIR / "test.jsonl")
    with pytest.MonkeyPatch.context() as monkeypatch:
        calls = install_transport(monkeypatch, optimization_handler)
        development_run = run_baseline_development(
            settings,
            development,
            confirm_budget=True,
            run_id="dev-s15",
            origin="live",
            runs_root=runs,
            programs_root=programs,
        )
        assert development_run.manifest.status == "completed"
        optimization_run = run_optimization(
            settings,
            train,
            development,
            baseline_development_run_id="dev-s15",
            confirm_budget=True,
            run_id="opt-s15",
            origin="live",
            runs_root=runs,
            programs_root=programs,
        )
        assert optimization_run.manifest.status == "completed"
        install_transport(monkeypatch, benchmark_handler(test))
        benchmark = run_benchmark(
            settings,
            test,
            baseline_development_run_id="dev-s15",
            optimization_run_id="opt-s15",
            confirm_budget=True,
            release_results=True,
            origin="live",
            run_id="bench-s15",
            runs_root=runs,
            programs_root=programs,
        )
        assert benchmark.manifest.status == "completed"
        assert benchmark.manifest.benchmark.released is True
        dispatch_count = len(calls)
    yield {
        "tmp": tmp,
        "runs": runs,
        "programs": programs,
        "dispatch_count": dispatch_count,
    }


def test_release_validation_passes_on_realistic_archive(
    release_workspace,
) -> None:
    runs = release_workspace["runs"]
    programs = release_workspace["programs"]
    before = release._tree_hash(runs / "benchmarks" / "bench-s15")
    report = release.validate_release(
        runs_root=runs,
        programs_root=programs,
        repeats=2,
    )
    after = release._tree_hash(runs / "benchmarks" / "bench-s15")
    assert before == after
    assert report.passed is True
    assert report.benchmark_run_id == "bench-s15"
    assert report.measured_archive is True
    assert report.verdict == "observed_heldout_improvement"
    assert report.exposure_index == 0
    checks = {check.check_id: check for check in report.checks}
    assert checks["metrics"].status == "pass"
    assert checks["paired_inputs"].status == "pass"
    assert checks["test_index"].status == "pass"
    assert checks["secret_scan"].status == "pass"
    acceptance = {item.check_id: item for item in report.acceptance}
    assert set(acceptance) == {f"A{i}" for i in range(1, 14)}
    assert acceptance["A1"].status == "partial"
    assert acceptance["A9"].status == "not_covered"
    assert acceptance["A10"].status == "not_covered"
    assert acceptance["A13"].status == "partial"
    payload = report.to_payload()
    assert payload["schema_version"] == release.RELEASE_REPORT_SCHEMA_VERSION
    assert payload["summary"]["metrics"]["baseline"]["article_count"] == 10
    assert SECRET not in json.dumps(payload, ensure_ascii=False)


def test_release_validation_does_not_dispatch_after_build(
    release_workspace,
) -> None:
    calls: list[dict] = []

    def forbidden(*args, **kwargs):
        calls.append(kwargs)
        raise AssertionError("release validation must not dispatch")

    original_lm = dspy_lm.litellm_completion
    original_dspy_lm = dspy.LM
    try:
        dspy_lm.litellm_completion = forbidden
        dspy.LM = forbidden
        report = release.validate_release(
            runs_root=release_workspace["runs"],
            programs_root=release_workspace["programs"],
            repeats=3,
        )
    finally:
        dspy_lm.litellm_completion = original_lm
        dspy.LM = original_dspy_lm
    assert report.passed is True
    assert release_workspace["dispatch_count"] > 0
    assert calls == []


def test_release_validation_rejects_unreleased_archive(
    monkeypatch, tmp_path
) -> None:
    from ui_fixture import build_ui_archive

    build_ui_archive(tmp_path, measured=False)
    with pytest.raises(release.ReleaseValidationError):
        release.validate_release(
            runs_root=tmp_path / "runs",
            programs_root=tmp_path / "programs",
        )


def test_release_validation_rejects_tampered_archive(
    release_workspace, tmp_path
) -> None:
    copied = tmp_path / "copy"
    shutil.copytree(release_workspace["tmp"], copied)
    summary = (
        copied
        / "runs"
        / "benchmarks"
        / "bench-s15"
        / "summary.json"
    )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["metrics"]["baseline"]["tp"] += 1
    summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(release.ReleaseValidationError, match="hash|corrupt"):
        release.validate_release(
            runs_root=copied / "runs",
            programs_root=copied / "programs",
        )


def test_release_cli_writes_outside_archive(
    release_workspace, tmp_path, capsys
) -> None:
    report_path = tmp_path / "acceptance.json"
    exit_code = release.main(
        [
            "validate",
            "--benchmark-run-id",
            "bench-s15",
            "--runs-root",
            str(release_workspace["runs"]),
            "--programs-root",
            str(release_workspace["programs"]),
            "--repeats",
            "2",
            "--json",
            str(report_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["benchmark_run_id"] == "bench-s15"
    captured = capsys.readouterr()
    assert "Release validation: PASS" in captured.out
    assert "Acceptance report written" in captured.out
    with pytest.raises(release.ReleaseValidationError, match="inside"):
        release._write_report(
            release.validate_release(
                runs_root=release_workspace["runs"],
                programs_root=release_workspace["programs"],
                repeats=2,
            ),
            release_workspace["runs"]
            / "benchmarks"
            / "bench-s15"
            / "acceptance.json",
        )


def test_baseline_command_requires_explicit_confirmation() -> None:
    assert release.main(["baseline-development"]) == 2


def test_release_validation_retains_invalid_outputs(
    release_workspace, tmp_path
) -> None:
    copied = tmp_path / "invalid"
    shutil.copytree(release_workspace["tmp"], copied)
    test = load_cases(DATA_DIR / "test.jsonl")
    settings = make_settings()
    with pytest.MonkeyPatch.context() as monkeypatch:
        install_transport(
            monkeypatch, benchmark_handler(test, baseline_invalid=True)
        )
        benchmark = run_benchmark(
            settings,
            test,
            baseline_development_run_id="dev-s15",
            optimization_run_id="opt-s15",
            confirm_budget=True,
            release_results=True,
            origin="live",
            run_id="bench-invalid",
            runs_root=copied / "runs",
            programs_root=copied / "programs",
        )
    assert benchmark.summary["verdict"] == "validity_problem"
    report = release.validate_release(
        benchmark_run_id="bench-invalid",
        runs_root=copied / "runs",
        programs_root=copied / "programs",
        repeats=2,
    )
    assert report.passed is True
    assert report.verdict == "validity_problem"
    baseline = report.summary["metrics"]["baseline"]
    assert baseline["article_count"] == 10
    assert baseline["invalid_count"] == 10
    assert baseline["parse_error_count"] == 10
    assert baseline["valid_count"] == 0
    checks = {check.check_id: check for check in report.checks}
    assert checks["paired_inputs"].status == "pass"
    assert checks["metrics"].status == "pass"


def test_perfect_baseline_skip_is_accepted_as_optimization_evidence() -> None:
    assert release._has_optimization_evidence(
        None,
        [
            "The baseline was perfect on the validation set and no "
            "reflection was attempted; GEPA's perfect-score skip left "
            "candidate search unexercised."
        ],
    )
    assert not release._has_optimization_evidence(
        None, ["The optimization run stopped at its budget."]
    )
    assert not release._has_optimization_evidence(
        None, ["The baseline was not perfect; search continued."]
    )
    assert release._has_optimization_evidence({}, [])


def test_release_validation_explicit_unreleased_archive_fails(
    tmp_path,
) -> None:
    from ui_fixture import build_ui_archive

    build_ui_archive(tmp_path, measured=False)
    report = release.validate_release(
        benchmark_run_id="bench-ui",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
        repeats=2,
    )
    assert report.measured_archive is False
    assert report.passed is False


def test_release_validation_warns_on_reused_test_exposure(
    release_workspace, tmp_path
) -> None:
    copied = tmp_path / "reused"
    shutil.copytree(release_workspace["tmp"], copied)
    test = load_cases(DATA_DIR / "test.jsonl")
    settings = make_settings()
    with pytest.MonkeyPatch.context() as monkeypatch:
        install_transport(monkeypatch, benchmark_handler(test))
        benchmark = run_benchmark(
            settings,
            test,
            baseline_development_run_id="dev-s15",
            optimization_run_id="opt-s15",
            confirm_budget=True,
            release_results=True,
            origin="live",
            run_id="bench-reuse",
            runs_root=copied / "runs",
            programs_root=copied / "programs",
        )
    assert benchmark.manifest.benchmark.exposure_index == 1
    report = release.validate_release(
        benchmark_run_id="bench-reuse",
        runs_root=copied / "runs",
        programs_root=copied / "programs",
        repeats=2,
    )
    assert report.passed is True
    assert report.exposure_index == 1
    checks = {check.check_id: check for check in report.checks}
    assert checks["first_exposure"].status == "warning"
    assert any(
        "Exploratory reevaluation" in warning for warning in report.warnings
    )


def test_release_report_cannot_be_written_into_program_tree(
    release_workspace,
) -> None:
    report = release.validate_release(
        runs_root=release_workspace["runs"],
        programs_root=release_workspace["programs"],
        repeats=2,
    )
    with pytest.raises(
        release.ReleaseValidationError, match="program artifact"
    ):
        release._write_report(
            report,
            release_workspace["programs"] / "program-id" / "acceptance.json",
            protected_roots=(release_workspace["programs"],),
        )


def test_release_validation_fails_on_credential_key(
    release_workspace, tmp_path
) -> None:
    copied = tmp_path / "credential"
    shutil.copytree(release_workspace["tmp"], copied)
    config_path = (
        copied / "runs" / "benchmarks" / "bench-s15" / "config.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["api_key"] = "fixture-credential-value"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path = config_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["config.json"] = release._sha256_file(config_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = release.validate_release(
        benchmark_run_id="bench-s15",
        runs_root=copied / "runs",
        programs_root=copied / "programs",
        repeats=2,
    )
    assert report.passed is False
    checks = {check.check_id: check for check in report.checks}
    assert checks["secret_scan"].status == "fail"
    assert "config.json" in checks["secret_scan"].detail


def test_release_does_not_certify_automated_only_gold_approval(
    release_workspace,
) -> None:
    report = release.validate_release(
        runs_root=release_workspace["runs"],
        programs_root=release_workspace["programs"],
        repeats=2,
    )
    checks = {check.check_id: check for check in report.checks}
    acceptance = {item.check_id: item for item in report.acceptance}
    assert checks["gold_approval"].status == "warning"
    assert "human gold sign-off" in checks["gold_approval"].detail
    assert acceptance["A2"].status == "partial"
    assert any(
        warning.startswith("gold_approval:")
        for warning in report.warnings
    )


def test_release_recomputes_verdict_and_economics(release_workspace) -> None:
    from dspy_lab.artifacts import open_replay_archive

    runs = release_workspace["runs"]
    programs = release_workspace["programs"]
    benchmark_dir = runs / "benchmarks" / "bench-s15"
    archive = open_replay_archive(
        benchmark_dir, runs_root=runs, programs_root=programs
    )
    summary = json.loads(
        (benchmark_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert release._outcome_consistency_problems(archive, summary) == []
    tampered = copy.deepcopy(summary)
    tampered["verdict"] = "baseline_preferred"
    problems = release._outcome_consistency_problems(archive, tampered)
    assert any("verdict" in problem for problem in problems)
    tampered = copy.deepcopy(summary)
    tampered["economics"]["break_even_articles"] += 1
    problems = release._outcome_consistency_problems(archive, tampered)
    assert any("economics" in problem for problem in problems)


def test_records_cost_empty_ledger_is_unavailable() -> None:
    assert release._records_cost([]) == (Decimal("0"), "unavailable")
