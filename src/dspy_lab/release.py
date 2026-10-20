"""Release validation and offline presentation checks for S15.

This module validates an already completed and released Live benchmark
archive. It never constructs an LM, calls a provider, opens the current test
split, or writes into the archive. It is the only S15 entry point that reads a
genuine released archive and is safe to run with no API key or Internet access.

The first held-out run itself remains an operator action: run the Live
development, optimization, and benchmark phases with their separate confirmed
budgets, then pass the resulting benchmark run ID to ``validate``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from .artifacts import (
    PROGRAMS_ROOT,
    RUNS_ROOT,
    ArtifactError,
    ReplayArchive,
    atomic_write_json,
    is_measured_archive,
    list_run_directories,
    open_replay_archive,
)
from .config import (
    PROJECT_ROOT,
    Settings,
    load_settings,
)
from .data import (
    DatasetValidationError,
    canonical_document_sha256,
    dataset_sha256,
    load_cases,
    load_review_audit,
    load_split_manifest,
    validate_review_audit,
)
from .experiments import (
    ArmMeasurement,
    ExperimentError,
    PairedMeasurement,
    run_baseline_development,
    summarize_benchmark,
)
from .metrics import CaseScore, CorpusScore, aggregate_scores
from .recorded_scoring import (
    RecordedOutcomeScoringError,
    rescore_recorded_outcome,
)
from .schemas import ALLOWED_CODES, CaseRecord

DATA_DIR = PROJECT_ROOT / "data/v1"
RELEASE_REPORT_SCHEMA_VERSION = "release_validation_v1"
SECRET_PATTERN = re.compile(rb"sk-or-v1-[A-Za-z0-9_-]{16,}")
CREDENTIAL_KEY_PATTERN = re.compile(
    rb'"(?:api[_-]?key|authorization|bearer|secret[_-]?key|access[_-]?token)"'
    rb'\s*:\s*"(?!\[redacted\])[^"]+"',
    re.IGNORECASE,
)
VERDICTS = {
    "observed_heldout_improvement",
    "regression",
    "tie",
    "mixed_tradeoff",
    "validity_problem",
    "not_comparable",
    "incomplete",
}
METRIC_KEYS = (
    "article_count",
    "valid_count",
    "invalid_count",
    "parse_error_count",
    "request_error_count",
    "missing_output_count",
    "evidence_failure_count",
    "duplicate_prediction_count",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "micro_f1",
    "macro_f1",
    "exact_match_count",
    "exact_match_rate",
    "parsing_success",
    "mean_reward",
)


class ReleaseValidationError(RuntimeError):
    """The archive cannot be validated as a genuine released experiment."""


@dataclass(frozen=True, slots=True)
class ReleaseCheck:
    check_id: str
    status: str
    detail: str


@dataclass(frozen=True, slots=True)
class ReleaseReport:
    schema_version: str
    generated_at: str
    benchmark_run_id: str
    archive_dir: Path
    passed: bool
    verdict: str | None
    exposure_index: int | None
    measured_archive: bool
    checks: tuple[ReleaseCheck, ...]
    acceptance: tuple[ReleaseCheck, ...]
    artifact_hashes: Mapping[str, str]
    summary: Mapping[str, object]
    warnings: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "benchmark_run_id": self.benchmark_run_id,
            "archive_dir": str(self.archive_dir),
            "passed": self.passed,
            "verdict": self.verdict,
            "exposure_index": self.exposure_index,
            "measured_archive": self.measured_archive,
            "checks": [
                {
                    "id": check.check_id,
                    "status": check.status,
                    "detail": check.detail,
                }
                for check in self.checks
            ],
            "acceptance": [
                {
                    "id": check.check_id,
                    "status": check.status,
                    "detail": check.detail,
                }
                for check in self.acceptance
            ],
            "artifact_hashes": dict(self.artifact_hashes),
            "summary": dict(self.summary),
            "warnings": list(self.warnings),
        }


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.is_dir():
        return digest.hexdigest()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleaseValidationError(f"missing file: {path}") from exc
    except ValueError as exc:
        raise ReleaseValidationError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseValidationError(f"expected a JSON object: {path}")
    return payload


def _resolve_benchmark_dir(
    benchmark_run_id: str | None, runs_root: Path
) -> Path:
    if benchmark_run_id is not None:
        benchmark_root = (Path(runs_root) / "benchmarks").resolve()
        run_dir = (benchmark_root / benchmark_run_id).resolve()
        if not run_dir.is_dir():
            raise ReleaseValidationError(
                f"benchmark run directory is missing: "
                f"{benchmark_root / benchmark_run_id}"
            )
        if not run_dir.is_relative_to(benchmark_root):
            raise ReleaseValidationError(
                "benchmark run path escapes the runs/benchmarks directory"
            )
        return run_dir
    candidates: list[Path] = []
    for entry in list_run_directories(Path(runs_root), kind="benchmark"):
        manifest = entry.manifest
        if (
            manifest is not None
            and manifest.status == "completed"
            and manifest.origin == "live"
            and manifest.benchmark is not None
            and manifest.benchmark.released
        ):
            candidates.append(entry.run_dir)
    if not candidates:
        raise ReleaseValidationError(
            "no completed, released Live benchmark archive exists under "
            f"{Path(runs_root) / 'benchmarks'}; run the confirmed Live "
            "workflow first"
        )
    if len(candidates) > 1:
        raise ReleaseValidationError(
            "multiple released Live benchmark archives exist; pass "
            "--benchmark-run-id explicitly"
        )
    return candidates[0]


def _close(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right or left == right
    try:
        return abs(float(left) - float(right)) <= 1e-9
    except (TypeError, ValueError):
        return left == right


def _rescore(case: CaseRecord, outcome) -> CaseScore:
    try:
        return rescore_recorded_outcome(
            case, outcome, strict_error_kind=True
        )
    except RecordedOutcomeScoringError as exc:
        raise ReleaseValidationError(str(exc)) from exc


def _corpus_for_arm(
    archive: ReplayArchive, arm: str
) -> tuple[CorpusScore, tuple[CaseScore, ...]]:
    case_by_id = {case.case_id: case for case in archive.cases}
    scores: list[CaseScore] = []
    for outcome in sorted(
        (item for item in archive.outcomes if item.phase == arm),
        key=lambda item: item.case_id,
    ):
        case = case_by_id.get(outcome.case_id)
        if case is None:
            raise ReleaseValidationError(
                f"archived outcome references unknown case "
                f"{outcome.case_id}"
            )
        scores.append(_rescore(case, outcome))
    return aggregate_scores(scores), tuple(scores)


def _compare_metrics(
    arm: str,
    stored: Mapping[str, object],
    corpus: CorpusScore,
) -> list[str]:
    problems: list[str] = []
    for key in METRIC_KEYS:
        expected = getattr(corpus, key, None)
        recorded = stored.get(key)
        if key in {
            "article_count",
            "valid_count",
            "invalid_count",
            "parse_error_count",
            "request_error_count",
            "missing_output_count",
            "evidence_failure_count",
            "duplicate_prediction_count",
            "tp",
            "fp",
            "fn",
            "exact_match_count",
        }:
            if isinstance(recorded, bool) or not isinstance(recorded, int):
                problems.append(f"{arm}.{key} is missing or not an integer")
            elif recorded != expected:
                problems.append(
                    f"{arm}.{key}: stored {recorded} != recomputed {expected}"
                )
        elif not _close(recorded, expected):
            problems.append(
                f"{arm}.{key}: stored {recorded!r} != recomputed "
                f"{expected!r}"
            )
    return problems


def _compare_deltas(
    stored: Mapping[str, object],
    baseline: CorpusScore,
    optimized: CorpusScore,
) -> list[str]:
    problems: list[str] = []
    for key in (
        "precision",
        "recall",
        "micro_f1",
        "macro_f1",
        "exact_match_rate",
        "parsing_success",
        "mean_reward",
    ):
        recorded = stored.get(key)
        left = getattr(baseline, key, None)
        right = getattr(optimized, key, None)
        expected = None if left is None or right is None else right - left
        if not _close(recorded, expected):
            problems.append(
                f"delta.{key}: stored {recorded!r} != recomputed "
                f"{expected!r}"
            )
    return problems


def _scan_for_secrets(
    paths: Sequence[Path],
) -> tuple[list[str], list[str]]:
    findings: list[str] = []
    unreadable: list[str] = []
    for root in paths:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                data = path.read_bytes()
            except OSError as exc:
                unreadable.append(f"{path}: {exc}")
                continue
            if SECRET_PATTERN.search(data) or CREDENTIAL_KEY_PATTERN.search(
                data
            ):
                findings.append(str(path))
    return findings, unreadable


def _dataset_contract(
    archive: ReplayArchive, data_dir: Path
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    notes: list[str] = []
    try:
        manifest = load_split_manifest(data_dir / "split_manifest.json")
    except Exception as exc:  # noqa: BLE001 - report, do not infer
        return [f"split manifest is unavailable: {exc}"], notes
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        return ["split manifest has no splits mapping"], notes
    expected = {"train": 30, "validation": 10, "test": 10}
    for name, count in expected.items():
        split = splits.get(name)
        recorded = (
            split.get("article_count")
            if isinstance(split, Mapping)
            else None
        )
        if recorded != count:
            failures.append(
                f"{name} split count is {recorded!r}, expected {count}"
            )
    test_split = splits.get("test")
    if isinstance(test_split, Mapping):
        hashes = test_split.get("case_hashes")
        if isinstance(hashes, Mapping):
            for case in archive.cases:
                recorded_hash = hashes.get(case.case_id)
                actual_hash = canonical_document_sha256(case)
                if recorded_hash != actual_hash:
                    failures.append(
                        f"test case {case.case_id} differs from the frozen "
                        "split manifest"
                    )
        else:
            failures.append("test split has no case hash index")
        code_counts = test_split.get("code_counts")
        if isinstance(code_counts, Mapping):
            missing = sorted(set(ALLOWED_CODES) - set(code_counts))
            if missing:
                failures.append(
                    "test split lacks category coverage: "
                    + ", ".join(missing)
                )
        else:
            failures.append("test split has no category counts")
    if manifest.get("test_sealed") is not True:
        failures.append("split manifest does not mark the test split sealed")
    if manifest.get("total_article_count") != 50:
        failures.append("split manifest total_article_count is not 50")
    recorded_dataset_hash = manifest.get("combined_dataset_sha256")
    try:
        recomputed_dataset_hash = dataset_sha256(
            [
                *load_cases(data_dir / "train.jsonl"),
                *load_cases(data_dir / "validation.jsonl"),
                *archive.cases,
            ]
        )
    except Exception as exc:  # noqa: BLE001 - report, do not infer
        failures.append(f"cannot recompute the combined dataset hash: {exc}")
    else:
        if recorded_dataset_hash != recomputed_dataset_hash:
            failures.append(
                "combined_dataset_sha256 does not match the current "
                "train/validation files plus the archived test snapshot"
            )
        notes.append(f"dataset manifest hash: {recorded_dataset_hash}")
    return failures, notes


def _gold_approval_check(
    archive: ReplayArchive, data_dir: Path
) -> tuple[list[str], list[str], bool]:
    """Validate archived approvals, reviewer independence, and human sign-off.

    Returns failures, notes, and whether every case has a human approval. A
    fully automated pre-review does not satisfy the human sign-off gate.
    """
    try:
        entries = load_review_audit(data_dir / "reviews.jsonl")
        current_cases = [
            *load_cases(data_dir / "train.jsonl"),
            *load_cases(data_dir / "validation.jsonl"),
            *archive.cases,
        ]
        validate_review_audit(
            current_cases, entries, require_approved_review=True
        )
    except (OSError, DatasetValidationError, ValueError) as exc:
        return [f"gold review audit is invalid: {exc}"], [], False
    problems: list[str] = []
    for entry in entries:
        if entry.author_id == entry.reviewer_id:
            problems.append(
                f"{entry.case_id}: reviewer is not independent of the author"
            )
    expected = {case.case_id for case in current_cases}
    entries_by_id = {entry.case_id: entry for entry in entries}
    human_approved = {
        entry.case_id
        for entry in entries
        if entry.reviewer_kind == "human"
        and entry.reviewer_decision == "approved"
        and entry.author_decision == "approved"
        and entry.reconciled
    }
    missing_human = sorted(expected - human_approved)
    notes = [
        f"{len(human_approved)} of {len(expected)} cases have an independent "
        "approved human sign-off."
    ]
    if missing_human:
        notes.append(
            "Human sign-off is still required for: "
            + ", ".join(missing_human)
        )
    if set(entries_by_id) != expected:
        problems.append(
            "reviews.jsonl does not contain exactly one approval per case"
        )
    return problems, notes, not missing_human and not problems


def _decimal_or_none(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        result = Decimal(str(value))
    except (ValueError, ArithmeticError):
        return None
    return result if result.is_finite() else None


def _records_cost(
    records: Sequence[object],
) -> tuple[Decimal, str]:
    """Reconstruct the ledger total bound and cost status from requests.

    Estimated/reconciled accounting retains conservative reservations until a
    charge is verified, so the bound is max(reservation, estimate) for every
    outstanding record and the actual cost for reconciled records.
    """
    total = Decimal("0")
    statuses: set[str] = set()
    for record in records:
        cost_status = str(getattr(record, "cost_status", "unavailable"))
        statuses.add(cost_status)
        status = str(getattr(record, "status", ""))
        if status == "released":
            continue
        if status == "reconciled":
            actual = getattr(record, "reported_cost_usd", None)
            if isinstance(actual, Decimal) and actual.is_finite():
                total += actual
            continue
        reserved = getattr(record, "reserved_cost_usd", None)
        if not isinstance(reserved, Decimal) or not reserved.is_finite():
            statuses.add("unavailable")
            continue
        bound = reserved
        reported = getattr(record, "reported_cost_usd", None)
        estimated = getattr(record, "estimated_cost_usd", None)
        for candidate in (reported, estimated):
            if (
                isinstance(candidate, Decimal)
                and candidate.is_finite()
                and candidate > bound
            ):
                bound = candidate
        total += bound
    if not records:
        return total, "unavailable"
    if "pending" in statuses:
        return total, "pending"
    if "estimated" in statuses:
        return total, "estimated"
    if "actual" in statuses:
        return total, "actual"
    return total, "unavailable"


def _one_time_cost_problems(
    archive: ReplayArchive, summary: Mapping[str, object]
) -> list[str]:
    costs = summary.get("costs")
    if not isinstance(costs, Mapping):
        return ["stored cost summary is missing"]
    problems: list[str] = []
    optimization_known, optimization_status = _records_cost(
        tuple(archive.optimization_requests.values())
    )
    development_known, development_status = _records_cost(
        tuple(archive.development_requests.values())
    )
    for label, stored_cost, stored_status, known, status in (
        (
            "optimization",
            costs.get("optimization_cost_usd"),
            costs.get("optimization_cost_status"),
            optimization_known,
            optimization_status,
        ),
        (
            "baseline development",
            costs.get("baseline_development_cost_usd"),
            costs.get("baseline_development_cost_status"),
            development_known,
            development_status,
        ),
    ):
        if stored_status != status:
            problems.append(
                f"{label} cost status {stored_status!r} != archived {status!r}"
            )
        stored_decimal = _decimal_or_none(stored_cost)
        if stored_decimal is None or not _close(stored_decimal, known):
            problems.append(
                f"{label} cost {stored_cost!r} != archived ledger total "
                f"{known}"
            )
    return problems


def _release_measurements(
    archive: ReplayArchive,
) -> list[PairedMeasurement]:
    case_by_id = {case.case_id: case for case in archive.cases}
    grouped: dict[str, dict[str, ArmMeasurement]] = {
        case.case_id: {} for case in archive.cases
    }
    for outcome in archive.outcomes:
        case = case_by_id.get(outcome.case_id)
        if case is None:
            raise ReleaseValidationError(
                f"archived outcome references unknown case {outcome.case_id}"
            )
        score = _rescore(case, outcome)
        grouped.setdefault(outcome.case_id, {})[outcome.phase] = ArmMeasurement(
            case_id=outcome.case_id,
            score=score,
            valid=outcome.status == "ok",
            cost_usd=outcome.cost_usd,
            cost_status=outcome.cost_status,
            latency_ms=outcome.latency_ms,
            evaluate_percent=None,
            input_sha256=outcome.input_sha256,
        )
    return [
        PairedMeasurement(
            case_id=case_id,
            baseline=arms.get("baseline"),
            optimized=arms.get("optimized"),
        )
        for case_id, arms in sorted(grouped.items())
    ]


def _outcome_consistency_problems(
    archive: ReplayArchive, summary: Mapping[str, object]
) -> list[str]:
    """Recompute the verdict, validity gates, and economics from raw results."""
    costs = summary.get("costs")
    if not isinstance(costs, Mapping):
        return ["stored cost summary is missing; economics cannot be verified"]
    test_reuse = summary.get("test_reuse")
    try:
        measurements = _release_measurements(archive)
        optimization_cost, optimization_status = _records_cost(
            tuple(archive.optimization_requests.values())
        )
        development_cost, development_status = _records_cost(
            tuple(archive.development_requests.values())
        )
        recomputed = summarize_benchmark(
            measurements,
            planned_case_count=len(archive.cases),
            run_complete=archive.benchmark.status == "completed",
            comparable=bool(summary.get("comparable")),
            comparability_warnings=[],
            optimization_cost_usd=optimization_cost,
            optimization_cost_status=optimization_status,
            baseline_development_cost_usd=development_cost,
            baseline_development_cost_status=development_status,
            protocol_sha256=str(summary.get("protocol_sha256") or ""),
            task_config_sha256=str(summary.get("task_config_sha256") or ""),
            test_reuse=(
                test_reuse if isinstance(test_reuse, Mapping) else {}
            ),
            optimization_cost_breakdown=(
                costs.get("optimization_breakdown")
                if isinstance(costs.get("optimization_breakdown"), Mapping)
                else None
            ),
        )
    except Exception as exc:  # noqa: BLE001 - report as a failed check
        return [f"cannot reconstruct raw results: {exc}"]
    problems = _one_time_cost_problems(archive, summary)
    for key in (
        "complete",
        "complete_with_errors",
        "comparable",
        "inputs_equal",
        "eligible_for_verdict",
        "quality_non_regression_gate",
        "verdict",
        "verdict_reason",
        "validity",
    ):
        if summary.get(key) != recomputed.get(key):
            problems.append(
                f"{key}: stored {summary.get(key)!r} != recomputed "
                f"{recomputed.get(key)!r}"
            )
    stored_economics = summary.get("economics")
    if not isinstance(stored_economics, Mapping):
        problems.append("stored economics are missing")
    elif dict(stored_economics) != recomputed.get("economics"):
        problems.append(
            "stored economics differ from the recalculated break-even"
        )
    return problems


def _config_for(run_dir: Path) -> Mapping[str, object]:
    return _read_json(run_dir / "config.json")


def _benchmark_config_checks(
    archive: ReplayArchive,
) -> list[str]:
    problems: list[str] = []
    benchmark_config = _config_for(archive.benchmark_dir)
    if benchmark_config.get("phase") != "benchmark":
        problems.append("benchmark config phase is not 'benchmark'")
    if benchmark_config.get("split") != "test":
        problems.append("benchmark config does not use the test split")
    if benchmark_config.get("release_requested") is not True:
        problems.append("benchmark config does not request release")
    if benchmark_config.get("protocol_sha256") != (
        archive.benchmark.benchmark.protocol_sha256
        if archive.benchmark.benchmark is not None
        else None
    ):
        problems.append(
            "benchmark config protocol hash differs from the manifest"
        )
    optimization_config = _config_for(
        archive.benchmark_dir.parent.parent
        / "optimization"
        / archive.optimization.run_id
    )
    development_config = _config_for(
        archive.benchmark_dir.parent.parent
        / "development"
        / archive.development.run_id
    )
    benchmark_task = benchmark_config.get("task")
    if not isinstance(benchmark_task, Mapping):
        problems.append("benchmark config has no task configuration")
    else:
        for label, config in (
            ("optimization", optimization_config),
            ("development", development_config),
        ):
            if config.get("task") != benchmark_task:
                problems.append(
                    f"{label} task configuration differs from benchmark"
                )
    optimization_split = optimization_config.get("split")
    if optimization_split == "test":
        problems.append("optimization config used the test split")
    optimization_ids = optimization_config.get("case_ids")
    if isinstance(optimization_ids, list) and any(
        str(case_id).startswith("test-") for case_id in optimization_ids
    ):
        problems.append("optimization config contains test case IDs")
    return problems


def _has_optimization_evidence(
    candidate_stats: Mapping[str, object] | None,
    warnings: Sequence[str],
) -> bool:
    if candidate_stats is not None:
        return True
    return any(
        "baseline was perfect" in warning.lower() for warning in warnings
    )


def _program_checks(archive: ReplayArchive) -> list[str]:
    problems: list[str] = []
    for label, program_id in (
        (
            "baseline",
            archive.benchmark.links.baseline_program_id,
        ),
        (
            "optimized",
            archive.benchmark.links.optimized_program_id,
        ),
    ):
        if program_id is None:
            problems.append(f"{label} program ID is missing")
            continue
        snapshot = archive.programs.get(program_id)
        if snapshot is None:
            problems.append(f"{label} program snapshot is missing")
            continue
        if not snapshot.instructions.strip():
            problems.append(f"{label} program has empty instructions")
        if snapshot.manifest.demos:
            problems.append(f"{label} program has non-empty demos")
        if snapshot.manifest.predictor_count != 1:
            problems.append(f"{label} program does not have one predictor")
        if not snapshot.schema:
            problems.append(f"{label} program has no signature schema")
    return problems


def _request_checks(archive: ReplayArchive) -> list[str]:
    problems: list[str] = []
    for outcome in archive.outcomes:
        if not outcome.request_ids:
            problems.append(
                f"{outcome.phase}/{outcome.case_id} has no recorded requests"
            )
            continue
        for request_id in outcome.request_ids:
            request = archive.requests.get(request_id)
            if request is None:
                problems.append(
                    f"{outcome.phase}/{outcome.case_id} references missing "
                    f"request {request_id}"
                )
            elif not request.messages:
                problems.append(f"request {request_id} has no messages")
    return problems


def _latency_problems(
    archive: ReplayArchive,
    arm: str,
    stored_arm: Mapping[str, object],
) -> list[str]:
    problems: list[str] = []
    latency = stored_arm.get("latency")
    if not isinstance(latency, Mapping):
        return ["latency summary is missing"]
    if latency.get("unit") != "ms":
        problems.append("latency unit is not milliseconds")
    values = [
        float(outcome.latency_ms)
        for outcome in archive.outcomes
        if outcome.phase == arm and outcome.latency_ms is not None
    ]
    expected_count = sum(
        1 for outcome in archive.outcomes if outcome.phase == arm
    )
    if len(values) != expected_count:
        problems.append(f"{arm} has outcomes without latency_ms")
    ordered = sorted(values)
    expected_median = statistics.median(ordered) if ordered else None
    expected_p95 = (
        ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
        if ordered
        else None
    )
    if latency.get("count") != expected_count:
        problems.append(
            f"{arm} latency count {latency.get('count')!r} != "
            f"{expected_count}"
        )
    if not _close(latency.get("median"), expected_median):
        problems.append(
            f"{arm} latency median {latency.get('median')!r} != "
            f"{expected_median!r}"
        )
    if not _close(latency.get("p95"), expected_p95):
        problems.append(
            f"{arm} latency p95 {latency.get('p95')!r} != "
            f"{expected_p95!r}"
        )
    return problems


def _cost_checks(
    archive: ReplayArchive, summary: Mapping[str, object]
) -> list[str]:
    problems: list[str] = []
    costs = summary.get("costs")
    if not isinstance(costs, Mapping):
        return ["benchmark summary has no costs mapping"]
    status = costs.get("benchmark_inference_cost_status")
    if status not in {"actual", "estimated", "pending", "unavailable", "none"}:
        problems.append(f"invalid benchmark cost status {status!r}")
    recorded = costs.get("benchmark_inference_known_cost_usd")
    try:
        recorded_decimal = Decimal(str(recorded))
    except Exception:  # noqa: BLE001 - report malformed cost
        return problems + [
            f"benchmark known cost is not decimal: {recorded!r}"
        ]
    known = sum(
        (
            outcome.cost_usd
            if outcome.cost_usd is not None
            else Decimal("0")
            for outcome in archive.outcomes
            if outcome.cost_usd is not None
            or outcome.cost_status == "none"
        ),
        Decimal("0"),
    )
    unknown = any(
        outcome.cost_usd is None and outcome.cost_status != "none"
        for outcome in archive.outcomes
    )
    if unknown:
        if status not in {"pending", "unavailable"}:
            problems.append(
                "some outcomes have unknown cost but the aggregate is "
                f"labeled {status!r}"
            )
    else:
        if status in {"pending", "unavailable"}:
            problems.append(
                "every outcome has known cost but the aggregate is labeled "
                f"{status!r}"
            )
        elif status in {"actual", "estimated", "none"} and not _close(
            recorded_decimal, known
        ):
            problems.append(
                f"recorded benchmark cost {recorded_decimal} != outcome sum "
                f"{known}"
            )
    return problems


def _acceptance_items(
    archive: ReplayArchive,
    summary: Mapping[str, object],
    checks: Sequence[ReleaseCheck],
) -> tuple[ReleaseCheck, ...]:
    by_id = {check.check_id: check for check in checks}
    def status(check_id: str) -> str:
        check = by_id.get(check_id)
        return "verified" if check is not None and check.status == "pass" else (
            "failed" if check is not None and check.status == "fail" else
            "not_covered"
        )

    items: list[ReleaseCheck] = []
    a1_status = (
        "failed" if status("environment") == "failed" else "partial"
    )
    items.append(
        ReleaseCheck(
            "A1",
            a1_status,
            "Validator-checkable subset: frozen runtime versions are "
            "recorded in every linked manifest. Fresh-checkout README/offline "
            "behavior still requires the operator setup run.",
        )
    )
    a2_states = {
        status("dataset_contract"),
        status("gold_approval"),
    }
    a2_status = (
        "failed"
        if "failed" in a2_states
        else "verified"
        if a2_states == {"verified"}
        else "partial"
    )
    items.append(
        ReleaseCheck(
            "A2",
            a2_status,
            "50 approved cases, 30/10/10 splits, sealed test, six-category "
            "coverage, frozen case hashes, archived approvals, and reviewer "
            "independence are checked; human gold sign-off is required before "
            "this item becomes verified.",
        )
    )
    items.append(
        ReleaseCheck(
            "A3",
            status("development_run"),
            "The linked baseline development run is completed Live and "
            "retains all validation outcomes.",
        )
    )
    items.append(
        ReleaseCheck(
            "A4",
            status("optimization_run"),
            "The linked optimization run is completed Live with native "
            "candidate statistics; an unchanged baseline is a valid outcome.",
        )
    )
    items.append(
        ReleaseCheck(
            "A5",
            status("program_artifacts"),
            "Both immutable program snapshots load with instructions, one "
            "predictor, zero demos, and a signature schema.",
        )
    )
    items.append(
        ReleaseCheck(
            "A6",
            status("paired_inputs"),
            "Exactly ten test cases have one baseline and one optimized "
            "outcome with identical input hashes; invalid outputs are "
            "retained.",
        )
    )
    items.append(
        ReleaseCheck(
            "A7",
            status("requests"),
            "Every outcome references recorded client messages; no preview "
            "is treated as sent.",
        )
    )
    a8_states = {
        status("metrics"),
        status("latency"),
        status("costs"),
    }
    a8_status = (
        "verified"
        if a8_states == {"verified"}
        else "failed"
        if "failed" in a8_states
        else "not_covered"
    )
    items.append(
        ReleaseCheck(
            "A8",
            a8_status,
            "Stored quality metrics, deltas, error counts, and costs are "
            "recomputed from archived raw findings; latency is checked "
            "against archived outcome timings in milliseconds.",
        )
    )
    items.append(
        ReleaseCheck(
            "A9",
            "not_covered",
            "Playground acceptance is exercised by the separately confirmed "
            "Live Playground and tests/test_playground.py; it is not stored "
            "inside a benchmark archive.",
        )
    )
    items.append(
        ReleaseCheck(
            "A10",
            "not_covered",
            "Budget/cancellation/secret tests are executed by the default "
            "test suite; this validator does not run tests.",
        )
    )
    items.append(
        ReleaseCheck(
            "A11",
            status("metadata"),
            "Run IDs, UTC timestamps, versions, snapshots, hashes, seed, "
            "request/metric counts, token usage, costs, and timing are "
            "present in linked manifests, configs, outcomes, and summary.",
        )
    )
    items.append(
        ReleaseCheck(
            "A12",
            status("honest_outcome"),
            f"Archived verdict is {summary.get('verdict', 'missing')!r}; "
            "the original warnings and validity gates are preserved.",
        )
    )
    a13_status = (
        "failed"
        if status("replay_immutability") == "failed"
        else "partial"
    )
    items.append(
        ReleaseCheck(
            "A13",
            a13_status,
            "Validator-checkable subset: repeated offline reads produced "
            "identical artifact hashes and zero dispatch. Restart/no-key "
            "presentation requires the operator acceptance run.",
        )
    )
    return tuple(items)


def validate_release(
    *,
    benchmark_run_id: str | None = None,
    runs_root: Path = RUNS_ROOT,
    programs_root: Path = PROGRAMS_ROOT,
    data_dir: Path = DATA_DIR,
    repeats: int = 3,
) -> ReleaseReport:
    """Validate one completed released Live archive without modifying it."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ReleaseValidationError("repeats must be a positive integer")
    runs_root = Path(runs_root)
    programs_root = Path(programs_root)
    benchmark_dir = _resolve_benchmark_dir(benchmark_run_id, runs_root)
    try:
        archive = open_replay_archive(
            benchmark_dir,
            runs_root=runs_root,
            programs_root=programs_root,
        )
    except ArtifactError as exc:
        raise ReleaseValidationError(str(exc)) from exc
    if archive.benchmark.benchmark is None:
        raise ReleaseValidationError("benchmark protocol metadata is missing")
    summary = archive.benchmark_summary
    if not isinstance(summary, Mapping):
        raise ReleaseValidationError("benchmark summary is not archived")

    evidence_paths = {
        "benchmark": archive.benchmark_dir,
        "development": benchmark_dir.parent.parent
        / "development"
        / archive.development.run_id,
        "optimization": benchmark_dir.parent.parent
        / "optimization"
        / archive.optimization.run_id,
    }
    for label in archive.programs:
        evidence_paths[f"program:{label}"] = Path(programs_root) / label
    before_hashes = {
        label: _tree_hash(path) for label, path in evidence_paths.items()
    }
    checks: list[ReleaseCheck] = []

    def add(check_id: str, passed: bool, detail: str) -> None:
        checks.append(
            ReleaseCheck(check_id, "pass" if passed else "fail", detail)
        )

    measured = is_measured_archive(archive.benchmark)
    add(
        "measured_archive",
        measured,
        "Archive is a completed, released Live benchmark."
        if measured
        else "Archive is mock/unreleased/incomplete and cannot be measured.",
    )
    protocol = archive.benchmark.benchmark
    add(
        "release_status",
        protocol.released,
        f"release flag={protocol.released}, exposure="
        f"{protocol.exposure_index}",
    )
    add(
        "environment",
        all(
            manifest.versions
            for manifest in (
                archive.benchmark,
                archive.development,
                archive.optimization,
            )
        ),
        "Linked manifests contain runtime versions: "
        + ", ".join(
            sorted(archive.benchmark.versions.keys())
        ),
    )
    dataset_failures, dataset_notes = _dataset_contract(archive, data_dir)
    add(
        "dataset_contract",
        not dataset_failures,
        "; ".join(dataset_failures) if dataset_failures else "; ".join(dataset_notes),
    )
    gold_failures, gold_notes, human_complete = _gold_approval_check(
        archive, data_dir
    )
    if gold_failures:
        add("gold_approval", False, "; ".join(gold_failures + gold_notes))
    elif human_complete:
        add("gold_approval", True, "; ".join(gold_notes))
    else:
        checks.append(
            ReleaseCheck(
                "gold_approval",
                "warning",
                "; ".join(
                    [
                        "Independent automated approvals are valid, but the "
                        "required human gold sign-off is incomplete.",
                        *gold_notes,
                    ]
                ),
            )
        )
    add(
        "development_run",
        archive.development.status == "completed"
        and archive.development.origin == "live"
        and len(archive.development_outcomes)
        == len(archive.development_cases),
        f"development {archive.development.status}/{archive.development.origin} "
        f"with {len(archive.development_outcomes)} outcomes",
    )
    optimization_evidence = _has_optimization_evidence(
        archive.candidate_stats, archive.optimization.warnings
    )
    add(
        "optimization_run",
        archive.optimization.status == "completed"
        and archive.optimization.origin == "live"
        and optimization_evidence,
        f"optimization {archive.optimization.status}/"
        f"{archive.optimization.origin}, candidate stats="
        f"{archive.candidate_stats is not None}, optimization evidence="
        f"{optimization_evidence}",
    )
    program_problems = _program_checks(archive)
    add(
        "program_artifacts",
        not program_problems,
        "; ".join(program_problems) or "Both immutable program snapshots load.",
    )
    expected_cases = {case.case_id for case in archive.cases}
    outcomes_by_case: dict[str, set[str]] = {
        case_id: set() for case_id in expected_cases
    }
    for outcome in archive.outcomes:
        outcomes_by_case.setdefault(outcome.case_id, set()).add(outcome.phase)
    paired_problems: list[str] = []
    if len(archive.cases) != 10:
        paired_problems.append(
            f"archived test case count is {len(archive.cases)}, expected 10"
        )
    for case_id, phases in outcomes_by_case.items():
        if phases != {"baseline", "optimized"}:
            paired_problems.append(
                f"{case_id} has phases {sorted(phases)}"
            )
    paired_by_case = {
        case_id: {
            outcome.phase: outcome
            for outcome in archive.outcomes
            if outcome.case_id == case_id
        }
        for case_id in expected_cases
    }
    for case_id, pair in paired_by_case.items():
        if set(pair) != {"baseline", "optimized"}:
            continue
        if pair["baseline"].input_sha256 != pair["optimized"].input_sha256:
            paired_problems.append(
                f"{case_id} baseline/optimized input hashes differ"
            )
    if len(archive.outcomes) != 2 * len(archive.cases):
        paired_problems.append(
            f"{len(archive.outcomes)} outcomes for {len(archive.cases)} cases"
        )
    add(
        "paired_inputs",
        not paired_problems,
        "; ".join(paired_problems)
        or "10 cases, both arms, identical input hashes, invalid outputs retained.",
    )
    request_problems = _request_checks(archive)
    add(
        "requests",
        not request_problems,
        "; ".join(request_problems)
        or "Every outcome references recorded request messages.",
    )
    baseline_corpus, _baseline_scores = _corpus_for_arm(
        archive, "baseline"
    )
    optimized_corpus, _optimized_scores = _corpus_for_arm(
        archive, "optimized"
    )
    stored_metrics = summary.get("metrics")
    if not isinstance(stored_metrics, Mapping):
        stored_metrics = {}
    stored_baseline = stored_metrics.get("baseline")
    stored_optimized = stored_metrics.get("optimized")
    metric_problems: list[str] = []
    if not isinstance(stored_baseline, Mapping):
        metric_problems.append("stored baseline metrics are missing")
    else:
        metric_problems.extend(
            _compare_metrics("baseline", stored_baseline, baseline_corpus)
        )
    if not isinstance(stored_optimized, Mapping):
        metric_problems.append("stored optimized metrics are missing")
    else:
        metric_problems.extend(
            _compare_metrics("optimized", stored_optimized, optimized_corpus)
        )
    stored_delta = stored_metrics.get("delta")
    if not isinstance(stored_delta, Mapping) or not stored_delta:
        metric_problems.append("stored metric deltas are missing")
    else:
        metric_problems.extend(
            _compare_deltas(
                stored_delta, baseline_corpus, optimized_corpus
            )
        )
    add(
        "metrics",
        not metric_problems,
        "; ".join(metric_problems)
        or "Stored metrics and deltas equal deterministic re-scoring of "
        "archived raw findings.",
    )
    latency_problems: list[str] = []
    for arm, stored_arm in (
        ("baseline", stored_baseline),
        ("optimized", stored_optimized),
    ):
        if isinstance(stored_arm, Mapping):
            latency_problems.extend(
                _latency_problems(archive, arm, stored_arm)
            )
        else:
            latency_problems.append(f"{arm} metrics are missing")
    add(
        "latency",
        not latency_problems,
        "; ".join(latency_problems)
        or "Stored latency counts, median, and exploratory p95 match the "
        "archived outcomes in milliseconds; latency is not recomputed from "
        "new inference.",
    )
    cost_problems = _cost_checks(archive, summary)
    add(
        "costs",
        not cost_problems,
        "; ".join(cost_problems)
        or "Recorded benchmark cost and cost status match archived outcomes.",
    )
    config_problems = _benchmark_config_checks(archive)
    add(
        "task_protocol",
        not config_problems,
        "; ".join(config_problems)
        or "Benchmark, development, and optimization task configurations match; "
        "no test data entered optimization.",
    )
    metadata_ok = bool(
        archive.benchmark.run_id
        and archive.benchmark.created_at
        and archive.benchmark.finished_at
        and archive.benchmark.versions
        and archive.benchmark.snapshot
        and summary.get("protocol_sha256")
        and summary.get("task_config_sha256")
        and summary.get("planned_cases") == len(archive.cases)
    )
    add(
        "metadata",
        metadata_ok,
        "Run IDs, UTC timestamps, versions, snapshots, protocol hashes, "
        "planned case count, seed, requests, and economics are present.",
    )
    verdict = summary.get("verdict")
    outcome_problems = _outcome_consistency_problems(archive, summary)
    add(
        "honest_outcome",
        verdict in VERDICTS
        and summary.get("complete") is True
        and not outcome_problems,
        "; ".join(outcome_problems)
        or (
            f"complete={summary.get('complete')!r}, verdict={verdict!r}, "
            f"validity={summary.get('validity', 'missing')!r}; verdict, "
            "gates, and economics match the reconstructed raw results."
        ),
    )
    test_reuse = summary.get("test_reuse")
    exposure = protocol.exposure_index
    exposure_ok = True
    exposure_detail = f"exposure index {exposure}"
    if isinstance(test_reuse, Mapping):
        recorded_exposure = test_reuse.get("exposure_index")
        if recorded_exposure != exposure:
            exposure_ok = False
            exposure_detail += f" differs from summary {recorded_exposure!r}"
        else:
            exposure_detail += (
                f", provenance {test_reuse.get('provenance', 'unknown')}"
            )
    else:
        exposure_ok = False
        exposure_detail += " but summary test_reuse is missing"
    add("test_index", exposure_ok, exposure_detail)
    first_exposure = exposure == 0
    if first_exposure:
        add(
            "first_exposure",
            True,
            "Exposure index 0: this is the first recorded test exposure.",
        )
    else:
        checks.append(
            ReleaseCheck(
                "first_exposure",
                "warning",
                f"Exposure index {exposure}: this is an exploratory "
                "reevaluation, not the first blind held-out run.",
            )
        )
    secret_paths, unreadable_paths = _scan_for_secrets(
        list(evidence_paths.values())
    )
    add(
        "secret_scan",
        not secret_paths and not unreadable_paths,
        "; ".join(secret_paths + unreadable_paths)
        or "No provider-style secret or credential-key pattern appears in "
        "readable archived artifacts.",
    )

    reopen_error: str | None = None
    for _ in range(repeats - 1):
        try:
            open_replay_archive(
                benchmark_dir,
                runs_root=runs_root,
                programs_root=programs_root,
            )
        except ArtifactError as exc:
            reopen_error = str(exc)
            break
    after_hashes = {
        label: _tree_hash(path) for label, path in evidence_paths.items()
    }
    if reopen_error is not None:
        add(
            "replay_immutability",
            False,
            f"repeated Replay read failed: {reopen_error}; artifact hashes "
            f"unchanged={before_hashes == after_hashes}",
        )
    else:
        add(
            "replay_immutability",
            before_hashes == after_hashes,
            f"{repeats} read(s); artifact hashes unchanged."
            if before_hashes == after_hashes
            else "Artifact hashes changed during repeated Replay reads.",
        )
    acceptance = _acceptance_items(archive, summary, checks)
    passed = all(check.status != "fail" for check in checks)
    selected_summary = {
        "verdict": summary.get("verdict"),
        "verdict_reason": summary.get("verdict_reason"),
        "complete": summary.get("complete"),
        "comparable": summary.get("comparable"),
        "metrics": summary.get("metrics"),
        "costs": summary.get("costs"),
        "economics": summary.get("economics"),
        "test_reuse": summary.get("test_reuse"),
        "protocol_sha256": summary.get("protocol_sha256"),
        "task_config_sha256": summary.get("task_config_sha256"),
    }
    return ReleaseReport(
        schema_version=RELEASE_REPORT_SCHEMA_VERSION,
        generated_at=_utc_now(),
        benchmark_run_id=archive.benchmark.run_id,
        archive_dir=benchmark_dir,
        passed=passed,
        verdict=str(verdict) if verdict is not None else None,
        exposure_index=exposure,
        measured_archive=measured,
        checks=tuple(checks),
        acceptance=acceptance,
        artifact_hashes=before_hashes,
        summary=selected_summary,
        warnings=tuple(
            dict.fromkeys(
                [
                    *archive.warnings,
                    *archive.development.warnings,
                    *archive.optimization.warnings,
                    *archive.benchmark.warnings,
                    *(
                        [
                            f"Exploratory reevaluation: exposure index "
                            f"{exposure}, not the first blind held-out run."
                        ]
                        if not first_exposure
                        else []
                    ),
                    *(
                        f"{check.check_id}: {check.detail}"
                        for check in checks
                        if check.status == "warning"
                    ),
                ]
            )
        ),
    )


def _human_summary(report: ReleaseReport) -> str:
    lines = [
        f"Release validation: {'PASS' if report.passed else 'FAIL'}",
        f"Benchmark run: {report.benchmark_run_id}",
        f"Archive: {report.archive_dir}",
        f"Measured: {report.measured_archive}",
        f"Verdict: {report.verdict}",
        f"Exposure index: {report.exposure_index}",
    ]
    for check in report.checks:
        lines.append(
            f"[{check.status.upper():4}] {check.check_id}: {check.detail}"
        )
    lines.append("Acceptance items:")
    for item in report.acceptance:
        lines.append(f"[{item.status.upper():11}] {item.check_id}: {item.detail}")
    for warning in report.warnings:
        lines.append(f"[WARN] {warning}")
    return "\n".join(lines)


def _write_report(
    report: ReleaseReport,
    path: Path,
    *,
    protected_roots: Sequence[Path] = (PROGRAMS_ROOT,),
) -> None:
    resolved = Path(path).resolve()
    runs_root = report.archive_dir.resolve().parent.parent
    allowed = runs_root / "release_validation"
    if resolved.is_relative_to(runs_root) and not resolved.is_relative_to(
        allowed
    ):
        raise ReleaseValidationError(
            "refusing to write the acceptance report inside a run archive; "
            "use a path outside runs/ or under runs/release_validation/"
        )
    for root in protected_roots:
        if resolved.is_relative_to(Path(root).resolve()):
            raise ReleaseValidationError(
                "refusing to write the acceptance report inside a program "
                "artifact tree"
            )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(resolved, report.to_payload())


def _run_baseline_command(args: argparse.Namespace) -> int:
    if not args.confirm_budget:
        print(
            "Refusing to run baseline development without "
            "--confirm-budget.",
            file=sys.stderr,
        )
        return 2
    try:
        settings: Settings = load_settings("live")
        cases = load_cases(DATA_DIR / "validation.jsonl")
        result = run_baseline_development(
            settings,
            cases,
            confirm_budget=True,
            run_id=args.run_id,
            origin="live",
            runs_root=args.runs_root,
            programs_root=args.programs_root,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"baseline development failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "run_id": result.manifest.run_id,
                "status": result.manifest.status,
                "origin": result.manifest.origin,
                "run_dir": str(result.run_dir),
            },
            indent=2,
        )
    )
    return 0 if result.manifest.status == "completed" else 1


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a released Live benchmark archive and prepare the "
            "baseline development run. Offline validation makes no network "
            "request."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate one released Live archive and print an acceptance record.",
    )
    validate_parser.add_argument("--benchmark-run-id")
    validate_parser.add_argument(
        "--runs-root", type=Path, default=RUNS_ROOT
    )
    validate_parser.add_argument(
        "--programs-root", type=Path, default=PROGRAMS_ROOT
    )
    validate_parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR
    )
    validate_parser.add_argument(
        "--repeats", type=int, default=3,
        help="Number of read-only archive loads used for immutability.",
    )
    validate_parser.add_argument(
        "--json", type=Path,
        help="Write the machine-readable acceptance report here.",
    )
    validate_parser.set_defaults(handler=_run_validate_command)

    baseline_parser = subparsers.add_parser(
        "baseline-development",
        help=(
            "Run the one confirmed Live baseline validation evaluation. This "
            "spends real money."
        ),
    )
    baseline_parser.add_argument(
        "--confirm-budget", action="store_true",
        help="Explicit confirmation of the baseline development budget.",
    )
    baseline_parser.add_argument("--run-id")
    baseline_parser.add_argument(
        "--runs-root", type=Path, default=RUNS_ROOT
    )
    baseline_parser.add_argument(
        "--programs-root", type=Path, default=PROGRAMS_ROOT
    )
    baseline_parser.set_defaults(handler=_run_baseline_command)
    return parser.parse_args(argv)


def _run_validate_command(args: argparse.Namespace) -> int:
    try:
        report = validate_release(
            benchmark_run_id=args.benchmark_run_id,
            runs_root=args.runs_root,
            programs_root=args.programs_root,
            data_dir=args.data_dir,
            repeats=args.repeats,
        )
    except ReleaseValidationError as exc:
        print(f"release validation failed: {exc}", file=sys.stderr)
        return 1
    print(_human_summary(report))
    if args.json is not None:
        try:
            _write_report(
                report,
                args.json,
                protected_roots=(args.programs_root,),
            )
        except (ReleaseValidationError, OSError) as exc:
            print(f"cannot write report: {exc}", file=sys.stderr)
            return 1
        print(f"Acceptance report written: {args.json}")
    return 0 if report.passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    return int(args.handler(args))


__all__ = [
    "ReleaseCheck",
    "ReleaseReport",
    "ReleaseValidationError",
    "main",
    "validate_release",
]


if __name__ == "__main__":
    raise SystemExit(main())
