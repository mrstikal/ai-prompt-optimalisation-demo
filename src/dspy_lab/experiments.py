"""Baseline development evaluation and genuine GEPA compilation.

S10 evaluates the frozen baseline on the development validation split and
compiles a fresh copy with the real ``dspy.GEPA`` optimizer. Both paid phases
require an explicit budget confirmation, share one pre-dispatch ledger per job,
and persist exact snapshots, outcomes, requests/responses, native candidate
statistics, and the selected state-only program. The sealed test split is never
read here.
"""

from __future__ import annotations

import copy
import json
import math
import random
import re
import secrets
import statistics
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import dspy
from dspy.teleprompt.gepa.gepa import DspyGEPAResult
from dspy.utils.exceptions import AdapterParseError

from .artifacts import (
    PROGRAMS_ROOT,
    RUNS_ROOT,
    RUN_GEPA_STATS_FILENAME,
    RUN_MANIFEST_FILENAME,
    RUN_OUTCOMES_FILENAME,
    RUN_REQUESTS_FILENAME,
    BenchmarkProtocol,
    LoadedProgram,
    ProgramManifest,
    ProgramSnapshot,
    RecordedOutcome,
    RunLinks,
    RunManifest,
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    canonical_json_text,
    case_input_sha256,
    cases_snapshot_sha256,
    file_index,
    load_program_artifact,
    program_reference,
    read_input_snapshot,
    read_program_snapshot,
    read_recorded_outcomes,
    read_recorded_requests,
    read_run_manifest,
    save_program_artifact,
    sha256_text,
    verify_file_index,
    write_input_snapshot,
    write_run_manifest,
)
from .config import (
    GEPA_CANDIDATE_SELECTION_STRATEGY,
    GEPA_REFLECTION_MINIBATCH_SIZE,
    GEPA_SEED,
    NUM_THREADS,
    PROJECT_ROOT,
    Settings,
    collect_runtime_versions,
)
from .data import (
    TEST_ARTICLE_COUNT,
    VALIDATION_ARTICLE_COUNT,
    canonical_document_sha256,
    load_benchmark_cases,
    load_cases,
    load_optimization_cases,
    validate_case_offsets,
)
from .llm import (
    BudgetLedger,
    BudgetSnapshot,
    BudgetStop,
    RequestRecord as LedgerRequestRecord,
    build_guarded_lm,
    conservative_token_counter,
)
from .metrics import (
    CaseScore,
    CorpusScore,
    InvalidPrediction,
    aggregate_scores,
    feedback_metric,
    scalar_metric,
    score_case,
)
from .program import (
    BASELINE_PROMPT_VERSION,
    STRICT_ADAPTER_VERSION,
    EditorialQA,
    StrictJSONAdapter,
    baseline_fingerprint,
    case_inputs,
    load_editorial_rules,
)
from .schemas import RULES_VERSION, CaseRecord, Finding

DATA_DIR = PROJECT_ROOT / "data" / "v1"
METRIC_VERSION = "deterministic_issue_match_v1"
EXPERIMENT_CONFIG_SCHEMA_VERSION = "experiment_config_v1"
GEPA_STATS_SCHEMA_VERSION = "gepa_stats_v1"
BENCHMARK_PLAN_SCHEMA_VERSION = "benchmark_plan_v1"
BENCHMARK_SUMMARY_SCHEMA_VERSION = "benchmark_summary_v1"
TEST_REUSE_SCHEMA_VERSION = "test_reuse_history_v1"
BENCHMARK_PROTOCOL_FILENAME = "benchmark_protocol.json"
BENCHMARK_SUMMARY_FILENAME = "summary.json"
TEST_REUSE_FILENAME = "test_reuse_history.json"
MIN_OPTIMIZATION_HEADROOM_CALLS = 40
FORMATTING_OVERHEAD_TOKENS = 4_096
BASELINE_CANDIDATE_ID = "baseline"
_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_TERMINAL_BUDGET_STOPS = frozenset(
    {"call_cap", "cost_cap", "overrun", "cancelled"}
)


class ExperimentError(RuntimeError):
    """Base class for experiment orchestration failures."""


class ExperimentNotAuthorized(ExperimentError):
    """Raised when a paid phase lacks an explicit budget confirmation."""


class ExperimentConfigurationError(ExperimentError):
    """Raised before dispatch when a configuration cannot be executed."""


class ExperimentRunExistsError(ExperimentError):
    """Raised instead of overwriting an existing run directory."""


class ExperimentIncompleteError(ExperimentError):
    """Raised when a linked run or artifact is missing or incomplete."""


@dataclass(frozen=True, slots=True)
class BaselineDevelopmentResult:
    run_id: str
    run_dir: Path
    manifest: RunManifest
    baseline_program: ProgramManifest
    outcomes: tuple[RecordedOutcome, ...]
    corpus: CorpusScore
    ledger: BudgetSnapshot
    requests: tuple[LedgerRequestRecord, ...]
    summary: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    run_id: str
    run_dir: Path
    manifest: RunManifest
    baseline_program: ProgramManifest
    selected_program: ProgramManifest
    stats: Mapping[str, object] | None
    ledger: BudgetSnapshot
    requests: tuple[LedgerRequestRecord, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArmMeasurement:
    """One arm's deterministic score plus its recorded economics."""

    case_id: str
    score: CaseScore
    valid: bool
    cost_usd: Decimal | None
    cost_status: str
    latency_ms: float | None
    evaluate_percent: float | None
    input_sha256: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    provider_latency_ms: float | None = None


@dataclass(frozen=True, slots=True)
class PairedMeasurement:
    case_id: str
    baseline: ArmMeasurement | None
    optimized: ArmMeasurement | None


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    run_id: str
    run_dir: Path
    manifest: RunManifest
    baseline_program: ProgramManifest
    optimized_program: ProgramManifest
    outcomes: tuple[RecordedOutcome, ...]
    measurements: tuple[PairedMeasurement, ...]
    summary: Mapping[str, object]
    ledger: BudgetSnapshot
    requests: tuple[LedgerRequestRecord, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _BenchmarkLinks:
    development_dir: Path
    development_manifest: RunManifest
    development_cases: tuple[CaseRecord, ...]
    development_rules: str
    optimization_dir: Path
    optimization_manifest: RunManifest
    optimization_cases: tuple[CaseRecord, ...]
    baseline_loaded: LoadedProgram
    optimized_loaded: LoadedProgram
    optimization_cost_usd: Decimal | None
    optimization_cost_status: str
    optimization_cost_breakdown: Mapping[str, object]
    development_cost_usd: Decimal | None
    development_cost_status: str
    cost_warnings: tuple[str, ...]


def load_validation_cases(data_dir: Path = DATA_DIR) -> list[CaseRecord]:
    """Load only the development validation split for baseline evaluation."""
    return load_cases(Path(data_dir) / "validation.jsonl")


def load_train_validation_cases(
    data_dir: Path = DATA_DIR,
) -> dict[str, list[CaseRecord]]:
    """Load only train and validation; the sealed test path is never opened."""
    base = Path(data_dir)
    return load_optimization_cases(
        base / "train.jsonl", base / "validation.jsonl"
    )


def load_sealed_benchmark_cases(
    data_dir: Path = DATA_DIR,
) -> list[CaseRecord]:
    """Explicitly load the sealed test split for an authorized benchmark."""
    return load_benchmark_cases(Path(data_dir))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utc_string(value: datetime | None = None) -> str:
    current = value or _utcnow()
    return current.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_run_id(prefix: str) -> str:
    stamp = _utcnow().strftime("%Y%m%dT%H%M%SZ").lower()
    return f"{prefix}-{stamp}-{secrets.token_hex(3)}"


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ExperimentConfigurationError(
            "run_id must match ^[a-z0-9][a-z0-9-]{0,63}$"
        )
    return run_id


def _require_live_settings(settings: Settings) -> None:
    if not isinstance(settings, Settings):
        raise ExperimentConfigurationError("settings must be a Settings instance")
    if settings.mode != "live":
        raise ExperimentConfigurationError(
            "baseline development and optimization require Live mode"
        )
    if (
        not settings.openrouter_api_key
        or not settings.openrouter_api_key.strip()
    ):
        raise ExperimentConfigurationError(
            "Live experiments require OPENROUTER_API_KEY"
        )


def _require_confirmation(confirm_budget: object) -> None:
    if confirm_budget is not True:
        raise ExperimentNotAuthorized(
            "an explicit confirmed phase budget is required before dispatch"
        )


def _frozen_rules_text(rules_text: str | None) -> str:
    frozen = load_editorial_rules()
    if rules_text is None:
        return frozen
    if not isinstance(rules_text, str) or rules_text != frozen:
        raise ExperimentConfigurationError(
            "experiments must use the frozen editorial rules text"
        )
    return frozen


def _validate_case_sequence(
    cases: Sequence[CaseRecord], *, name: str
) -> list[CaseRecord]:
    case_list = list(cases)
    if not case_list:
        raise ExperimentConfigurationError(f"{name} must contain at least one case")
    if not all(isinstance(case, CaseRecord) for case in case_list):
        raise ExperimentConfigurationError(f"{name} must contain CaseRecord values")
    case_ids = [case.case_id for case in case_list]
    if len(case_ids) != len(set(case_ids)):
        raise ExperimentConfigurationError(f"{name} case IDs must be unique")
    for case in case_list:
        if case.rules_version != RULES_VERSION:
            raise ExperimentConfigurationError(
                f"{case.case_id}: unsupported rules version"
            )
        validate_case_offsets(case)
    return case_list


def _validate_optimization_split(
    train_cases: Sequence[CaseRecord],
    validation_cases: Sequence[CaseRecord],
) -> tuple[list[CaseRecord], list[CaseRecord]]:
    train = _validate_case_sequence(train_cases, name="train cases")
    validation = _validate_case_sequence(
        validation_cases, name="validation cases"
    )
    if {case.case_id for case in train} & {
        case.case_id for case in validation
    }:
        raise ExperimentConfigurationError(
            "train and validation case IDs must be disjoint"
        )
    if {case.family_id for case in train} & {
        case.family_id for case in validation
    }:
        raise ExperimentConfigurationError(
            "train and validation families must be disjoint"
        )
    return train, validation


def _validate_optimization_budget(settings: Settings) -> int:
    metric_calls = settings.max_optimization_metric_calls
    if metric_calls < 1:
        raise ExperimentConfigurationError(
            "MAX_OPTIMIZATION_METRIC_CALLS must be positive"
        )
    if metric_calls > settings.max_optimization_calls:
        raise ExperimentConfigurationError(
            "MAX_OPTIMIZATION_METRIC_CALLS must not exceed "
            "MAX_OPTIMIZATION_CALLS"
        )
    headroom = settings.max_optimization_calls - metric_calls
    # One third of the metric target is reserved for reflection, compression,
    # and final-batch overhang, capped at the 40 slots required by the default
    # 120/160 configuration. Never silently raise the confirmed call cap.
    required = min(
        MIN_OPTIMIZATION_HEADROOM_CALLS,
        max(1, math.ceil(metric_calls / 3)),
    )
    if headroom < required:
        raise ExperimentConfigurationError(
            "optimization budget has insufficient headroom for reflection and "
            f"final-batch overhang: calls={settings.max_optimization_calls}, "
            f"metric_calls={metric_calls}, required_headroom={required}. "
            "Do not silently raise the cap; require a newly confirmed budget."
        )
    return headroom


def _validate_budget(limits: object) -> None:
    try:
        max_calls = getattr(limits, "max_calls")
        max_cost = getattr(limits, "max_cost_usd")
    except AttributeError as exc:
        raise ExperimentConfigurationError(
            "a named phase budget is required"
        ) from exc
    if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
        raise ExperimentConfigurationError("budget max_calls must be positive")
    if not isinstance(max_cost, Decimal) or not max_cost.is_finite() or max_cost <= 0:
        raise ExperimentConfigurationError(
            "budget max_cost_usd must be a positive finite Decimal"
        )


def _sanitize_text(text: str, secret: str | None) -> str:
    if secret:
        return text.replace(secret, "***")
    return text


def _json_ready(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        return _json_ready(value.model_dump(mode="json"))
    if hasattr(value, "item"):
        return _json_ready(value.item())
    return str(value)


def _assert_finite_json(value: object, *, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ExperimentError(f"{context} contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_finite_json(item, context=context)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_finite_json(item, context=context)


def _pretty_json(value: object) -> str:
    return (
        json.dumps(
            _json_ready(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _case_score_payload(score: CaseScore) -> dict[str, object]:
    return {
        "case_id": score.case_id,
        "valid": score.valid,
        "error_kind": score.error_kind,
        "error": score.error,
        "reward": score.reward,
        "exact_match": score.exact_match,
        "tp": score.tp,
        "fp": score.fp,
        "fn": score.fn,
        "precision": score.precision,
        "recall": score.recall,
        "f1": score.f1,
        "matched_issue_ids": list(score.matched_issue_ids),
        "false_positives": [
            {
                "prediction_index": item.prediction_index,
                "code": item.code,
                "evidence": item.evidence,
                "reason": item.reason,
                "occurrences": item.occurrences,
                "duplicate_of": item.duplicate_of,
                "matching_gold_issue_id": item.matching_gold_issue_id,
                "matching_gold_code": item.matching_gold_code,
                "evidence_failure": item.evidence_failure,
            }
            for item in score.false_positives
        ],
        "missed_gold": [
            {
                "issue_id": item.issue_id,
                "code": item.code,
                "evidence": item.evidence,
            }
            for item in score.missed_gold
        ],
        "duplicate_count": score.duplicate_count,
        "evidence_failure_count": score.evidence_failure_count,
        "clean_case": score.clean_case,
        "prediction_count": score.prediction_count,
        "gold_count": score.gold_count,
    }


def _corpus_payload(corpus: CorpusScore) -> dict[str, object]:
    return {
        "article_count": corpus.article_count,
        "valid_count": corpus.valid_count,
        "invalid_count": corpus.invalid_count,
        "parse_error_count": corpus.parse_error_count,
        "request_error_count": corpus.request_error_count,
        "missing_output_count": corpus.missing_output_count,
        "evidence_failure_count": corpus.evidence_failure_count,
        "duplicate_prediction_count": corpus.duplicate_prediction_count,
        "tp": corpus.tp,
        "fp": corpus.fp,
        "fn": corpus.fn,
        "precision": corpus.precision,
        "recall": corpus.recall,
        "micro_f1": corpus.micro_f1,
        "macro_f1": corpus.macro_f1,
        "exact_match_count": corpus.exact_match_count,
        "exact_match_rate": corpus.exact_match_rate,
        "parsing_success": corpus.parsing_success,
        "clean_article_count": corpus.clean_article_count,
        "clean_articles_falsely_flagged": (
            corpus.clean_articles_falsely_flagged
        ),
        "mean_reward": corpus.mean_reward,
        "per_category": [
            {
                "code": category.code,
                "tp": category.tp,
                "fp": category.fp,
                "fn": category.fn,
                "precision": category.precision,
                "recall": category.recall,
                "f1": category.f1,
            }
            for category in corpus.per_category
        ],
    }


def _budget_payload(snapshot: BudgetSnapshot) -> dict[str, object]:
    return {
        "max_calls": snapshot.max_calls,
        "max_cost_usd": str(snapshot.max_cost_usd),
        "admitted_calls": snapshot.admitted_calls,
        "delegated_calls": snapshot.delegated_calls,
        "response_calls": snapshot.response_calls,
        "succeeded_calls": snapshot.succeeded_calls,
        "failed_calls": snapshot.failed_calls,
        "pending_cost_calls": snapshot.pending_cost_calls,
        "rejected_calls": snapshot.rejected_calls,
        "cancelled_before_dispatch_calls": (
            snapshot.cancelled_before_dispatch_calls
        ),
        "estimated_cost_usd": str(snapshot.estimated_cost_usd),
        "actual_cost_usd": str(snapshot.actual_cost_usd),
        "committed_cost_usd": str(snapshot.committed_cost_usd),
        "unresolved_cost_usd": str(snapshot.unresolved_cost_usd),
        "total_cost_usd": str(snapshot.total_cost_usd),
        "remaining_calls": snapshot.remaining_calls,
        "remaining_cost_usd": str(snapshot.remaining_cost_usd),
        "stop_reason": snapshot.stop_reason,
        "cancelled": snapshot.cancelled,
        "exhausted": snapshot.exhausted,
        "overrun": snapshot.overrun,
    }


def _redact_value(value: object, secret: str | None) -> object:
    if isinstance(value, str):
        return _sanitize_text(value, secret)
    if isinstance(value, Mapping):
        return {
            str(key): _redact_value(item, secret)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_value(item, secret) for item in value]
    return value


def _request_payload(
    record: LedgerRequestRecord,
    secret: str | None,
) -> dict[str, object]:
    return {
        "call_id": record.call_id,
        "sequence": record.sequence,
        "phase": record.phase,
        "model": record.model,
        "provider": record.provider,
        "status": record.status,
        "cost_status": record.cost_status,
        "reserved_cost_usd": str(record.reserved_cost_usd),
        "estimated_cost_usd": (
            None
            if record.estimated_cost_usd is None
            else str(record.estimated_cost_usd)
        ),
        "reported_cost_usd": (
            None
            if record.reported_cost_usd is None
            else str(record.reported_cost_usd)
        ),
        "input_token_bound": record.input_token_bound,
        "output_token_bound": record.output_token_bound,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "generation_id": record.generation_id,
        "response_model": record.response_model,
        "finish_reason": record.finish_reason,
        "error": (
            None if record.error is None else _sanitize_text(record.error, secret)
        ),
        "released_reason": record.released_reason,
        "messages": _redact_value(
            [dict(message) for message in record.messages], secret
        ),
        "request_kwargs": _redact_value(
            dict(record.request_kwargs), secret
        ),
        "usage": _redact_value(dict(record.usage), secret),
        "started_at": _utc_string(record.started_at),
        "finished_at": (
            None
            if record.finished_at is None
            else _utc_string(record.finished_at)
        ),
        "response_text": (
            None
            if record.response_text is None
            else _sanitize_text(record.response_text, secret)
        ),
    }


def _write_requests(
    run_dir: Path, ledger: BudgetLedger, secret: str | None
) -> None:
    atomic_write_jsonl(
        run_dir / RUN_REQUESTS_FILENAME,
        [
            _request_payload(record, secret)
            for record in ledger.records
        ],
    )


def _append_event(run_dir: Path, event: Mapping[str, object]) -> None:
    append_jsonl(
        run_dir / "events.jsonl",
        {"at": _utc_string(), **dict(event)},
    )


def _case_cost(
    records: Sequence[LedgerRequestRecord],
) -> tuple[Decimal | None, str]:
    if not records:
        return None, "none"
    statuses = {record.cost_status for record in records}
    if "pending" in statuses:
        return None, "pending"
    if "unavailable" in statuses:
        return None, "unavailable"
    costs = [
        record.reported_cost_usd
        if record.reported_cost_usd is not None
        else record.estimated_cost_usd
        for record in records
    ]
    total = sum((cost for cost in costs if cost is not None), Decimal("0"))
    if "estimated" in statuses:
        return total, "estimated"
    if "actual" in statuses:
        return total, "actual"
    # A released/no-dispatch record contributes zero known cost; a sibling
    # request with a known cost is still retained.
    return (total if total else None), "none"


def _make_example(case: CaseRecord, rules_text: str) -> dspy.Example:
    inputs = case_inputs(case, rules_text)
    return dspy.Example(
        case_id=case.case_id,
        document_text=inputs["document_text"],
        product_facts=inputs["product_facts"],
        sources=inputs["sources"],
        editorial_rules=inputs["editorial_rules"],
        gold_findings=list(case.gold_findings),
    ).with_inputs(
        "document_text",
        "product_facts",
        "sources",
        "editorial_rules",
    )


def _assert_program_contract(
    program: EditorialQA, *, context: str
) -> str:
    if not isinstance(program, EditorialQA):
        raise ExperimentError(f"{context} is not an EditorialQA program")
    predictors = list(program.named_predictors())
    if len(predictors) != 1 or predictors[0][0] != "predict":
        raise ExperimentError(
            f"{context} must contain exactly the named predictor 'predict'"
        )
    if program.predict.lm is not None:
        raise ExperimentError(f"{context} must not persist an LM")
    if program.predict.demos:
        raise ExperimentError(f"{context} must contain zero demos")
    instructions = program.predict.signature.instructions
    if not isinstance(instructions, str) or not instructions.strip():
        raise ExperimentError(f"{context} instructions must be non-empty")
    return instructions


def _assert_baseline_immutable(
    baseline: EditorialQA,
    *,
    expected_instructions: str,
    expected_fingerprint: str,
) -> None:
    actual = _assert_program_contract(baseline, context="baseline")
    if actual != expected_instructions:
        raise ExperimentError("GEPA mutated the frozen baseline instructions")
    if baseline_fingerprint() != expected_fingerprint:
        raise ExperimentError("the frozen baseline fingerprint changed")


def _save_program(
    program: EditorialQA,
    *,
    origin: str,
    programs_root: Path,
    secret: str | None,
    parent_program_id: str | None = None,
) -> tuple[ProgramManifest, Path]:
    manifest = save_program_artifact(
        program,
        origin=origin,  # type: ignore[arg-type]
        programs_root=Path(programs_root),
        parent_program_id=parent_program_id,
        secret_values=(secret,) if secret else (),
    )
    return manifest, Path(programs_root) / manifest.program_id


def _task_config_payload(settings: Settings) -> dict[str, object]:
    task_route = settings.task_route()
    return {
        "model": task_route.model,
        "provider": task_route.provider,
        "temperature": settings.temperature,
        "max_output_tokens": settings.max_output_tokens,
        "schema_mode": task_route.expected_schema_mode,
        "adapter": "StrictJSONAdapter",
        "adapter_version": STRICT_ADAPTER_VERSION,
    }


def _config_payload(
    *,
    phase: str,
    run_id: str,
    origin: str,
    settings: Settings,
    split: str,
    cases: Sequence[CaseRecord],
    baseline_instructions: str,
    limits: object,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": EXPERIMENT_CONFIG_SCHEMA_VERSION,
        "phase": phase,
        "run_id": run_id,
        "origin": origin,
        "mode": settings.mode,
        "created_at": _utc_string(),
        "metric_version": METRIC_VERSION,
        "dataset_version": "v1",
        "split": split,
        "case_ids": [case.case_id for case in cases],
        "baseline_fingerprint": baseline_fingerprint(),
        "baseline_instructions_sha256": sha256_text(
            baseline_instructions
        ),
        "task": _task_config_payload(settings),
        "budget": {
            "max_calls": getattr(limits, "max_calls"),
            "max_cost_usd": str(getattr(limits, "max_cost_usd")),
        },
        "num_threads": NUM_THREADS,
        "prompt_version": BASELINE_PROMPT_VERSION,
        "rules_version": RULES_VERSION,
    }
    if phase == "optimization":
        reflection_route = settings.reflection_route()
        payload["reflection"] = {
            "model": reflection_route.model,
            "provider": reflection_route.provider,
            "temperature": settings.reflection_temperature,
            "max_output_tokens": settings.reflection_max_output_tokens,
            "schema_mode": reflection_route.expected_schema_mode,
        }
    if extra:
        payload.update(dict(extra))
    return payload


def _file_names(run_dir: Path, names: Sequence[str]) -> list[str]:
    return [name for name in names if (run_dir / name).is_file()]


def _run_manifest(
    *,
    run_id: str,
    kind: str,
    origin: str,
    status: str,
    created_at: str,
    finished_at: str | None,
    snapshot: object,
    programs: Sequence[object],
    links: RunLinks,
    run_dir: Path,
    file_names: Sequence[str],
    stop_reason: str | None,
    warnings: Sequence[str],
    benchmark: BenchmarkProtocol | None = None,
) -> RunManifest:
    files = file_index(run_dir, _file_names(run_dir, file_names))
    return RunManifest(
        run_id=run_id,
        kind=kind,  # type: ignore[arg-type]
        origin=origin,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        created_at=created_at,
        finished_at=finished_at,
        versions=collect_runtime_versions(),
        snapshot=snapshot,  # type: ignore[arg-type]
        programs=list(programs),  # type: ignore[arg-type]
        links=links,
        benchmark=benchmark,
        files=files,
        stop_reason=stop_reason,
        warnings=list(warnings),
    )


def _write_manifest(
    run_dir: Path, manifest: RunManifest, secret: str | None
) -> RunManifest:
    write_run_manifest(
        run_dir,
        manifest,
        secret_values=(secret,) if secret else (),
    )
    return manifest


def _new_run_dir(runs_root: Path, kind_folder: str, run_id: str) -> Path:
    run_dir = Path(runs_root) / kind_folder / run_id
    if run_dir.exists():
        raise ExperimentRunExistsError(
            f"run directory already exists: {run_dir}"
        )
    run_dir.mkdir(parents=True)
    return run_dir


def _outcomes_payload(
    outcomes: Sequence[RecordedOutcome],
) -> list[dict[str, object]]:
    return [outcome.model_dump(mode="json") for outcome in outcomes]


def run_baseline_development(
    settings: Settings,
    cases: Sequence[CaseRecord],
    *,
    confirm_budget: bool = False,
    run_id: str | None = None,
    origin: str = "live",
    runs_root: Path = RUNS_ROOT,
    programs_root: Path = PROGRAMS_ROOT,
    rules_text: str | None = None,
) -> BaselineDevelopmentResult:
    """Evaluate the frozen baseline once per validation case."""
    _require_live_settings(settings)
    _require_confirmation(confirm_budget)
    limits = settings.baseline_dev_budget()
    _validate_budget(limits)
    case_list = _validate_case_sequence(cases, name="validation cases")
    rules = _frozen_rules_text(rules_text)
    if origin not in ("live", "mock_test"):
        raise ExperimentConfigurationError("unsupported run origin")
    run_id = _validate_run_id(run_id or _new_run_id("dev"))
    secret = settings.openrouter_api_key
    run_dir = _new_run_dir(Path(runs_root), "development", run_id)
    baseline_program = EditorialQA()
    baseline_instructions = _assert_program_contract(
        baseline_program, context="baseline"
    )
    expected_fingerprint = baseline_fingerprint()
    baseline_manifest, baseline_dir = _save_program(
        baseline_program,
        origin=origin,
        programs_root=Path(programs_root),
        secret=secret,
    )
    baseline_ref = program_reference(baseline_dir)
    snapshot = write_input_snapshot(
        run_dir,
        cases=case_list,
        rules_text=rules,
        split="validation",
        dataset_version="v1",
    )
    config = _config_payload(
        phase="baseline_development",
        run_id=run_id,
        origin=origin,
        settings=settings,
        split="validation",
        cases=case_list,
        baseline_instructions=baseline_instructions,
        limits=limits,
    )
    atomic_write_json(run_dir / "config.json", config)
    atomic_write_jsonl(run_dir / RUN_OUTCOMES_FILENAME, [])
    atomic_write_jsonl(run_dir / RUN_REQUESTS_FILENAME, [])
    _append_event(
        run_dir,
        {
            "event": "baseline_development_started",
            "run_id": run_id,
            "case_count": len(case_list),
            "baseline_program_id": baseline_manifest.program_id,
        },
    )
    _append_event(
        run_dir,
        {
            "event": "baseline_frozen",
            "program_id": baseline_manifest.program_id,
            "fingerprint": expected_fingerprint,
        },
    )
    ledger = BudgetLedger.from_limits(limits)
    write_lock = threading.Lock()

    def listener(record: LedgerRequestRecord) -> None:
        with write_lock:
            _write_requests(run_dir, ledger, secret)

    task_lm = build_guarded_lm(
        settings,
        role="task",
        ledger=ledger,
        token_counter=conservative_token_counter,
        formatting_overhead_tokens=FORMATTING_OVERHEAD_TOKENS,
        record_listener=listener,
    )
    program = baseline_program
    outcomes: list[RecordedOutcome] = []
    scores: list[CaseScore] = []
    stopped_early = False

    with dspy.context(lm=task_lm, adapter=StrictJSONAdapter()):
        for case in case_list:
            request_count = len(ledger.records)
            started = time.perf_counter()
            error: str | None = None
            prediction = None
            budget_stop = False
            parse_error = False
            try:
                prediction = program(**case_inputs(case, rules))
            except BudgetStop as exc:
                error = _sanitize_text(
                    f"{type(exc).__name__}: {exc}", secret
                )
                budget_stop = True
            except AdapterParseError as exc:
                error = _sanitize_text(
                    f"{type(exc).__name__}: {exc}", secret
                )
                parse_error = True
            except Exception as exc:
                error = _sanitize_text(
                    f"{type(exc).__name__}: {exc}", secret
                )
            case_requests = ledger.records[request_count:]
            if prediction is not None:
                score = score_case(case, prediction)
            else:
                score = score_case(
                    case,
                    InvalidPrediction(  # type: ignore[arg-type]
                        error or "missing output",
                        "parse_error" if parse_error else "request_error",
                    ),
                )
            if error is None and not score.valid:
                error = score.error or "invalid prediction"
            scores.append(score)
            cost_usd, cost_status = _case_cost(case_requests)
            findings = None
            status = "error"
            if error is None and prediction is not None:
                raw_findings = getattr(prediction, "findings", None)
                if isinstance(raw_findings, list):
                    findings = [
                        item
                        if isinstance(item, Finding)
                        else Finding.model_validate(item)
                        for item in raw_findings
                    ]
                    status = "ok"
                else:
                    error = "prediction did not expose a findings list"
                    score = score_case(
                        case,
                        InvalidPrediction(error, "missing_output"),
                    )
                    scores[-1] = score
            if status == "error" and error is None:
                error = "invalid or failed prediction"
            latency_ms = (time.perf_counter() - started) * 1000.0
            outcome = _recorded_outcome(
                run_id=run_id,
                case=case,
                rules_text=rules,
                program_id=baseline_manifest.program_id,
                candidate_id=BASELINE_CANDIDATE_ID,
                phase="development",
                split="validation",
                status=status,
                findings=findings,
                error=error,
                score=score,
                request_ids=[record.call_id for record in case_requests],
                cost_usd=cost_usd,
                cost_status=cost_status,
                latency_ms=latency_ms,
            )
            outcomes.append(outcome)
            atomic_write_jsonl(
                run_dir / RUN_OUTCOMES_FILENAME,
                _outcomes_payload(outcomes),
            )
            _append_event(
                run_dir,
                {
                    "event": "case_evaluated",
                    "case_id": case.case_id,
                    "status": status,
                    "reward": score.reward,
                    "request_count": len(case_requests),
                },
            )
            if budget_stop:
                stopped_early = True
                break

    if stopped_early:
        evaluated = {outcome.case_id for outcome in outcomes}
        for case in case_list:
            if case.case_id in evaluated:
                continue
            error = _sanitize_text(
                "budget stopped before this case was evaluated", secret
            )
            score = score_case(
                case, InvalidPrediction(error, "request_error")
            )
            scores.append(score)
            outcomes.append(
                _recorded_outcome(
                    run_id=run_id,
                    case=case,
                    rules_text=rules,
                    program_id=baseline_manifest.program_id,
                    candidate_id=BASELINE_CANDIDATE_ID,
                    phase="development",
                    split="validation",
                    status="error",
                    findings=None,
                    error=error,
                    score=score,
                    request_ids=[],
                    cost_usd=None,
                    cost_status="none",
                    latency_ms=None,
                )
            )
        atomic_write_jsonl(
            run_dir / RUN_OUTCOMES_FILENAME,
            _outcomes_payload(outcomes),
        )

    _assert_baseline_immutable(
        program,
        expected_instructions=baseline_instructions,
        expected_fingerprint=expected_fingerprint,
    )
    _write_requests(run_dir, ledger, secret)
    ledger_snapshot = ledger.snapshot()
    corpus = aggregate_scores(scores)
    always_empty_reward = (
        sum(1 for case in case_list if not case.gold_findings)
        / len(case_list)
    )
    summary: dict[str, object] = {
        "schema_version": "baseline_summary_v1",
        "run_id": run_id,
        "split": "validation",
        "metric_version": METRIC_VERSION,
        "baseline_program_id": baseline_manifest.program_id,
        "baseline_instructions": baseline_instructions,
        "case_count": len(case_list),
        "expected_case_count": VALIDATION_ARTICLE_COUNT,
        "corpus": _corpus_payload(corpus),
        "budget": _budget_payload(ledger_snapshot),
        "always_empty_reference": {
            "mean_reward": always_empty_reward,
            "label": "calculated_reference_not_a_model_run",
        },
    }
    atomic_write_text(run_dir / "summary.json", _pretty_json(summary))
    warnings: list[str] = []
    if len(case_list) != VALIDATION_ARTICLE_COUNT:
        warnings.append(
            f"development run contains {len(case_list)} cases; the frozen "
            f"validation split has {VALIDATION_ARTICLE_COUNT}."
        )
    if corpus.invalid_count:
        warnings.append(
            f"{corpus.invalid_count} baseline outcomes are invalid or failed; "
            "they remain counted."
        )
    stop_reason = ledger_snapshot.stop_reason
    status = "completed"
    if stop_reason in _TERMINAL_BUDGET_STOPS:
        status = "stopped"
        warnings.append(
            f"baseline development stopped at the confirmed budget: {stop_reason}."
        )
    _append_event(
        run_dir,
        {
            "event": "baseline_development_finished",
            "status": status,
            "stop_reason": stop_reason,
        },
    )
    created_at = str(_read_json_object(run_dir / "config.json")["created_at"])
    finished_at = _utc_string()
    manifest = _run_manifest(
        run_id=run_id,
        kind="development",
        origin=origin,
        status=status,
        created_at=created_at,
        finished_at=finished_at,
        snapshot=snapshot,
        programs=[baseline_ref],
        links=RunLinks(baseline_program_id=baseline_manifest.program_id),
        run_dir=run_dir,
        file_names=(
            "config.json",
            "summary.json",
            "events.jsonl",
            RUN_OUTCOMES_FILENAME,
            RUN_REQUESTS_FILENAME,
            snapshot.cases_file,
            snapshot.rules_file,
        ),
        stop_reason=stop_reason,
        warnings=warnings,
    )
    _write_manifest(run_dir, manifest, secret)
    return BaselineDevelopmentResult(
        run_id=run_id,
        run_dir=run_dir,
        manifest=manifest,
        baseline_program=baseline_manifest,
        outcomes=tuple(outcomes),
        corpus=corpus,
        ledger=ledger_snapshot,
        requests=ledger.records,
        summary=summary,
    )


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExperimentError(f"{path} must contain one JSON object")
    return payload


def _recorded_outcome(
    *,
    run_id: str,
    case: CaseRecord,
    rules_text: str,
    program_id: str,
    candidate_id: str,
    phase: str,
    split: str,
    status: str,
    findings: list[Finding] | None,
    error: str | None,
    score: CaseScore,
    request_ids: Sequence[str],
    cost_usd: Decimal | None,
    cost_status: str,
    latency_ms: float | None,
) -> RecordedOutcome:
    return RecordedOutcome(
        run_id=run_id,
        case_id=case.case_id,
        program_id=program_id,
        candidate_id=candidate_id,
        phase=phase,  # type: ignore[arg-type]
        split=split,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        findings=findings,
        error=error,
        metric=_case_score_payload(score),
        request_ids=list(request_ids),
        cost_usd=cost_usd,
        cost_status=cost_status,  # type: ignore[arg-type]
        latency_ms=latency_ms,
        input_sha256=case_input_sha256(case, rules_text),
        created_at=_utc_string(),
    )


def _load_development_run(
    *,
    runs_root: Path,
    run_id: str,
    validation_cases: Sequence[CaseRecord],
    rules_text: str,
    expected_baseline_program_id: str,
    expected_task: Mapping[str, object],
    origin: str,
) -> tuple[Path, RunManifest, list[RecordedOutcome]]:
    run_dir = Path(runs_root) / "development" / run_id
    if not run_dir.is_dir():
        raise ExperimentIncompleteError(
            f"baseline development run is missing: {run_id}"
        )
    manifest = read_run_manifest(run_dir)
    if manifest.kind != "development":
        raise ExperimentIncompleteError(
            f"linked run {run_id!r} is not a development run"
        )
    if manifest.status != "completed":
        raise ExperimentIncompleteError(
            f"baseline development run is {manifest.status!r}, not completed"
        )
    if manifest.origin != origin:
        raise ExperimentIncompleteError(
            "baseline development and optimization must share one origin"
        )
    if manifest.links.baseline_program_id != expected_baseline_program_id:
        raise ExperimentIncompleteError(
            "baseline development used a different baseline program"
        )
    if manifest.snapshot.split != "validation":
        raise ExperimentIncompleteError(
            "baseline development must use a validation snapshot"
        )
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise ExperimentIncompleteError(
            "baseline development run has no protocol configuration"
        )
    development_config = _read_json_object(config_path)
    if development_config.get("task") != dict(expected_task):
        raise ExperimentIncompleteError(
            "task model or request protocol changed between baseline "
            "development and optimization"
        )
    if development_config.get("metric_version") != METRIC_VERSION:
        raise ExperimentIncompleteError(
            "baseline development used a different metric version"
        )
    if development_config.get("prompt_version") != BASELINE_PROMPT_VERSION:
        raise ExperimentIncompleteError(
            "baseline development used a different baseline prompt"
        )
    if development_config.get("dataset_version") != "v1":
        raise ExperimentIncompleteError(
            "baseline development used a different dataset version"
        )
    if development_config.get("rules_version") != RULES_VERSION:
        raise ExperimentIncompleteError(
            "baseline development used a different rules version"
        )
    verify_file_index(run_dir, manifest.files)
    if RUN_OUTCOMES_FILENAME not in manifest.files:
        raise ExperimentIncompleteError(
            "baseline development run does not retain outcomes"
        )
    cases, rules = read_input_snapshot(run_dir, manifest.snapshot)
    if rules != rules_text:
        raise ExperimentIncompleteError(
            "baseline development used different editorial rules"
        )
    expected_ids = [case.case_id for case in validation_cases]
    if [case.case_id for case in cases] != expected_ids:
        raise ExperimentIncompleteError(
            "baseline development cases do not match optimization validation"
        )
    expected_payload = [
        canonical_json_text(case.model_dump(mode="json"))
        for case in validation_cases
    ]
    actual_payload = [
        canonical_json_text(case.model_dump(mode="json")) for case in cases
    ]
    if actual_payload != expected_payload:
        raise ExperimentIncompleteError(
            "baseline development snapshot content differs from optimization "
            "validation inputs"
        )
    outcomes = read_recorded_outcomes(
        run_dir / RUN_OUTCOMES_FILENAME, run_id=run_id
    )
    if len(outcomes) != len(validation_cases):
        raise ExperimentIncompleteError(
            "baseline development outcomes do not cover every validation case"
        )
    seen: set[str] = set()
    for outcome in outcomes:
        if outcome.case_id in seen:
            raise ExperimentIncompleteError(
                "baseline development contains duplicate outcomes"
            )
        seen.add(outcome.case_id)
        if (
            outcome.phase != "development"
            or outcome.split != "validation"
            or outcome.program_id != expected_baseline_program_id
        ):
            raise ExperimentIncompleteError(
                "baseline development outcome provenance is inconsistent"
            )
    if seen != set(expected_ids):
        raise ExperimentIncompleteError(
            "baseline development outcomes do not match its snapshot"
        )
    return run_dir, manifest, outcomes


def _build_gepa_optimizer(
    *,
    settings: Settings,
    reflection_lm: dspy.LM,
    log_dir: Path,
) -> dspy.GEPA:
    return dspy.GEPA(
        metric=feedback_metric,
        auto=None,
        max_full_evals=None,
        max_metric_calls=settings.max_optimization_metric_calls,
        reflection_minibatch_size=GEPA_REFLECTION_MINIBATCH_SIZE,
        candidate_selection_strategy=GEPA_CANDIDATE_SELECTION_STRATEGY,
        reflection_lm=reflection_lm,
        skip_perfect_score=True,
        add_format_failure_as_feedback=True,
        use_merge=False,
        num_threads=NUM_THREADS,
        failure_score=0.0,
        perfect_score=1.0,
        log_dir=str(log_dir),
        track_stats=True,
        track_best_outputs=False,
        seed=GEPA_SEED,
    )


def _gepa_settings_payload(settings: Settings, headroom: int) -> dict[str, object]:
    return {
        "auto": None,
        "max_full_evals": None,
        "max_metric_calls": settings.max_optimization_metric_calls,
        "reflection_minibatch_size": GEPA_REFLECTION_MINIBATCH_SIZE,
        "candidate_selection_strategy": GEPA_CANDIDATE_SELECTION_STRATEGY,
        "use_merge": False,
        "skip_perfect_score": True,
        "add_format_failure_as_feedback": True,
        "failure_score": 0.0,
        "perfect_score": 1.0,
        "track_stats": True,
        "track_best_outputs": False,
        "seed": GEPA_SEED,
        "num_threads": NUM_THREADS,
        "budget_headroom_calls": headroom,
    }


def _candidate_instructions(
    stats: Mapping[str, object],
) -> list[dict[str, object]]:
    candidates = stats.get("candidates")
    parents = stats.get("parents")
    scores = stats.get("val_aggregate_scores")
    discovery = stats.get("discovery_eval_counts")
    best_idx = stats.get("best_idx")
    if not isinstance(candidates, list):
        raise ExperimentError("GEPA stats do not contain candidates")
    records: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ExperimentError("GEPA candidate is not an object")
        instruction = candidate.get("predict")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ExperimentError(
                "GEPA candidate does not contain predictor instructions"
            )
        parents_value = None
        if isinstance(parents, list) and index < len(parents):
            parent_items = parents[index]
            if isinstance(parent_items, list):
                parents_value = list(parent_items)
        records.append(
            {
                "candidate_id": f"candidate-{index}",
                "candidate_index": index,
                "instructions": {"predict": instruction},
                "parents": parents_value,
                "val_aggregate_score": (
                    scores[index]
                    if isinstance(scores, list) and index < len(scores)
                    else None
                ),
                "discovery_eval_count": (
                    discovery[index]
                    if isinstance(discovery, list)
                    and index < len(discovery)
                    else None
                ),
                "selected": index == best_idx,
            }
        )
    return records


def _valid_proposal_count(
    records: Sequence[LedgerRequestRecord],
) -> int:
    count = 0
    for record in records:
        if record.phase != "reflection" or record.status not in (
            "succeeded",
            "reconciled",
        ):
            continue
        text = record.response_text
        if not text:
            continue
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                continue
            try:
                payload = json.loads(text[start : end + 1])
            except (TypeError, ValueError):
                continue
        if not isinstance(payload, Mapping):
            continue
        instruction = payload.get("new_instruction")
        if isinstance(instruction, str) and instruction.strip():
            count += 1
    return count


def _search_assessment(
    *,
    detailed: DspyGEPAResult,
    records: Sequence[LedgerRequestRecord],
    baseline_instructions: str,
) -> tuple[str, list[str], dict[str, object]]:
    candidates = detailed.candidates
    candidate_count = len(candidates)
    if candidate_count == 0:
        raise ExperimentError("GEPA returned no candidate programs")
    selected_index = detailed.best_idx
    selected_components = {
        name: predictor.signature.instructions
        for name, predictor in candidates[selected_index].named_predictors()
    }
    selected_instructions = selected_components.get("predict")
    if not isinstance(selected_instructions, str):
        raise ExperimentError(
            "GEPA selected program has no predictor instructions"
        )
    reflection_records = [
        record for record in records if record.phase == "reflection"
    ]
    task_records = [record for record in records if record.phase == "task"]
    reflection_attempts = len(reflection_records)
    reflection_successes = sum(
        1
        for record in reflection_records
        if record.status in ("succeeded", "reconciled")
    )
    valid_proposals = _valid_proposal_count(reflection_records)
    task_attempts = len(task_records)
    selected_equals_baseline = selected_instructions == baseline_instructions
    baseline_score = (
        detailed.val_aggregate_scores[0]
        if detailed.val_aggregate_scores
        else None
    )
    warnings: list[str] = []
    status_override: str | None = None
    if candidate_count == 1 and selected_equals_baseline:
        if reflection_attempts == 0:
            if baseline_score is not None and baseline_score >= 1.0 - 1e-9:
                search_status = "perfect_baseline_skip"
                warnings.append(
                    "The baseline was perfect on the validation set and no "
                    "reflection was attempted; GEPA's perfect-score skip left "
                    "candidate search unexercised."
                )
            else:
                search_status = "not_exercised"
                status_override = "failed"
                warnings.append(
                    "No reflection or proposal request was made; candidate "
                    "search was not exercised."
                )
        elif reflection_successes == 0:
            search_status = "all_proposals_failed"
            status_override = "failed"
            warnings.append(
                "Every reflection request failed; this is a failed search, "
                "not evidence that GEPA found no improvement."
            )
        elif valid_proposals == 0:
            search_status = "all_proposals_failed"
            status_override = "failed"
            warnings.append(
                "Every reflection response lacked a usable new instruction; "
                "this is a failed proposal search, not evidence that GEPA "
                "found no improvement."
            )
        else:
            search_status = "unchanged"
            warnings.append(
                "GEPA selected the unchanged baseline after a completed "
                "search."
            )
    elif selected_equals_baseline:
        search_status = "unchanged"
        warnings.append(
            "GEPA evaluated candidate changes but selected the unchanged "
            "baseline."
        )
    else:
        search_status = "changed"
    assessment = {
        "candidate_count": candidate_count,
        "accepted_candidate_count": max(0, candidate_count - 1),
        "selected_candidate_index": selected_index,
        "selected_equals_baseline": selected_equals_baseline,
        "baseline_val_aggregate": baseline_score,
        "selected_val_aggregate": (
            detailed.val_aggregate_scores[selected_index]
            if detailed.val_aggregate_scores
            else None
        ),
        "reflection_attempts": reflection_attempts,
        "reflection_successes": reflection_successes,
        "valid_proposal_responses": valid_proposals,
        "task_attempts": task_attempts,
        "total_attempts": len(records),
        "reported_total_metric_calls": detailed.total_metric_calls,
        "num_full_val_evals": detailed.num_full_val_evals,
        "search_status": search_status,
        "status_override": status_override,
        "warnings": warnings,
    }
    return search_status, warnings, assessment


def run_optimization(
    settings: Settings,
    train_cases: Sequence[CaseRecord],
    validation_cases: Sequence[CaseRecord],
    *,
    baseline_development_run_id: str,
    confirm_budget: bool = False,
    run_id: str | None = None,
    origin: str = "live",
    runs_root: Path = RUNS_ROOT,
    programs_root: Path = PROGRAMS_ROOT,
    rules_text: str | None = None,
    ledger: BudgetLedger | None = None,
) -> OptimizationResult:
    """Compile a fresh baseline copy with genuine ``dspy.GEPA``.

    ``ledger`` may be supplied by a server-side job coordinator so the same
    per-job ledger can be cancelled before the next dispatch. It must use the
    exact confirmed optimization limits; a mismatched ledger is rejected
    before any dispatch.
    """
    _require_live_settings(settings)
    _require_confirmation(confirm_budget)
    limits = settings.optimization_budget()
    _validate_budget(limits)
    if ledger is None:
        ledger = BudgetLedger.from_limits(limits)
    elif not isinstance(ledger, BudgetLedger):
        raise ExperimentConfigurationError(
            "optimization ledger must be a BudgetLedger"
        )
    elif (
        ledger.max_calls != limits.max_calls
        or ledger.max_cost_usd != limits.max_cost_usd
    ):
        raise ExperimentConfigurationError(
            "injected optimization ledger limits do not match the confirmed "
            "optimization budget"
        )
    headroom = _validate_optimization_budget(settings)
    train, validation = _validate_optimization_split(
        train_cases, validation_cases
    )
    rules = _frozen_rules_text(rules_text)
    if origin not in ("live", "mock_test"):
        raise ExperimentConfigurationError("unsupported run origin")
    baseline_development_run_id = _validate_run_id(
        baseline_development_run_id
    )
    run_id = _validate_run_id(run_id or _new_run_id("opt"))
    secret = settings.openrouter_api_key
    baseline_program = EditorialQA()
    baseline_instructions = _assert_program_contract(
        baseline_program, context="baseline"
    )
    expected_fingerprint = baseline_fingerprint()
    baseline_manifest, baseline_dir = _save_program(
        baseline_program,
        origin=origin,
        programs_root=Path(programs_root),
        secret=secret,
    )
    _load_development_run(
        runs_root=Path(runs_root),
        run_id=baseline_development_run_id,
        validation_cases=validation,
        rules_text=rules,
        expected_baseline_program_id=baseline_manifest.program_id,
        expected_task=_task_config_payload(settings),
        origin=origin,
    )
    run_dir = _new_run_dir(Path(runs_root), "optimization", run_id)
    snapshot = write_input_snapshot(
        run_dir,
        cases=train + validation,
        rules_text=rules,
        split="train_validation",
        dataset_version="v1",
    )
    config = _config_payload(
        phase="optimization",
        run_id=run_id,
        origin=origin,
        settings=settings,
        split="train_validation",
        cases=train + validation,
        baseline_instructions=baseline_instructions,
        limits=limits,
        extra={
            "baseline_development_run_id": baseline_development_run_id,
            "gepa": _gepa_settings_payload(settings, headroom),
        },
    )
    atomic_write_json(run_dir / "config.json", config)
    atomic_write_jsonl(run_dir / RUN_REQUESTS_FILENAME, [])
    _append_event(
        run_dir,
        {
            "event": "optimization_started",
            "run_id": run_id,
            "baseline_development_run_id": baseline_development_run_id,
            "baseline_program_id": baseline_manifest.program_id,
            "max_metric_calls": settings.max_optimization_metric_calls,
            "budget_headroom_calls": headroom,
        },
    )
    _append_event(
        run_dir,
        {
            "event": "baseline_frozen",
            "program_id": baseline_manifest.program_id,
            "fingerprint": expected_fingerprint,
        },
    )
    write_lock = threading.Lock()

    def listener(record: LedgerRequestRecord) -> None:
        with write_lock:
            _write_requests(run_dir, ledger, secret)

    task_lm = build_guarded_lm(
        settings,
        role="task",
        ledger=ledger,
        token_counter=conservative_token_counter,
        formatting_overhead_tokens=FORMATTING_OVERHEAD_TOKENS,
        record_listener=listener,
    )
    reflection_lm = build_guarded_lm(
        settings,
        role="reflection",
        ledger=ledger,
        token_counter=conservative_token_counter,
        formatting_overhead_tokens=FORMATTING_OVERHEAD_TOKENS,
        record_listener=listener,
    )
    trainset = [_make_example(case, rules) for case in train]
    valset = [_make_example(case, rules) for case in validation]
    student = copy.deepcopy(baseline_program)
    compiled = None
    compile_error: str | None = None
    detailed: DspyGEPAResult | None = None
    status = "completed"
    stop_reason: str | None = None
    warnings: list[str] = []
    assessment: Mapping[str, object] = {}
    stats_payload: dict[str, object] | None = None
    selected_program = baseline_manifest
    selected_dir = baseline_dir
    try:
        optimizer = _build_gepa_optimizer(
            settings=settings,
            reflection_lm=reflection_lm,
            log_dir=run_dir / "gepa_native",
        )
        _append_event(
            run_dir,
            {
                "event": "compile_started",
                "train_cases": len(train),
                "validation_cases": len(validation),
            },
        )
        with dspy.context(lm=task_lm, adapter=StrictJSONAdapter()):
            compiled = optimizer.compile(
                student, trainset=trainset, valset=valset
            )
        _append_event(run_dir, {"event": "compile_returned"})
        _assert_baseline_immutable(
            baseline_program,
            expected_instructions=baseline_instructions,
            expected_fingerprint=expected_fingerprint,
        )
        compiled_instructions = _assert_program_contract(
            compiled, context="selected GEPA program"
        )
        detailed = getattr(compiled, "detailed_results", None)
        if not isinstance(detailed, DspyGEPAResult):
            raise ExperimentError(
                "GEPA did not return native detailed_results"
            )
        native_stats = detailed.to_dict()
        _assert_finite_json(native_stats, context="GEPA native statistics")
        _candidate_instructions(native_stats)
        best_program = detailed.candidates[detailed.best_idx]
        best_instructions = {
            name: predictor.signature.instructions
            for name, predictor in best_program.named_predictors()
        }.get("predict")
        if compiled_instructions != best_instructions:
            raise ExperimentError(
                "GEPA returned a program that does not match its selected "
                "native candidate"
            )
        search_status, search_warnings, assessment = _search_assessment(
            detailed=detailed,
            records=ledger.records,
            baseline_instructions=baseline_instructions,
        )
        warnings.extend(search_warnings)
        reported_calls = detailed.total_metric_calls
        recorded_tasks = assessment.get("task_attempts")
        if (
            isinstance(reported_calls, int)
            and isinstance(recorded_tasks, int)
            and reported_calls != recorded_tasks
        ):
            warnings.append(
                "GEPA reported "
                f"{reported_calls} metric calls while the ledger recorded "
                f"{recorded_tasks} task attempts; the ledger remains "
                "authoritative for cost and admission."
            )
        if (
            isinstance(reported_calls, int)
            and reported_calls > settings.max_optimization_metric_calls
        ):
            warnings.append(
                "GEPA exceeded the metric-call target by "
                f"{reported_calls - settings.max_optimization_metric_calls} "
                "call(s) as final-batch overhang; the confirmed ledger cap "
                "remains authoritative."
            )
        status_override = assessment.get("status_override")
        if status_override == "failed":
            status = "failed"
        if compiled_instructions == baseline_instructions:
            selected_program = baseline_manifest
            selected_dir = baseline_dir
            _append_event(
                run_dir,
                {
                    "event": "selected_unchanged_baseline",
                    "search_status": search_status,
                },
            )
        else:
            selected_program, selected_dir = _save_program(
                compiled,
                origin=origin,
                programs_root=Path(programs_root),
                secret=secret,
                parent_program_id=baseline_manifest.program_id,
            )
            _append_event(
                run_dir,
                {
                    "event": "selected_candidate_saved",
                    "program_id": selected_program.program_id,
                    "parent_program_id": baseline_manifest.program_id,
                },
            )
        loaded = load_program_artifact(
            selected_dir, allow_mock=(origin == "mock_test")
        )
        if (
            loaded.program.predict.signature.instructions
            != selected_program.instructions
        ):
            raise ExperimentError(
                "saved selected program does not round-trip its instructions"
            )
        if loaded.program.predict.demos:
            raise ExperimentError("saved selected program contains demos")
        if loaded.program.predict.lm is not None:
            raise ExperimentError("saved selected program persisted an LM")
        stats_payload = {
            "schema_version": GEPA_STATS_SCHEMA_VERSION,
            "status": status,
            "run_id": run_id,
            "baseline_program_id": baseline_manifest.program_id,
            "selected_program_id": selected_program.program_id,
            "selection": {
                **assessment,
                "budget": _budget_payload(ledger.snapshot()),
            },
            "detailed_results": native_stats,
        }
    except Exception as exc:
        compile_error = _sanitize_text(
            f"{type(exc).__name__}: {exc}", secret
        )
        snapshot_budget = ledger.snapshot()
        if snapshot_budget.stop_reason in _TERMINAL_BUDGET_STOPS:
            status = "stopped"
            stop_reason = snapshot_budget.stop_reason
            warnings.append(
                "Optimization stopped at the confirmed budget: "
                f"{snapshot_budget.stop_reason}."
            )
        else:
            status = "failed"
            stop_reason = compile_error
            warnings.append(
                "Optimization failed before a genuine selected candidate "
                "was produced."
            )
        warnings.append(
            "No selected candidate is recorded; the baseline program ID is "
            "a schema placeholder, and this run must not be treated as a "
            "completed optimization."
        )
        selected_program = baseline_manifest
        selected_dir = baseline_dir
        stats_payload = {
            "schema_version": GEPA_STATS_SCHEMA_VERSION,
            "status": status,
            "run_id": run_id,
            "baseline_program_id": baseline_manifest.program_id,
            "selected_program_id": None,
            "error": compile_error,
            "selection": {
                "search_status": "not_completed",
                "candidate_count": 0,
                "budget": _budget_payload(snapshot_budget),
            },
            "detailed_results": None,
        }
        _append_event(
            run_dir,
            {
                "event": "compile_failed",
                "status": status,
                "error": compile_error,
            },
        )

    final_budget = ledger.snapshot()
    if final_budget.stop_reason in _TERMINAL_BUDGET_STOPS and status == "completed":
        status = "stopped"
        stop_reason = final_budget.stop_reason
        warnings.append(
            "A budget stop invalidated the apparent completion: "
            f"{final_budget.stop_reason}."
        )
        if stats_payload is not None:
            stats_payload["status"] = status
    if stats_payload is not None:
        if status != "completed":
            stats_payload["stop_reason"] = stop_reason
        atomic_write_text(
            run_dir / RUN_GEPA_STATS_FILENAME, _pretty_json(stats_payload)
        )
        if isinstance(stats_payload.get("detailed_results"), Mapping):
            candidate_records = _candidate_instructions(
                stats_payload["detailed_results"]  # type: ignore[arg-type]
            )
            atomic_write_jsonl(
                run_dir / "candidate_instructions.jsonl",
                _json_ready(candidate_records),
            )
    _write_requests(run_dir, ledger, secret)
    _append_event(
        run_dir,
        {
            "event": "optimization_finished",
            "status": status,
            "stop_reason": stop_reason,
            "selected_program_id": selected_program.program_id,
            "admitted_calls": final_budget.admitted_calls,
        },
    )
    created_at = str(_read_json_object(run_dir / "config.json")["created_at"])
    program_refs = [program_reference(baseline_dir)]
    selected_ref = program_reference(selected_dir)
    if selected_ref.program_id != program_refs[0].program_id:
        program_refs.append(selected_ref)
    manifest = _run_manifest(
        run_id=run_id,
        kind="optimization",
        origin=origin,
        status=status,
        created_at=created_at,
        finished_at=_utc_string(),
        snapshot=snapshot,
        programs=program_refs,
        links=RunLinks(
            baseline_development_run_id=baseline_development_run_id,
            baseline_program_id=baseline_manifest.program_id,
            optimized_program_id=selected_program.program_id,
        ),
        run_dir=run_dir,
        file_names=(
            "config.json",
            "events.jsonl",
            RUN_REQUESTS_FILENAME,
            RUN_GEPA_STATS_FILENAME,
            "candidate_instructions.jsonl",
            snapshot.cases_file,
            snapshot.rules_file,
        ),
        stop_reason=stop_reason,
        warnings=warnings,
    )
    _write_manifest(run_dir, manifest, secret)
    return OptimizationResult(
        run_id=run_id,
        run_dir=run_dir,
        manifest=manifest,
        baseline_program=baseline_manifest,
        selected_program=selected_program,
        stats=stats_payload,
        ledger=final_budget,
        requests=ledger.records,
        warnings=tuple(warnings),
    )


def _program_ref_for_id(
    manifest: RunManifest, program_id: str
) -> object:
    for reference in manifest.programs:
        if reference.program_id == program_id:
            return reference
    raise ExperimentIncompleteError(
        f"run {manifest.run_id!r} does not archive program {program_id!r}"
    )


def _requests_cost_status(
    records: Sequence[LedgerRequestRecord],
) -> str:
    if not records:
        return "none"
    statuses = {record.cost_status for record in records}
    if "pending" in statuses:
        return "pending"
    if "estimated" in statuses:
        return "estimated"
    if "actual" in statuses:
        return "actual"
    return "unavailable"


def _read_optimization_cost(
    optimization_dir: Path, manifest: RunManifest
) -> tuple[Decimal | None, str, list[str]]:
    warnings: list[str] = []
    stats_path = optimization_dir / RUN_GEPA_STATS_FILENAME
    if not stats_path.is_file():
        return None, "unavailable", [
            "Optimization candidate statistics are missing; optimization "
            "cost is unknown."
        ]
    try:
        payload = _read_json_object(stats_path)
    except Exception as exc:
        return None, "unavailable", [
            f"Optimization statistics are unreadable: {exc}"
        ]
    selection = payload.get("selection")
    budget = (
        selection.get("budget")
        if isinstance(selection, Mapping)
        else None
    )
    total = budget.get("total_cost_usd") if isinstance(budget, Mapping) else None
    setup_cost = _known_cost(total)
    requests: Sequence[LedgerRequestRecord] = ()
    requests_path = optimization_dir / RUN_REQUESTS_FILENAME
    if requests_path.is_file():
        try:
            recorded = read_recorded_requests(requests_path)
            requests = tuple(recorded.values())
        except Exception as exc:
            warnings.append(
                f"Optimization requests are unreadable; cost certainty is "
                f"unavailable: {exc}"
            )
    request_status = _requests_cost_status(requests)
    if setup_cost is None:
        return None, "unavailable", warnings + [
            "Optimization cost is not a finite non-negative decimal."
        ]
    if isinstance(budget, Mapping) and budget.get("pending_cost_calls"):
        return setup_cost, "pending", warnings
    if request_status in ("pending", "unavailable"):
        return setup_cost, request_status, warnings
    if request_status == "none":
        return setup_cost, "unavailable", warnings
    return setup_cost, request_status, warnings


def _optimization_cost_breakdown(
    optimization_dir: Path,
    *,
    validation_cases: Sequence[CaseRecord],
    optimization_cases: Sequence[CaseRecord],
) -> dict[str, object]:
    path = optimization_dir / RUN_REQUESTS_FILENAME
    if not path.is_file():
        return {
            "task_train_cost_usd": None,
            "task_validation_cost_usd": None,
            "reflection_cost_usd": None,
            "unknown_task_cost_usd": None,
            "cost_status": "unavailable",
            "task_train_validation_split": "unavailable",
            "note": (
                "Optimization requests are not archived; task and reflection "
                "costs cannot be separated."
            ),
        }
    try:
        requests = tuple(read_recorded_requests(path).values())
    except Exception as exc:
        return {
            "task_train_cost_usd": None,
            "task_validation_cost_usd": None,
            "reflection_cost_usd": None,
            "unknown_task_cost_usd": None,
            "cost_status": "unavailable",
            "task_train_validation_split": "unavailable",
            "note": f"Optimization requests are unreadable: {exc}",
        }
    validation_ids = {case.case_id for case in validation_cases}
    validation_titles = {case.title for case in validation_cases}
    train_titles = {
        case.title
        for case in optimization_cases
        if case.case_id not in validation_ids
    }

    def record_cost(record: LedgerRequestRecord) -> Decimal | None:
        if record.reported_cost_usd is not None:
            return record.reported_cost_usd
        return record.estimated_cost_usd

    totals = {
        "task_train": Decimal("0"),
        "task_validation": Decimal("0"),
        "reflection": Decimal("0"),
        "unknown_task": Decimal("0"),
    }
    unknown_count = 0
    for record in requests:
        cost = record_cost(record)
        if cost is None:
            continue
        if record.phase == "reflection":
            totals["reflection"] += cost
            continue
        if record.phase != "task":
            totals["unknown_task"] += cost
            unknown_count += 1
            continue
        text = json.dumps(record.messages, ensure_ascii=False)
        in_validation = any(title in text for title in validation_titles)
        in_train = any(title in text for title in train_titles)
        if in_validation and not in_train:
            totals["task_validation"] += cost
        elif in_train and not in_validation:
            totals["task_train"] += cost
        else:
            totals["unknown_task"] += cost
            unknown_count += 1
    status = _requests_cost_status(requests)
    split_status = (
        "classified"
        if unknown_count == 0
        else "partial"
    )
    return {
        "task_train_cost_usd": str(totals["task_train"]),
        "task_validation_cost_usd": str(totals["task_validation"]),
        "reflection_cost_usd": str(totals["reflection"]),
        "unknown_task_cost_usd": str(totals["unknown_task"]),
        "cost_status": status,
        "task_train_validation_split": split_status,
        "note": (
            "Task requests are classified by matching their article titles "
            "to the optimization snapshot; reflection and compression calls "
            "are reported separately."
        ),
    }


def _read_development_cost(
    development_dir: Path,
) -> tuple[Decimal | None, str, list[str]]:
    warnings: list[str] = []
    summary_path = development_dir / BENCHMARK_SUMMARY_FILENAME
    if not summary_path.is_file():
        return None, "unavailable", [
            "Baseline development summary is missing; its cost is unknown."
        ]
    try:
        summary = _read_json_object(summary_path)
    except Exception as exc:
        return None, "unavailable", [
            f"Baseline development summary is unreadable: {exc}"
        ]
    budget = summary.get("budget")
    total = budget.get("total_cost_usd") if isinstance(budget, Mapping) else None
    cost = _known_cost(total)
    requests: Sequence[LedgerRequestRecord] = ()
    requests_path = development_dir / RUN_REQUESTS_FILENAME
    if requests_path.is_file():
        try:
            recorded = read_recorded_requests(requests_path)
            requests = tuple(recorded.values())
        except Exception as exc:
            warnings.append(
                f"Development requests are unreadable; cost certainty is "
                f"unavailable: {exc}"
            )
    status = _requests_cost_status(requests)
    if cost is None:
        return None, "unavailable", warnings
    if status == "none":
        return cost, "unavailable", warnings
    return cost, status, warnings


def _load_benchmark_links(
    *,
    runs_root: Path,
    programs_root: Path,
    baseline_development_run_id: str,
    optimization_run_id: str,
    origin: str,
    settings: Settings,
) -> _BenchmarkLinks:
    expected_task = _task_config_payload(settings)
    development_dir = Path(runs_root) / "development" / baseline_development_run_id
    if not development_dir.is_dir():
        raise ExperimentIncompleteError(
            "baseline development run is missing"
        )
    development_manifest = read_run_manifest(development_dir)
    development_cases, development_rules = read_input_snapshot(
        development_dir, development_manifest.snapshot
    )
    if development_rules != _frozen_rules_text(None):
        raise ExperimentIncompleteError(
            "baseline development used non-frozen editorial rules"
        )
    _load_development_run(
        runs_root=Path(runs_root),
        run_id=baseline_development_run_id,
        validation_cases=development_cases,
        rules_text=development_rules,
        expected_baseline_program_id=development_manifest.links.baseline_program_id,  # type: ignore[arg-type]
        expected_task=expected_task,
        origin=origin,
    )
    development_config = _read_json_object(development_dir / "config.json")
    baseline_program_id = development_manifest.links.baseline_program_id
    assert baseline_program_id is not None

    optimization_dir = Path(runs_root) / "optimization" / optimization_run_id
    if not optimization_dir.is_dir():
        raise ExperimentIncompleteError("optimization run is missing")
    optimization_manifest = read_run_manifest(optimization_dir)
    if optimization_manifest.kind != "optimization":
        raise ExperimentIncompleteError(
            "linked optimization run has the wrong kind"
        )
    if optimization_manifest.status != "completed":
        raise ExperimentIncompleteError(
            f"optimization run is {optimization_manifest.status!r}, "
            "not completed"
        )
    if optimization_manifest.origin != origin:
        raise ExperimentIncompleteError(
            "benchmark and optimization runs must share one origin"
        )
    verify_file_index(optimization_dir, optimization_manifest.files)
    links = optimization_manifest.links
    if links.baseline_development_run_id != baseline_development_run_id:
        raise ExperimentIncompleteError(
            "optimization run does not link the benchmark development run"
        )
    if links.baseline_program_id != baseline_program_id:
        raise ExperimentIncompleteError(
            "optimization baseline program does not match the benchmark"
        )
    optimized_program_id = links.optimized_program_id
    if optimized_program_id is None:
        raise ExperimentIncompleteError(
            "optimization run has no selected program"
        )
    if optimization_manifest.snapshot.split != "train_validation":
        raise ExperimentIncompleteError(
            "optimization run does not archive train_validation inputs"
        )
    optimization_cases, optimization_rules = read_input_snapshot(
        optimization_dir, optimization_manifest.snapshot
    )
    if optimization_rules != development_rules:
        raise ExperimentIncompleteError(
            "development and optimization runs used different rules"
        )
    optimization_config = _read_json_object(optimization_dir / "config.json")
    for key in (
        "task",
        "metric_version",
        "prompt_version",
        "dataset_version",
        "rules_version",
    ):
        if optimization_config.get(key) != development_config.get(key):
            raise ExperimentIncompleteError(
                f"development and optimization runs differ on {key}"
            )
    if optimization_config.get("task") != expected_task:
        raise ExperimentIncompleteError(
            "optimization task protocol does not match the benchmark"
        )
    optimization_by_id = {
        case.case_id: case for case in optimization_cases
    }
    for case in development_cases:
        optimization_case = optimization_by_id.get(case.case_id)
        if optimization_case is None:
            raise ExperimentIncompleteError(
                "optimization snapshot omits a development validation case"
            )
        if canonical_json_text(
            optimization_case.model_dump(mode="json")
        ) != canonical_json_text(case.model_dump(mode="json")):
            raise ExperimentIncompleteError(
                "optimization validation content differs from development"
            )

    baseline_ref = _program_ref_for_id(
        development_manifest, baseline_program_id
    )
    optimized_ref = _program_ref_for_id(
        optimization_manifest, optimized_program_id
    )
    baseline_snapshot = read_program_snapshot(
        Path(programs_root) / baseline_program_id
    )
    if (
        baseline_snapshot.manifest_sha256 != baseline_ref.manifest_sha256
        or baseline_snapshot.state_sha256 != baseline_ref.state_sha256
    ):
        raise ExperimentIncompleteError(
            "baseline program artifact does not match its run reference"
        )
    if optimized_program_id == baseline_program_id:
        optimized_snapshot = baseline_snapshot
    else:
        optimized_snapshot = read_program_snapshot(
            Path(programs_root) / optimized_program_id
        )
        if (
            optimized_snapshot.manifest_sha256 != optimized_ref.manifest_sha256
            or optimized_snapshot.state_sha256 != optimized_ref.state_sha256
        ):
            raise ExperimentIncompleteError(
                "optimized program artifact does not match its run reference"
            )
    if canonical_json_text(
        baseline_snapshot.manifest.signature_schema
    ) != canonical_json_text(optimized_snapshot.manifest.signature_schema):
        raise ExperimentIncompleteError(
            "benchmark arms use different signature schemas"
        )
    if canonical_json_text(
        baseline_snapshot.manifest.signature_fields
    ) != canonical_json_text(optimized_snapshot.manifest.signature_fields):
        raise ExperimentIncompleteError(
            "benchmark arms use different signature field contracts"
        )
    allow_mock = origin == "mock_test"
    baseline_loaded = load_program_artifact(
        Path(programs_root) / baseline_program_id, allow_mock=allow_mock
    )
    if optimized_program_id == baseline_program_id:
        optimized_loaded = baseline_loaded
    else:
        optimized_loaded = load_program_artifact(
            Path(programs_root) / optimized_program_id,
            allow_mock=allow_mock,
        )
    for label, loaded in (
        ("baseline", baseline_loaded),
        ("optimized", optimized_loaded),
    ):
        _assert_program_contract(loaded.program, context=f"{label} program")
        if (
            loaded.program.predict.signature.instructions
            != loaded.manifest.instructions
        ):
            raise ExperimentIncompleteError(
                f"{label} program instructions do not match its manifest"
            )

    optimization_cost, optimization_cost_status, optimization_warnings = (
        _read_optimization_cost(optimization_dir, optimization_manifest)
    )
    optimization_breakdown = _optimization_cost_breakdown(
        optimization_dir,
        validation_cases=development_cases,
        optimization_cases=optimization_cases,
    )
    development_cost, development_cost_status, development_warnings = (
        _read_development_cost(development_dir)
    )
    return _BenchmarkLinks(
        development_dir=development_dir,
        development_manifest=development_manifest,
        development_cases=tuple(development_cases),
        development_rules=development_rules,
        optimization_dir=optimization_dir,
        optimization_manifest=optimization_manifest,
        optimization_cases=tuple(optimization_cases),
        baseline_loaded=baseline_loaded,
        optimized_loaded=optimized_loaded,
        optimization_cost_usd=optimization_cost,
        optimization_cost_status=optimization_cost_status,
        optimization_cost_breakdown=optimization_breakdown,
        development_cost_usd=development_cost,
        development_cost_status=development_cost_status,
        cost_warnings=tuple(optimization_warnings + development_warnings),
    )


def _validate_test_isolation(
    test_cases: Sequence[CaseRecord],
    training_cases: Sequence[CaseRecord],
) -> None:
    test_ids = {case.case_id for case in test_cases}
    test_families = {case.family_id for case in test_cases}
    training_ids = {case.case_id for case in training_cases}
    training_families = {case.family_id for case in training_cases}
    if test_ids & training_ids:
        raise ExperimentConfigurationError(
            "sealed test cases overlap development/optimization case IDs"
        )
    if test_families & training_families:
        raise ExperimentConfigurationError(
            "sealed test families overlap development/optimization families"
        )
    test_hashes = {canonical_document_sha256(case) for case in test_cases}
    training_hashes = {
        canonical_document_sha256(case) for case in training_cases
    }
    if test_hashes & training_hashes:
        raise ExperimentConfigurationError(
            "sealed test documents duplicate development/optimization inputs"
        )


def _benchmark_reuse_history(
    runs_root: Path,
    *,
    planned_cases_sha256: str,
    rules_sha256: str,
    origin: str,
) -> dict[str, object]:
    root = Path(runs_root) / "benchmarks"
    prior: list[dict[str, object]] = []
    if root.exists():
        for directory in sorted(root.iterdir()):
            if not directory.is_dir():
                continue
            manifest_path = directory / RUN_MANIFEST_FILENAME
            if not manifest_path.is_file():
                config_path = directory / "config.json"
                if config_path.is_file():
                    try:
                        interrupted = _read_json_object(config_path)
                    except Exception as exc:
                        raise ExperimentIncompleteError(
                            "cannot determine test exposure from an "
                            f"interrupted benchmark {directory}: {exc}"
                        ) from exc
                    if (
                        interrupted.get("origin") != origin
                        or interrupted.get("dataset_version") != "v1"
                    ):
                        continue
                    prior.append(
                        {
                            "run_id": str(interrupted.get("run_id", directory.name)),
                            "created_at": str(interrupted.get("created_at", "")),
                            "status": "interrupted",
                            "released": False,
                            "case_count": len(
                                interrupted.get("case_ids", [])
                                if isinstance(
                                    interrupted.get("case_ids"), list
                                )
                                else []
                            ),
                            "same_test_inputs": False,
                        }
                    )
                    continue
                raise ExperimentIncompleteError(
                    "cannot determine test exposure because a prior "
                    f"benchmark directory is incomplete: {directory}"
                )
            try:
                manifest = read_run_manifest(directory)
            except Exception as exc:
                raise ExperimentIncompleteError(
                    "cannot determine test exposure from "
                    f"{directory}: {exc}"
                ) from exc
            if manifest.kind != "benchmark":
                raise ExperimentIncompleteError(
                    f"non-benchmark run found under benchmarks: {directory}"
                )
            if (
                manifest.origin == origin
                and manifest.snapshot.dataset_version == "v1"
                and manifest.snapshot.rules_sha256 == rules_sha256
            ):
                prior.append(
                    {
                        "run_id": manifest.run_id,
                        "created_at": manifest.created_at,
                        "status": manifest.status,
                        "released": bool(
                            manifest.benchmark
                            and manifest.benchmark.released
                        ),
                        "case_count": manifest.snapshot.case_count,
                        "same_test_inputs": (
                            manifest.snapshot.cases_sha256
                            == planned_cases_sha256
                        ),
                    }
                )
    prior.sort(key=lambda item: (str(item["created_at"]), str(item["run_id"])))
    exposure = len(prior)
    provenance = (
        "first_blind" if exposure == 0 else "exploratory_reevaluation"
    )
    warnings: list[str] = []
    if exposure:
        prior_ids = ", ".join(str(item["run_id"]) for item in prior)
        warnings.append(
            "Exploratory reevaluation: the sealed test split has exposure "
            f"index {exposure}; prior benchmark runs: {prior_ids}. This is "
            "not fresh confirmatory evidence."
        )
    return {
        "schema_version": TEST_REUSE_SCHEMA_VERSION,
        "test_cases_sha256": planned_cases_sha256,
        "rules_sha256": rules_sha256,
        "origin": origin,
        "exposure_index": exposure,
        "provenance": provenance,
        "count_rule": (
            "same origin, dataset version, and rules; same_test_inputs "
            "flags identical ordered case hashes"
        ),
        "prior_runs": prior,
        "warnings": warnings,
    }


def _build_benchmark_plan(
    *,
    settings: Settings,
    test_cases: Sequence[CaseRecord],
    rules_text: str,
    baseline_snapshot: ProgramSnapshot,
    optimized_snapshot: ProgramSnapshot,
    seed: int,
    case_orders: Mapping[str, Sequence[str]],
    limits: object,
) -> tuple[dict[str, object], str]:
    task = _task_config_payload(settings)
    task_config_sha256 = sha256_text(canonical_json_text(task))
    plan: dict[str, object] = {
        "schema_version": BENCHMARK_PLAN_SCHEMA_VERSION,
        "metric_version": METRIC_VERSION,
        "dataset_version": "v1",
        "test_cases_sha256": cases_snapshot_sha256(test_cases),
        "rules_version": RULES_VERSION,
        "rules_sha256": sha256_text(rules_text),
        "baseline_program": {
            "program_id": baseline_snapshot.manifest.program_id,  # type: ignore[attr-defined]
            "manifest_sha256": baseline_snapshot.manifest_sha256,  # type: ignore[attr-defined]
            "state_sha256": baseline_snapshot.state_sha256,  # type: ignore[attr-defined]
            "instructions_sha256": sha256_text(
                baseline_snapshot.manifest.instructions  # type: ignore[attr-defined]
            ),
        },
        "optimized_program": {
            "program_id": optimized_snapshot.manifest.program_id,  # type: ignore[attr-defined]
            "manifest_sha256": optimized_snapshot.manifest_sha256,  # type: ignore[attr-defined]
            "state_sha256": optimized_snapshot.state_sha256,  # type: ignore[attr-defined]
            "instructions_sha256": sha256_text(
                optimized_snapshot.manifest.instructions  # type: ignore[attr-defined]
            ),
        },
        "task": task,
        "task_config_sha256": task_config_sha256,
        "adapter": "StrictJSONAdapter",
        "retry_policy": "none",
        "cache_policy": "disabled",
        "concurrency": 1,
        "seed": seed,
        "case_ids": [case.case_id for case in test_cases],
        "case_orders": {
            case_id: list(order) for case_id, order in case_orders.items()
        },
        "planned_arms": len(test_cases) * 2,
        "budget": {
            "max_calls": getattr(limits, "max_calls"),
            "max_cost_usd": str(getattr(limits, "max_cost_usd")),
        },
        "evaluation_backend": (
            "dspy.Evaluate on one-case devsets with scalar_metric, "
            "num_threads=1, failure_score=0.0, max_errors=2"
        ),
        "latency_definition": (
            "monotonic prediction plus deterministic validation, milliseconds"
        ),
        "verdict_rule": (
            "observed_heldout_improvement only when complete, comparable, "
            "both arms 100% valid, optimized micro-F1 strictly higher, and "
            "optimized exact match not lower"
        ),
        "p95_label": "exploratory",
    }
    return plan, sha256_text(canonical_json_text(plan))


def _evaluate_benchmark_arm(
    *,
    run_id: str,
    case: CaseRecord,
    rules_text: str,
    arm: str,
    program: object,
    program_id: str,
    task_lm: object,
    ledger: BudgetLedger,
    secret: str | None,
) -> tuple[ArmMeasurement, RecordedOutcome, bool, str | None]:
    request_start = len(ledger.records)
    started = time.perf_counter()
    evaluation = None
    escaped: Exception | None = None
    with dspy.context(lm=task_lm, adapter=StrictJSONAdapter()):
        try:
            evaluator = dspy.Evaluate(
                devset=[_make_example(case, rules_text)],
                metric=scalar_metric,
                num_threads=1,
                display_progress=False,
                display_table=False,
                max_errors=2,
                failure_score=0.0,
            )
            evaluation = evaluator(program)  # type: ignore[arg-type]
        except BudgetStop as exc:
            escaped = exc
        except Exception as exc:
            escaped = exc
    records = ledger.records[request_start:]
    prediction = None
    evaluate_percent: float | None = None
    if evaluation is not None and len(evaluation.results) == 1:
        _, prediction, _ = evaluation.results[0]
        evaluate_percent = float(evaluation.score)
    error: str | None = None
    kind = "request_error"
    fatal: str | None = None
    if escaped is not None:
        error = _sanitize_text(
            f"{type(escaped).__name__}: {escaped}", secret
        )
        if not isinstance(escaped, BudgetStop):
            fatal = error
    else:
        failed_request = next(
            (
                record
                for record in records
                if record.status in ("in_flight", "released")
                or (record.status == "pending" and record.error)
            ),
            None,
        )
        stop_reason = ledger.snapshot().stop_reason
        if failed_request is not None:
            error = _sanitize_text(
                failed_request.error
                or f"task request {failed_request.call_id} failed",
                secret,
            )
        elif stop_reason in _TERMINAL_BUDGET_STOPS and not records:
            error = f"budget stop before dispatch: {stop_reason}"
            kind = "request_error"
        elif getattr(prediction, "findings", None) is None:
            error = (
                "prediction did not expose findings after a successful task "
                "request (schema/parse failure)"
            )
            kind = "parse_error"
    if error is None and prediction is not None:
        score = score_case(case, prediction)
        if not score.valid:
            error = score.error or "prediction failed deterministic validation"
            kind = (
                "parse_error"
                if score.error_kind == "parse_error"
                else "request_error"
            )
    elif prediction is None:
        score = score_case(
            case,
            InvalidPrediction(  # type: ignore[arg-type]
                error or "missing output", kind  # type: ignore[arg-type]
            ),
        )
    if error is not None:
        score = score_case(
            case,
            InvalidPrediction(  # type: ignore[arg-type]
                error, kind  # type: ignore[arg-type]
            ),
        )
    latency_ms = (time.perf_counter() - started) * 1000.0
    cost_usd, cost_status = _case_cost(records)
    findings: list[Finding] | None = None
    status = "error"
    if error is None and prediction is not None:
        raw_findings = getattr(prediction, "findings", None)
        if isinstance(raw_findings, list):
            findings = [
                item
                if isinstance(item, Finding)
                else Finding.model_validate(item)
                for item in raw_findings
            ]
            status = "ok"
        else:
            error = "prediction did not expose a findings list"
            score = score_case(
                case, InvalidPrediction(error, "missing_output")
            )
    outcome = _recorded_outcome(
        run_id=run_id,
        case=case,
        rules_text=rules_text,
        program_id=program_id,
        candidate_id=arm,
        phase=arm,
        split="test",
        status=status,
        findings=findings,
        error=error,
        score=score,
        request_ids=[record.call_id for record in records],
        cost_usd=cost_usd,
        cost_status=cost_status,
        latency_ms=latency_ms,
    )
    input_values = [record.input_tokens for record in records]
    output_values = [record.output_tokens for record in records]
    input_tokens = (
        sum(value for value in input_values if value is not None)
        if input_values and all(value is not None for value in input_values)
        else None
    )
    output_tokens = (
        sum(value for value in output_values if value is not None)
        if output_values and all(value is not None for value in output_values)
        else None
    )
    provider_values = [
        (record.finished_at - record.started_at).total_seconds() * 1000.0
        for record in records
        if record.finished_at is not None
    ]
    provider_latency_ms = (
        sum(provider_values)
        if provider_values and len(provider_values) == len(records)
        else None
    )
    measurement = ArmMeasurement(
        case_id=case.case_id,
        score=score,
        valid=status == "ok",
        cost_usd=cost_usd,
        cost_status=cost_status,
        latency_ms=latency_ms,
        evaluate_percent=evaluate_percent,
        input_sha256=case_input_sha256(case, rules_text),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        provider_latency_ms=provider_latency_ms,
    )
    terminal = ledger.snapshot().stop_reason in _TERMINAL_BUDGET_STOPS
    return measurement, outcome, terminal, fatal


def run_benchmark(
    settings: Settings,
    test_cases: Sequence[CaseRecord],
    *,
    baseline_development_run_id: str,
    optimization_run_id: str,
    confirm_budget: bool = False,
    seed: int = GEPA_SEED,
    release_results: bool = False,
    run_id: str | None = None,
    origin: str = "live",
    runs_root: Path = RUNS_ROOT,
    programs_root: Path = PROGRAMS_ROOT,
    rules_text: str | None = None,
) -> BenchmarkResult:
    """Re-evaluate two saved programs on sealed test cases, without compiling."""
    _require_live_settings(settings)
    _require_confirmation(confirm_budget)
    limits = settings.benchmark_budget()
    _validate_budget(limits)
    test_list = _validate_case_sequence(test_cases, name="sealed test cases")
    rules = _frozen_rules_text(rules_text)
    if origin not in ("live", "mock_test"):
        raise ExperimentConfigurationError("unsupported run origin")
    if not isinstance(release_results, bool):
        raise ExperimentConfigurationError(
            "release_results must be a boolean"
        )
    if release_results and origin != "live":
        raise ExperimentConfigurationError(
            "mock/test benchmark archives cannot be released"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ExperimentConfigurationError("benchmark seed must be an integer")
    if origin == "live":
        sealed_cases = load_benchmark_cases(DATA_DIR)
        if [
            canonical_json_text(case.model_dump(mode="json"))
            for case in sealed_cases
        ] != [
            canonical_json_text(case.model_dump(mode="json"))
            for case in test_list
        ]:
            raise ExperimentConfigurationError(
                "a Live benchmark must use the frozen sealed test split"
            )
    baseline_development_run_id = _validate_run_id(
        baseline_development_run_id
    )
    optimization_run_id = _validate_run_id(optimization_run_id)
    links = _load_benchmark_links(
        runs_root=Path(runs_root),
        programs_root=Path(programs_root),
        baseline_development_run_id=baseline_development_run_id,
        optimization_run_id=optimization_run_id,
        origin=origin,
        settings=settings,
    )
    _validate_test_isolation(
        test_list, links.development_cases + links.optimization_cases
    )
    test_cases_sha256 = cases_snapshot_sha256(test_list)
    rules_sha256 = sha256_text(rules)
    reuse = _benchmark_reuse_history(
        Path(runs_root),
        planned_cases_sha256=test_cases_sha256,
        rules_sha256=rules_sha256,
        origin=origin,
    )
    case_orders = derive_paired_orders(test_list, seed)
    baseline_snapshot = read_program_snapshot(
        Path(programs_root) / links.baseline_loaded.manifest.program_id
    )
    optimized_snapshot = read_program_snapshot(
        Path(programs_root) / links.optimized_loaded.manifest.program_id
    )
    plan, protocol_sha256 = _build_benchmark_plan(
        settings=settings,
        test_cases=test_list,
        rules_text=rules,
        baseline_snapshot=baseline_snapshot,
        optimized_snapshot=optimized_snapshot,
        seed=seed,
        case_orders=case_orders,
        limits=limits,
    )
    task_config_sha256 = str(plan["task_config_sha256"])
    run_id = _validate_run_id(run_id or _new_run_id("bench"))
    secret = settings.openrouter_api_key
    run_dir = _new_run_dir(Path(runs_root), "benchmarks", run_id)
    snapshot = write_input_snapshot(
        run_dir,
        cases=test_list,
        rules_text=rules,
        split="test",
        dataset_version="v1",
    )
    if snapshot.cases_sha256 != test_cases_sha256:
        raise ExperimentError("test snapshot hash changed before dispatch")
    config = {
        "schema_version": EXPERIMENT_CONFIG_SCHEMA_VERSION,
        "phase": "benchmark",
        "run_id": run_id,
        "origin": origin,
        "mode": settings.mode,
        "created_at": _utc_string(),
        "metric_version": METRIC_VERSION,
        "dataset_version": "v1",
        "split": "test",
        "case_ids": [case.case_id for case in test_list],
        "rules_version": RULES_VERSION,
        "task": _task_config_payload(settings),
        "budget": {
            "max_calls": limits.max_calls,
            "max_cost_usd": str(limits.max_cost_usd),
        },
        "num_threads": NUM_THREADS,
        "seed": seed,
        "case_orders": case_orders,
        "protocol_sha256": protocol_sha256,
        "baseline_development_run_id": baseline_development_run_id,
        "optimization_run_id": optimization_run_id,
        "baseline_program_id": links.baseline_loaded.manifest.program_id,
        "optimized_program_id": links.optimized_loaded.manifest.program_id,
        "release_requested": bool(release_results),
        "exposure_index": reuse["exposure_index"],
        "provenance": reuse["provenance"],
        "test_reuse": reuse,
    }
    atomic_write_json(run_dir / "config.json", config)
    atomic_write_jsonl(run_dir / RUN_OUTCOMES_FILENAME, [])
    atomic_write_jsonl(run_dir / RUN_REQUESTS_FILENAME, [])
    atomic_write_text(
        run_dir / TEST_REUSE_FILENAME, _pretty_json(reuse)
    )
    # Commit provenance (programs, task config, seed, per-case order, budget)
    # before the first paid benchmark call.
    atomic_write_text(
        run_dir / BENCHMARK_PROTOCOL_FILENAME,
        _pretty_json(
            {
                "schema_version": BENCHMARK_PLAN_SCHEMA_VERSION,
                "protocol_sha256": protocol_sha256,
                "plan": plan,
                "pre_dispatch_comparability_warnings": list(
                    reuse["warnings"]
                ),
            }
        ),
    )
    _append_event(
        run_dir,
        {
            "event": "benchmark_started",
            "run_id": run_id,
            "exposure_index": reuse["exposure_index"],
            "provenance": reuse["provenance"],
            "protocol_sha256": protocol_sha256,
            "planned_arms": len(test_list) * 2,
        },
    )
    for label, loaded in (
        ("baseline", links.baseline_loaded),
        ("optimized", links.optimized_loaded),
    ):
        _append_event(
            run_dir,
            {
                "event": "program_loaded",
                "arm": label,
                "program_id": loaded.manifest.program_id,
                "instructions_sha256": sha256_text(
                    loaded.manifest.instructions
                ),
            },
        )
    ledger = BudgetLedger.from_limits(limits)
    write_lock = threading.Lock()

    def listener(record: LedgerRequestRecord) -> None:
        with write_lock:
            _write_requests(run_dir, ledger, secret)

    task_lm = build_guarded_lm(
        settings,
        role="task",
        ledger=ledger,
        token_counter=conservative_token_counter,
        formatting_overhead_tokens=FORMATTING_OVERHEAD_TOKENS,
        record_listener=listener,
    )
    programs = {
        "baseline": links.baseline_loaded.program,
        "optimized": links.optimized_loaded.program,
    }
    program_ids = {
        "baseline": links.baseline_loaded.manifest.program_id,
        "optimized": links.optimized_loaded.manifest.program_id,
    }
    measurements: list[PairedMeasurement] = []
    outcomes: list[RecordedOutcome] = []
    status = "completed"
    stop_reason: str | None = None
    fatal = False
    try:
        for case in test_list:
            pair: dict[str, ArmMeasurement] = {}
            for arm in case_orders[case.case_id]:
                measurement, outcome, terminal, arm_fatal = (
                    _evaluate_benchmark_arm(
                        run_id=run_id,
                        case=case,
                        rules_text=rules,
                        arm=arm,
                        program=programs[arm],
                        program_id=program_ids[arm],
                        task_lm=task_lm,
                        ledger=ledger,
                        secret=secret,
                    )
                )
                pair[arm] = measurement
                outcomes.append(outcome)
                atomic_write_jsonl(
                    run_dir / RUN_OUTCOMES_FILENAME,
                    _outcomes_payload(outcomes),
                )
                _append_event(
                    run_dir,
                    {
                        "event": "arm_evaluated",
                        "case_id": case.case_id,
                        "arm": arm,
                        "status": outcome.status,
                        "reward": measurement.score.reward,
                        "valid": measurement.valid,
                        "cost_status": measurement.cost_status,
                    },
                )
                if arm_fatal is not None:
                    fatal = True
                    stop_reason = arm_fatal
                    break
                if terminal:
                    stop_reason = ledger.snapshot().stop_reason
                    break
            measurements.append(
                PairedMeasurement(
                    case_id=case.case_id,
                    baseline=pair.get("baseline"),
                    optimized=pair.get("optimized"),
                )
            )
            if fatal or (
                stop_reason in _TERMINAL_BUDGET_STOPS
            ):
                break
    except Exception as exc:
        fatal = True
        stop_reason = _sanitize_text(
            f"{type(exc).__name__}: {exc}", secret
        )
        _append_event(
            run_dir,
            {"event": "benchmark_fatal", "error": stop_reason},
        )
    final_budget = ledger.snapshot()
    if final_budget.stop_reason in _TERMINAL_BUDGET_STOPS:
        status = "stopped"
        stop_reason = final_budget.stop_reason
    elif fatal:
        status = "failed"
    elif len(outcomes) != len(test_list) * 2:
        status = "failed"
        stop_reason = "benchmark ended before all planned arm evaluations"
    _write_requests(run_dir, ledger, secret)
    comparability_warnings = list(reuse["warnings"])
    comparability_warnings.extend(links.cost_warnings)
    if links.optimization_cost_status in (
        "estimated",
        "pending",
        "unavailable",
    ):
        comparability_warnings.append(
            "Optimization cost status is "
            f"{links.optimization_cost_status}."
        )
    if links.development_cost_status in (
        "estimated",
        "pending",
        "unavailable",
    ):
        comparability_warnings.append(
            "Development cost status is "
            f"{links.development_cost_status}."
        )
    if len(test_list) != TEST_ARTICLE_COUNT:
        comparability_warnings.append(
            f"Benchmark fixture contains {len(test_list)} test cases; the "
            f"production split contains {TEST_ARTICLE_COUNT}."
        )
    comparability_warnings = list(dict.fromkeys(comparability_warnings))
    summary = summarize_benchmark(
        measurements,
        planned_case_count=len(test_list),
        run_complete=status == "completed",
        comparable=True,
        comparability_warnings=comparability_warnings,
        optimization_cost_usd=links.optimization_cost_usd,
        optimization_cost_status=links.optimization_cost_status,
        optimization_cost_breakdown=links.optimization_cost_breakdown,
        baseline_development_cost_usd=links.development_cost_usd,
        baseline_development_cost_status=links.development_cost_status,
        protocol_sha256=protocol_sha256,
        task_config_sha256=task_config_sha256,
        test_reuse=reuse,
    )
    if status != "completed":
        summary["verdict"] = "incomplete"
        summary["verdict_reason"] = (
            f"Benchmark {status}: only {len(outcomes)} of "
            f"{len(test_list) * 2} planned arm evaluations were recorded."
        )
        summary["eligible_for_verdict"] = False
        summary["quality_non_regression_gate"] = False
    summary["run_status"] = status
    summary["stop_reason"] = stop_reason
    summary_warnings = list(summary.get("warnings", []))
    if links.optimization_cost_status in ("estimated", "pending", "unavailable"):
        summary_warnings.append(
            "Optimization cost status is "
            f"{links.optimization_cost_status}; break-even is conditional."
        )
    if links.development_cost_status in (
        "estimated",
        "pending",
        "unavailable",
    ):
        summary_warnings.append(
            "Development cost status is "
            f"{links.development_cost_status}."
        )
    if status == "stopped":
        summary_warnings.append(
            "Benchmark incomplete: a budget or cancellation stop occurred "
            "before every planned arm finished."
        )
    if status == "failed":
        summary_warnings.append(
            f"Benchmark failed: {stop_reason}"
        )
    summary["warnings"] = list(dict.fromkeys(summary_warnings))
    released = bool(
        release_results and origin == "live" and status == "completed"
    )
    protocol = BenchmarkProtocol(
        exposure_index=int(reuse["exposure_index"]),
        paired_order=list(case_orders[test_list[0].case_id]),
        case_orders={
            case_id: list(order) for case_id, order in case_orders.items()
        },
        seed=seed,
        protocol_sha256=protocol_sha256,
        released=released,
        comparability_warnings=comparability_warnings,
    )
    atomic_write_text(
        run_dir / BENCHMARK_SUMMARY_FILENAME,
        _pretty_json(summary),
    )
    _append_event(
        run_dir,
        {
            "event": "benchmark_finished",
            "status": status,
            "verdict": summary["verdict"],
            "admitted_calls": final_budget.admitted_calls,
            "exposure_index": reuse["exposure_index"],
        },
    )
    created_at = str(_read_json_object(run_dir / "config.json")["created_at"])
    program_refs = [program_reference(Path(programs_root) / program_ids["baseline"])]
    if program_ids["optimized"] != program_ids["baseline"]:
        program_refs.append(
            program_reference(Path(programs_root) / program_ids["optimized"])
        )
    manifest_warnings = list(summary.get("warnings", []))
    if released:
        manifest_warnings.append(
            "Released live benchmark; original measurements and reuse "
            "history are frozen."
        )
    manifest = _run_manifest(
        run_id=run_id,
        kind="benchmark",
        origin=origin,
        status=status,
        created_at=created_at,
        finished_at=_utc_string(),
        snapshot=snapshot,
        programs=program_refs,
        links=RunLinks(
            baseline_development_run_id=baseline_development_run_id,
            optimization_run_id=optimization_run_id,
            baseline_program_id=program_ids["baseline"],
            optimized_program_id=program_ids["optimized"],
        ),
        run_dir=run_dir,
        file_names=(
            "config.json",
            BENCHMARK_PROTOCOL_FILENAME,
            BENCHMARK_SUMMARY_FILENAME,
            TEST_REUSE_FILENAME,
            "events.jsonl",
            RUN_OUTCOMES_FILENAME,
            RUN_REQUESTS_FILENAME,
            snapshot.cases_file,
            snapshot.rules_file,
        ),
        stop_reason=stop_reason,
        warnings=manifest_warnings,
        benchmark=protocol,
    )
    _write_manifest(run_dir, manifest, secret)
    return BenchmarkResult(
        run_id=run_id,
        run_dir=run_dir,
        manifest=manifest,
        baseline_program=links.baseline_loaded.manifest,
        optimized_program=links.optimized_loaded.manifest,
        outcomes=tuple(outcomes),
        measurements=tuple(measurements),
        summary=summary,
        ledger=final_budget,
        requests=ledger.records,
        warnings=tuple(manifest_warnings),
    )


def derive_paired_orders(
    cases: Sequence[CaseRecord], seed: int
) -> dict[str, list[str]]:
    """Deterministically randomize one A/B or B/A order per case."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ExperimentConfigurationError("benchmark seed must be an integer")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ExperimentConfigurationError(
            "benchmark case IDs must be unique"
        )
    rng = random.Random(seed)
    orders: dict[str, list[str]] = {}
    for case in cases:
        order = ["baseline", "optimized"]
        rng.shuffle(order)
        orders[case.case_id] = order
    return orders


def latency_summary(values: Sequence[float | None]) -> dict[str, object]:
    """Return raw, median, and nearest-rank exploratory p95 in milliseconds."""
    raw = [float(value) for value in values if value is not None]
    if not raw:
        return {
            "raw": [],
            "count": 0,
            "median": None,
            "p95": None,
            "unit": "ms",
            "p95_label": "exploratory",
        }
    ordered = sorted(raw)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "raw": raw,
        "count": len(raw),
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "unit": "ms",
        "p95_label": "exploratory",
    }


def _delta(left: object, right: object) -> float | None:
    if left is None or right is None:
        return None
    try:
        return float(right) - float(left)
    except (TypeError, ValueError):
        return None


def _cost_status_summary(measurements: Sequence[ArmMeasurement]) -> dict[str, object]:
    if not measurements:
        return {
            "known_cost_usd": "0",
            "currency": "USD",
            "cost_status": "none",
            "cost_per_article_usd": None,
            "attempted_articles": 0,
        }
    statuses = [item.cost_status for item in measurements]
    known = sum(
        (
            item.cost_usd
            for item in measurements
            if item.cost_usd is not None
        ),
        Decimal("0"),
    )
    if "pending" in statuses:
        status = "pending"
        per_article = None
    elif "unavailable" in statuses:
        status = "unavailable"
        per_article = None
    elif "estimated" in statuses:
        status = "estimated"
        per_article = known / len(measurements)
    elif any(item == "actual" for item in statuses):
        status = "actual"
        per_article = known / len(measurements)
    else:
        status = "none"
        per_article = Decimal("0")
    return {
        "known_cost_usd": str(known),
        "currency": "USD",
        "cost_status": status,
        "cost_per_article_usd": (
            None if per_article is None else str(per_article)
        ),
        "attempted_articles": len(measurements),
    }


def _combine_cost_statuses(
    known_cost: Decimal, statuses: Sequence[str]
) -> str:
    if "pending" in statuses:
        return "pending"
    if "unavailable" in statuses:
        return "unavailable"
    if "estimated" in statuses:
        return "estimated"
    if any(status == "actual" for status in statuses):
        return "actual"
    return "none"


def _arm_summary(measurements: Sequence[ArmMeasurement]) -> dict[str, object]:
    scores = [item.score for item in measurements]
    corpus = aggregate_scores(scores)
    payload = _corpus_payload(corpus)
    attempted = len(measurements)
    valid_count = sum(1 for item in measurements if item.valid)
    evaluate_values = [
        item.evaluate_percent
        for item in measurements
        if item.evaluate_percent is not None
    ]
    input_token_values = [
        item.input_tokens
        for item in measurements
        if item.input_tokens is not None
    ]
    output_token_values = [
        item.output_tokens
        for item in measurements
        if item.output_tokens is not None
    ]
    unknown_tokens = sum(
        1
        for item in measurements
        if item.input_tokens is None or item.output_tokens is None
    )
    payload.update(
        {
            "attempted_articles": attempted,
            "valid_count": valid_count,
            "invalid_count": attempted - valid_count,
            "parsing_success": (
                None if attempted == 0 else valid_count / attempted
            ),
            "evaluate_percent_mean": (
                None
                if not evaluate_values
                else statistics.mean(evaluate_values)
            ),
            "evaluate_score_semantics": (
                "rounded percentage mean of per-case rewards; not corpus "
                "micro-F1"
            ),
            "latency": latency_summary(
                [item.latency_ms for item in measurements]
            ),
            "provider_latency": latency_summary(
                [item.provider_latency_ms for item in measurements]
            ),
            "latency_semantics": (
                "latency includes prediction plus deterministic validation; "
                "provider_latency is the recorded LM request duration only"
            ),
            "tokens": {
                "input_tokens_known": (
                    None
                    if unknown_tokens
                    else sum(input_token_values)
                ),
                "output_tokens_known": (
                    None
                    if unknown_tokens
                    else sum(output_token_values)
                ),
                "total_tokens_known": (
                    None
                    if unknown_tokens
                    else sum(input_token_values)
                    + sum(output_token_values)
                ),
                "unknown_measurements": unknown_tokens,
                "note": (
                    "null means at least one evaluated arm lacked recorded "
                    "token usage"
                ),
            },
            "cost": _cost_status_summary(measurements),
        }
    )
    return payload


def _known_cost(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        decimal = Decimal(str(value))
    except (ArithmeticError, ValueError):
        return None
    if not decimal.is_finite() or decimal < 0:
        return None
    return decimal


def _cost_settled(status: str, value: Decimal | None) -> bool:
    return status in {"actual", "estimated", "none"} and (
        value is not None or status == "none"
    )


def compute_break_even(
    *,
    optimization_cost_usd: Decimal | None,
    optimization_cost_status: str,
    baseline_cost_per_article_usd: Decimal | None,
    baseline_cost_status: str,
    optimized_cost_per_article_usd: Decimal | None,
    optimized_cost_status: str,
    quality_gate: bool,
    complete: bool,
) -> dict[str, object]:
    """Compute the predeclared cost-only break-even with Decimal arithmetic."""
    base_cost = _known_cost(baseline_cost_per_article_usd)
    opt_cost = _known_cost(optimized_cost_per_article_usd)
    setup_cost = _known_cost(optimization_cost_usd)
    result: dict[str, object] = {
        "formula": "ceil(C_opt / (c_A - c_B))",
        "currency": "USD",
        "optimization_cost_usd": (
            None if setup_cost is None else str(setup_cost)
        ),
        "optimization_cost_status": optimization_cost_status,
        "baseline_inference_cost_per_article_usd": (
            None if base_cost is None else str(base_cost)
        ),
        "baseline_cost_status": baseline_cost_status,
        "optimized_inference_cost_per_article_usd": (
            None if opt_cost is None else str(opt_cost)
        ),
        "optimized_cost_status": optimized_cost_status,
        "savings_per_article_usd": None,
        "savings_direction": "unknown",
        "quality_non_regression_gate": bool(quality_gate),
        "break_even_articles": None,
        "status": "not_applicable",
        "reason": None,
        "costs_estimated": any(
            status == "estimated"
            for status in (
                optimization_cost_status,
                baseline_cost_status,
                optimized_cost_status,
            )
        ),
        "assumptions": [
            "Only measured inference cost is included in savings.",
            "Holdout per-article cost is used as the production inference-cost "
            "proxy.",
            "Optimization cost is the recorded one-time optimization run cost.",
            "Annotation, engineering, and human-review costs are excluded.",
        ],
    }
    if not complete:
        result["reason"] = "incomplete_benchmark"
        return result
    if not quality_gate:
        result["reason"] = "quality_non_regression_gate_not_met"
        return result
    if base_cost is None and baseline_cost_status == "none":
        base_cost = Decimal("0")
        result["baseline_inference_cost_per_article_usd"] = "0"
    if opt_cost is None and optimized_cost_status == "none":
        opt_cost = Decimal("0")
        result["optimized_inference_cost_per_article_usd"] = "0"
    if setup_cost is None and optimization_cost_status == "none":
        setup_cost = Decimal("0")
        result["optimization_cost_usd"] = "0"
    if not (
        _cost_settled(baseline_cost_status, base_cost)
        and _cost_settled(optimized_cost_status, opt_cost)
        and _cost_settled(optimization_cost_status, setup_cost)
    ):
        result["reason"] = "cost_status_not_settled"
        return result
    assert base_cost is not None and opt_cost is not None
    assert setup_cost is not None
    savings = base_cost - opt_cost
    result["savings_per_article_usd"] = str(savings)
    if savings == 0:
        result["savings_direction"] = "zero"
        result["status"] = "no_finite_break_even"
        result["reason"] = "zero_inference_savings"
        return result
    if savings < 0:
        result["savings_direction"] = "negative"
        result["status"] = "no_finite_break_even"
        result["reason"] = "optimized_inference_is_more_expensive"
        return result
    result["savings_direction"] = "positive"
    result["status"] = "finite"
    result["break_even_articles"] = math.ceil(setup_cost / savings)
    result["reason"] = "finite_cost_only_break_even"
    return result


def summarize_benchmark(
    measurements: Sequence[PairedMeasurement],
    *,
    planned_case_count: int,
    run_complete: bool,
    comparable: bool,
    comparability_warnings: Sequence[str],
    optimization_cost_usd: Decimal | None,
    optimization_cost_status: str,
    baseline_development_cost_usd: Decimal | None,
    baseline_development_cost_status: str,
    protocol_sha256: str,
    task_config_sha256: str,
    test_reuse: Mapping[str, object],
    optimization_cost_breakdown: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Aggregate exact paired metrics, validity gates, costs, and economics."""
    case_ids = [pair.case_id for pair in measurements]
    if len(case_ids) != len(set(case_ids)):
        raise ExperimentError("paired measurements contain duplicate cases")
    baseline_measurements = [
        pair.baseline for pair in measurements if pair.baseline is not None
    ]
    optimized_measurements = [
        pair.optimized for pair in measurements if pair.optimized is not None
    ]
    full_pairs = [
        pair
        for pair in measurements
        if pair.baseline is not None and pair.optimized is not None
    ]
    attempted_pairs = len(full_pairs)
    attempted_arms = len(baseline_measurements) + len(optimized_measurements)
    planned_arms = planned_case_count * 2
    complete = bool(
        run_complete
        and len(measurements) == planned_case_count
        and attempted_pairs == planned_case_count
        and attempted_arms == planned_arms
    )
    inputs_equal = (
        attempted_pairs == planned_case_count
        and all(
            pair.baseline.input_sha256 == pair.optimized.input_sha256  # type: ignore[union-attr]
            for pair in full_pairs
        )
    )
    both_full_validity = bool(
        complete
        and all(item.valid for item in baseline_measurements)
        and all(item.valid for item in optimized_measurements)
    )
    baseline = _arm_summary(baseline_measurements)
    optimized = _arm_summary(optimized_measurements)
    metric_keys = (
        "precision",
        "recall",
        "micro_f1",
        "macro_f1",
        "exact_match_rate",
        "parsing_success",
        "mean_reward",
    )
    deltas = {
        key: _delta(baseline.get(key), optimized.get(key))
        for key in metric_keys
    }
    baseline_exact = sum(
        1 for item in baseline_measurements if item.score.exact_match
    )
    optimized_exact = sum(
        1 for item in optimized_measurements if item.score.exact_match
    )
    corrected_cases = [
        pair.case_id
        for pair in full_pairs
        if not pair.baseline.score.exact_match  # type: ignore[union-attr]
        and pair.optimized.score.exact_match  # type: ignore[union-attr]
    ]
    regressed_cases = [
        pair.case_id
        for pair in full_pairs
        if pair.baseline.score.exact_match  # type: ignore[union-attr]
        and not pair.optimized.score.exact_match  # type: ignore[union-attr]
    ]
    base_categories = {
        str(item["code"]): item
        for item in baseline.get("per_category", [])  # type: ignore[union-attr]
        if isinstance(item, Mapping)
    }
    optimized_categories = {
        str(item["code"]): item
        for item in optimized.get("per_category", [])  # type: ignore[union-attr]
        if isinstance(item, Mapping)
    }
    category_regressions: list[dict[str, object]] = []
    for code, base_category in base_categories.items():
        opt_category = optimized_categories.get(code)
        if opt_category is None:
            continue
        base_f1 = base_category.get("f1")
        opt_f1 = opt_category.get("f1")
        if (
            isinstance(base_f1, (int, float))
            and isinstance(opt_f1, (int, float))
            and opt_f1 < base_f1
        ):
            category_regressions.append(
                {
                    "code": code,
                    "baseline_f1": base_f1,
                    "optimized_f1": opt_f1,
                    "delta": float(opt_f1) - float(base_f1),
                }
            )
    invalid_cases = {
        "baseline": [
            item.case_id for item in baseline_measurements if not item.valid
        ],
        "optimized": [
            item.case_id for item in optimized_measurements if not item.valid
        ],
    }
    eligible_for_verdict = bool(
        complete and comparable and inputs_equal and both_full_validity
    )
    micro_base = baseline.get("micro_f1")
    micro_opt = optimized.get("micro_f1")
    exact_base = baseline.get("exact_match_rate")
    exact_opt = optimized.get("exact_match_rate")
    quality_gate = bool(
        eligible_for_verdict
        and isinstance(micro_base, (int, float))
        and isinstance(micro_opt, (int, float))
        and micro_opt >= micro_base
        and isinstance(exact_base, (int, float))
        and isinstance(exact_opt, (int, float))
        and exact_opt >= exact_base
    )
    if not complete:
        verdict = "incomplete"
        verdict_reason = (
            f"Only {attempted_arms} of {planned_arms} planned arm "
            "evaluations were recorded."
        )
    elif not comparable or not inputs_equal:
        verdict = "not_comparable"
        verdict_reason = (
            "Arm inputs, protocol fingerprints, or frozen configuration "
            "differ."
        )
    elif not both_full_validity:
        verdict = "validity_problem"
        verdict_reason = (
            "At least one arm did not reach 100% execution/schema validity."
        )
    elif (
        isinstance(micro_base, (int, float))
        and isinstance(micro_opt, (int, float))
        and micro_opt > micro_base
        and isinstance(exact_base, (int, float))
        and isinstance(exact_opt, (int, float))
        and exact_opt >= exact_base
    ):
        verdict = "observed_heldout_improvement"
        verdict_reason = (
            "Both arms are complete and valid; optimized micro-F1 is "
            "strictly higher and exact match is not lower."
        )
    elif (
        isinstance(micro_base, (int, float))
        and isinstance(micro_opt, (int, float))
        and micro_opt < micro_base
        and isinstance(exact_base, (int, float))
        and isinstance(exact_opt, (int, float))
        and exact_opt < exact_base
    ):
        verdict = "regression"
        verdict_reason = (
            "Both arms are complete and valid, but optimized micro-F1 and "
            "exact match are lower."
        )
    elif micro_opt == micro_base and exact_opt == exact_base:
        verdict = "tie"
        verdict_reason = (
            "Both arms are complete and valid with equal micro-F1 and exact "
            "match."
        )
    else:
        verdict = "mixed_tradeoff"
        verdict_reason = (
            "Both arms are complete and valid, but the aggregate metric "
            "deltas point in different directions."
        )
    baseline_cost = _cost_status_summary(baseline_measurements)
    optimized_cost = _cost_status_summary(optimized_measurements)
    economics = compute_break_even(
        optimization_cost_usd=optimization_cost_usd,
        optimization_cost_status=optimization_cost_status,
        baseline_cost_per_article_usd=_known_cost(
            baseline_cost["cost_per_article_usd"]
        ),
        baseline_cost_status=str(baseline_cost["cost_status"]),
        optimized_cost_per_article_usd=_known_cost(
            optimized_cost["cost_per_article_usd"]
        ),
        optimized_cost_status=str(optimized_cost["cost_status"]),
        quality_gate=quality_gate,
        complete=complete,
    )
    benchmark_known = _known_cost(baseline_cost["known_cost_usd"]) or Decimal(
        "0"
    )
    benchmark_known += _known_cost(optimized_cost["known_cost_usd"]) or Decimal(
        "0"
    )
    benchmark_status = _combine_cost_statuses(
        benchmark_known,
        [
            str(baseline_cost["cost_status"]),
            str(optimized_cost["cost_status"]),
        ],
    )
    optimization_known = _known_cost(optimization_cost_usd)
    development_known = _known_cost(baseline_development_cost_usd)
    one_time_known = (optimization_known or Decimal("0")) + (
        development_known or Decimal("0")
    )
    one_time_status = _combine_cost_statuses(
        one_time_known,
        [optimization_cost_status, baseline_development_cost_status],
    )
    warnings = list(dict.fromkeys(comparability_warnings))
    if complete and not both_full_validity:
        warnings.append(
            "Complete-with-errors benchmark: validity gate failed, so no "
            "official improvement verdict is available."
        )
    if category_regressions:
        warnings.append(
            "Category F1 regressed for: "
            + ", ".join(
                str(item["code"]) for item in category_regressions
            )
        )
    if economics["costs_estimated"]:
        warnings.append(
            "Costs are estimated, not provider-invoiced; break-even inherits "
            "that uncertainty."
        )
    if baseline.get("micro_f1") is None or optimized.get("micro_f1") is None:
        warnings.append(
            "Micro-F1 is undefined for at least one arm; compare raw TP/FP/FN "
            "instead of treating an empty denominator as perfect."
        )
    return {
        "schema_version": BENCHMARK_SUMMARY_SCHEMA_VERSION,
        "protocol_sha256": protocol_sha256,
        "task_config_sha256": task_config_sha256,
        "planned_cases": planned_case_count,
        "planned_arms": planned_arms,
        "attempted_pairs": attempted_pairs,
        "attempted_arms": attempted_arms,
        "complete": complete,
        "complete_with_errors": complete and not both_full_validity,
        "comparable": bool(comparable and inputs_equal),
        "inputs_equal": inputs_equal,
        "validity": {
            "both_arms_full_validity": both_full_validity,
            "baseline_invalid_cases": invalid_cases["baseline"],
            "optimized_invalid_cases": invalid_cases["optimized"],
        },
        "metrics": {
            "baseline": baseline,
            "optimized": optimized,
            "delta": deltas,
        },
        "exact_match_counts": {
            "baseline": baseline_exact,
            "optimized": optimized_exact,
            "total": planned_case_count,
        },
        "corrected_cases": corrected_cases,
        "regressed_cases": regressed_cases,
        "category_regressions": category_regressions,
        "latency_delta": {
            "median_ms": _delta(
                (baseline.get("latency") or {}).get("median"),  # type: ignore[union-attr]
                (optimized.get("latency") or {}).get("median"),  # type: ignore[union-attr]
            ),
            "p95_ms": _delta(
                (baseline.get("latency") or {}).get("p95"),  # type: ignore[union-attr]
                (optimized.get("latency") or {}).get("p95"),  # type: ignore[union-attr]
            ),
            "unit": "ms",
            "p95_label": "exploratory",
        },
        "costs": {
            "currency": "USD",
            "baseline": baseline_cost,
            "optimized": optimized_cost,
            "benchmark_inference_known_cost_usd": str(benchmark_known),
            "benchmark_inference_cost_status": benchmark_status,
            "optimization_cost_usd": (
                None
                if optimization_known is None
                else str(optimization_known)
            ),
            "optimization_cost_status": optimization_cost_status,
            "optimization_breakdown": dict(
                optimization_cost_breakdown or {}
            ),
            "baseline_development_cost_usd": (
                None
                if development_known is None
                else str(development_known)
            ),
            "baseline_development_cost_status": (
                baseline_development_cost_status
            ),
            "one_time_known_cost_usd": str(one_time_known),
            "one_time_cost_status": one_time_status,
        },
        "economics": economics,
        "eligible_for_verdict": eligible_for_verdict,
        "quality_non_regression_gate": quality_gate,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "test_reuse": dict(test_reuse),
        "warnings": warnings,
    }


__all__ = [
    "BASELINE_CANDIDATE_ID",
    "BENCHMARK_PLAN_SCHEMA_VERSION",
    "BENCHMARK_PROTOCOL_FILENAME",
    "BENCHMARK_SUMMARY_FILENAME",
    "BENCHMARK_SUMMARY_SCHEMA_VERSION",
    "DATA_DIR",
    "EXPERIMENT_CONFIG_SCHEMA_VERSION",
    "ExperimentConfigurationError",
    "ExperimentError",
    "ExperimentIncompleteError",
    "ExperimentNotAuthorized",
    "ExperimentRunExistsError",
    "GEPA_STATS_SCHEMA_VERSION",
    "METRIC_VERSION",
    "MIN_OPTIMIZATION_HEADROOM_CALLS",
    "TEST_REUSE_FILENAME",
    "TEST_REUSE_SCHEMA_VERSION",
    "ArmMeasurement",
    "BaselineDevelopmentResult",
    "BenchmarkResult",
    "OptimizationResult",
    "PairedMeasurement",
    "compute_break_even",
    "derive_paired_orders",
    "latency_summary",
    "load_sealed_benchmark_cases",
    "load_train_validation_cases",
    "load_validation_cases",
    "run_baseline_development",
    "run_benchmark",
    "run_optimization",
    "summarize_benchmark",
]
