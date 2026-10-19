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
import re
import secrets
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
    RUN_OUTCOMES_FILENAME,
    RUN_REQUESTS_FILENAME,
    ProgramManifest,
    RecordedOutcome,
    RunLinks,
    RunManifest,
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    canonical_json_text,
    case_input_sha256,
    file_index,
    load_program_artifact,
    program_reference,
    read_input_snapshot,
    read_recorded_outcomes,
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
    VALIDATION_ARTICLE_COUNT,
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
    costs = [
        record.reported_cost_usd
        if record.reported_cost_usd is not None
        else record.estimated_cost_usd
        for record in records
    ]
    if any(cost is None for cost in costs):
        return None, "pending" if "pending" in statuses else "none"
    total = sum((cost for cost in costs if cost is not None), Decimal("0"))
    if statuses == {"actual"}:
        return total, "actual"
    if statuses & {"estimated", "actual"}:
        return total, "estimated"
    return None, "none"


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
) -> OptimizationResult:
    """Compile a fresh baseline copy with genuine ``dspy.GEPA``."""
    _require_live_settings(settings)
    _require_confirmation(confirm_budget)
    limits = settings.optimization_budget()
    _validate_budget(limits)
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


__all__ = [
    "BASELINE_CANDIDATE_ID",
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
    "BaselineDevelopmentResult",
    "OptimizationResult",
    "load_train_validation_cases",
    "load_validation_cases",
    "run_baseline_development",
    "run_optimization",
]
