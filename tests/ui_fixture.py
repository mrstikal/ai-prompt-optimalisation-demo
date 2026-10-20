"""Deterministic UI fixture archives for the Streamlit foundation tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import dspy

from dspy_lab.artifacts import (
    BenchmarkProtocol,
    RecordedOutcome,
    RecordedRequest,
    RunLinks,
    RunManifest,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    case_input_sha256,
    program_reference,
    save_program_artifact,
    sha256_file,
    write_input_snapshot,
    write_run_manifest,
)
from dspy_lab.config import Settings, collect_runtime_versions
from dspy_lab.data import load_cases
from dspy_lab.experiments import _task_config_payload
from dspy_lab.metrics import (
    InvalidPrediction,
    aggregate_scores,
    score_case,
)
from dspy_lab.program import EditorialQA, load_editorial_rules
from dspy_lab.schemas import Finding

DATA_DIR = Path(__file__).resolve().parents[1] / "data/v1"
UTC = "2026-10-19T12:00:00Z"
MODEL = "openrouter/deepseek/deepseek-v4.1-flash-1017"


def _files_for(directory: Path, names: list[str]) -> dict[str, str]:
    return {name: sha256_file(directory / name) for name in names}


def _metric_payload(case, findings) -> dict[str, object]:
    score = score_case(case, dspy.Prediction(findings=findings))
    return _metric_payload_from_score(score)


def _invalid_metric_payload(case, kind: str = "parse_error") -> dict[str, object]:
    score = score_case(
        case,
        InvalidPrediction(
            "fixture invalid output",
            kind=kind,  # type: ignore[arg-type]
        ),
    )
    return _metric_payload_from_score(score)


def _metric_payload_from_score(score) -> dict[str, object]:
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


def _corpus_payload(corpus) -> dict[str, object]:
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
        "clean_articles_falsely_flagged": corpus.clean_articles_falsely_flagged,
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


def _request(
    call_id: str,
    sequence: int,
    phase: str,
    messages: list[dict[str, object]],
    response_text: str,
) -> RecordedRequest:
    return RecordedRequest(
        call_id=call_id,
        sequence=sequence,
        phase=phase,
        model=MODEL,
        provider="deepinfra/fp8",
        status="succeeded",
        cost_status="estimated",
        reserved_cost_usd=Decimal("0.01"),
        estimated_cost_usd=Decimal("0.0001"),
        input_token_bound=2000,
        output_token_bound=256,
        input_tokens=100,
        output_tokens=20,
        generation_id=f"gen-{call_id}",
        response_model="deepseek/deepseek-v4.1-flash-1017",
        finish_reason="stop",
        messages=messages,
        request_kwargs={"model": MODEL, "temperature": 0.0},
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
        started_at=datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc),
        finished_at=datetime(
            2026, 10, 19, 12, 0, 2, tzinfo=timezone.utc
        ),
        response_text=response_text,
    )


def _finding(case, index: int) -> Finding:
    gold = case.gold_findings[index]
    return Finding(
        code=gold.code,
        evidence=gold.evidence,
        explanation=gold.explanation,
    )


def _outcome(
    *,
    run_id: str,
    case,
    program_id: str,
    candidate_id: str,
    phase: str,
    split: str,
    findings: list[Finding],
    rules: str,
    request_ids: list[str],
    index: int,
) -> RecordedOutcome:
    return RecordedOutcome(
        run_id=run_id,
        case_id=case.case_id,
        program_id=program_id,
        candidate_id=candidate_id,
        phase=phase,
        split=split,
        status="ok",
        findings=findings,
        metric=_metric_payload(case, findings),
        request_ids=request_ids,
        cost_usd=Decimal("0.0001"),
        cost_status="estimated",
        latency_ms=120.0 + index,
        input_sha256=case_input_sha256(case, rules),
        created_at=UTC,
    )


def _error_outcome(
    *,
    run_id: str,
    case,
    program_id: str,
    candidate_id: str,
    phase: str,
    split: str,
    rules: str,
    request_ids: list[str],
    index: int,
) -> RecordedOutcome:
    return RecordedOutcome(
        run_id=run_id,
        case_id=case.case_id,
        program_id=program_id,
        candidate_id=candidate_id,
        phase=phase,
        split=split,
        status="error",
        findings=None,
        error="parse_error: fixture invalid output",
        metric=_invalid_metric_payload(case),
        request_ids=request_ids,
        cost_usd=Decimal("0.0001"),
        cost_status="estimated",
        latency_ms=160.0 + index,
        input_sha256=case_input_sha256(case, rules),
        created_at=UTC,
    )


def build_ui_archive(
    root: Path,
    *,
    changed: bool = True,
    with_requests: bool = True,
    with_stats: bool = True,
    measured: bool = False,
) -> Path:
    """Create linked development, optimization, and benchmark artifacts."""
    origin = "live" if measured else "mock_test"
    runs = root / "runs"
    programs = root / "programs"
    rules = load_editorial_rules()
    train_cases = load_cases(DATA_DIR / "train.jsonl")[:2]
    validation = load_cases(DATA_DIR / "validation.jsonl")[:5]
    test = load_cases(DATA_DIR / "test.jsonl")[:4]

    baseline_program = EditorialQA()
    baseline_manifest = save_program_artifact(
        baseline_program,
        origin=origin,
        programs_root=programs,
        created_at=UTC,
    )
    baseline_dir = programs / baseline_manifest.program_id
    if changed:
        optimized_program = EditorialQA()
        optimized_program.predict.signature = (
            optimized_program.predict.signature.with_instructions(
                baseline_manifest.instructions
                + "\n\nOptimized rule: quote the narrowest exact evidence span."
            )
        )
        optimized_manifest = save_program_artifact(
            optimized_program,
            origin=origin,
            programs_root=programs,
            parent_program_id=baseline_manifest.program_id,
            created_at=UTC,
        )
        optimized_dir = programs / optimized_manifest.program_id
    else:
        optimized_manifest = baseline_manifest
        optimized_dir = baseline_dir

    development_dir = runs / "development" / "dev-ui"
    development_dir.mkdir(parents=True)
    development_snapshot = write_input_snapshot(
        development_dir,
        cases=validation,
        rules_text=rules,
        split="validation",
    )
    development_predictions: list[list[Finding]] = [
        [], [], [], [], []
    ]
    development_predictions[1] = [
        Finding(
            code="UNWARRANTED_CERTAINTY",
            evidence=validation[1].title,
            explanation="Fixture false positive on a clean article.",
        )
    ]
    development_predictions[2] = [_finding(validation[2], 0)]
    development_requests = []
    development_outcomes = []
    development_scores = []
    for index, case in enumerate(validation):
        call_id = f"req-dev-{index}"
        messages = [
            {
                "role": "system",
                "content": baseline_manifest.instructions,
            },
            {"role": "user", "content": case.article_text},
        ]
        if with_requests:
            development_requests.append(
                _request(
                    call_id,
                    index,
                    "task",
                    messages,
                    json.dumps({"findings": []}),
                )
            )
        findings = development_predictions[index]
        if index == 4:
            development_scores.append(
                score_case(
                    case,
                    InvalidPrediction(
                        "fixture invalid output", kind="parse_error"
                    ),
                )
            )
            development_outcomes.append(
                _error_outcome(
                    run_id="dev-ui",
                    case=case,
                    program_id=baseline_manifest.program_id,
                    candidate_id="baseline",
                    phase="development",
                    split="validation",
                    rules=rules,
                    request_ids=[call_id] if with_requests else [],
                    index=index,
                )
            )
        else:
            development_scores.append(
                score_case(case, dspy.Prediction(findings=findings))
            )
            development_outcomes.append(
                _outcome(
                    run_id="dev-ui",
                    case=case,
                    program_id=baseline_manifest.program_id,
                    candidate_id="baseline",
                    phase="development",
                    split="validation",
                    findings=findings,
                    rules=rules,
                    request_ids=[call_id] if with_requests else [],
                    index=index,
                )
            )
    development_summary = {
        "schema_version": "baseline_summary_v1",
        "run_id": "dev-ui",
        "split": "validation",
        "metric_version": "deterministic_issue_match_v1",
        "baseline_program_id": baseline_manifest.program_id,
        "baseline_instructions": baseline_manifest.instructions,
        "case_count": len(validation),
        "expected_case_count": 10,
        "corpus": _corpus_payload(
            aggregate_scores(development_scores)
        ),
        "budget": {
            "max_calls": 10,
            "max_cost_usd": "0.50",
            "admitted_calls": len(development_requests),
            "delegated_calls": len(development_requests),
            "response_calls": len(development_requests),
            "succeeded_calls": len(development_requests),
            "failed_calls": 0,
            "rejected_calls": 0,
            "cancelled_before_dispatch_calls": 0,
            "pending_cost_calls": 0,
            "estimated_cost_usd": "0.0004",
            "actual_cost_usd": "0.0",
            "committed_cost_usd": "0.0004",
            "unresolved_cost_usd": "0.0",
            "total_cost_usd": "0.0004",
            "remaining_calls": 10 - len(development_requests),
            "remaining_cost_usd": "0.4996",
            "stop_reason": None,
            "cancelled": False,
            "exhausted": False,
            "overrun": False,
        },
        "always_empty_reference": {
            "mean_reward": 0.5,
            "label": "calculated_reference_not_a_model_run",
        },
    }
    atomic_write_json(
        development_dir / "config.json",
        {
            "run_id": "dev-ui",
            "phase": "baseline_development",
            "created_at": UTC,
            "task": {
                "task_model": MODEL,
                "temperature": 0.0,
                "max_output_tokens": 256,
            },
        },
    )
    atomic_write_json(development_dir / "summary.json", development_summary)
    atomic_write_jsonl(
        development_dir / "events.jsonl", [{"event": "fixture"}]
    )
    atomic_write_jsonl(
        development_dir / "outcomes.jsonl",
        [outcome.model_dump(mode="json") for outcome in development_outcomes],
    )
    if with_requests:
        atomic_write_jsonl(
            development_dir / "requests.jsonl",
            [
                request.model_dump(mode="json")
                for request in development_requests
            ],
        )
    development_files = _files_for(
        development_dir,
        [
            "config.json",
            "summary.json",
            "events.jsonl",
            "outcomes.jsonl",
            *(["requests.jsonl"] if with_requests else []),
            development_snapshot.cases_file,
            development_snapshot.rules_file,
        ],
    )
    development_manifest = RunManifest(
        run_id="dev-ui",
        kind="development",
        origin=origin,
        status="completed",
        created_at=UTC,
        finished_at="2026-10-19T12:05:00Z",
        versions=collect_runtime_versions(),
        snapshot=development_snapshot,
        programs=[program_reference(baseline_dir)],
        links=RunLinks(
            baseline_program_id=baseline_manifest.program_id
        ),
        files=development_files,
    )
    write_run_manifest(development_dir, development_manifest)

    optimization_dir = runs / "optimization" / "opt-ui"
    optimization_dir.mkdir(parents=True)
    optimization_snapshot = write_input_snapshot(
        optimization_dir,
        cases=train_cases + validation,
        rules_text=rules,
        split="train_validation",
    )
    optimization_requests = [
        _request(
            "req-opt-task",
            0,
            "task",
            [{"role": "user", "content": "task"}],
            "{}",
        ),
        _request(
            "req-opt-reflect",
            1,
            "reflection",
            [
                {"role": "system", "content": "reflection"},
                {"role": "user", "content": "propose"},
            ],
            json.dumps({"new_instruction": "Optimized rule"}),
        ),
    ]
    candidates = [
        baseline_manifest.instructions,
        optimized_manifest.instructions,
    ]
    candidate_stats = {
        "schema_version": "gepa_stats_v1",
        "status": "completed",
        "run_id": "opt-ui",
        "baseline_program_id": baseline_manifest.program_id,
        "selected_program_id": optimized_manifest.program_id,
        "selection": {
            "candidate_count": 2,
            "search_status": "completed",
        },
        "detailed_results": {
            "candidates": [{"predict": item} for item in candidates],
            "parents": [[], [0]],
            "val_aggregate_scores": [0.5, 0.75],
            "discovery_eval_counts": [0, 10],
            "best_idx": 1 if changed else 0,
        },
    }
    atomic_write_json(
        optimization_dir / "config.json",
        {
            "run_id": "opt-ui",
            "phase": "optimization",
            "created_at": UTC,
            "task": _task_config_payload(
                Settings(
                    mode="live",
                    openrouter_api_key="sk-or-v1-fixture",
                )
            ),
        },
    )
    atomic_write_jsonl(
        optimization_dir / "events.jsonl", [{"event": "fixture"}]
    )
    if with_requests:
        atomic_write_jsonl(
            optimization_dir / "requests.jsonl",
            [
                request.model_dump(mode="json")
                for request in optimization_requests
            ],
        )
    if with_stats:
        atomic_write_json(
            optimization_dir / "gepa_stats.json", candidate_stats
        )
    atomic_write_jsonl(
        optimization_dir / "candidate_instructions.jsonl",
        [
            {
                "candidate_id": f"candidate-{index}",
                "candidate_index": index,
                "instructions": {"predict": instruction},
                "parents": [] if index == 0 else [0],
                "val_aggregate_score": 0.5 if index == 0 else 0.75,
                "discovery_eval_count": 0 if index == 0 else 10,
                "selected": (
                    index == 0 if not changed else index == 1
                ),
            }
            for index, instruction in enumerate(candidates)
        ],
    )
    optimization_files = _files_for(
        optimization_dir,
        [
            "config.json",
            "events.jsonl",
            *(["requests.jsonl"] if with_requests else []),
            *(["gepa_stats.json"] if with_stats else []),
            "candidate_instructions.jsonl",
            optimization_snapshot.cases_file,
            optimization_snapshot.rules_file,
        ],
    )
    optimization_refs = [program_reference(baseline_dir)]
    if optimized_manifest.program_id != baseline_manifest.program_id:
        optimization_refs.append(program_reference(optimized_dir))
    optimization_manifest = RunManifest(
        run_id="opt-ui",
        kind="optimization",
        origin=origin,
        status="completed",
        created_at=UTC,
        finished_at="2026-10-19T12:10:00Z",
        versions=collect_runtime_versions(),
        snapshot=optimization_snapshot,
        programs=optimization_refs,
        links=RunLinks(
            baseline_development_run_id="dev-ui",
            baseline_program_id=baseline_manifest.program_id,
            optimized_program_id=optimized_manifest.program_id,
        ),
        files=optimization_files,
    )
    write_run_manifest(optimization_dir, optimization_manifest)

    benchmark_dir = runs / "benchmarks" / "bench-ui"
    benchmark_dir.mkdir(parents=True)
    benchmark_snapshot = write_input_snapshot(
        benchmark_dir,
        cases=test,
        rules_text=rules,
        split="test",
    )
    baseline_predictions: list[list[Finding]] = [[], [], [], []]
    optimized_predictions: list[list[Finding]] = [[], [], [], []]
    baseline_predictions[1] = [
        Finding(
            code="UNWARRANTED_CERTAINTY",
            evidence=test[1].title,
            explanation="Fixture false positive.",
        )
    ]
    optimized_predictions[2] = [_finding(test[2], 0)]
    baseline_predictions[3] = [_finding(test[3], 0)]
    benchmark_requests = []
    benchmark_outcomes = []
    arm_scores: dict[str, list] = {"baseline": [], "optimized": []}
    for index, case in enumerate(test):
        for arm, predictions in (
            ("baseline", baseline_predictions),
            ("optimized", optimized_predictions),
        ):
            call_id = f"req-{arm}-{index}"
            instructions = (
                baseline_manifest.instructions
                if arm == "baseline"
                else optimized_manifest.instructions
            )
            messages = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": case.article_text},
            ]
            if with_requests:
                benchmark_requests.append(
                    _request(
                        call_id,
                        index,
                        "task",
                        messages,
                        json.dumps({"findings": []}),
                    )
                )
            findings = predictions[index]
            arm_scores[arm].append(
                score_case(case, dspy.Prediction(findings=findings))
            )
            benchmark_outcomes.append(
                _outcome(
                    run_id="bench-ui",
                    case=case,
                    program_id=(
                        baseline_manifest.program_id
                        if arm == "baseline"
                        else optimized_manifest.program_id
                    ),
                    candidate_id=arm,
                    phase=arm,
                    split="test",
                    findings=findings,
                    rules=rules,
                    request_ids=[call_id] if with_requests else [],
                    index=index,
                )
            )
    benchmark_summary = {
        "schema_version": "benchmark_summary_v1",
        "run_id": "bench-ui",
        "verdict": "mixed_tradeoff",
        "verdict_reason": "Fixture archive for UI inspection only.",
        "complete": True,
        "metrics": {
            "baseline": _corpus_payload(
                aggregate_scores(arm_scores["baseline"])
            ),
            "optimized": _corpus_payload(
                aggregate_scores(arm_scores["optimized"])
            ),
            "delta": {},
        },
        "costs": {
            "currency": "USD",
            "optimization_cost_usd": "0.0123",
            "optimization_cost_status": "estimated",
            "baseline_development_cost_usd": "0.0004",
            "baseline_development_cost_status": "estimated",
            "benchmark_inference_known_cost_usd": "0.0008",
            "benchmark_inference_cost_status": "estimated",
            "one_time_known_cost_usd": "0.0127",
            "one_time_cost_status": "estimated",
        },
        "warnings": [],
    }
    atomic_write_json(
        benchmark_dir / "config.json",
        {
            "run_id": "bench-ui",
            "phase": "benchmark",
            "created_at": UTC,
        },
    )
    atomic_write_json(benchmark_dir / "summary.json", benchmark_summary)
    atomic_write_jsonl(
        benchmark_dir / "events.jsonl", [{"event": "fixture"}]
    )
    atomic_write_jsonl(
        benchmark_dir / "outcomes.jsonl",
        [outcome.model_dump(mode="json") for outcome in benchmark_outcomes],
    )
    if with_requests:
        atomic_write_jsonl(
            benchmark_dir / "requests.jsonl",
            [
                request.model_dump(mode="json")
                for request in benchmark_requests
            ],
        )
    atomic_write_json(
        benchmark_dir / "benchmark_protocol.json",
        {
            "schema_version": "benchmark_plan_v1",
            "metric_version": "deterministic_issue_match_v1",
            "protocol_sha256": "a" * 64,
            "task_config_sha256": "b" * 64,
            "selected_program_ids": [
                baseline_manifest.program_id,
                optimized_manifest.program_id,
            ],
            "case_count": len(test),
            "seed": 42,
            "pre_dispatch_comparability_warnings": [],
        },
    )
    atomic_write_json(
        benchmark_dir / "test_reuse_history.json",
        {
            "schema_version": "test_reuse_history_v1",
            "test_cases_sha256": benchmark_snapshot.cases_sha256,
            "rules_sha256": benchmark_snapshot.rules_sha256,
            "origin": origin,
            "exposure_index": 0,
            "provenance": "first_exposure",
            "count_rule": (
                "same origin, dataset version, and rules; same_test_inputs "
                "flags identical ordered case hashes"
            ),
            "prior_runs": [],
            "warnings": [],
        },
    )
    benchmark_files = _files_for(
        benchmark_dir,
        [
            "config.json",
            "summary.json",
            "events.jsonl",
            "outcomes.jsonl",
            *(["requests.jsonl"] if with_requests else []),
            "benchmark_protocol.json",
            "test_reuse_history.json",
            benchmark_snapshot.cases_file,
            benchmark_snapshot.rules_file,
        ],
    )
    benchmark_manifest = RunManifest(
        run_id="bench-ui",
        kind="benchmark",
        origin=origin,
        status="completed",
        created_at=UTC,
        finished_at="2026-10-19T12:20:00Z",
        versions=collect_runtime_versions(),
        snapshot=benchmark_snapshot,
        programs=optimization_refs,
        links=RunLinks(
            baseline_development_run_id="dev-ui",
            optimization_run_id="opt-ui",
            baseline_program_id=baseline_manifest.program_id,
            optimized_program_id=optimized_manifest.program_id,
        ),
        benchmark=BenchmarkProtocol(
            exposure_index=0,
            paired_order=["baseline", "optimized"],
            case_orders={
                case.case_id: ["baseline", "optimized"]
                for case in test
            },
            seed=42,
            protocol_sha256="a" * 64,
            released=measured,
        ),
        files=benchmark_files,
    )
    write_run_manifest(benchmark_dir, benchmark_manifest)
    return benchmark_dir
