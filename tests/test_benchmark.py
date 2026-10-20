from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import dspy
import dspy.clients.lm as dspy_lm
import pytest

import dspy_lab.experiments as experiments
from dspy_lab.artifacts import (
    RUN_REQUESTS_FILENAME,
    ArtifactIncompleteError,
    open_replay_archive,
    read_recorded_requests,
    read_run_manifest,
)
from dspy_lab.config import Settings
from dspy_lab.data import file_sha256, load_cases
from dspy_lab.llm import BudgetLedger
from dspy_lab.experiments import (
    BENCHMARK_PROTOCOL_FILENAME,
    BENCHMARK_SUMMARY_FILENAME,
    ArmMeasurement,
    PairedMeasurement,
    compute_break_even,
    derive_paired_orders,
    latency_summary,
    load_sealed_benchmark_cases,
    run_baseline_development,
    run_benchmark,
    run_optimization,
    summarize_benchmark,
)
from dspy_lab.metrics import InvalidPrediction, score_case
from dspy_lab.schemas import RULES_VERSION, Finding

DATA_DIR = Path(__file__).resolve().parents[1] / "data/v1"
SECRET = "sk-or-v1-s11-benchmark-sentinel"
TASK_MODEL = "openrouter/deepseek/deepseek-v4.1-flash-1017"
REFLECTION_MODEL = "openrouter/openai/gpt-6-sol"
DEV_IDS = ("val-001", "val-002")
TRAIN_IDS = ("train-001", "train-006")
TEST_IDS = ("val-003", "val-004")


class Reply:
    def __init__(self, content: str, cost: float) -> None:
        self.content = content
        self.cost = cost


def response_for(reply: Reply, request: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id="gen-s11",
        model=request.get("model"),
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason="stop",
                message=SimpleNamespace(
                    content=reply.content,
                    tool_calls=None,
                    reasoning_content=None,
                ),
                logprobs=None,
            )
        ],
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        _hidden_params={"response_cost": reply.cost},
    )


def install_transport(monkeypatch, handler):
    calls: list[dict] = []

    def fake_completion(*, request, num_retries=0):
        calls.append(request)
        outcome = handler(request, len(calls))
        if isinstance(outcome, BaseException):
            raise outcome
        content, cost = outcome
        return response_for(Reply(content, cost), request)

    monkeypatch.setattr(dspy_lm, "litellm_completion", fake_completion)
    return calls


def make_settings(**overrides) -> Settings:
    values = {
        "mode": "live",
        "openrouter_api_key": SECRET,
        "max_baseline_dev_calls": 10,
        "max_baseline_dev_cost_usd": Decimal("0.50"),
        "max_optimization_calls": 160,
        "max_optimization_metric_calls": 10,
        "max_optimization_cost_usd": Decimal("5.00"),
        "max_benchmark_calls": 20,
        "max_benchmark_cost_usd": Decimal("0.50"),
        "request_timeout_seconds": 5.0,
    }
    values.update(overrides)
    return Settings(**values)


def clean(case):
    return case.model_copy(update={"gold_findings": []})


def fixture_cases():
    train_all = {case.case_id: case for case in load_cases(DATA_DIR / "train.jsonl")}
    validation_all = {
        case.case_id: case for case in load_cases(DATA_DIR / "validation.jsonl")
    }
    train = [clean(train_all[case_id]) for case_id in TRAIN_IDS]
    development = [clean(validation_all[case_id]) for case_id in DEV_IDS]
    test = [validation_all[case_id] for case_id in TEST_IDS]
    return train, development, test


def hidden_findings(case) -> str:
    return json.dumps(
        {
            "findings": [
                {
                    "code": finding.code,
                    "evidence": finding.evidence,
                    "explanation": finding.explanation,
                }
                for finding in case.gold_findings
            ]
        }
    )


def optimization_handler(request: dict, _index: int):
    model = str(request.get("model"))
    if model == REFLECTION_MODEL:
        return json.dumps(
            {"new_instruction": "MARKER improved editorial instructions"}
        ), 0.00001
    payload = json.dumps(
        request.get("messages") or request.get("prompt"),
        ensure_ascii=False,
    )
    if "MARKER" in payload:
        return '{"findings": []}', 0.00001
    return "not json", 0.00002


def make_saved_pair(monkeypatch, tmp_path, test_cases):
    train, development, _test = fixture_cases()
    settings = make_settings()
    install_transport(monkeypatch, optimization_handler)
    development_run = run_baseline_development(
        settings,
        development,
        confirm_budget=True,
        run_id="dev-s11",
        origin="mock_test",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )
    optimization_run = run_optimization(
        settings,
        train,
        development,
        baseline_development_run_id="dev-s11",
        confirm_budget=True,
        run_id="opt-s11",
        origin="mock_test",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )
    assert optimization_run.manifest.status == "completed"
    assert optimization_run.selected_program.program_id != (
        optimization_run.baseline_program.program_id
    )
    return development_run, optimization_run, settings


def benchmark_handler(test_cases, *, baseline_invalid: bool = False):
    by_title = {case.title: case for case in test_cases}

    def handler(request: dict, _index: int):
        model = str(request.get("model"))
        if model == REFLECTION_MODEL:
            return json.dumps({"new_instruction": "MARKER"}), 0.00001
        text = json.dumps(
            request.get("messages") or request.get("prompt"),
            ensure_ascii=False,
        )
        if "MARKER" in text:
            for title, case in by_title.items():
                if title in text:
                    return hidden_findings(case), 0.00001
            return '{"findings": []}', 0.00001
        if baseline_invalid:
            return "not json", 0.00002
        return '{"findings": []}', 0.00002

    return handler


def find_mixed_order_seed(cases) -> int:
    for seed in range(200):
        orders = derive_paired_orders(cases, seed)
        first = orders[cases[0].case_id][0]
        second = orders[cases[1].case_id][0]
        if first == "baseline" and second == "optimized":
            return seed
    raise AssertionError("no mixed-order seed found")


def arm(score, *, cost="0.004", status="estimated", latency=100.0, percent=90.0):
    return ArmMeasurement(
        case_id=score.case_id,
        score=score,
        valid=score.valid,
        cost_usd=Decimal(cost),
        cost_status=status,
        latency_ms=latency,
        evaluate_percent=percent,
        input_sha256="a" * 64,
    )


def hand_scores():
    validation = {
        case.case_id: case
        for case in load_cases(DATA_DIR / "validation.jsonl")
    }
    one = validation["val-003"]
    two = validation["val-007"]

    def prediction(case, indexes):
        return dspy.Prediction(
            findings=[
                Finding(
                    code=case.gold_findings[index].code,
                    evidence=case.gold_findings[index].evidence,
                    explanation=case.gold_findings[index].explanation,
                )
                for index in indexes
            ]
        )

    baseline_one = score_case(one, prediction(one, [0]))
    baseline_two = score_case(two, prediction(two, [0]))
    optimized_one = score_case(one, prediction(one, [0]))
    optimized_two = score_case(two, prediction(two, [0, 1]))
    return one, two, baseline_one, baseline_two, optimized_one, optimized_two


def hand_pairs():
    _one, _two, b1, b2, o1, o2 = hand_scores()
    return [
        PairedMeasurement(
            "val-003",
            arm(b1, cost="0.004", latency=100.0, percent=90.0),
            arm(o1, cost="0.003", latency=90.0, percent=90.0),
        ),
        PairedMeasurement(
            "val-007",
            arm(b2, cost="0.004", latency=200.0, percent=80.0),
            arm(o2, cost="0.003", latency=180.0, percent=100.0),
        ),
    ]


def summarize_hand(
    pairs,
    *,
    complete=True,
    run_complete=True,
    cost_status="estimated",
):
    return summarize_benchmark(
        pairs,
        planned_case_count=2,
        run_complete=run_complete,
        comparable=True,
        comparability_warnings=[],
        optimization_cost_usd=Decimal("0.01"),
        optimization_cost_status=cost_status,
        baseline_development_cost_usd=Decimal("0.02"),
        baseline_development_cost_status=cost_status,
        protocol_sha256="b" * 64,
        task_config_sha256="c" * 64,
        test_reuse={"exposure_index": 0, "provenance": "first_blind"},
    )


def test_paired_orders_can_cover_ab_and_ba() -> None:
    _train, _development, test = fixture_cases()
    seed = find_mixed_order_seed(test)
    orders = derive_paired_orders(test, seed)
    assert orders[test[0].case_id][0] == "baseline"
    assert orders[test[1].case_id][0] == "optimized"
    assert all(
        set(order) == {"baseline", "optimized"} for order in orders.values()
    )


def test_latency_summary_uses_milliseconds_and_exploratory_p95() -> None:
    summary = latency_summary([1000.0, 3000.0, 2000.0, None])
    assert summary["unit"] == "ms"
    assert summary["count"] == 3
    assert summary["median"] == 2000.0
    assert summary["p95"] == 3000.0
    assert summary["p95_label"] == "exploratory"
    assert latency_summary([])["median"] is None


def test_case_cost_retains_known_sibling_costs() -> None:
    known = SimpleNamespace(
        cost_status="estimated",
        reported_cost_usd=Decimal("0.003"),
        estimated_cost_usd=Decimal("0.003"),
    )
    no_dispatch = SimpleNamespace(
        cost_status="none",
        reported_cost_usd=None,
        estimated_cost_usd=None,
    )
    cost, status = experiments._case_cost([known, no_dispatch])
    assert cost == Decimal("0.003")
    assert status == "estimated"


def test_break_even_positive_zero_negative_and_gates() -> None:
    positive = compute_break_even(
        optimization_cost_usd=Decimal("0.01"),
        optimization_cost_status="estimated",
        baseline_cost_per_article_usd=Decimal("0.01"),
        baseline_cost_status="estimated",
        optimized_cost_per_article_usd=Decimal("0.008"),
        optimized_cost_status="estimated",
        quality_gate=True,
        complete=True,
    )
    assert positive["status"] == "finite"
    assert positive["break_even_articles"] == 5
    assert positive["savings_direction"] == "positive"
    assert positive["costs_estimated"] is True
    assert positive["currency"] == "USD"

    zero = compute_break_even(
        optimization_cost_usd=Decimal("0.01"),
        optimization_cost_status="actual",
        baseline_cost_per_article_usd=Decimal("0.01"),
        baseline_cost_status="actual",
        optimized_cost_per_article_usd=Decimal("0.01"),
        optimized_cost_status="actual",
        quality_gate=True,
        complete=True,
    )
    assert zero["status"] == "no_finite_break_even"
    assert zero["savings_direction"] == "zero"

    negative = compute_break_even(
        optimization_cost_usd=Decimal("0.01"),
        optimization_cost_status="actual",
        baseline_cost_per_article_usd=Decimal("0.01"),
        baseline_cost_status="actual",
        optimized_cost_per_article_usd=Decimal("0.012"),
        optimized_cost_status="actual",
        quality_gate=True,
        complete=True,
    )
    assert negative["status"] == "no_finite_break_even"
    assert negative["savings_direction"] == "negative"

    gate_failed = compute_break_even(
        optimization_cost_usd=Decimal("0.01"),
        optimization_cost_status="actual",
        baseline_cost_per_article_usd=Decimal("0.01"),
        baseline_cost_status="actual",
        optimized_cost_per_article_usd=Decimal("0.008"),
        optimized_cost_status="actual",
        quality_gate=False,
        complete=True,
    )
    assert gate_failed["status"] == "not_applicable"
    assert gate_failed["reason"] == "quality_non_regression_gate_not_met"

    pending = compute_break_even(
        optimization_cost_usd=None,
        optimization_cost_status="pending",
        baseline_cost_per_article_usd=Decimal("0.01"),
        baseline_cost_status="pending",
        optimized_cost_per_article_usd=Decimal("0.008"),
        optimized_cost_status="estimated",
        quality_gate=True,
        complete=True,
    )
    assert pending["status"] == "not_applicable"
    assert pending["reason"] == "cost_status_not_settled"


def test_paired_summary_is_hand_checkable_and_separates_micro_from_mean() -> None:
    summary = summarize_hand(hand_pairs())
    baseline = summary["metrics"]["baseline"]
    optimized = summary["metrics"]["optimized"]
    assert baseline["tp"] == 2 and baseline["fn"] == 1
    assert baseline["micro_f1"] == pytest.approx(0.8)
    assert baseline["mean_reward"] == pytest.approx(0.8333333333)
    assert baseline["evaluate_percent_mean"] == pytest.approx(85.0)
    assert baseline["micro_f1"] != baseline["mean_reward"]
    assert optimized["micro_f1"] == pytest.approx(1.0)
    assert optimized["mean_reward"] == pytest.approx(1.0)
    assert summary["exact_match_counts"] == {
        "baseline": 1,
        "optimized": 2,
        "total": 2,
    }
    assert summary["corrected_cases"] == ["val-007"]
    assert summary["regressed_cases"] == []
    assert summary["verdict"] == "observed_heldout_improvement"
    assert summary["eligible_for_verdict"] is True
    assert summary["quality_non_regression_gate"] is True
    assert summary["economics"]["status"] == "finite"
    assert summary["economics"]["break_even_articles"] == 10
    assert summary["costs"]["benchmark_inference_cost_status"] == "estimated"
    assert summary["latency_delta"]["median_ms"] == pytest.approx(-15.0)


def test_paired_summary_reports_category_regressions() -> None:
    one, two, b1, _b2, o1, _o2 = hand_scores()
    baseline_two = score_case(
        two,
        dspy.Prediction(
            findings=[
                Finding(
                    code=finding.code,
                    evidence=finding.evidence,
                    explanation=finding.explanation,
                )
                for finding in two.gold_findings
            ]
        ),
    )
    optimized_two = score_case(
        two,
        dspy.Prediction(
            findings=[
                Finding(
                    code=two.gold_findings[0].code,
                    evidence=two.gold_findings[0].evidence,
                    explanation=two.gold_findings[0].explanation,
                )
            ]
        ),
    )
    summary = summarize_hand(
        [
            PairedMeasurement("val-003", arm(b1), arm(o1)),
            PairedMeasurement(
                "val-007",
                arm(baseline_two),
                arm(optimized_two),
            ),
        ]
    )
    assert summary["verdict"] == "regression"
    assert summary["category_regressions"]
    assert summary["category_regressions"][0]["code"] == (
        two.gold_findings[1].code
    )
    assert summary["regressed_cases"] == ["val-007"]


def test_paired_summary_preserves_actual_cost_status() -> None:
    _one, _two, b1, b2, o1, o2 = hand_scores()
    actual_pairs = [
        PairedMeasurement(
            "val-003",
            arm(b1, cost="0.004", status="actual"),
            arm(o1, cost="0.003", status="actual"),
        ),
        PairedMeasurement(
            "val-007",
            arm(b2, cost="0.004", status="actual"),
            arm(o2, cost="0.003", status="actual"),
        ),
    ]
    summary = summarize_hand(actual_pairs, cost_status="actual")
    assert summary["costs"]["baseline"]["cost_status"] == "actual"
    assert summary["costs"]["optimized"]["cost_status"] == "actual"
    assert summary["costs"]["benchmark_inference_cost_status"] == "actual"
    assert summary["economics"]["costs_estimated"] is False
    assert summary["economics"]["status"] == "finite"


def test_paired_summary_invalid_and_incomplete_gates() -> None:
    one, two, b1, b2, o1, _o2 = hand_scores()
    invalid_two = score_case(two, InvalidPrediction("bad", "parse_error"))
    invalid_pairs = [
        PairedMeasurement("val-003", arm(b1), arm(o1)),
        PairedMeasurement(
            "val-007",
            arm(b2),
            ArmMeasurement(
                case_id=two.case_id,
                score=invalid_two,
                valid=False,
                cost_usd=None,
                cost_status="pending",
                latency_ms=None,
                evaluate_percent=0.0,
                input_sha256="a" * 64,
            ),
        ),
    ]
    invalid = summarize_hand(invalid_pairs)
    assert invalid["complete"] is True
    assert invalid["complete_with_errors"] is True
    assert invalid["validity"]["both_arms_full_validity"] is False
    assert invalid["verdict"] == "validity_problem"
    assert invalid["validity"]["optimized_invalid_cases"] == ["val-007"]
    assert invalid["economics"]["status"] == "not_applicable"

    incomplete = summarize_hand(
        [
            PairedMeasurement(
                "val-003", arm(b1), arm(o1)
            )
        ],
        run_complete=False,
    )
    assert incomplete["complete"] is False
    assert incomplete["attempted_arms"] == 2
    assert incomplete["planned_arms"] == 4
    assert incomplete["verdict"] == "incomplete"
    assert incomplete["economics"]["status"] == "not_applicable"


def test_loader_verifies_a_synthetic_sealed_test_fixture(tmp_path) -> None:
    source = load_cases(DATA_DIR / "validation.jsonl")[:2]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    test_path = data_dir / "test.jsonl"
    test_path.write_text(
        "".join(
            json.dumps(
                case.model_dump(mode="python"),
                ensure_ascii=False,
                default=lambda value: (
                    float(value)
                    if isinstance(value, Decimal)
                    else str(value)
                ),
            )
            + "\n"
            for case in source
        ),
        encoding="utf-8",
    )
    manifest = {
        "test_sealed": True,
        "rules_version": RULES_VERSION,
        "splits": {
            "test": {
                "file_sha256": file_sha256(test_path),
                "case_ids": [case.case_id for case in source],
                "family_ids": [case.family_id for case in source],
            }
        },
    }
    (data_dir / "split_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    loaded = load_sealed_benchmark_cases(data_dir)
    assert [case.case_id for case in loaded] == [
        case.case_id for case in source
    ]
    test_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(Exception, match="hash"):
        load_sealed_benchmark_cases(data_dir)


def test_benchmark_rejects_mismatched_or_overlapping_inputs(
    monkeypatch, tmp_path
) -> None:
    _train, _development, test = fixture_cases()
    development_run, optimization_run, settings = make_saved_pair(
        monkeypatch, tmp_path, test
    )
    install_transport(monkeypatch, benchmark_handler(test))
    with pytest.raises(experiments.ExperimentNotAuthorized):
        run_benchmark(
            settings,
            test,
            baseline_development_run_id=development_run.run_id,
            optimization_run_id=optimization_run.run_id,
            confirm_budget=False,
            origin="mock_test",
            runs_root=tmp_path / "runs",
            programs_root=tmp_path / "programs",
        )
    with pytest.raises(experiments.ExperimentConfigurationError):
        run_benchmark(
            settings,
            [test[0], fixture_cases()[1][0]],
            baseline_development_run_id=development_run.run_id,
            optimization_run_id=optimization_run.run_id,
            confirm_budget=True,
            origin="mock_test",
            run_id="bench-overlap",
            runs_root=tmp_path / "runs",
            programs_root=tmp_path / "programs",
        )
    with pytest.raises(experiments.ExperimentIncompleteError, match="protocol"):
        run_benchmark(
            make_settings(temperature=0.5),
            test,
            baseline_development_run_id=development_run.run_id,
            optimization_run_id=optimization_run.run_id,
            confirm_budget=True,
            origin="mock_test",
            run_id="bench-protocol",
            runs_root=tmp_path / "runs",
            programs_root=tmp_path / "programs",
        )
    install_transport(monkeypatch, optimization_handler)
    other_development = run_baseline_development(
        settings,
        fixture_cases()[1],
        confirm_budget=True,
        run_id="dev-other",
        origin="mock_test",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )
    with pytest.raises(experiments.ExperimentIncompleteError, match="link"):
        run_benchmark(
            settings,
            test,
            baseline_development_run_id=other_development.run_id,
            optimization_run_id=optimization_run.run_id,
            confirm_budget=True,
            origin="mock_test",
            run_id="bench-links",
            runs_root=tmp_path / "runs",
            programs_root=tmp_path / "programs",
        )


def test_benchmark_runs_saved_programs_without_compiling(
    monkeypatch, tmp_path
) -> None:
    _train, _development, test = fixture_cases()
    development_run, optimization_run, settings = make_saved_pair(
        monkeypatch, tmp_path, test
    )
    install_transport(monkeypatch, benchmark_handler(test))
    seed = find_mixed_order_seed(test)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("benchmark must never compile")

    monkeypatch.setattr(experiments, "_build_gepa_optimizer", forbidden)
    monkeypatch.setattr(experiments, "save_program_artifact", forbidden)
    result = run_benchmark(
        settings,
        test,
        baseline_development_run_id=development_run.run_id,
        optimization_run_id=optimization_run.run_id,
        confirm_budget=True,
        seed=seed,
        origin="mock_test",
        run_id="bench-s11",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )

    assert result.manifest.status == "completed"
    assert result.manifest.kind == "benchmark"
    assert result.manifest.origin == "mock_test"
    assert result.manifest.benchmark.released is False
    assert result.manifest.benchmark.exposure_index == 0
    assert result.summary["complete"] is True
    assert result.summary["validity"]["both_arms_full_validity"] is True
    assert result.summary["verdict"] == "observed_heldout_improvement"
    assert result.summary["metrics"]["baseline"]["micro_f1"] == pytest.approx(0.0)
    assert result.summary["metrics"]["optimized"]["micro_f1"] == pytest.approx(1.0)
    assert result.summary["exact_match_counts"]["baseline"] == 0
    assert result.summary["exact_match_counts"]["optimized"] == 2
    assert result.summary["test_reuse"]["exposure_index"] == 0
    assert result.summary["test_reuse"]["provenance"] == "first_blind"
    assert result.summary["economics"]["status"] == "finite"
    assert result.summary["economics"]["break_even_articles"] > 0
    assert result.summary["costs"]["benchmark_inference_cost_status"] == "estimated"
    assert result.summary["costs"]["currency"] == "USD"
    breakdown = result.summary["costs"]["optimization_breakdown"]
    assert Decimal(breakdown["task_train_cost_usd"]) > 0
    assert Decimal(breakdown["task_validation_cost_usd"]) > 0
    assert Decimal(breakdown["reflection_cost_usd"]) > 0
    assert breakdown["task_train_validation_split"] == "classified"
    assert result.summary["metrics"]["baseline"]["tokens"][
        "input_tokens_known"
    ] == 20
    assert result.summary["metrics"]["baseline"]["tokens"][
        "output_tokens_known"
    ] == 10
    assert result.summary["metrics"]["baseline"]["provider_latency"][
        "unit"
    ] == "ms"
    assert result.summary["metrics"]["baseline"]["provider_latency"][
        "count"
    ] == 2
    assert (
        result.summary["metrics"]["baseline"]["provider_latency"]["median"]
        <= result.summary["metrics"]["baseline"]["latency"]["median"]
    )
    assert len(result.outcomes) == 4
    assert len(result.requests) == 4
    assert {outcome.phase for outcome in result.outcomes} == {
        "baseline",
        "optimized",
    }
    assert all(outcome.status == "ok" for outcome in result.outcomes)
    by_case = {}
    for outcome in result.outcomes:
        by_case.setdefault(outcome.case_id, {})[outcome.phase] = outcome
    for outcomes in by_case.values():
        assert outcomes["baseline"].input_sha256 == (
            outcomes["optimized"].input_sha256
        )
    orders = result.manifest.benchmark.case_orders
    assert orders[test[0].case_id][0] == "baseline"
    assert orders[test[1].case_id][0] == "optimized"
    requests = read_recorded_requests(
        result.run_dir / RUN_REQUESTS_FILENAME
    )
    assert len(requests) == 4
    for case_id, order in orders.items():
        first_arm = order[0]
        outcome = by_case[case_id][first_arm]
        assert requests[outcome.request_ids[0]].sequence < requests[
            by_case[case_id][order[1]].request_ids[0]
        ].sequence
    assert SECRET not in (result.run_dir / "config.json").read_text(
        encoding="utf-8"
    )
    assert all(
        request.model == TASK_MODEL for request in requests.values()
    )
    for request in requests.values():
        payload = json.dumps(request.messages, ensure_ascii=False)
        assert "gold_findings" not in payload
        assert "test-" not in payload
    protocol = json.loads(
        (result.run_dir / BENCHMARK_PROTOCOL_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert protocol["protocol_sha256"] == (
        result.manifest.benchmark.protocol_sha256
    )
    summary = json.loads(
        (result.run_dir / BENCHMARK_SUMMARY_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert summary["verdict"] == "observed_heldout_improvement"
    verify = read_run_manifest(result.run_dir)
    assert verify.status == "completed"
    archive = open_replay_archive(
        result.run_dir,
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )
    assert archive.is_measured is False
    assert len(archive.outcomes) == 4
    assert any("Mock/test archive" in warning for warning in archive.warnings)
    assert any("not released" in warning for warning in archive.warnings)


def test_benchmark_retains_errors_and_reuse_history(
    monkeypatch, tmp_path
) -> None:
    _train, _development, test = fixture_cases()
    development_run, optimization_run, settings = make_saved_pair(
        monkeypatch, tmp_path, test
    )
    install_transport(
        monkeypatch, benchmark_handler(test, baseline_invalid=True)
    )
    seed = find_mixed_order_seed(test)
    first = run_benchmark(
        settings,
        test,
        baseline_development_run_id=development_run.run_id,
        optimization_run_id=optimization_run.run_id,
        confirm_budget=True,
        seed=seed,
        origin="mock_test",
        run_id="bench-errors",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )
    assert first.manifest.status == "completed"
    assert first.summary["verdict"] == "validity_problem"
    assert first.summary["complete_with_errors"] is True
    assert first.summary["validity"]["baseline_invalid_cases"] == [
        test[0].case_id,
        test[1].case_id,
    ]
    assert all(
        outcome.status == "error"
        for outcome in first.outcomes
        if outcome.phase == "baseline"
    )
    requests = read_recorded_requests(
        first.run_dir / RUN_REQUESTS_FILENAME
    )
    assert any(
        request.response_text == "not json"
        for request in requests.values()
    )
    assert all(
        outcome.request_ids for outcome in first.outcomes
    )

    second = run_benchmark(
        settings,
        test,
        baseline_development_run_id=development_run.run_id,
        optimization_run_id=optimization_run.run_id,
        confirm_budget=True,
        seed=seed,
        origin="mock_test",
        run_id="bench-reuse",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )
    assert second.manifest.benchmark.exposure_index == 1
    assert second.summary["test_reuse"]["provenance"] == (
        "exploratory_reevaluation"
    )
    assert second.summary["test_reuse"]["prior_runs"][0]["run_id"] == (
        "bench-errors"
    )
    assert second.manifest.benchmark.protocol_sha256 == (
        first.manifest.benchmark.protocol_sha256
    )
    assert any(
        "exposure index 1" in warning
        for warning in second.manifest.warnings
    )
    history = json.loads(
        (second.run_dir / "test_reuse_history.json").read_text(
            encoding="utf-8"
        )
    )
    assert history["exposure_index"] == 1
    assert history["prior_runs"][0]["run_id"] == "bench-errors"


def test_interrupted_benchmark_still_counts_as_exposure(
    monkeypatch, tmp_path
) -> None:
    _train, _development, test = fixture_cases()
    development_run, optimization_run, settings = make_saved_pair(
        monkeypatch, tmp_path, test
    )
    orphan = tmp_path / "runs" / "benchmarks" / "bench-interrupted"
    orphan.mkdir(parents=True)
    (orphan / "config.json").write_text(
        json.dumps(
            {
                "run_id": "bench-interrupted",
                "origin": "mock_test",
                "dataset_version": "v1",
                "created_at": "2026-10-19T20:00:00Z",
                "case_ids": [case.case_id for case in test],
            }
        ),
        encoding="utf-8",
    )
    install_transport(monkeypatch, benchmark_handler(test))
    result = run_benchmark(
        settings,
        test,
        baseline_development_run_id=development_run.run_id,
        optimization_run_id=optimization_run.run_id,
        confirm_budget=True,
        seed=find_mixed_order_seed(test),
        origin="mock_test",
        run_id="bench-after-interrupt",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )
    assert result.manifest.status == "completed"
    assert result.manifest.benchmark.exposure_index == 1
    assert result.summary["test_reuse"]["prior_runs"][0]["run_id"] == (
        "bench-interrupted"
    )
    assert result.summary["test_reuse"]["prior_runs"][0]["status"] == (
        "interrupted"
    )


def test_benchmark_budget_stop_is_incomplete(monkeypatch, tmp_path) -> None:
    _train, _development, test = fixture_cases()
    development_run, optimization_run, settings = make_saved_pair(
        monkeypatch, tmp_path, test
    )
    install_transport(monkeypatch, benchmark_handler(test))
    result = run_benchmark(
        make_settings(max_benchmark_calls=2),
        test,
        baseline_development_run_id=development_run.run_id,
        optimization_run_id=optimization_run.run_id,
        confirm_budget=True,
        seed=find_mixed_order_seed(test),
        origin="mock_test",
        run_id="bench-stop",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )
    assert result.manifest.status == "stopped"
    assert result.manifest.stop_reason == "call_cap"
    assert result.ledger.stop_reason == "call_cap"
    assert result.summary["verdict"] == "incomplete"
    assert result.summary["complete"] is False
    assert result.summary["attempted_arms"] < result.summary["planned_arms"]
    assert len(result.outcomes) < 4
    rejected = result.outcomes[-1]
    assert rejected.status == "error"
    assert rejected.metric["error_kind"] == "request_error"
    assert "budget stop" in rejected.error
    assert rejected.request_ids == []
    assert result.summary["economics"]["status"] == "not_applicable"
    with pytest.raises(ArtifactIncompleteError):
        open_replay_archive(
            result.run_dir,
            runs_root=tmp_path / "runs",
            programs_root=tmp_path / "programs",
        )


def test_benchmark_rejects_mismatched_injected_ledger(
    monkeypatch, tmp_path
) -> None:
    _train, _development, test = fixture_cases()
    development_run, optimization_run, settings = make_saved_pair(
        monkeypatch, tmp_path, test
    )
    calls = install_transport(monkeypatch, benchmark_handler(test))
    mismatched = BudgetLedger(max_calls=19, max_cost_usd="0.50")
    with pytest.raises(
        experiments.ExperimentConfigurationError, match="ledger limits"
    ):
        run_benchmark(
            settings,
            test,
            baseline_development_run_id=development_run.run_id,
            optimization_run_id=optimization_run.run_id,
            confirm_budget=True,
            origin="mock_test",
            run_id="bench-ledger-mismatch",
            runs_root=tmp_path / "runs",
            programs_root=tmp_path / "programs",
            ledger=mismatched,
        )
    assert calls == []


def test_benchmark_cancelled_injected_ledger_dispatches_nothing(
    monkeypatch, tmp_path
) -> None:
    _train, _development, test = fixture_cases()
    development_run, optimization_run, settings = make_saved_pair(
        monkeypatch, tmp_path, test
    )
    calls = install_transport(monkeypatch, benchmark_handler(test))
    ledger = BudgetLedger.from_limits(settings.benchmark_budget())
    ledger.cancel("cancelled")
    result = run_benchmark(
        settings,
        test,
        baseline_development_run_id=development_run.run_id,
        optimization_run_id=optimization_run.run_id,
        confirm_budget=True,
        origin="mock_test",
        run_id="bench-ledger-cancelled",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
        ledger=ledger,
    )
    assert calls == []
    assert result.ledger.admitted_calls == 0
    assert result.manifest.status == "stopped"
    assert result.summary["complete"] is False
