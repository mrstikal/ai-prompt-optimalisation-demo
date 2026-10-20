from __future__ import annotations

import copy
import json
import shutil
import time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import dspy
import dspy.clients.lm as dspy_lm
import pytest
from dspy.teleprompt.gepa.gepa import DspyGEPAResult

import dspy_lab.experiments as experiments
from dspy_lab.artifacts import (
    RUN_GEPA_STATS_FILENAME,
    RUN_OUTCOMES_FILENAME,
    RUN_REQUESTS_FILENAME,
    load_program_artifact,
    read_input_snapshot,
    read_program_manifest,
    read_recorded_outcomes,
    read_recorded_requests,
    read_run_manifest,
    verify_file_index,
)
from dspy_lab.config import Settings
from dspy_lab.data import load_cases
from dspy_lab.experiments import (
    ExperimentConfigurationError,
    ExperimentIncompleteError,
    ExperimentNotAuthorized,
    ExperimentRunExistsError,
    load_train_validation_cases,
    load_validation_cases,
    run_baseline_development,
    run_optimization,
)
from dspy_lab.metrics import feedback_metric, scalar_metric
from dspy_lab.program import EditorialQA

DATA_DIR = Path(__file__).resolve().parents[1] / "data/v1"
SECRET = "sk-or-v1-s10-experiment-sentinel"
TASK_MODEL = "openrouter/deepseek/deepseek-v4.1-flash-1017"
REFLECTION_MODEL = "openrouter/openai/gpt-6-sol"


class Reply:
    def __init__(
        self,
        content: str,
        *,
        cost: float | None = 0.00001,
        generation_id: str = "gen-s10",
    ) -> None:
        self.content = content
        self.cost = cost
        self.generation_id = generation_id


def response_for(reply: Reply, request: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=reply.generation_id,
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
        _hidden_params=(
            {}
            if reply.cost is None
            else {"response_cost": reply.cost}
        ),
    )


def install_transport(monkeypatch, handler):
    calls: list[dict] = []

    def fake_completion(*, request, num_retries=0):
        calls.append(request)
        content = handler(request, len(calls))
        if isinstance(content, BaseException):
            raise content
        return response_for(Reply(content), request)

    monkeypatch.setattr(dspy_lm, "litellm_completion", fake_completion)
    return calls


def valid_empty(request: dict, _index: int) -> str:
    return '{"findings": []}'


def make_settings(**overrides) -> Settings:
    values = {
        "mode": "live",
        "openrouter_api_key": SECRET,
        "max_baseline_dev_calls": 10,
        "max_baseline_dev_cost_usd": Decimal("0.50"),
        "max_optimization_calls": 160,
        "max_optimization_metric_calls": 10,
        "max_optimization_cost_usd": Decimal("5.00"),
        "request_timeout_seconds": 5.0,
    }
    values.update(overrides)
    return Settings(**values)


def clean_cases(cases, count: int = 2):
    return [
        case.model_copy(update={"gold_findings": []})
        for case in cases[:count]
    ]


def validation_cases():
    """Two clean validation cases for offline cadence fixtures, not the 10-case split."""
    return clean_cases(load_cases(DATA_DIR / "validation.jsonl"), count=2)


def train_cases():
    """Two clean training cases for offline GEPA fixtures."""
    return clean_cases(load_cases(DATA_DIR / "train.jsonl"), count=2)


def make_dev_run(
    monkeypatch,
    tmp_path,
    *,
    settings: Settings | None = None,
    cases=None,
    run_id: str = "dev-s10",
):
    install_transport(monkeypatch, valid_empty)
    return run_baseline_development(
        settings or make_settings(),
        cases or validation_cases(),
        confirm_budget=True,
        run_id=run_id,
        origin="mock_test",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )


def marker_handler(request: dict, _index: int) -> str:
    model = str(request.get("model"))
    if model == REFLECTION_MODEL:
        return json.dumps(
            {"new_instruction": "MARKER improved editorial instructions"}
        )
    payload = json.dumps(
        request.get("messages") or request.get("prompt"),
        ensure_ascii=False,
    )
    if "MARKER" in payload:
        return '{"findings": []}'
    return "not json"


def test_loaders_never_open_the_sealed_test_split(tmp_path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(DATA_DIR / "train.jsonl", data / "train.jsonl")
    shutil.copy(DATA_DIR / "validation.jsonl", data / "validation.jsonl")
    (data / "test.jsonl").mkdir()
    (data / "test.jsonl" / "sentinel").write_text(
        "test split must not be opened", encoding="utf-8"
    )

    splits = load_train_validation_cases(data)
    assert set(splits) == {"train", "validation"}
    assert len(splits["train"]) == 30
    assert len(splits["validation"]) == 10
    assert len(load_validation_cases(data)) == 10


def test_paid_phases_require_explicit_confirmation_and_live_mode(
    tmp_path,
) -> None:
    cases = validation_cases()
    with pytest.raises(ExperimentNotAuthorized):
        run_baseline_development(
            make_settings(), cases, runs_root=tmp_path / "runs"
        )
    with pytest.raises(ExperimentConfigurationError, match="Live mode"):
        run_baseline_development(
            Settings(mode="replay", openrouter_api_key=SECRET),
            cases,
            confirm_budget=True,
            runs_root=tmp_path / "runs",
        )
    with pytest.raises(ExperimentConfigurationError, match="OPENROUTER"):
        run_baseline_development(
            Settings(mode="live", openrouter_api_key=None),
            cases,
            confirm_budget=True,
            runs_root=tmp_path / "runs",
        )
    assert not (tmp_path / "runs").exists()


def test_baseline_development_persists_all_cases_and_requests(
    monkeypatch, tmp_path
) -> None:
    calls = install_transport(monkeypatch, valid_empty)
    cases = clean_cases(
        load_cases(DATA_DIR / "validation.jsonl"),
        count=10,
    )
    result = run_baseline_development(
        make_settings(max_baseline_dev_calls=10),
        cases,
        confirm_budget=True,
        run_id="dev-s10-contract",
        origin="mock_test",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )

    assert result.manifest.status == "completed"
    assert result.manifest.kind == "development"
    assert result.manifest.origin == "mock_test"
    assert len(result.outcomes) == len(cases) == 10
    assert all(outcome.status == "ok" for outcome in result.outcomes)
    assert len(result.requests) == 10
    assert len(calls) == 10
    assert all(record.phase == "task" for record in result.requests)
    assert all(
        record.response_text == '{"findings": []}'
        for record in result.requests
    )
    assert all(
        record.model == TASK_MODEL for record in result.requests
    )
    assert result.corpus.article_count == 10
    assert result.corpus.mean_reward == 1.0

    manifest = read_run_manifest(result.run_dir)
    verify_file_index(result.run_dir, manifest.files)
    assert {
        RUN_OUTCOMES_FILENAME,
        RUN_REQUESTS_FILENAME,
        "config.json",
        "summary.json",
        "events.jsonl",
        "inputs_snapshot.jsonl",
        "editorial_rules_snapshot.md",
    } <= set(manifest.files)
    outcomes = read_recorded_outcomes(
        result.run_dir / RUN_OUTCOMES_FILENAME, run_id=result.run_id
    )
    assert [outcome.case_id for outcome in outcomes] == [
        case.case_id for case in cases
    ]
    requests = read_recorded_requests(
        result.run_dir / RUN_REQUESTS_FILENAME
    )
    assert len(requests) == 10
    assert all(
        request.response_text == '{"findings": []}'
        for request in requests.values()
    )
    assert {
        request_id
        for outcome in outcomes
        for request_id in outcome.request_ids
    } == set(requests)
    snapshot_cases, _rules = read_input_snapshot(
        result.run_dir, manifest.snapshot
    )
    assert [case.case_id for case in snapshot_cases] == [
        case.case_id for case in cases
    ]
    assert manifest.links.baseline_program_id == result.baseline_program.program_id
    assert result.baseline_program.instructions == (
        EditorialQA().predict.signature.instructions
    )
    assert result.baseline_program.demos == []
    assert SECRET not in (result.run_dir / "config.json").read_text(
        encoding="utf-8"
    )
    summary = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["always_empty_reference"]["label"] == (
        "calculated_reference_not_a_model_run"
    )


def test_baseline_latency_includes_deterministic_scoring(
    monkeypatch, tmp_path
) -> None:
    install_transport(monkeypatch, valid_empty)
    real_score_case = experiments.score_case

    def slow_score_case(*args, **kwargs):
        time.sleep(0.01)
        return real_score_case(*args, **kwargs)

    monkeypatch.setattr(experiments, "score_case", slow_score_case)
    result = run_baseline_development(
        make_settings(max_baseline_dev_calls=1),
        validation_cases()[:1],
        confirm_budget=True,
        run_id="dev-s10-latency",
        origin="mock_test",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )
    assert result.outcomes[0].latency_ms >= 5.0


def test_baseline_parse_errors_are_retained_and_counted(
    monkeypatch, tmp_path
) -> None:
    responses = iter(["not json", '{"findings": []}'])
    install_transport(monkeypatch, lambda _request, _index: next(responses))
    cases = validation_cases()
    result = run_baseline_development(
        make_settings(max_baseline_dev_calls=2),
        cases,
        confirm_budget=True,
        run_id="dev-s10-errors",
        origin="mock_test",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )

    assert result.manifest.status == "completed"
    assert [outcome.status for outcome in result.outcomes] == [
        "error",
        "ok",
    ]
    first = result.outcomes[0]
    assert first.metric["valid"] is False
    assert first.metric["error_kind"] == "parse_error"
    assert "AdapterParseError" in first.error
    assert first.findings is None
    assert len(result.requests) == 2
    assert result.requests[0].response_text == "not json"
    assert result.corpus.article_count == 2
    assert result.corpus.invalid_count == 1


def test_baseline_budget_stop_keeps_unattempted_outcomes(
    monkeypatch, tmp_path
) -> None:
    calls = install_transport(monkeypatch, valid_empty)
    cases = validation_cases()
    result = run_baseline_development(
        make_settings(max_baseline_dev_calls=1),
        cases,
        confirm_budget=True,
        run_id="dev-s10-stop",
        origin="mock_test",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )

    assert len(calls) == 1
    assert len(result.requests) == 1
    assert result.manifest.status == "stopped"
    assert result.manifest.stop_reason == "call_cap"
    assert len(result.outcomes) == 2
    assert result.outcomes[0].status == "ok"
    assert result.outcomes[1].status == "error"
    assert "budget stop" in result.outcomes[1].error
    assert result.outcomes[1].request_ids == []
    assert result.ledger.rejected_calls == 1
    assert result.ledger.stop_reason == "call_cap"


def test_baseline_run_id_is_never_overwritten(monkeypatch, tmp_path) -> None:
    install_transport(monkeypatch, valid_empty)
    cases = validation_cases()
    first = run_baseline_development(
        make_settings(max_baseline_dev_calls=2),
        cases,
        confirm_budget=True,
        run_id="dev-s10-once",
        origin="mock_test",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )
    with pytest.raises(ExperimentRunExistsError):
        run_baseline_development(
            make_settings(max_baseline_dev_calls=2),
            cases,
            confirm_budget=True,
            run_id="dev-s10-once",
            origin="mock_test",
            runs_root=tmp_path / "runs",
            programs_root=tmp_path / "programs",
        )
    assert first.run_dir.is_dir()


def test_optimization_budget_headroom_is_checked_before_dispatch(
    tmp_path,
) -> None:
    assert (
        experiments._validate_optimization_budget(
            make_settings(max_optimization_metric_calls=120)
        )
        == 40
    )
    with pytest.raises(ExperimentConfigurationError, match="headroom"):
        run_optimization(
            make_settings(
                max_optimization_calls=150,
                max_optimization_metric_calls=120,
            ),
            train_cases(),
            validation_cases(),
            baseline_development_run_id="dev-missing",
            confirm_budget=True,
            runs_root=tmp_path / "runs",
        )
    with pytest.raises(ExperimentConfigurationError, match="headroom"):
        run_optimization(
            make_settings(
                max_optimization_calls=160,
                max_optimization_metric_calls=160,
            ),
            train_cases(),
            validation_cases(),
            baseline_development_run_id="dev-missing",
            confirm_budget=True,
            runs_root=tmp_path / "runs",
        )
    with pytest.raises(ExperimentConfigurationError, match="headroom"):
        run_optimization(
            make_settings(
                max_optimization_calls=120,
                max_optimization_metric_calls=119,
            ),
            train_cases(),
            validation_cases(),
            baseline_development_run_id="dev-missing",
            confirm_budget=True,
            runs_root=tmp_path / "runs",
        )


def test_optimization_requires_a_completed_linked_development_run(
    tmp_path,
) -> None:
    with pytest.raises(ExperimentIncompleteError, match="missing"):
        run_optimization(
            make_settings(),
            train_cases(),
            validation_cases(),
            baseline_development_run_id="dev-missing",
            confirm_budget=True,
            origin="mock_test",
            runs_root=tmp_path / "runs",
            programs_root=tmp_path / "programs",
        )


def test_optimization_rejects_a_mismatched_development_snapshot(
    monkeypatch, tmp_path
) -> None:
    dev_cases = validation_cases()
    make_dev_run(monkeypatch, tmp_path, cases=dev_cases)
    changed = [dev_cases[0].model_copy(update={"title": "Changed title"})]
    changed += dev_cases[1:]
    with pytest.raises(ExperimentIncompleteError, match="content"):
        run_optimization(
            make_settings(),
            train_cases(),
            changed,
            baseline_development_run_id="dev-s10",
            confirm_budget=True,
            origin="mock_test",
            runs_root=tmp_path / "runs",
            programs_root=tmp_path / "programs",
        )


def test_optimization_rejects_a_changed_task_protocol(
    monkeypatch, tmp_path
) -> None:
    validation = validation_cases()
    make_dev_run(monkeypatch, tmp_path, cases=validation)
    with pytest.raises(ExperimentIncompleteError, match="protocol"):
        run_optimization(
            make_settings(temperature=0.5),
            train_cases(),
            validation,
            baseline_development_run_id="dev-s10",
            confirm_budget=True,
            origin="mock_test",
            runs_root=tmp_path / "runs",
            programs_root=tmp_path / "programs",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("metric_version", "different_metric"),
        ("prompt_version", "different_prompt"),
        ("dataset_version", "v2"),
        ("rules_version", "editorial_rules_v2"),
    ],
)
def test_optimization_rejects_changed_linked_versions(
    monkeypatch, tmp_path, field, value
) -> None:
    validation = validation_cases()
    development = make_dev_run(monkeypatch, tmp_path, cases=validation)
    config_path = development.run_dir / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload[field] = value
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExperimentIncompleteError, match="different"):
        run_optimization(
            make_settings(),
            train_cases(),
            validation,
            baseline_development_run_id="dev-s10",
            confirm_budget=True,
            origin="mock_test",
            runs_root=tmp_path / "runs",
            programs_root=tmp_path / "programs",
        )


def test_optimization_runs_genuine_gepa_and_round_trips_selected_program(
    monkeypatch, tmp_path
) -> None:
    train = train_cases()
    validation = validation_cases()
    assert len(train) == 2 and len(validation) == 2
    # With a 2-case valset, GEPA's seed val evaluation costs 2 of the 10
    # metric calls. The observed cadence below is 10 task calls + 1
    # reflection, not a 10-case seed evaluation.
    make_dev_run(monkeypatch, tmp_path, cases=validation)
    calls = install_transport(monkeypatch, marker_handler)
    result = run_optimization(
        make_settings(
            max_optimization_calls=160,
            max_optimization_metric_calls=10,
        ),
        train,
        validation,
        baseline_development_run_id="dev-s10",
        confirm_budget=True,
        run_id="opt-s10-genuine",
        origin="mock_test",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )

    assert result.manifest.status == "completed"
    assert result.ledger.stop_reason is None
    assert result.selected_program.program_id != (
        result.baseline_program.program_id
    )
    assert "MARKER" in result.selected_program.instructions
    assert result.baseline_program.instructions == (
        EditorialQA().predict.signature.instructions
    )
    baseline_manifest_after = read_program_manifest(
        Path(tmp_path / "programs") / result.baseline_program.program_id
    )
    assert baseline_manifest_after == result.baseline_program
    assert result.selected_program.parent_program_id == (
        result.baseline_program.program_id
    )
    assert result.selected_program.demos == []
    assert result.selected_program.lm is None
    loaded = load_program_artifact(
        Path(tmp_path / "programs") / result.selected_program.program_id,
        allow_mock=True,
    )
    assert loaded.program.predict.signature.instructions == (
        result.selected_program.instructions
    )
    assert loaded.program.predict.demos == []
    assert loaded.program.predict.lm is None

    manifest = read_run_manifest(result.run_dir)
    verify_file_index(result.run_dir, manifest.files)
    assert manifest.links.baseline_development_run_id == "dev-s10"
    assert manifest.links.baseline_program_id == result.baseline_program.program_id
    assert manifest.links.optimized_program_id == (
        result.selected_program.program_id
    )
    assert RUN_GEPA_STATS_FILENAME in manifest.files
    assert "candidate_instructions.jsonl" in manifest.files
    stats = json.loads(
        (result.run_dir / RUN_GEPA_STATS_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    detailed = stats["detailed_results"]
    assert len(detailed["candidates"]) >= 2
    assert detailed["best_idx"] == stats["selection"][
        "selected_candidate_index"
    ]
    assert stats["selection"]["selected_equals_baseline"] is False
    assert stats["selection"]["task_attempts"] == 10
    assert stats["selection"]["reflection_attempts"] >= 1
    assert stats["selection"]["valid_proposal_responses"] >= 1
    assert stats["selection"]["reported_total_metric_calls"] == 10
    assert stats["selection"]["budget"]["admitted_calls"] == 11
    assert stats["selection"]["budget"]["stop_reason"] is None
    assert stats["status"] == "completed"
    candidate_records = [
        json.loads(line)
        for line in (
            result.run_dir / "candidate_instructions.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert len(candidate_records) == len(detailed["candidates"])
    assert sum(record["selected"] for record in candidate_records) == 1

    requests = read_recorded_requests(
        result.run_dir / RUN_REQUESTS_FILENAME
    )
    assert len(requests) == len(result.requests) == 11
    assert {request.phase for request in requests.values()} == {
        "task",
        "reflection",
    }
    reflection = [
        request for request in requests.values() if request.phase == "reflection"
    ]
    assert reflection
    assert all(
        "MARKER" in (request.response_text or "")
        for request in reflection
    )
    reflection_prompt = json.dumps(
        reflection[0].messages, ensure_ascii=False
    )
    assert (
        "failed to parse" in reflection_prompt
        or "Follow this structure" in reflection_prompt
    )
    assert any(
        request.response_text == "not json"
        for request in requests.values()
        if request.phase == "task"
    )
    assert all(
        request.model == TASK_MODEL
        for request in requests.values()
        if request.phase == "task"
    )
    assert all(
        request.model == REFLECTION_MODEL
        for request in requests.values()
        if request.phase == "reflection"
    )
    for request in requests.values():
        payload = json.dumps(request.messages, ensure_ascii=False)
        assert "gold_findings" not in payload
        assert "test-" not in payload
    assert len(calls) == 11
    assert all(call["model"] in (TASK_MODEL, REFLECTION_MODEL) for call in calls)
    config_text = (result.run_dir / "config.json").read_text(
        encoding="utf-8"
    )
    assert SECRET not in config_text
    assert "gold_findings" not in config_text
    config = json.loads(config_text)
    assert config["reflection"]["temperature"] is None
    assert config["reflection"]["supports_temperature"] is False


def test_optimization_labels_a_perfect_baseline_as_unexercised(
    monkeypatch, tmp_path
) -> None:
    validation = validation_cases()
    train = train_cases()
    make_dev_run(monkeypatch, tmp_path, cases=validation)
    install_transport(monkeypatch, valid_empty)
    result = run_optimization(
        make_settings(),
        train,
        validation,
        baseline_development_run_id="dev-s10",
        confirm_budget=True,
        run_id="opt-s10-perfect",
        origin="mock_test",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )

    assert result.manifest.status == "completed"
    assert result.selected_program.program_id == (
        result.baseline_program.program_id
    )
    assert result.manifest.links.optimized_program_id == (
        result.baseline_program.program_id
    )
    assessment = result.stats["selection"]
    assert assessment["candidate_count"] == 1
    assert assessment["selected_equals_baseline"] is True
    assert assessment["search_status"] == "perfect_baseline_skip"
    assert any(
        "unexercised" in warning for warning in result.warnings
    )
    assert RUN_GEPA_STATS_FILENAME in result.manifest.files


def test_optimization_budget_stop_is_not_marked_completed(
    monkeypatch, tmp_path
) -> None:
    validation = validation_cases()
    train = train_cases()
    make_dev_run(monkeypatch, tmp_path, cases=validation)
    install_transport(monkeypatch, valid_empty)

    class OverrunningOptimizer:
        def __init__(self, **_kwargs) -> None:
            pass

        def compile(self, student, trainset, valset):
            for _ in range(80):
                student(**trainset[0].inputs())
            return student

    monkeypatch.setattr(
        experiments,
        "_build_gepa_optimizer",
        lambda **_kwargs: OverrunningOptimizer(),
    )
    result = run_optimization(
        make_settings(
            max_optimization_calls=20,
            max_optimization_metric_calls=10,
        ),
        train,
        validation,
        baseline_development_run_id="dev-s10",
        confirm_budget=True,
        run_id="opt-s10-stop",
        origin="mock_test",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )

    assert result.manifest.status == "stopped"
    assert result.manifest.stop_reason == "call_cap"
    assert result.ledger.stop_reason == "call_cap"
    assert result.ledger.admitted_calls == 20
    assert result.selected_program.program_id == (
        result.baseline_program.program_id
    )
    assert result.stats["status"] == "stopped"
    assert result.stats["detailed_results"] is None
    assert any(
        "must not be treated as a completed optimization" in warning
        for warning in result.warnings
    )
    assert read_run_manifest(result.run_dir).status == "stopped"


def test_returned_compile_is_downgraded_after_a_budget_stop(
    monkeypatch, tmp_path
) -> None:
    validation = validation_cases()
    train = train_cases()
    make_dev_run(monkeypatch, tmp_path, cases=validation)
    install_transport(monkeypatch, valid_empty)

    class DowngradeOptimizer:
        def __init__(self, **_kwargs) -> None:
            pass

        def compile(self, student, trainset, valset):
            try:
                for _ in range(80):
                    student(**trainset[0].inputs())
            except experiments.BudgetStop:
                pass
            improved = copy.deepcopy(student)
            improved.predict.signature = (
                improved.predict.signature.with_instructions(
                    "DOWNGRADE candidate reached after a budget stop"
                )
            )
            improved.detailed_results = DspyGEPAResult(
                candidates=[student, improved],
                parents=[[None], [0]],
                val_aggregate_scores=[0.0, 1.0],
                val_subscores=[{}, {}],
                per_val_instance_best_candidates={0: {1}},
                discovery_eval_counts=[1, 1],
                total_metric_calls=99,
                num_full_val_evals=1,
            )
            return improved

    monkeypatch.setattr(
        experiments,
        "_build_gepa_optimizer",
        lambda **_kwargs: DowngradeOptimizer(),
    )
    result = run_optimization(
        make_settings(
            max_optimization_calls=20,
            max_optimization_metric_calls=10,
        ),
        train,
        validation,
        baseline_development_run_id="dev-s10",
        confirm_budget=True,
        run_id="opt-s10-downgrade",
        origin="mock_test",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )

    assert result.manifest.status == "stopped"
    assert result.manifest.stop_reason == "call_cap"
    assert result.selected_program.program_id != (
        result.baseline_program.program_id
    )
    assert result.stats["status"] == "stopped"
    assert result.stats["stop_reason"] == "call_cap"
    assert result.stats["detailed_results"] is not None
    assert any("invalidated the apparent completion" in w for w in result.warnings)
    assert any("ledger recorded" in w for w in result.warnings)
    assert any("exceeded the metric-call target" in w for w in result.warnings)
    assert read_run_manifest(result.run_dir).status == "stopped"


def test_optimization_compile_exception_is_a_failed_record(
    monkeypatch, tmp_path
) -> None:
    validation = validation_cases()
    make_dev_run(monkeypatch, tmp_path, cases=validation)

    class FailingOptimizer:
        def __init__(self, **_kwargs) -> None:
            pass

        def compile(self, student, trainset, valset):
            raise RuntimeError("simulated compile failure")

    monkeypatch.setattr(
        experiments,
        "_build_gepa_optimizer",
        lambda **_kwargs: FailingOptimizer(),
    )
    result = run_optimization(
        make_settings(),
        train_cases(),
        validation,
        baseline_development_run_id="dev-s10",
        confirm_budget=True,
        run_id="opt-s10-failed",
        origin="mock_test",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )

    assert result.manifest.status == "failed"
    assert result.manifest.stop_reason == "compile_error"
    assert "simulated compile failure" in (result.manifest.error_message or "")
    assert result.stats["detailed_results"] is None
    assert result.selected_program.program_id == (
        result.baseline_program.program_id
    )
    assert read_run_manifest(result.run_dir).status == "failed"


def test_all_proposal_failure_is_assessed_as_failed() -> None:
    program = EditorialQA()
    detailed = DspyGEPAResult(
        candidates=[program],
        parents=[[None]],
        val_aggregate_scores=[0.0],
        val_subscores=[{}],
        per_val_instance_best_candidates={},
        discovery_eval_counts=[1],
        total_metric_calls=1,
        num_full_val_evals=1,
    )
    no_reflection = experiments._search_assessment(
        detailed=detailed,
        records=[],
        baseline_instructions=program.predict.signature.instructions,
    )
    assert no_reflection[0] == "not_exercised"
    assert no_reflection[2]["status_override"] == "failed"

    failed_reflection = SimpleNamespace(
        phase="reflection", status="pending", response_text=None
    )
    assessment = experiments._search_assessment(
        detailed=detailed,
        records=[failed_reflection],
        baseline_instructions=program.predict.signature.instructions,
    )
    assert assessment[0] == "all_proposals_failed"
    assert assessment[2]["status_override"] == "failed"

    unusable_proposal = SimpleNamespace(
        phase="reflection",
        status="succeeded",
        response_text='{"unexpected": "field"}',
    )
    assessment = experiments._search_assessment(
        detailed=detailed,
        records=[unusable_proposal],
        baseline_instructions=program.predict.signature.instructions,
    )
    assert assessment[0] == "all_proposals_failed"
    assert assessment[2]["status_override"] == "failed"
    assert assessment[2]["valid_proposal_responses"] == 0


def test_feedback_metric_matches_scalar_metric_and_gepa_contract() -> None:
    case = load_cases(DATA_DIR / "validation.jsonl")[0]
    prediction = dspy.Prediction(findings=[])
    feedback = feedback_metric(
        case, prediction, None, "predict", None
    )
    scalar = scalar_metric(case, prediction)
    assert feedback.score == scalar
    assert isinstance(feedback.feedback, str)
    assert feedback.feedback
    assert len(feedback.feedback) <= 400


def test_live_phase_rejects_price_increase_before_creating_run(
    monkeypatch, tmp_path
) -> None:
    original = experiments.fetch_endpoint_catalog

    def inflated_fetch(model: str):
        payload = original(model)
        payload["data"]["endpoints"][0]["pricing"]["prompt"] = "1"
        return payload

    monkeypatch.setattr(
        experiments, "fetch_endpoint_catalog", inflated_fetch
    )
    runs_root = tmp_path / "runs"
    with pytest.raises(
        ExperimentConfigurationError, match="catalog validation failed"
    ):
        run_baseline_development(
            make_settings(),
            load_validation_cases(),
            confirm_budget=True,
            origin="live",
            run_id="dev-price",
            runs_root=runs_root,
            programs_root=tmp_path / "programs",
        )
    assert not runs_root.exists()


def test_live_optimization_rejects_held_out_records(tmp_path) -> None:
    held_out = load_cases(DATA_DIR / "test.jsonl")
    with pytest.raises(
        ExperimentConfigurationError, match="frozen train split"
    ):
        run_optimization(
            make_settings(),
            held_out,
            load_validation_cases(),
            baseline_development_run_id="dev-missing",
            confirm_budget=True,
            origin="live",
            runs_root=tmp_path / "runs",
            programs_root=tmp_path / "programs",
        )


def test_live_phases_fetch_current_route_catalog_snapshots(
    monkeypatch, tmp_path
) -> None:
    calls: list[str] = []
    original = experiments.fetch_endpoint_catalog

    def recording_fetch(model: str):
        calls.append(model)
        return original(model)

    monkeypatch.setattr(
        experiments, "fetch_endpoint_catalog", recording_fetch
    )
    result = run_baseline_development(
        make_settings(),
        load_validation_cases(),
        confirm_budget=True,
        origin="live",
        run_id="dev-catalog",
        runs_root=tmp_path / "runs",
        programs_root=tmp_path / "programs",
    )
    settings = make_settings()
    assert calls == [settings.task_model, settings.reflection_model]
    config = json.loads(
        (result.run_dir / "config.json").read_text(encoding="utf-8")
    )
    snapshots = config["route_catalog_snapshots"]
    assert snapshots["task"]["model"] == settings.task_model
    assert snapshots["reflection"]["model"] == settings.reflection_model
