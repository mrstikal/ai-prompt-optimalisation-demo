from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import dspy
import dspy.clients.lm as dspy_lm
import pytest
from pydantic import ValidationError

import dspy_lab.artifacts as artifacts
from dspy_lab.artifacts import (
    RUN_GEPA_STATS_FILENAME,
    RUN_OUTCOMES_FILENAME,
    RUN_REQUESTS_FILENAME,
    SNAPSHOT_CASES_FILENAME,
    SNAPSHOT_RULES_FILENAME,
    ArtifactCompatibilityError,
    ArtifactCorruptError,
    ArtifactError,
    ArtifactIncompleteError,
    ArtifactNotFoundError,
    ArtifactSchemaError,
    ArtifactSecretError,
    BenchmarkProtocol,
    ProgramManifest,
    RecordedOutcome,
    RecordedRequest,
    ReplayError,
    RunLinks,
    RunManifest,
    SentTrace,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    case_input_sha256,
    file_index,
    format_preview_trace,
    format_sent_trace,
    is_measured_archive,
    load_program_artifact,
    open_replay_archive,
    program_reference,
    read_program_manifest,
    read_program_snapshot,
    save_program_artifact,
    scan_tree_for_secrets,
    sha256_file,
    sha256_text,
    write_input_snapshot,
    write_run_manifest,
)
from dspy_lab.config import collect_runtime_versions
from dspy_lab.data import load_cases
from dspy_lab.llm import (
    BudgetLedger,
    GuardedLM,
    ModelPricing,
    ModelRoute,
    RequestRecord,
)
from dspy_lab.program import (
    EditorialQA,
    StrictJSONAdapter,
    case_inputs,
    load_editorial_rules,
)
from dspy_lab.schemas import CaseRecord

DATA_DIR = Path(__file__).resolve().parents[1] / "data/v1"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SECRET = "sk-or-v1-s9-artifact-sentinel"
RULES = load_editorial_rules()
TRAIN_CASES = load_cases(DATA_DIR / "train.jsonl")


@dataclass
class MockArchive:
    root: Path
    programs_root: Path
    runs_root: Path
    development_dir: Path
    optimization_dir: Path
    benchmark_dir: Path
    development: RunManifest
    optimization: RunManifest
    benchmark: RunManifest
    baseline_manifest: ProgramManifest
    optimized_manifest: ProgramManifest
    validation_cases: list[CaseRecord]
    benchmark_cases: list[CaseRecord]


def rebrand(case: CaseRecord, case_id: str, family_id: str) -> CaseRecord:
    payload = case.model_dump()
    payload["case_id"] = case_id
    payload["family_id"] = family_id
    return CaseRecord.model_validate(payload)


def _reference(root: Path, manifest: ProgramManifest):
    return program_reference(root / manifest.program_id)


def _outcome(
    run_id: str,
    case: CaseRecord,
    phase: str,
    split: str,
    program_id: str,
    candidate_id: str,
    request_ids: tuple[str, ...] = (),
) -> RecordedOutcome:
    return RecordedOutcome(
        run_id=run_id,
        case_id=case.case_id,
        program_id=program_id,
        candidate_id=candidate_id,
        phase=phase,
        split=split,
        status="ok",
        findings=[],
        metric={"f1": 0.0},
        request_ids=list(request_ids),
        cost_usd=Decimal("0.001"),
        cost_status="estimated",
        latency_ms=12.0,
        input_sha256=case_input_sha256(case, RULES),
    )


def _request(call_id: str, sequence: int) -> dict[str, object]:
    return {
        "call_id": call_id,
        "sequence": sequence,
        "phase": "task",
        "model": "openrouter/deepseek/deepseek-v4.1-flash-1017",
        "provider": "deepinfra/fp8",
        "status": "succeeded",
        "cost_status": "estimated",
        "reserved_cost_usd": "0.10336256",
        "estimated_cost_usd": "0.001",
        "reported_cost_usd": "0.001",
        "input_token_bound": 1_048_576,
        "output_token_bound": 2048,
        "input_tokens": 100,
        "output_tokens": 5,
        "generation_id": f"gen-{sequence}",
        "response_model": "openrouter/deepseek/deepseek-v4.1-flash-1017",
        "finish_reason": "stop",
        "error": None,
        "released_reason": None,
        "messages": [{"role": "user", "content": "mock request"}],
        "request_kwargs": {"max_tokens": 2048},
        "usage": {"prompt_tokens": 100, "completion_tokens": 5},
        "started_at": "2026-10-19T10:30:00Z",
        "finished_at": "2026-10-19T10:30:01Z",
    }


def build_mock_archive(
    root: Path,
    *,
    origin: str = "mock_test",
    released: bool = False,
    benchmark_status: str = "completed",
    optimized_parent: bool = True,
) -> MockArchive:
    programs_root = root / "programs"
    runs_root = root / "runs"
    validation_cases = [
        rebrand(TRAIN_CASES[0], "mock-val-001", "mock-fam-001"),
        rebrand(TRAIN_CASES[1], "mock-val-002", "mock-fam-002"),
    ]
    train_cases = [
        rebrand(TRAIN_CASES[0], "mock-train-001", "mock-fam-003"),
        rebrand(TRAIN_CASES[1], "mock-train-002", "mock-fam-004"),
    ]
    benchmark_cases = [
        rebrand(TRAIN_CASES[2], "mock-test-001", "mock-fam-005"),
        rebrand(TRAIN_CASES[3], "mock-test-002", "mock-fam-006"),
    ]
    baseline_manifest = save_program_artifact(
        EditorialQA(), origin=origin, programs_root=programs_root
    )
    optimized_program = EditorialQA()
    optimized_program.predict.signature = (
        optimized_program.predict.signature.with_instructions(
            "Optimized mock instruction for replay tests."
        )
    )
    optimized_manifest = save_program_artifact(
        optimized_program,
        origin=origin,
        programs_root=programs_root,
        parent_program_id=(
            baseline_manifest.program_id if optimized_parent else None
        ),
    )
    baseline_ref = _reference(programs_root, baseline_manifest)
    optimized_ref = _reference(programs_root, optimized_manifest)
    versions = collect_runtime_versions()

    development_dir = runs_root / "development" / "run-dev-001"
    development_dir.mkdir(parents=True)
    development_snapshot = write_input_snapshot(
        development_dir,
        cases=validation_cases,
        rules_text=RULES,
        split="validation",
    )
    development_outcomes = [
        _outcome(
            "run-dev-001",
            case,
            "development",
            "validation",
            baseline_manifest.program_id,
            "baseline",
        )
        for case in validation_cases
    ]
    atomic_write_jsonl(
        development_dir / RUN_OUTCOMES_FILENAME,
        [
            outcome.model_dump(mode="json")
            for outcome in development_outcomes
        ],
    )
    development_files = file_index(
        development_dir,
        [
            SNAPSHOT_CASES_FILENAME,
            SNAPSHOT_RULES_FILENAME,
            RUN_OUTCOMES_FILENAME,
        ],
    )
    development = RunManifest(
        run_id="run-dev-001",
        kind="development",
        origin=origin,
        status="completed",
        created_at="2026-10-19T10:00:00Z",
        finished_at="2026-10-19T10:05:00Z",
        versions=versions,
        snapshot=development_snapshot,
        programs=[baseline_ref],
        links=RunLinks(
            baseline_program_id=baseline_manifest.program_id
        ),
        files=development_files,
    )
    write_run_manifest(development_dir, development)

    optimization_dir = runs_root / "optimization" / "run-opt-001"
    optimization_dir.mkdir(parents=True)
    optimization_snapshot = write_input_snapshot(
        optimization_dir,
        cases=train_cases + validation_cases,
        rules_text=RULES,
        split="train_validation",
    )
    atomic_write_json(
        optimization_dir / RUN_GEPA_STATS_FILENAME,
        {
            "candidates": [
                {"id": "baseline"},
                {"id": "candidate-1"},
            ]
        },
    )
    optimization_files = file_index(
        optimization_dir,
        [
            SNAPSHOT_CASES_FILENAME,
            SNAPSHOT_RULES_FILENAME,
            RUN_GEPA_STATS_FILENAME,
        ],
    )
    optimization = RunManifest(
        run_id="run-opt-001",
        kind="optimization",
        origin=origin,
        status="completed",
        created_at="2026-10-19T10:10:00Z",
        finished_at="2026-10-19T10:20:00Z",
        versions=versions,
        snapshot=optimization_snapshot,
        programs=[baseline_ref, optimized_ref],
        links=RunLinks(
            baseline_development_run_id="run-dev-001",
            baseline_program_id=baseline_manifest.program_id,
            optimized_program_id=optimized_manifest.program_id,
        ),
        files=optimization_files,
    )
    write_run_manifest(optimization_dir, optimization)

    benchmark_dir = runs_root / "benchmarks" / "run-bench-001"
    benchmark_dir.mkdir(parents=True)
    benchmark_snapshot = write_input_snapshot(
        benchmark_dir,
        cases=benchmark_cases,
        rules_text=RULES,
        split="test",
    )
    benchmark_outcomes: list[RecordedOutcome] = []
    for index, case in enumerate(benchmark_cases):
        benchmark_outcomes.append(
            _outcome(
                "run-bench-001",
                case,
                "baseline",
                "test",
                baseline_manifest.program_id,
                "baseline",
                (f"call-{index * 2}",),
            )
        )
        benchmark_outcomes.append(
            _outcome(
                "run-bench-001",
                case,
                "optimized",
                "test",
                optimized_manifest.program_id,
                "candidate-1",
                (f"call-{index * 2 + 1}",),
            )
        )
    atomic_write_jsonl(
        benchmark_dir / RUN_OUTCOMES_FILENAME,
        [
            outcome.model_dump(mode="json")
            for outcome in benchmark_outcomes
        ],
    )
    atomic_write_jsonl(
        benchmark_dir / RUN_REQUESTS_FILENAME,
        [_request(f"call-{index}", index) for index in range(4)],
    )
    benchmark_files = file_index(
        benchmark_dir,
        [
            SNAPSHOT_CASES_FILENAME,
            SNAPSHOT_RULES_FILENAME,
            RUN_OUTCOMES_FILENAME,
            RUN_REQUESTS_FILENAME,
        ],
    )
    benchmark = RunManifest(
        run_id="run-bench-001",
        kind="benchmark",
        origin=origin,
        status=benchmark_status,
        created_at="2026-10-19T10:30:00Z",
        finished_at=(
            None
            if benchmark_status == "running"
            else "2026-10-19T10:40:00Z"
        ),
        versions=versions,
        snapshot=benchmark_snapshot,
        programs=[baseline_ref, optimized_ref],
        links=RunLinks(
            baseline_development_run_id="run-dev-001",
            optimization_run_id="run-opt-001",
            baseline_program_id=baseline_manifest.program_id,
            optimized_program_id=optimized_manifest.program_id,
        ),
        benchmark=BenchmarkProtocol(
            exposure_index=0,
            paired_order=["baseline", "optimized"],
            seed=42,
            protocol_sha256=sha256_text("mock benchmark protocol"),
            released=released,
        ),
        files=benchmark_files,
    )
    write_run_manifest(benchmark_dir, benchmark)
    return MockArchive(
        root=root,
        programs_root=programs_root,
        runs_root=runs_root,
        development_dir=development_dir,
        optimization_dir=optimization_dir,
        benchmark_dir=benchmark_dir,
        development=development,
        optimization=optimization,
        benchmark=benchmark,
        baseline_manifest=baseline_manifest,
        optimized_manifest=optimized_manifest,
        validation_cases=validation_cases,
        benchmark_cases=benchmark_cases,
    )


def _rewrite_manifest(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    atomic_write_json(path, payload)


def _rewrite_outcomes(
    archive: MockArchive,
    run_dir: Path,
    outcomes: list[RecordedOutcome],
) -> None:
    path = run_dir / RUN_OUTCOMES_FILENAME
    atomic_write_jsonl(
        path, [outcome.model_dump(mode="json") for outcome in outcomes]
    )
    _rewrite_manifest(
        run_dir / "manifest.json",
        lambda payload: payload["files"].update(
            {RUN_OUTCOMES_FILENAME: sha256_file(path)}
        ),
    )


def _read_outcomes(archive: MockArchive, run_dir: Path) -> list[RecordedOutcome]:
    from dspy_lab.artifacts import read_recorded_outcomes

    manifest = archive.benchmark if run_dir == archive.benchmark_dir else (
        archive.development
    )
    return read_recorded_outcomes(
        run_dir / RUN_OUTCOMES_FILENAME, run_id=manifest.run_id
    )


class Reply:
    def __init__(
        self,
        content: str,
        *,
        generation_id: str = "gen-loaded",
        cost: float = 0.00001,
    ) -> None:
        self.content = content
        self.generation_id = generation_id
        self.cost = cost


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
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "total_tokens": 105,
        },
        _hidden_params={"response_cost": reply.cost},
    )


def install_transport(monkeypatch, reply: Reply):
    calls: list[dict[str, object]] = []

    def fake_completion(*, request, num_retries=0):
        calls.append({"request": request, "num_retries": num_retries})
        return response_for(reply, request)

    monkeypatch.setattr(
        dspy_lm, "litellm_completion", fake_completion
    )
    return calls


def test_state_only_program_round_trip_and_identity(tmp_path) -> None:
    manifest = save_program_artifact(
        EditorialQA(), origin="mock_test", programs_root=tmp_path
    )
    assert manifest.baseline is True
    assert manifest.origin == "mock_test"
    assert manifest.demos == []
    assert manifest.lm is None
    state = json.loads(
        (tmp_path / manifest.program_id / "program.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["predict"]["lm"] is None
    assert state["predict"]["demos"] == []
    instructions = (
        tmp_path / manifest.program_id / "instructions.txt"
    ).read_text(encoding="utf-8")
    assert instructions == manifest.instructions
    loaded = load_program_artifact(
        tmp_path / manifest.program_id, allow_mock=True
    )
    assert loaded.manifest.program_id == manifest.program_id
    assert (
        loaded.program.predict.signature.instructions
        == manifest.instructions
    )
    assert loaded.program.predict.demos == []
    assert loaded.program.predict.lm is None
    snapshot = read_program_snapshot(tmp_path / manifest.program_id)
    assert snapshot.instructions == manifest.instructions
    assert snapshot.schema == manifest.signature_schema
    assert snapshot.state_sha256 == manifest.state_sha256
    reference = program_reference(tmp_path / manifest.program_id)
    assert reference.program_id == manifest.program_id
    assert reference.manifest_sha256 == sha256_file(
        tmp_path / manifest.program_id / "manifest.json"
    )


def test_optimized_program_round_trip_preserves_schema(tmp_path) -> None:
    baseline = save_program_artifact(
        EditorialQA(), origin="mock_test", programs_root=tmp_path
    )
    program = EditorialQA()
    program.predict.signature = program.predict.signature.with_instructions(
        "Optimized artifact instructions."
    )
    optimized = save_program_artifact(
        program,
        origin="mock_test",
        programs_root=tmp_path,
        parent_program_id=baseline.program_id,
    )
    assert optimized.parent_program_id == baseline.program_id
    assert optimized.baseline is False
    assert optimized.signature_schema == baseline.signature_schema
    assert optimized.instructions != baseline.instructions
    loaded = load_program_artifact(
        tmp_path / optimized.program_id, allow_mock=True
    )
    assert loaded.program.predict.signature.instructions == (
        "Optimized artifact instructions."
    )


def test_program_identity_is_content_deterministic(tmp_path) -> None:
    first = save_program_artifact(
        EditorialQA(), origin="mock_test", programs_root=tmp_path / "a"
    )
    second = save_program_artifact(
        EditorialQA(), origin="mock_test", programs_root=tmp_path / "b"
    )
    assert first.program_id == second.program_id
    assert first.identity_sha256 == second.identity_sha256
    assert first.state_sha256 == second.state_sha256


def test_mock_program_requires_explicit_allow_mock(tmp_path) -> None:
    manifest = save_program_artifact(
        EditorialQA(), origin="mock_test", programs_root=tmp_path
    )
    with pytest.raises(ArtifactCompatibilityError, match="mock"):
        load_program_artifact(tmp_path / manifest.program_id)


def _append_byte(path: Path) -> None:
    with path.open("ab") as handle:
        handle.write(b" ")


def test_missing_corrupt_and_tampered_program_files(tmp_path) -> None:
    for name, mutation, error in (
        (
            "missing",
            lambda directory: (directory / "manifest.json").unlink(),
            ArtifactNotFoundError,
        ),
        (
            "corrupt",
            lambda directory: atomic_write_text(
                directory / "manifest.json", "{"
            ),
            ArtifactCorruptError,
        ),
        (
            "tampered",
            lambda directory: _append_byte(
                directory / "program.json"
            ),
            ArtifactCorruptError,
        ),
    ):
        root = tmp_path / name
        manifest = save_program_artifact(
            EditorialQA(), origin="mock_test", programs_root=root
        )
        directory = root / manifest.program_id
        mutation(directory)
        with pytest.raises(error):
            read_program_manifest(directory)


def test_program_schema_version_and_unsafe_state_are_rejected(
    tmp_path,
) -> None:
    manifest = save_program_artifact(
        EditorialQA(), origin="mock_test", programs_root=tmp_path / "schema"
    )
    artifact_dir = tmp_path / "schema" / manifest.program_id

    def wrong_schema_version(payload):
        payload["schema_version"] = "program_manifest_v999"

    _rewrite_manifest(artifact_dir / "manifest.json", wrong_schema_version)
    with pytest.raises(ArtifactSchemaError):
        read_program_manifest(artifact_dir)

    manifest = save_program_artifact(
        EditorialQA(), origin="mock_test", programs_root=tmp_path / "unsafe"
    )
    artifact_dir = tmp_path / "unsafe" / manifest.program_id
    state_path = artifact_dir / "program.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["predict"]["api_key"] = "leaked"
    state_bytes = json.dumps(state, ensure_ascii=False).encode("utf-8")
    state_path.write_bytes(state_bytes)
    _rewrite_manifest(
        artifact_dir / "manifest.json",
        lambda payload: payload.update(
            {
                "state_sha256": sha256_text(state_bytes.decode("utf-8")),
                "state_bytes": len(state_bytes),
            }
        ),
    )
    with pytest.raises(ArtifactSecretError):
        load_program_artifact(artifact_dir, allow_mock=True)


def test_version_mismatch_is_rejected_before_dspy_load(
    tmp_path, monkeypatch
) -> None:
    original_versions = artifacts.collect_runtime_versions
    fake_versions = original_versions()
    fake_versions["dspy"] = "0.0.0-incompatible"
    monkeypatch.setattr(
        artifacts, "collect_runtime_versions", lambda: fake_versions
    )
    manifest = save_program_artifact(
        EditorialQA(), origin="mock_test", programs_root=tmp_path
    )
    monkeypatch.setattr(
        artifacts, "collect_runtime_versions", original_versions
    )
    artifact_dir = tmp_path / manifest.program_id

    def forbidden_load(*args, **kwargs):
        raise AssertionError("DSPy load must not run on incompatible state")

    monkeypatch.setattr(EditorialQA, "load", forbidden_load)
    with pytest.raises(ArtifactCompatibilityError, match="dspy"):
        load_program_artifact(artifact_dir, allow_mock=True)


def test_secret_scan_fails_closed_and_cleans_up(tmp_path, monkeypatch) -> None:
    manifest = save_program_artifact(
        EditorialQA(), origin="mock_test", programs_root=tmp_path
    )
    assert scan_tree_for_secrets(tmp_path, [SECRET]) == []
    secret_file = tmp_path / "leak.txt"
    atomic_write_text(secret_file, f"prefix {SECRET} suffix")
    assert scan_tree_for_secrets(tmp_path, [SECRET]) == [secret_file]

    monkeypatch.setattr(
        artifacts,
        "collect_runtime_versions",
        lambda: {**collect_runtime_versions(), "leak": SECRET},
    )
    root = tmp_path / "leaky"
    with pytest.raises(ArtifactSecretError):
        save_program_artifact(
            EditorialQA(),
            origin="mock_test",
            programs_root=root,
            secret_values=[SECRET],
        )
    assert not root.exists() or not any(
        path.name.startswith(".program-")
        for path in root.iterdir()
    )


def test_atomic_write_leaves_no_partial_destination(
    tmp_path, monkeypatch
) -> None:
    target = tmp_path / "record.json"
    def failing_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(artifacts.os, "replace", failing_replace)
    with pytest.raises(OSError):
        atomic_write_text(target, "new contents")
    assert not target.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_program_save_is_atomic_on_directory_replace(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "programs"
    original_replace = artifacts.os.replace

    def failing_replace(source, destination):
        if Path(destination).name.startswith("program-"):
            raise OSError("simulated directory replace failure")
        return original_replace(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", failing_replace)
    with pytest.raises(OSError):
        save_program_artifact(
            EditorialQA(), origin="mock_test", programs_root=root
        )
    assert not any(
        path.name.startswith("program-") for path in root.iterdir()
    )
    assert not any(
        path.name.startswith(".program-") for path in root.iterdir()
    )


def test_runtime_lm_reattachment_uses_guarded_lm(
    tmp_path, monkeypatch
) -> None:
    manifest = save_program_artifact(
        EditorialQA(), origin="mock_test", programs_root=tmp_path
    )
    loaded = load_program_artifact(
        tmp_path / manifest.program_id, allow_mock=True
    )
    calls = install_transport(
        monkeypatch, Reply('{"findings": []}')
    )
    ledger = BudgetLedger(max_calls=1, max_cost_usd=Decimal("1"))
    route = ModelRoute(
        model="deepseek/deepseek-v4.1-flash-1017",
        provider="deepinfra/fp8",
        context_limit=1_048_576,
        max_output_tokens=2048,
        pricing=ModelPricing(
            Decimal("0.098"), Decimal("0.294")
        ),
        expected_schema_mode="native",
    )
    lm = GuardedLM.from_route(
        route,
        ledger=ledger,
        phase="task",
        api_key=SECRET,
        temperature=0.0,
    )
    case = TRAIN_CASES[0]
    with dspy.context(lm=lm, adapter=StrictJSONAdapter()):
        prediction = loaded.program(**case_inputs(case, RULES))
    assert prediction.findings == []
    assert len(calls) == 1
    assert len(ledger.records) == 1
    assert ledger.records[0].phase == "task"
    assert loaded.program.predict.lm is None


def test_trace_formatting_distinguishes_sent_and_preview() -> None:
    fixture = json.loads(
        (FIXTURES / "s8_preflight_call_v1.json").read_text(
            encoding="utf-8"
        )
    )
    sent = format_sent_trace(fixture)
    assert isinstance(sent, SentTrace)
    assert sent.source == "sent"
    assert sent.sent is True
    assert sent.generation_id == "gen-s8-fixture-1"
    assert sent.messages == fixture["messages"]
    assert sent.estimated_cost_usd == "0.0000123"
    assert sent.reported_cost_usd == "0.0000123"
    assert sent.cost_status == "estimated"
    assert sent.usage["prompt_tokens"] == 512

    preview = format_preview_trace(fixture["messages"])
    assert preview.source == "preview"
    assert preview.sent is False
    assert preview.messages == fixture["messages"]
    preview_payload = preview.model_dump()
    assert "generation_id" not in preview_payload
    assert "usage" not in preview_payload
    assert "estimated_cost_usd" not in preview_payload
    with pytest.raises(ValidationError):
        SentTrace.model_validate(preview_payload)
    with pytest.raises(ArtifactSchemaError):
        format_sent_trace(preview_payload)


def test_sent_trace_requires_complete_record() -> None:
    with pytest.raises(ArtifactSchemaError, match="missing"):
        format_sent_trace(
            {
                "phase": "task",
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "request_kwargs": {},
            }
        )
    with pytest.raises(ArtifactSchemaError, match="usage object"):
        format_sent_trace(
            {
                "phase": "task",
                "model": "m",
                "provider": "p",
                "messages": [{"role": "user", "content": "x"}],
                "request_kwargs": {},
                "generation_id": "g",
                "usage": 1,
                "estimated_cost_usd": "0.0",
                "status": "succeeded",
            }
        )


def test_sent_trace_accepts_slots_request_record() -> None:
    record = RequestRecord(
        call_id="call-dataclass",
        sequence=1,
        phase="task",
        model="openrouter/deepseek/deepseek-v4.1-flash-1017",
        provider="deepinfra/fp8",
        status="succeeded",
        cost_status="estimated",
        reserved_cost_usd=Decimal("0.10336256"),
        estimated_cost_usd=Decimal("0.001"),
        reported_cost_usd=Decimal("0.001"),
        input_token_bound=1_048_576,
        output_token_bound=2048,
        input_tokens=100,
        output_tokens=5,
        generation_id="gen-dataclass",
        response_model="openrouter/deepseek/deepseek-v4.1-flash-1017",
        finish_reason="stop",
        error=None,
        released_reason=None,
        messages=({"role": "user", "content": "hello"},),
        request_kwargs={"max_tokens": 2048},
        usage={"prompt_tokens": 100, "completion_tokens": 5},
        started_at=datetime(2026, 10, 19, 17, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 10, 19, 17, 0, 1, tzinfo=timezone.utc),
    )
    sent = format_sent_trace(record)
    assert sent.generation_id == "gen-dataclass"
    assert sent.messages == [{"role": "user", "content": "hello"}]
    assert sent.estimated_cost_usd == "0.001"


def test_artifact_reads_reject_symlinks(tmp_path) -> None:
    manifest = save_program_artifact(
        EditorialQA(), origin="mock_test", programs_root=tmp_path
    )
    state_path = tmp_path / manifest.program_id / "program.json"
    target = tmp_path / "outside.json"
    target.write_bytes(state_path.read_bytes())
    state_path.unlink()
    state_path.symlink_to(target)
    with pytest.raises(artifacts.ArtifactError, match="symlink"):
        read_program_manifest(tmp_path / manifest.program_id)


def test_state_signature_field_tamper_is_rejected(tmp_path) -> None:
    manifest = save_program_artifact(
        EditorialQA(), origin="mock_test", programs_root=tmp_path
    )
    artifact_dir = tmp_path / manifest.program_id
    state_path = artifact_dir / "program.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["predict"]["signature"]["fields"][0]["description"] = "tampered"
    state_bytes = json.dumps(state, ensure_ascii=False).encode("utf-8")
    state_path.write_bytes(state_bytes)
    _rewrite_manifest(
        artifact_dir / "manifest.json",
        lambda payload: payload.update(
            {
                "state_sha256": sha256_text(state_bytes.decode("utf-8")),
                "state_bytes": len(state_bytes),
            }
        ),
    )
    with pytest.raises(ArtifactCorruptError, match="signature fields"):
        load_program_artifact(artifact_dir, allow_mock=True)


def test_replay_reads_mock_archive_without_lm_or_native_load(
    tmp_path, monkeypatch
) -> None:
    archive = build_mock_archive(tmp_path)

    def forbidden_load(*args, **kwargs):
        raise AssertionError("Replay must not call EditorialQA.load")

    def forbidden_lm(*args, **kwargs):
        raise AssertionError("Replay must not construct an LM")

    monkeypatch.setattr(EditorialQA, "load", forbidden_load)
    monkeypatch.setattr(dspy.LM, "__init__", forbidden_lm)
    monkeypatch.setattr(dspy, "Predict", forbidden_lm)
    monkeypatch.setattr(dspy, "Evaluate", forbidden_lm)
    replay = open_replay_archive(
        archive.benchmark_dir, programs_root=archive.programs_root
    )
    assert replay.is_measured is False
    assert replay.additional_api_calls == 0
    assert replay.additional_api_cost_usd == Decimal("0.00")
    assert [case.case_id for case in replay.cases] == [
        "mock-test-001",
        "mock-test-002",
    ]
    assert len(replay.outcomes) == 4
    assert len(replay.development_outcomes) == 2
    assert set(replay.requests) == {
        "call-0",
        "call-1",
        "call-2",
        "call-3",
    }
    assert replay.candidate_stats is not None
    assert any("Mock/test archive" in warning for warning in replay.warnings)
    assert replay.warnings[-1] == "This archive is not released."
    pair = replay.outcome_pairs("mock-test-001")
    assert pair["baseline"].candidate_id == "baseline"
    assert pair["optimized"].candidate_id == "candidate-1"
    assert replay.sent_trace("call-0").generation_id == "gen-0"
    baseline_snapshot = replay.programs[
        archive.baseline_manifest.program_id
    ]
    optimized_snapshot = replay.programs[
        archive.optimized_manifest.program_id
    ]
    assert baseline_snapshot.instructions != optimized_snapshot.instructions
    assert baseline_snapshot.schema == optimized_snapshot.schema


def test_replay_retains_error_outcome(tmp_path) -> None:
    archive = build_mock_archive(tmp_path)
    outcomes = _read_outcomes(archive, archive.benchmark_dir)
    outcomes[1] = outcomes[1].model_copy(
        update={
            "status": "error",
            "findings": None,
            "error": "provider timeout",
        }
    )
    _rewrite_outcomes(archive, archive.benchmark_dir, outcomes)
    replay = open_replay_archive(
        archive.benchmark_dir, programs_root=archive.programs_root
    )
    pair = replay.outcome_pairs("mock-test-001")
    assert pair["optimized"].status == "error"
    assert pair["optimized"].error == "provider timeout"


def test_replay_rejects_missing_outcome(tmp_path) -> None:
    archive = build_mock_archive(tmp_path)
    outcomes = _read_outcomes(archive, archive.benchmark_dir)
    _rewrite_outcomes(archive, archive.benchmark_dir, outcomes[:-1])
    with pytest.raises(ArtifactIncompleteError, match="no optimized outcome"):
        open_replay_archive(
            archive.benchmark_dir, programs_root=archive.programs_root
        )


def test_replay_rejects_wrong_input_hash(tmp_path) -> None:
    archive = build_mock_archive(tmp_path)
    outcomes = _read_outcomes(archive, archive.benchmark_dir)
    outcomes[0] = outcomes[0].model_copy(
        update={"input_sha256": "0" * 64}
    )
    _rewrite_outcomes(archive, archive.benchmark_dir, outcomes)
    with pytest.raises(ReplayError, match="input hash"):
        open_replay_archive(
            archive.benchmark_dir, programs_root=archive.programs_root
        )


def test_replay_rejects_wrong_phase_program(tmp_path) -> None:
    archive = build_mock_archive(tmp_path)
    outcomes = _read_outcomes(archive, archive.benchmark_dir)
    outcomes[1] = outcomes[1].model_copy(
        update={"program_id": archive.baseline_manifest.program_id}
    )
    _rewrite_outcomes(archive, archive.benchmark_dir, outcomes)
    with pytest.raises(ReplayError, match="used program"):
        open_replay_archive(
            archive.benchmark_dir, programs_root=archive.programs_root
        )


def test_replay_rejects_missing_request_reference(tmp_path) -> None:
    archive = build_mock_archive(tmp_path)
    requests = [
        _request(f"call-{index}", index) for index in range(3)
    ]
    path = archive.benchmark_dir / RUN_REQUESTS_FILENAME
    atomic_write_jsonl(path, requests)
    _rewrite_manifest(
        archive.benchmark_dir / "manifest.json",
        lambda payload: payload["files"].update(
            {RUN_REQUESTS_FILENAME: sha256_file(path)}
        ),
    )
    with pytest.raises(ReplayError, match="missing requests"):
        open_replay_archive(
            archive.benchmark_dir, programs_root=archive.programs_root
        )


def test_replay_rejects_tampered_snapshot(tmp_path) -> None:
    archive = build_mock_archive(tmp_path)
    cases_path = archive.benchmark_dir / SNAPSHOT_CASES_FILENAME
    lines = cases_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    payload["title"] = "Tampered title"
    lines[0] = json.dumps(payload, ensure_ascii=False)
    atomic_write_text(cases_path, "\n".join(lines) + "\n")
    _rewrite_manifest(
        archive.benchmark_dir / "manifest.json",
        lambda payload: payload["files"].update(
            {SNAPSHOT_CASES_FILENAME: sha256_file(cases_path)}
        ),
    )
    with pytest.raises(ArtifactCorruptError, match="snapshot cases hash"):
        open_replay_archive(
            archive.benchmark_dir, programs_root=archive.programs_root
        )


def test_replay_rejects_missing_linked_run(tmp_path) -> None:
    archive = build_mock_archive(tmp_path)
    import shutil

    shutil.rmtree(archive.optimization_dir)
    with pytest.raises(ArtifactNotFoundError, match="optimization run"):
        open_replay_archive(
            archive.benchmark_dir, programs_root=archive.programs_root
        )


def test_replay_rejects_mismatched_optimization_link(tmp_path) -> None:
    archive = build_mock_archive(tmp_path)
    _rewrite_manifest(
        archive.optimization_dir / "manifest.json",
        lambda payload: payload["links"].update(
            {"baseline_development_run_id": "run-dev-999"}
        ),
    )
    with pytest.raises(ReplayError, match="does not link"):
        open_replay_archive(
            archive.benchmark_dir, programs_root=archive.programs_root
        )


def test_replay_rejects_mock_program_in_live_archive(tmp_path) -> None:
    archive = build_mock_archive(tmp_path)
    for run_dir in (
        archive.development_dir,
        archive.optimization_dir,
        archive.benchmark_dir,
    ):
        _rewrite_manifest(
            run_dir / "manifest.json",
            lambda payload: payload.update({"origin": "live"}),
        )
    with pytest.raises(ReplayError, match="origin does not match"):
        open_replay_archive(
            archive.benchmark_dir, programs_root=archive.programs_root
        )


def test_replay_rejects_shared_request_reference(tmp_path) -> None:
    archive = build_mock_archive(tmp_path)
    outcomes = _read_outcomes(archive, archive.benchmark_dir)
    outcomes[1] = outcomes[1].model_copy(
        update={"request_ids": outcomes[0].request_ids}
    )
    _rewrite_outcomes(archive, archive.benchmark_dir, outcomes)
    with pytest.raises(ReplayError, match="shared by outcomes"):
        open_replay_archive(
            archive.benchmark_dir, programs_root=archive.programs_root
        )


def test_whitespace_only_api_key_is_ignored(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
    assert artifacts._effective_secret_values(("explicit",)) == (
        "explicit",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    assert artifacts._effective_secret_values(("explicit",)) == (
        "explicit",
        SECRET,
    )


def test_configured_api_key_is_scanned_by_default(
    tmp_path, monkeypatch
) -> None:
    archive = build_mock_archive(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    leaky = archive.benchmark.model_copy(
        update={"warnings": [f"leaked {SECRET}"]}
    )
    with pytest.raises(ArtifactSecretError):
        write_run_manifest(
            archive.benchmark_dir, leaky
        )


def test_run_manifest_requires_snapshot_files_and_hashes(
    tmp_path,
) -> None:
    archive = build_mock_archive(tmp_path)
    missing = {
        name: digest
        for name, digest in archive.benchmark.files.items()
        if name != SNAPSHOT_CASES_FILENAME
    }
    bad_manifest = archive.benchmark.model_copy(
        update={"files": missing}
    )
    with pytest.raises(ArtifactError, match="snapshot files"):
        write_run_manifest(archive.benchmark_dir, bad_manifest)

    cases_path = archive.benchmark_dir / SNAPSHOT_CASES_FILENAME
    atomic_write_text(cases_path, "tampered\n")
    with pytest.raises(ArtifactCorruptError):
        write_run_manifest(
            archive.benchmark_dir, archive.benchmark
        )


def test_jsonl_round_trip_preserves_unicode_line_separators(
    tmp_path,
) -> None:
    path = tmp_path / "requests.jsonl"
    request = RecordedRequest.model_validate(_request("call-1", 1))
    request = request.model_copy(
        update={"error": "provider said   stop   now"}
    )
    atomic_write_jsonl(path, [request.model_dump(mode="json")])
    loaded = artifacts.read_recorded_requests(path)
    assert loaded["call-1"].error == "provider said   stop   now"


def test_append_jsonl_is_durable_and_valid(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    artifacts.append_jsonl(path, {"event": "started"})
    artifacts.append_jsonl(path, {"event": "finished"})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["event"] for line in lines] == [
        "started",
        "finished",
    ]
    assert path.stat().st_mode & 0o777 == 0o600


def test_append_jsonl_handles_partial_writes(tmp_path, monkeypatch) -> None:
    path = tmp_path / "events.jsonl"
    original_write = artifacts.os.write

    def partial_write(descriptor, data):
        return original_write(descriptor, bytes(data[:1]))

    monkeypatch.setattr(artifacts.os, "write", partial_write)
    artifacts.append_jsonl(path, {"event": "started"})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "event": "started"
    }


def test_replay_rejects_non_task_request_phase(tmp_path) -> None:
    archive = build_mock_archive(tmp_path)
    path = archive.benchmark_dir / RUN_REQUESTS_FILENAME
    requests = [_request(f"call-{index}", index) for index in range(4)]
    requests[0]["phase"] = "reflection"
    atomic_write_jsonl(path, requests)
    _rewrite_manifest(
        archive.benchmark_dir / "manifest.json",
        lambda payload: payload["files"].update(
            {RUN_REQUESTS_FILENAME: sha256_file(path)}
        ),
    )
    with pytest.raises(ReplayError, match="non-task request"):
        open_replay_archive(
            archive.benchmark_dir, programs_root=archive.programs_root
        )


def test_replay_rejects_optimized_program_without_parent(
    tmp_path,
) -> None:
    archive = build_mock_archive(tmp_path, optimized_parent=False)
    with pytest.raises(ReplayError, match="parent lineage"):
        open_replay_archive(
            archive.benchmark_dir, programs_root=archive.programs_root
        )


def test_replay_rejects_incomplete_benchmark(tmp_path) -> None:
    archive = build_mock_archive(
        tmp_path, benchmark_status="stopped"
    )
    with pytest.raises(ArtifactIncompleteError, match="not completed"):
        open_replay_archive(
            archive.benchmark_dir, programs_root=archive.programs_root
        )


def test_measured_archive_requires_live_released_completed(
    tmp_path,
) -> None:
    archive = build_mock_archive(tmp_path)
    mock_manifest = archive.benchmark
    assert is_measured_archive(mock_manifest) is False
    live = mock_manifest.model_copy(
        update={
            "origin": "live",
            "benchmark": mock_manifest.benchmark.model_copy(
                update={"released": True}
            ),
        }
    )
    assert is_measured_archive(live) is True
    assert is_measured_archive(
        live.model_copy(update={"status": "stopped"})
    ) is False


def test_snapshot_write_and_read_are_exact(tmp_path) -> None:
    archive = build_mock_archive(tmp_path)
    cases, rules = artifacts.read_input_snapshot(
        archive.benchmark_dir, archive.benchmark.snapshot
    )
    assert [case.case_id for case in cases] == [
        "mock-test-001",
        "mock-test-002",
    ]
    assert rules == RULES
    assert cases[0].title == archive.benchmark_cases[0].title


def test_run_manifest_rejects_secret_and_tampered_file(tmp_path) -> None:
    archive = build_mock_archive(tmp_path)
    leaky = archive.benchmark.model_copy(
        update={"warnings": [f"leaked {SECRET}"]}
    )
    with pytest.raises(ArtifactSecretError):
        write_run_manifest(
            archive.benchmark_dir,
            leaky,
            secret_values=[SECRET],
        )
    (archive.development_dir / RUN_OUTCOMES_FILENAME).write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(ArtifactCorruptError, match="hash mismatch"):
        artifacts.verify_file_index(
            archive.development_dir, archive.development.files
        )


def test_run_manifest_schema_version_and_run_id_are_enforced(
    tmp_path,
) -> None:
    archive = build_mock_archive(tmp_path)
    path = archive.benchmark_dir / "manifest.json"
    _rewrite_manifest(
        path, lambda payload: payload.update({"schema_version": "run_v999"})
    )
    with pytest.raises(ArtifactSchemaError):
        artifacts.read_run_manifest(archive.benchmark_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "run_manifest_v1"
    payload["run_id"] = "another-run"
    atomic_write_json(path, payload)
    with pytest.raises(ArtifactCorruptError, match="does not match"):
        artifacts.read_run_manifest(archive.benchmark_dir)
