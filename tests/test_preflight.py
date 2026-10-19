from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import dspy.clients.lm as dspy_lm
import pytest

from dspy_lab.config import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_PREFLIGHT_COST_USD,
    DEFAULT_TASK_MODEL,
    DEFAULT_TASK_PROVIDER,
    ProviderRouteProfile,
    Settings,
    load_settings,
)
from dspy_lab.data import load_cases
from dspy_lab.preflight import (
    PREFLIGHT_REFLECTION_MAX_TOKENS,
    PreflightError,
    PreflightNotAuthorized,
    fetch_endpoint_catalog,
    parse_endpoint_catalog,
    run_preflight,
    select_preflight_cases,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data/v1"
SECRET = "sk-or-v1-preflight-secret"
TASK_MODEL = "google/gemini-2.5-flash-lite"
REFLECTION_MODEL = "deepseek/deepseek-chat-v3.1"


def live_settings(**overrides: str) -> Settings:
    env = {
        "OPENROUTER_API_KEY": SECRET,
        **overrides,
    }
    return load_settings(mode="live", env=env)


def catalog_payload(
    profile: ProviderRouteProfile,
    *,
    prompt: str = "0.0000001",
    completion: str = "0.0000004",
    cache_write: str | None = None,
    overrides: list[dict[str, object]] | None = None,
    context_length: int | None = None,
    max_completion_tokens: int | None = None,
    supported_parameters: list[str] | None = None,
    tag: str | None = None,
    service_tier: str | None = None,
) -> dict[str, object]:
    pricing: dict[str, object] = {
        "prompt": prompt,
        "completion": completion,
    }
    if cache_write is not None:
        pricing["input_cache_write"] = cache_write
    if overrides is not None:
        pricing["overrides"] = overrides
    endpoint = {
        "name": f"Test | {profile.model}",
        "model_id": profile.model,
        "context_length": (
            profile.context_limit
            if context_length is None
            else context_length
        ),
        "max_completion_tokens": (
            profile.endpoint_max_output_tokens
            if max_completion_tokens is None
            else max_completion_tokens
        ),
        "pricing": pricing,
        "provider_name": "Test Provider",
        "tag": profile.provider if tag is None else tag,
        "service_tier": service_tier,
        "quantization": "test",
        "supports_implicit_caching": False,
        "supported_parameters": (
            ["structured_outputs", "response_format", "reasoning", "max_tokens"]
            if supported_parameters is None
            else supported_parameters
        ),
    }
    return {"data": {"id": profile.model, "endpoints": [endpoint]}}


class Reply:
    def __init__(
        self,
        content: str,
        *,
        generation_id: str,
        cost: float | None = 0.00001,
        prompt_tokens: int = 100,
        completion_tokens: int = 5,
        finish_reason: str = "stop",
    ) -> None:
        self.content = content
        self.generation_id = generation_id
        self.cost = cost
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.finish_reason = finish_reason


def response_for(reply: Reply, request: dict) -> SimpleNamespace:
    hidden = (
        {}
        if reply.cost is None
        else {"response_cost": reply.cost}
    )
    return SimpleNamespace(
        id=reply.generation_id,
        model=request.get("model"),
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason=reply.finish_reason,
                message=SimpleNamespace(
                    content=reply.content,
                    tool_calls=None,
                    reasoning_content=None,
                ),
                logprobs=None,
            )
        ],
        usage={
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
            "total_tokens": reply.prompt_tokens + reply.completion_tokens,
        },
        _hidden_params=hidden,
    )


def install_transport(monkeypatch, responses):
    calls: list[dict[str, object]] = []

    def fake_completion(*, request, num_retries=0):
        calls.append(
            {"request": request, "num_retries": num_retries}
        )
        if not responses:
            raise AssertionError("unexpected extra dispatch")
        item = responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return response_for(item, request)

    monkeypatch.setattr(
        dspy_lm, "litellm_completion", fake_completion
    )
    return calls


def gold_response(case_id: str) -> str:
    case = next(
        case
        for case in load_cases(DATA_DIR / "train.jsonl")
        if case.case_id == case_id
    )
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
        },
        ensure_ascii=False,
    )


def test_catalog_parser_accepts_the_frozen_route() -> None:
    settings = live_settings()
    profile = settings.task_route()
    snapshot = parse_endpoint_catalog(
        catalog_payload(profile),
        profile=profile,
        configured_output_tokens=settings.max_output_tokens,
    )
    assert snapshot.provider_tag == DEFAULT_TASK_PROVIDER
    assert snapshot.model == DEFAULT_TASK_MODEL
    assert snapshot.context_length == profile.context_limit
    assert snapshot.input_usd_per_million == profile.input_usd_per_million
    assert snapshot.output_usd_per_million == profile.output_usd_per_million
    assert "structured_outputs" in snapshot.supported_parameters


def test_catalog_parser_uses_maximum_override_price() -> None:
    settings = live_settings(
        REFLECTION_MODEL="anthropic/claude-sonnet-4.5",
        REFLECTION_PROVIDER="claude-on-aws",
    )
    profile = settings.reflection_route()
    payload = catalog_payload(
        profile,
        prompt="0.000003",
        completion="0.000015",
        cache_write="0.00000375",
        overrides=[
            {
                "min_prompt_tokens": 200000,
                "prompt": "0.000006",
                "completion": "0.0000225",
                "input_cache_write": "0.0000075",
                "input_cache_write_1h": "0.000012",
            }
        ],
    )
    snapshot = parse_endpoint_catalog(
        payload,
        profile=profile,
        configured_output_tokens=settings.reflection_max_output_tokens,
    )
    assert snapshot.input_usd_per_million == profile.input_usd_per_million
    assert snapshot.output_usd_per_million == profile.output_usd_per_million
    assert (
        snapshot.cache_write_usd_per_million
        == profile.cache_write_usd_per_million
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda profile, payload: payload["data"]["endpoints"][0].update({"tag": "other/provider"}), "exactly one endpoint"),
        (lambda profile, payload: payload["data"]["endpoints"][0].update({"context_length": 10}), "context"),
        (lambda profile, payload: payload["data"]["endpoints"][0].update({"max_completion_tokens": 1}), "max_completion_tokens"),
        (lambda profile, payload: payload["data"]["endpoints"][0].update({"supported_parameters": ["response_format"]}), "structured_outputs"),
        (lambda profile, payload: payload["data"]["endpoints"][0].update({"service_tier": "flex"}), "service tier"),
        (lambda profile, payload: payload["data"]["endpoints"][0]["pricing"].update({"prompt": "9"}), "exceeds"),
        (lambda profile, payload: payload["data"]["endpoints"][0]["pricing"].update({"completion": "9"}), "exceeds"),
    ],
)
def test_catalog_parser_rejects_route_drift(mutate, message) -> None:
    settings = live_settings()
    profile = settings.task_route()
    payload = catalog_payload(profile)
    mutate(profile, payload)
    with pytest.raises(PreflightError, match=message):
        parse_endpoint_catalog(
            payload,
            profile=profile,
            configured_output_tokens=settings.max_output_tokens,
        )


def test_catalog_parser_rejects_override_above_ceiling() -> None:
    settings = live_settings()
    profile = settings.task_route()
    payload = catalog_payload(
        profile,
        overrides=[{"prompt": "0.0000002"}],
    )
    with pytest.raises(PreflightError, match="exceeds"):
        parse_endpoint_catalog(
            payload,
            profile=profile,
            configured_output_tokens=settings.max_output_tokens,
        )


def test_select_preflight_cases_uses_development_cases_only() -> None:
    selected = select_preflight_cases(
        load_cases(DATA_DIR / "train.jsonl")
    )
    assert [kind for _case, kind in selected] == ["clean", "multi_issue"]
    assert selected[0][0].case_id == "train-001"
    assert selected[1][0].case_id == "train-002"
    assert all(
        case.case_id.startswith("train-")
        for case, _kind in selected
    )


def test_preflight_requires_explicit_confirmation(monkeypatch) -> None:
    settings = live_settings()

    def forbidden(*args, **kwargs):
        raise AssertionError("no catalog or model request is allowed")

    monkeypatch.setattr(
        "dspy_lab.preflight.fetch_endpoint_catalog", forbidden
    )
    install_transport(monkeypatch, [])
    with pytest.raises(PreflightNotAuthorized):
        run_preflight(settings, confirm=False)


def test_preflight_rejects_a_cap_below_the_reservation(
    monkeypatch, tmp_path
) -> None:
    settings = live_settings(MAX_PREFLIGHT_COST_USD="0.01")

    def forbidden(*args, **kwargs):
        raise AssertionError("no catalog or model request is allowed")

    monkeypatch.setattr(
        "dspy_lab.preflight.fetch_endpoint_catalog", forbidden
    )
    calls = install_transport(monkeypatch, [])
    with pytest.raises(PreflightError, match="reservation"):
        run_preflight(
            settings, confirm=True, output_dir=tmp_path
        )
    assert calls == []
    records = list(tmp_path.glob("preflight-*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["calls"] == []


def test_preflight_passes_with_exactly_two_task_and_one_reflection_call(
    monkeypatch, tmp_path
) -> None:
    settings = live_settings()
    task_profile = settings.task_route()
    reflection_profile = settings.reflection_route()

    def fake_catalog(model, **kwargs):
        if model == task_profile.model:
            return catalog_payload(
                task_profile,
                cache_write="0.0000000833333333333333",
            )
        if model == reflection_profile.model:
            return catalog_payload(
                reflection_profile,
                prompt="0.00000025",
                completion="0.00000095",
            )
        raise AssertionError(f"unexpected model {model}")

    monkeypatch.setattr(
        "dspy_lab.preflight.fetch_endpoint_catalog", fake_catalog
    )
    calls = install_transport(
        monkeypatch,
        [
            Reply('{"findings": []}', generation_id="gen-task-1"),
            Reply(gold_response("train-002"), generation_id="gen-task-2"),
            Reply(
                '{"instruction": "Keep each finding atomic."}',
                generation_id="gen-reflection",
            ),
        ],
    )
    result = run_preflight(
        settings, confirm=True, output_dir=tmp_path
    )
    assert result.status == "passed"
    assert len(calls) == 3
    task_records = [
        record
        for record in result.record["calls"]
        if record["phase"] == "task"
    ]
    reflection_records = [
        record
        for record in result.record["calls"]
        if record["phase"] == "reflection"
    ]
    assert len(task_records) == 2
    assert len(reflection_records) == 1
    assert all(
        record["request_kwargs"]["max_tokens"]
        == settings.max_output_tokens
        for record in task_records
    )
    assert all(
        record["request_kwargs"]["response_format"]["type"] == "json_schema"
        for record in task_records
    )
    assert all(
        record["request_kwargs"]["response_format"]
        == {"type": "json_object"}
        for record in reflection_records
    )
    assert (
        reflection_records[0]["request_kwargs"]["max_tokens"]
        == PREFLIGHT_REFLECTION_MAX_TOKENS
    )
    assert all(
        record["request_kwargs"]["extra_body"]["provider"]["only"]
        == [record["provider"]]
        for record in task_records + reflection_records
    )
    assert all(
        record["cost_status"] == "estimated"
        for record in task_records + reflection_records
    )
    assert all(
        record["generation_id"]
        for record in task_records + reflection_records
    )
    assert result.record["ledger"]["admitted_calls"] == 3
    assert result.record["ledger"]["delegated_calls"] == 3
    assert result.record["ledger"]["rejected_calls"] == 0
    assert result.record["ledger"]["stop_reason"] is None
    assert all(
        check["passed"] for check in result.record["checks"]
    )
    strict_checks = [
        check
        for check in result.record["checks"]
        if check["name"].endswith(".strict_schema")
    ]
    assert strict_checks
    assert all(check["passed"] for check in strict_checks)
    assert result.record["routes"]["task"]["snapshot"]["provider_tag"] == (
        task_profile.provider
    )
    assert result.record["settings"]["openrouter_api_key"] == "***"
    text = result.record_path.read_text(encoding="utf-8")
    assert SECRET not in text
    assert stat.S_IMODE(result.record_path.stat().st_mode) == 0o600
    assert list(tmp_path.glob("*.tmp")) == []


def test_preflight_records_failure_and_does_not_retry(
    monkeypatch, tmp_path
) -> None:
    settings = live_settings()
    task_profile = settings.task_route()
    reflection_profile = settings.reflection_route()

    def fake_catalog(model, **kwargs):
        profile = (
            task_profile
            if model == task_profile.model
            else reflection_profile
        )
        return catalog_payload(profile)

    monkeypatch.setattr(
        "dspy_lab.preflight.fetch_endpoint_catalog", fake_catalog
    )
    calls = install_transport(
        monkeypatch,
        [
            Reply('{"findings": []}', generation_id="gen-task-1"),
            RuntimeError("provider exploded"),
        ],
    )
    with pytest.raises(PreflightError, match="provider exploded"):
        run_preflight(
            settings, confirm=True, output_dir=tmp_path
        )
    assert len(calls) == 2
    records = list(tmp_path.glob("preflight-*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert "provider exploded" in record["error"]
    assert len(record["calls"]) == 2
    assert record["ledger"]["admitted_calls"] == 2
    assert record["ledger"]["stop_reason"] is None
    assert record["task_results"] == [
        {
            "case_id": "train-001",
            "kind": "clean",
            "findings_count": 0,
            "codes": [],
        }
    ]


def test_preflight_records_interrupt_as_failure_and_reraises(
    monkeypatch, tmp_path
) -> None:
    settings = live_settings()
    task_profile = settings.task_route()
    reflection_profile = settings.reflection_route()

    def fake_catalog(model, **kwargs):
        profile = (
            task_profile
            if model == task_profile.model
            else reflection_profile
        )
        return catalog_payload(profile)

    monkeypatch.setattr(
        "dspy_lab.preflight.fetch_endpoint_catalog", fake_catalog
    )
    calls = install_transport(
        monkeypatch, [KeyboardInterrupt("operator interrupted")]
    )
    with pytest.raises(KeyboardInterrupt):
        run_preflight(
            settings, confirm=True, output_dir=tmp_path
        )
    assert len(calls) == 1
    record = json.loads(
        next(tmp_path.glob("preflight-*.json")).read_text(encoding="utf-8")
    )
    assert record["status"] == "failed"
    assert "KeyboardInterrupt" in record["error"]
    assert record["ledger"]["admitted_calls"] == 1
    assert record["ledger"]["failed_calls"] == 1


def test_main_reports_missing_configuration_without_traceback(
    monkeypatch, capsys
) -> None:
    from dspy_lab.preflight import main

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    exit_code = main(["--confirm-preflight-budget"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Configuration error" in captured.err
    assert "Traceback" not in captured.err


def test_preflight_records_a_missing_cost_as_failure(
    monkeypatch, tmp_path
) -> None:
    settings = live_settings()
    task_profile = settings.task_route()
    reflection_profile = settings.reflection_route()

    def fake_catalog(model, **kwargs):
        profile = (
            task_profile
            if model == task_profile.model
            else reflection_profile
        )
        return catalog_payload(profile)

    monkeypatch.setattr(
        "dspy_lab.preflight.fetch_endpoint_catalog", fake_catalog
    )
    calls = install_transport(
        monkeypatch,
        [Reply('{"findings": []}', generation_id="gen-task-1", cost=None)],
    )
    with pytest.raises(PreflightError, match="cost"):
        run_preflight(
            settings, confirm=True, output_dir=tmp_path
        )
    assert len(calls) == 1
    record = json.loads(
        next(tmp_path.glob("preflight-*.json")).read_text(encoding="utf-8")
    )
    assert record["status"] == "failed"
    assert record["ledger"]["pending_cost_calls"] == 1


def test_preflight_catalog_failure_makes_no_model_call(
    monkeypatch, tmp_path
) -> None:
    settings = live_settings()
    task_profile = settings.task_route()

    def wrong_catalog(model, **kwargs):
        return catalog_payload(task_profile, tag="wrong/provider")

    monkeypatch.setattr(
        "dspy_lab.preflight.fetch_endpoint_catalog", wrong_catalog
    )
    calls = install_transport(monkeypatch, [])
    with pytest.raises(PreflightError, match="exactly one endpoint"):
        run_preflight(
            settings, confirm=True, output_dir=tmp_path
        )
    assert calls == []
    record = json.loads(
        next(tmp_path.glob("preflight-*.json")).read_text(encoding="utf-8")
    )
    assert record["status"] == "failed"
    assert record["calls"] == []
    assert record["task_results"] == []
    assert record["ledger"]["admitted_calls"] == 0
    assert record["ledger"]["delegated_calls"] == 0


def test_preflight_unknown_route_fails_before_dispatch(
    monkeypatch, tmp_path
) -> None:
    settings = live_settings(
        TASK_MODEL="openai/gpt-4.1-mini",
        TASK_PROVIDER="openai",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("no catalog or model request is allowed")

    monkeypatch.setattr(
        "dspy_lab.preflight.fetch_endpoint_catalog", forbidden
    )
    calls = install_transport(monkeypatch, [])
    with pytest.raises(PreflightError, match="Unsupported OpenRouter route"):
        run_preflight(
            settings, confirm=True, output_dir=tmp_path
        )
    assert calls == []
    record = json.loads(
        next(tmp_path.glob("preflight-*.json")).read_text(encoding="utf-8")
    )
    assert record["status"] == "failed"
    assert record["calls"] == []


def test_preflight_budget_defaults_are_frozen() -> None:
    settings = live_settings()
    assert settings.preflight_budget().phase == "preflight"
    assert settings.preflight_budget().max_calls == 3
    assert (
        settings.preflight_budget().max_cost_usd
        == DEFAULT_MAX_PREFLIGHT_COST_USD
    )
    assert DEFAULT_MAX_OUTPUT_TOKENS == 2048


def test_fetch_endpoint_catalog_uses_public_endpoint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, amount):
            return json.dumps({"data": {"endpoints": []}}).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        return FakeResponse()

    monkeypatch.setattr(
        "dspy_lab.preflight.urllib.request.urlopen", fake_urlopen
    )
    payload = fetch_endpoint_catalog("google/gemini-2.5-flash-lite")
    assert payload["data"]["endpoints"] == []
    assert captured["url"] == (
        "https://openrouter.ai/api/v1/models/"
        "google/gemini-2.5-flash-lite/endpoints"
    )
    assert captured["timeout"] == 30.0
    assert "Authorization" not in captured["headers"]


@pytest.mark.live
def test_live_openrouter_preflight_records_routes() -> None:
    if os.environ.get("RUN_LIVE_PREFLIGHT") != "1":
        pytest.skip(
            "set RUN_LIVE_PREFLIGHT=1 after approving the preflight budget"
        )
    settings = load_settings("live")
    result = run_preflight(settings, confirm=True)
    assert result.status == "passed"
    assert result.record["ledger"]["admitted_calls"] == 3
