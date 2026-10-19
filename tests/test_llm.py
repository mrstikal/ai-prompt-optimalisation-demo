from __future__ import annotations

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace

import dspy
import dspy.clients.lm as dspy_lm
import pytest
from dspy.utils.exceptions import LMUnexpectedError

from dspy_lab.config import (
    BudgetLimits,
    InvalidConfigurationError,
    load_settings,
)
from dspy_lab.llm import (
    BudgetCancelled,
    BudgetConfigurationError,
    BudgetExceeded,
    BudgetLedger,
    BudgetLedgerError,
    BudgetStop,
    GuardedLM,
    ModelPricing,
    ModelRoute,
    sanitize_request_kwargs,
)
from dspy_lab.schemas import EditorialQAOutput

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing OpenAI-style message dictionaries.*:DeprecationWarning"
)

MODEL = "openrouter/google/gemini-2.5-flash-lite"
MODEL_BARE = "google/gemini-2.5-flash-lite"
PROVIDER = "google-vertex/eu"
SECRET = "sk-or-v1-super-secret"
CONTEXT_LIMIT = 1000
MAX_OUTPUT = 100
PRICING = ModelPricing(Decimal("0.10"), Decimal("0.40"))
BIG_PRICING = ModelPricing(Decimal("1000"), Decimal("1000"))
RESERVATION = Decimal("0.00014")
BIG_RESERVATION = Decimal("1.1")


@dataclass
class Reply:
    content: str = "ok"
    cost: float | None = 0.00001
    prompt_tokens: int = 10
    completion_tokens: int = 5
    generation_id: str = "gen-1"
    finish_reason: str = "stop"


def make_ledger(
    *,
    max_calls: int = 10,
    max_cost_usd: Decimal = Decimal("10"),
) -> BudgetLedger:
    return BudgetLedger(
        max_calls=max_calls, max_cost_usd=max_cost_usd
    )


def make_lm(
    ledger: BudgetLedger,
    *,
    pricing: ModelPricing = PRICING,
    context_limit: int = CONTEXT_LIMIT,
    max_output_tokens: int = MAX_OUTPUT,
    expected_schema_mode: str = "none",
    phase: str = "task",
    **kwargs,
) -> GuardedLM:
    api_key = kwargs.pop("api_key", SECRET)
    return GuardedLM(
        MODEL,
        ledger=ledger,
        pricing=pricing,
        context_limit=context_limit,
        max_output_tokens=max_output_tokens,
        expected_schema_mode=expected_schema_mode,
        provider_label=PROVIDER,
        phase=phase,
        api_key=api_key,
        temperature=0.0,
        **kwargs,
    )


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
            "total_tokens": reply.prompt_tokens
            + reply.completion_tokens,
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


def test_exact_call_cap_and_no_unreserved_dispatch(monkeypatch) -> None:
    ledger = make_ledger(max_calls=2)
    lm = make_lm(ledger)
    calls = install_transport(
        monkeypatch, [Reply("one"), Reply("two")]
    )
    assert lm("first") == ["one"]
    assert lm("second") == ["two"]
    with pytest.raises(BudgetExceeded) as exc_info:
        lm("third")
    assert exc_info.value.reason == "call_cap"
    assert len(calls) == 2
    snapshot = ledger.snapshot()
    assert snapshot.admitted_calls == 2
    assert snapshot.delegated_calls == 2
    assert snapshot.rejected_calls == 1
    assert snapshot.exhausted is True
    assert snapshot.stop_reason == "call_cap"
    assert len(ledger.records) == 2


def test_cost_boundary_is_inclusive(monkeypatch) -> None:
    ledger = make_ledger(max_calls=2, max_cost_usd=BIG_RESERVATION)
    lm = make_lm(ledger, pricing=BIG_PRICING)
    calls = install_transport(
        monkeypatch, [Reply(cost=1.1), Reply(cost=0.0)]
    )
    assert lm("first") == ["ok"]
    snapshot = ledger.snapshot()
    assert snapshot.committed_cost_usd == BIG_RESERVATION
    assert snapshot.unresolved_cost_usd == Decimal("0")
    with pytest.raises(BudgetExceeded) as exc_info:
        lm("second")
    assert exc_info.value.reason == "cost_cap"
    assert len(calls) == 1


def test_cost_boundary_below_cap_is_rejected(monkeypatch) -> None:
    ledger = make_ledger(
        max_calls=1, max_cost_usd=Decimal("1.099999")
    )
    lm = make_lm(ledger, pricing=BIG_PRICING)
    calls = install_transport(monkeypatch, [])
    with pytest.raises(BudgetExceeded) as exc_info:
        lm("first")
    assert exc_info.value.reason == "cost_cap"
    assert len(calls) == 0
    assert ledger.snapshot().admitted_calls == 0


def test_unknown_cost_retains_reservation_and_reconcile(
    monkeypatch,
) -> None:
    ledger = make_ledger(
        max_calls=5, max_cost_usd=BIG_RESERVATION
    )
    lm = make_lm(ledger, pricing=BIG_PRICING)
    calls = install_transport(monkeypatch, [Reply(cost=None)])
    assert lm("first") == ["ok"]
    record = ledger.records[0]
    assert record.status == "pending"
    assert record.cost_status == "pending"
    snapshot = ledger.snapshot()
    assert snapshot.unresolved_cost_usd == BIG_RESERVATION
    assert snapshot.committed_cost_usd == Decimal("0")
    assert snapshot.pending_cost_calls == 1
    with pytest.raises(BudgetExceeded) as exc_info:
        lm("second")
    assert exc_info.value.reason == "cost_cap"
    assert len(calls) == 1
    ledger.reconcile(
        record.call_id,
        actual_cost_usd=Decimal("0.25"),
        input_tokens=10,
        output_tokens=5,
        generation_id="gen-reconciled",
    )
    record = ledger.record(record.call_id)
    assert record.status == "reconciled"
    assert record.cost_status == "actual"
    assert record.reported_cost_usd == Decimal("0.25")
    assert record.generation_id == "gen-reconciled"
    snapshot = ledger.snapshot()
    assert snapshot.unresolved_cost_usd == Decimal("0")
    assert snapshot.actual_cost_usd == Decimal("0.25")
    assert snapshot.pending_cost_calls == 0
    assert snapshot.stop_reason == "cost_cap"
    with pytest.raises(BudgetExceeded):
        lm("third")
    assert len(calls) == 1


def test_timeout_retains_reservation(monkeypatch) -> None:
    ledger = make_ledger(
        max_calls=2, max_cost_usd=BIG_RESERVATION
    )
    lm = make_lm(ledger, pricing=BIG_PRICING)
    calls = install_transport(
        monkeypatch, [TimeoutError("timed out")]
    )
    with pytest.raises(LMUnexpectedError):
        lm("first")
    record = ledger.records[0]
    assert record.status == "pending"
    assert "timed out" in (record.error or "")
    snapshot = ledger.snapshot()
    assert snapshot.unresolved_cost_usd == BIG_RESERVATION
    assert snapshot.failed_calls == 1
    assert snapshot.response_calls == 0
    with pytest.raises(BudgetExceeded):
        lm("second")
    assert len(calls) == 1


def test_cancellation_prevents_dispatch(monkeypatch) -> None:
    ledger = make_ledger(max_calls=2)
    lm = make_lm(ledger)
    calls = install_transport(monkeypatch, [])
    ledger.cancel("operator")
    with pytest.raises(BudgetCancelled):
        lm("first")
    with pytest.raises(BudgetCancelled):
        lm("second")
    assert len(calls) == 0
    snapshot = ledger.snapshot()
    assert snapshot.cancelled is True
    assert snapshot.stop_reason == "cancelled"
    assert snapshot.rejected_calls == 2


def test_both_lms_share_one_ledger(monkeypatch) -> None:
    ledger = make_ledger(max_calls=2)
    task_lm = make_lm(ledger, phase="task")
    reflection_lm = make_lm(
        ledger,
        phase="reflection",
        pricing=ModelPricing(Decimal("0.25"), Decimal("0.95")),
    )
    calls = install_transport(
        monkeypatch, [Reply("task"), Reply("reflection")]
    )
    assert task_lm("task") == ["task"]
    assert reflection_lm("reflection") == ["reflection"]
    with pytest.raises(BudgetExceeded):
        task_lm("third")
    assert len(calls) == 2
    records = ledger.records
    assert [record.phase for record in records] == [
        "task",
        "reflection",
    ]
    assert ledger.snapshot().admitted_calls == 2


def test_copied_lms_share_the_ledger(monkeypatch) -> None:
    ledger = make_ledger(max_calls=2)
    lm = make_lm(ledger)
    shallow = copy.copy(lm)
    deep = copy.deepcopy(lm)
    method_copy = lm.copy()
    assert shallow.ledger is ledger
    assert deep.ledger is ledger
    assert method_copy.ledger is ledger
    calls = install_transport(
        monkeypatch, [Reply("one"), Reply("two")]
    )
    assert lm("one") == ["one"]
    assert deep("two") == ["two"]
    with pytest.raises(BudgetExceeded):
        method_copy("three")
    assert len(calls) == 2
    assert ledger.snapshot().admitted_calls == 2


def test_concurrent_admission_respects_call_cap() -> None:
    ledger = make_ledger(max_calls=5, max_cost_usd=Decimal("100"))
    barrier = threading.Barrier(20)

    def admit_one(index: int):
        barrier.wait()
        try:
            reservation = ledger.admit(
                phase="test",
                model=MODEL,
                provider=PROVIDER,
                reservation_usd=Decimal("0.01"),
                input_token_bound=1,
                output_token_bound=1,
            )
            return reservation.call_id
        except BudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(admit_one, range(20)))
    assert sum(result is not None for result in results) == 5
    snapshot = ledger.snapshot()
    assert snapshot.admitted_calls == 5
    assert snapshot.rejected_calls == 15
    assert snapshot.stop_reason == "call_cap"


def test_sdk_settings_are_enforced(monkeypatch) -> None:
    ledger = make_ledger(max_calls=5)
    lm = make_lm(ledger)
    assert lm.cache is False
    assert lm.num_retries == 0
    calls = install_transport(monkeypatch, [Reply("ok")])
    with pytest.raises(BudgetConfigurationError):
        lm("x", n=2)
    with pytest.raises(BudgetConfigurationError):
        lm("x", cache=True)
    with pytest.raises(BudgetConfigurationError):
        lm("x", num_retries=1)
    with pytest.raises(BudgetConfigurationError):
        lm("x", n=1.0)
    with pytest.raises(BudgetConfigurationError):
        lm("x", n="1")
    with pytest.raises(BudgetConfigurationError):
        lm("x", n=0)
    with pytest.raises(BudgetConfigurationError):
        lm("x", num_retries=0.0)
    with pytest.raises(BudgetConfigurationError):
        lm("x", max_completion_tokens=50)
    with pytest.raises(BudgetConfigurationError):
        make_lm(ledger, max_completion_tokens=50)
    with pytest.raises(BudgetConfigurationError):
        make_lm(ledger, num_retries=False)
    with pytest.raises(BudgetConfigurationError):
        make_lm(ledger, num_retries=0.0)
    assert len(calls) == 0
    assert lm("x") == ["ok"]
    request = calls[0]["request"]
    assert request["n"] == 1
    assert request["max_tokens"] == MAX_OUTPUT
    assert request["temperature"] == 0.0
    assert calls[0]["num_retries"] == 0
    assert lm.cache is False
    assert lm.num_retries == 0


def test_cache_disabled_calls_transport_twice(monkeypatch) -> None:
    ledger = make_ledger(max_calls=2)
    lm = make_lm(ledger)
    calls = install_transport(
        monkeypatch, [Reply("one"), Reply("two")]
    )
    assert lm("same") == ["one"]
    assert lm("same") == ["two"]
    assert len(calls) == 2
    assert ledger.snapshot().admitted_calls == 2


def test_fallback_style_second_attempt_is_guarded(monkeypatch) -> None:
    ledger = make_ledger(max_calls=1)
    lm = make_lm(ledger)
    calls = install_transport(monkeypatch, [Reply("first")])
    assert lm("first") == ["first"]
    with pytest.raises(BudgetExceeded) as exc_info:
        lm("fallback")
    assert exc_info.value.reason == "call_cap"
    assert len(calls) == 1


def test_schema_mode_downgrade_rejected(monkeypatch) -> None:
    ledger = make_ledger(max_calls=5)
    native_lm = make_lm(ledger, expected_schema_mode="native")
    calls = install_transport(
        monkeypatch, [Reply("native"), Reply("json")]
    )
    with pytest.raises(BudgetConfigurationError):
        native_lm("x")
    with pytest.raises(BudgetConfigurationError):
        native_lm("x", response_format={"type": "json_object"})
    assert len(calls) == 0
    assert ledger.snapshot().admitted_calls == 0
    assert native_lm(
        "x", response_format=EditorialQAOutput
    ) == ["native"]
    assert len(calls) == 1
    json_lm = make_lm(ledger, expected_schema_mode="json_object")
    with pytest.raises(BudgetConfigurationError):
        json_lm("x", response_format=EditorialQAOutput)
    assert json_lm(
        "x", response_format={"type": "json_object"}
    ) == ["json"]
    assert len(calls) == 2


def test_price_unit_conversion_and_worked_quality_check() -> None:
    pricing = ModelPricing.from_usd_per_token(
        Decimal("0.0000001"), Decimal("0.0000004")
    )
    assert pricing.input_usd_per_million == Decimal("0.1")
    assert pricing.output_usd_per_million == Decimal("0.4")
    assert pricing.input_usd_per_token() == Decimal("0.0000001")
    assert pricing.output_usd_per_token() == Decimal("0.0000004")
    assert pricing.reservation(1_000_000, 1_000_000) == Decimal(
        "0.5"
    )
    quality = ModelPricing(Decimal("6.00"), Decimal("22.50"))
    assert quality.reservation(1_000_000, 4_096) == Decimal(
        "6.09216"
    )


def test_invalid_prices_and_routes_are_rejected() -> None:
    with pytest.raises(BudgetConfigurationError):
        ModelPricing(Decimal("-1"), Decimal("1"))
    with pytest.raises(BudgetConfigurationError):
        ModelPricing(Decimal("NaN"), Decimal("1"))
    with pytest.raises(BudgetConfigurationError):
        ModelPricing(Decimal("Infinity"), Decimal("1"))
    with pytest.raises(ValueError):
        ModelRoute(
            model="openrouter/google/gemini-2.5-flash-lite",
            provider=PROVIDER,
            context_limit=1000,
            max_output_tokens=100,
            pricing=PRICING,
        )
    with pytest.raises(BudgetConfigurationError):
        ModelRoute(
            model=MODEL_BARE,
            provider=PROVIDER,
            context_limit=0,
            max_output_tokens=100,
            pricing=PRICING,
        )
    with pytest.raises(BudgetConfigurationError):
        ModelRoute(
            model=MODEL_BARE,
            provider=PROVIDER,
            context_limit=100,
            max_output_tokens=101,
            pricing=PRICING,
        )


def test_secret_redaction(monkeypatch) -> None:
    ledger = make_ledger(max_calls=2)
    lm = make_lm(ledger)
    assert SECRET not in repr(lm)
    assert SECRET not in repr(ledger)
    with pytest.raises(TypeError):
        lm.dump_state()
    install_transport(
        monkeypatch, [RuntimeError(f"bad key {SECRET}")]
    )
    with pytest.raises(LMUnexpectedError) as exc_info:
        lm("x")
    assert SECRET not in str(exc_info.value)
    assert "***" in str(exc_info.value)
    assert SECRET not in str(exc_info.value.__cause__)
    record = ledger.records[0]
    assert SECRET not in (record.error or "")
    assert "***" in (record.error or "")
    assert SECRET not in str(record.request_kwargs)
    assert SECRET not in str(ledger.snapshot())


def test_request_record_captures_messages_and_kwargs(
    monkeypatch,
) -> None:
    ledger = make_ledger(max_calls=2)
    lm = make_lm(ledger)
    install_transport(monkeypatch, [Reply("ok")])
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert lm(
        messages=messages, temperature=0.2, max_tokens=50
    ) == ["ok"]
    record = ledger.records[0]
    assert record.messages == tuple(messages)
    assert record.request_kwargs["temperature"] == 0.2
    assert record.request_kwargs["max_tokens"] == 50
    assert record.request_kwargs["n"] == 1
    assert record.request_kwargs["cache"] is False
    assert "api_key" not in record.request_kwargs
    assert record.input_token_bound == CONTEXT_LIMIT
    assert record.output_token_bound == MAX_OUTPUT
    assert record.reserved_cost_usd == PRICING.reservation(
        CONTEXT_LIMIT, MAX_OUTPUT
    )


def test_request_record_usage_generation_and_cost(
    monkeypatch,
) -> None:
    ledger = make_ledger(max_calls=1)
    lm = make_lm(ledger)
    install_transport(
        monkeypatch,
        [
            Reply(
                content="ok",
                cost=0.000123,
                prompt_tokens=10,
                completion_tokens=5,
                generation_id="gen-abc",
            )
        ],
    )
    assert lm("x") == ["ok"]
    record = ledger.records[0]
    assert record.status == "succeeded"
    assert record.cost_status == "estimated"
    assert record.reported_cost_usd == Decimal("0.000123")
    assert record.input_tokens == 10
    assert record.output_tokens == 5
    assert record.generation_id == "gen-abc"
    assert record.response_model == MODEL
    assert record.usage["prompt_tokens"] == 10
    snapshot = ledger.snapshot()
    assert snapshot.estimated_cost_usd == Decimal("0.000123")
    assert snapshot.actual_cost_usd == Decimal("0")
    assert snapshot.unresolved_cost_usd == Decimal("0")
    assert snapshot.response_calls == 1


def test_overrun_sets_stop(monkeypatch) -> None:
    ledger = make_ledger(
        max_calls=3, max_cost_usd=BIG_RESERVATION
    )
    lm = make_lm(ledger, pricing=BIG_PRICING)
    calls = install_transport(monkeypatch, [Reply(cost=2.0)])
    assert lm("x") == ["ok"]
    snapshot = ledger.snapshot()
    assert snapshot.overrun is True
    assert snapshot.stop_reason == "overrun"
    assert snapshot.committed_cost_usd == Decimal("2.0")
    with pytest.raises(BudgetExceeded) as exc_info:
        lm("y")
    assert exc_info.value.reason == "overrun"
    assert len(calls) == 1


def test_release_before_dispatch() -> None:
    ledger = make_ledger(max_calls=2)
    reservation = ledger.admit(
        phase="test",
        model=MODEL,
        provider=PROVIDER,
        reservation_usd=Decimal("1"),
        input_token_bound=1,
        output_token_bound=1,
    )
    ledger.release(reservation.call_id, reason="cancelled")
    snapshot = ledger.snapshot()
    assert snapshot.unresolved_cost_usd == Decimal("0")
    assert snapshot.cancelled_before_dispatch_calls == 1
    record = ledger.record(reservation.call_id)
    assert record.status == "released"
    assert record.released_reason == "cancelled"
    with pytest.raises(BudgetLedgerError):
        ledger.release(reservation.call_id, reason="again")


def test_no_history_marks_pending(monkeypatch) -> None:
    ledger = make_ledger(max_calls=1)
    lm = make_lm(ledger)
    install_transport(monkeypatch, [Reply("ok")])
    with dspy.context(disable_history=True):
        assert lm("x") == ["ok"]
    record = ledger.records[0]
    assert record.status == "pending"
    assert "no LM history record" in (record.error or "")
    snapshot = ledger.snapshot()
    assert snapshot.unresolved_cost_usd == RESERVATION
    assert snapshot.pending_cost_calls == 1
    assert snapshot.response_calls == 1


def test_nan_cost_is_treated_as_pending(monkeypatch) -> None:
    ledger = make_ledger(max_calls=1)
    lm = make_lm(ledger)
    install_transport(monkeypatch, [Reply(cost=float("nan"))])
    assert lm("x") == ["ok"]
    record = ledger.records[0]
    assert record.status == "pending"
    assert record.cost_status == "pending"
    assert ledger.snapshot().unresolved_cost_usd == RESERVATION


def test_reconcile_estimated_to_actual(monkeypatch) -> None:
    ledger = make_ledger(max_calls=1)
    lm = make_lm(ledger)
    install_transport(monkeypatch, [Reply(cost=0.00001)])
    assert lm("x") == ["ok"]
    record = ledger.records[0]
    assert record.cost_status == "estimated"
    ledger.reconcile(
        record.call_id, actual_cost_usd=Decimal("0.00002")
    )
    record = ledger.record(record.call_id)
    assert record.status == "reconciled"
    assert record.cost_status == "actual"
    assert record.estimated_cost_usd == Decimal("0.00001")
    assert record.input_tokens == 10
    assert record.output_tokens == 5
    snapshot = ledger.snapshot()
    assert snapshot.estimated_cost_usd == Decimal("0")
    assert snapshot.actual_cost_usd == Decimal("0.00002")
    assert snapshot.committed_cost_usd == Decimal("0.00002")
    assert snapshot.response_calls == 1


def test_budget_limits_from_settings() -> None:
    settings = load_settings(
        mode="live",
        env={"OPENROUTER_API_KEY": "sk-or-v1-test"},
    )
    assert settings.optimization_budget() == BudgetLimits(
        "optimization", 160, Decimal("2.00")
    )
    assert settings.baseline_dev_budget() == BudgetLimits(
        "baseline_dev", 10, Decimal("0.50")
    )
    assert settings.benchmark_budget() == BudgetLimits(
        "benchmark", 20, Decimal("0.50")
    )
    assert settings.playground_budget() == BudgetLimits(
        "playground", 2, Decimal("0.10")
    )
    ledger = BudgetLedger.from_limits(
        settings.optimization_budget()
    )
    assert ledger.max_calls == 160
    assert ledger.max_cost_usd == Decimal("2.00")


def test_budget_configuration_validation() -> None:
    with pytest.raises(BudgetConfigurationError):
        BudgetLedger(max_calls=0, max_cost_usd=Decimal("1"))
    with pytest.raises(BudgetConfigurationError):
        BudgetLedger(max_calls=True, max_cost_usd=Decimal("1"))
    with pytest.raises(BudgetConfigurationError):
        BudgetLedger(max_calls=1, max_cost_usd=Decimal("0"))
    with pytest.raises(BudgetConfigurationError):
        BudgetLedger(max_calls=1, max_cost_usd=Decimal("NaN"))
    with pytest.raises(InvalidConfigurationError):
        BudgetLimits("bad phase", 1, Decimal("1"))


def test_model_route_and_from_route(monkeypatch) -> None:
    route = ModelRoute(
        model=MODEL_BARE,
        provider=PROVIDER,
        context_limit=CONTEXT_LIMIT,
        max_output_tokens=MAX_OUTPUT,
        pricing=PRICING,
        expected_schema_mode="native",
    )
    ledger = make_ledger(max_calls=1)
    lm = GuardedLM.from_route(
        route,
        ledger=ledger,
        phase="task",
        api_key=SECRET,
    )
    assert lm.model == MODEL
    assert lm.phase == "task"
    assert lm.ledger is ledger
    assert lm.pricing == PRICING
    assert lm.context_limit == CONTEXT_LIMIT
    assert lm.max_output_tokens == MAX_OUTPUT
    assert lm.expected_schema_mode == "native"
    with pytest.raises(BudgetConfigurationError):
        GuardedLM.from_route(
            route,
            ledger=ledger,
            phase="task",
            max_tokens=10,
        )
    with pytest.raises(BudgetConfigurationError):
        GuardedLM.from_route(
            route,
            ledger=ledger,
            phase="task",
            max_completion_tokens=10,
        )


def test_token_counter_reduces_input_bound(monkeypatch) -> None:
    ledger = make_ledger(max_calls=1)
    lm = make_lm(
        ledger,
        token_counter=lambda **kwargs: 123,
        formatting_overhead_tokens=7,
    )
    install_transport(monkeypatch, [Reply("ok")])
    assert lm("x") == ["ok"]
    record = ledger.records[0]
    assert record.input_token_bound == 130
    assert record.reserved_cost_usd == PRICING.reservation(
        130, MAX_OUTPUT
    )


def test_token_counter_above_context_is_rejected(
    monkeypatch,
) -> None:
    ledger = make_ledger(max_calls=1)
    lm = make_lm(
        ledger,
        token_counter=lambda **kwargs: CONTEXT_LIMIT + 1,
    )
    calls = install_transport(monkeypatch, [])
    with pytest.raises(BudgetConfigurationError):
        lm("x")
    assert len(calls) == 0
    assert ledger.snapshot().admitted_calls == 0


def test_sanitize_request_kwargs() -> None:
    sanitized = sanitize_request_kwargs(
        {
            "api_key": SECRET,
            "authorization": "Bearer secret",
            "headers": {"Authorization": "Bearer secret"},
            "access_token": "token-secret",
            "azure_ad_token": "azure-secret",
            "max_tokens": 10,
            "provider": {"only": ["p"], "api_key": "nested"},
        }
    )
    assert "api_key" not in sanitized
    assert "authorization" not in sanitized
    assert "headers" not in sanitized
    assert "access_token" not in sanitized
    assert "azure_ad_token" not in sanitized
    assert sanitized["max_tokens"] == 10
    assert sanitized["provider"] == {"only": ["p"]}


def test_budget_stop_is_distinct_and_cannot_be_retried(
    monkeypatch,
) -> None:
    ledger = make_ledger(max_calls=1)
    lm = make_lm(ledger)
    calls = install_transport(monkeypatch, [Reply("ok")])
    assert lm("x") == ["ok"]
    with pytest.raises(BudgetStop) as exc_info:
        lm("y")
    assert isinstance(exc_info.value, BudgetExceeded)
    assert exc_info.value.snapshot.stop_reason == "call_cap"
    assert len(calls) == 1
    with pytest.raises(BudgetExceeded):
        lm("z")
    assert len(calls) == 1


def test_stop_cannot_be_reset_by_copy(monkeypatch) -> None:
    ledger = make_ledger(max_calls=1)
    lm = make_lm(ledger)
    calls = install_transport(monkeypatch, [Reply("ok")])
    assert lm("x") == ["ok"]
    clone = copy.deepcopy(lm)
    with pytest.raises(BudgetExceeded):
        clone("y")
    assert len(calls) == 1
    assert ledger.snapshot().admitted_calls == 1


def test_records_are_defensive_copies(monkeypatch) -> None:
    ledger = make_ledger(max_calls=1)
    lm = make_lm(ledger)
    install_transport(monkeypatch, [Reply("ok")])
    assert lm("x") == ["ok"]
    first = ledger.records[0]
    first.request_kwargs["injected"] = True
    assert "injected" not in ledger.records[0].request_kwargs


def test_real_chat_adapter_fallback_is_guarded(monkeypatch) -> None:
    class SimpleSignature(dspy.Signature):
        text: str = dspy.InputField()
        answer: str = dspy.OutputField()

    ledger = make_ledger(max_calls=2)
    lm = make_lm(ledger)
    calls = install_transport(
        monkeypatch,
        [
            Reply("not valid chat format"),
            Reply('{"answer": "ok"}'),
        ],
    )
    adapter = dspy.ChatAdapter(use_native_function_calling=False)
    with dspy.context(lm=lm, adapter=adapter):
        prediction = dspy.Predict(SimpleSignature)(text="x")
    assert prediction.answer == "ok"
    assert len(calls) == 2
    assert len(ledger.records) == 2
    assert ledger.snapshot().admitted_calls == 2
    assert ledger.snapshot().response_calls == 2


def test_real_chat_adapter_fallback_cannot_exceed_cap(
    monkeypatch,
) -> None:
    class SimpleSignature(dspy.Signature):
        text: str = dspy.InputField()
        answer: str = dspy.OutputField()

    ledger = make_ledger(max_calls=1)
    lm = make_lm(ledger)
    calls = install_transport(
        monkeypatch,
        [
            Reply("not valid chat format"),
            Reply('{"answer": "ok"}'),
        ],
    )
    adapter = dspy.ChatAdapter(use_native_function_calling=False)
    with dspy.context(lm=lm, adapter=adapter):
        with pytest.raises(BudgetStop):
            dspy.Predict(SimpleSignature)(text="x")
    assert len(calls) == 1
    assert ledger.snapshot().admitted_calls == 1
    assert ledger.snapshot().stop_reason == "call_cap"


def test_lower_max_tokens_is_dispatched_but_reserved_at_cap(
    monkeypatch,
) -> None:
    ledger = make_ledger(max_calls=1)
    lm = make_lm(ledger, max_tokens=50)
    calls = install_transport(monkeypatch, [Reply("ok")])
    assert lm("x") == ["ok"]
    assert calls[0]["request"]["max_tokens"] == 50
    record = ledger.records[0]
    assert record.output_token_bound == MAX_OUTPUT
    assert record.reserved_cost_usd == PRICING.reservation(
        CONTEXT_LIMIT, MAX_OUTPUT
    )


def test_finish_reason_is_recorded(monkeypatch) -> None:
    ledger = make_ledger(max_calls=1)
    lm = make_lm(ledger)
    install_transport(
        monkeypatch,
        [Reply(content="", finish_reason="length")],
    )
    assert lm("x") == [""]
    record = ledger.records[0]
    assert record.finish_reason == "length"
    assert record.status == "succeeded"


def test_reconcile_can_trigger_overrun(monkeypatch) -> None:
    ledger = make_ledger(
        max_calls=2, max_cost_usd=BIG_RESERVATION
    )
    lm = make_lm(ledger, pricing=BIG_PRICING)
    install_transport(monkeypatch, [Reply(cost=None)])
    assert lm("x") == ["ok"]
    record = ledger.records[0]
    ledger.reconcile(
        record.call_id, actual_cost_usd=Decimal("2.0")
    )
    snapshot = ledger.snapshot()
    assert snapshot.actual_cost_usd == Decimal("2.0")
    assert snapshot.overrun is True
    assert snapshot.stop_reason == "overrun"
    with pytest.raises(BudgetExceeded):
        lm("y")


def test_cache_write_premium_is_reserved() -> None:
    pricing = ModelPricing(
        Decimal("0.10"),
        Decimal("0.40"),
        cache_write_usd_per_million=Decimal("0.20"),
    )
    assert pricing.reservation(
        1_000_000, 0, cache_write_tokens=1_000_000
    ) == Decimal("0.30")
