"""Guarded DSPy LM, shared budget ledger, and request telemetry.

The guard reserves one application-level attempt and a conservative Decimal
USD amount before delegating to the real LiteLLM-backed DSPy LM. The ledger is
shared by task and reflection LMs and by copied LM objects, so no copy can
reset a budget. Budget stops are execution-control exceptions, not quality
failures, and the ledger remains stopped after a stop.
"""

from __future__ import annotations

import copy
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Literal
from uuid import uuid4

import dspy
from pydantic import BaseModel

from .config import BudgetLimits, to_litellm_model_id, validate_model_id

SchemaMode = Literal["native", "json_object", "none"]
RequestStatus = Literal[
    "admitted",
    "in_flight",
    "succeeded",
    "pending",
    "released",
    "reconciled",
]
CostStatus = Literal["none", "estimated", "actual", "pending"]
StopReason = Literal["call_cap", "cost_cap", "cancelled", "overrun"]

_MILLION = Decimal(1_000_000)
_SECRET_KEY_NAMES = {
    "api_key",
    "authorization",
    "headers",
    "proxy_authorization",
    "openrouter_api_key",
    "password",
    "secret",
}


class BudgetStop(RuntimeError):
    """Base class for execution-control budget stops."""


class BudgetExceeded(BudgetStop):
    """Raised before dispatch when a call or USD cap would be exceeded."""

    def __init__(self, reason: StopReason, snapshot: BudgetSnapshot):
        super().__init__(f"budget stop: {reason}")
        self.reason = reason
        self.snapshot = snapshot


class BudgetCancelled(BudgetStop):
    """Raised when a cancelled ledger refuses a new dispatch."""

    def __init__(self, message: str, snapshot: BudgetSnapshot):
        super().__init__(message)
        self.snapshot = snapshot


class BudgetConfigurationError(ValueError):
    """Raised when a guard/reservation configuration is not enforceable."""


class BudgetLedgerError(RuntimeError):
    """Raised when a ledger transition is invalid."""


def _to_decimal(value: object, *, field_name: str) -> Decimal:
    if isinstance(value, bool):
        raise BudgetConfigurationError(
            f"{field_name} must be a finite non-negative decimal"
        )
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, int):
        result = Decimal(value)
    elif isinstance(value, str):
        try:
            result = Decimal(value)
        except InvalidOperation as exc:
            raise BudgetConfigurationError(
                f"{field_name} must be a finite non-negative decimal"
            ) from exc
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise BudgetConfigurationError(
                f"{field_name} must be a finite non-negative decimal"
            )
        result = Decimal(str(value))
    else:
        raise BudgetConfigurationError(
            f"{field_name} must be a finite non-negative decimal"
        )
    if not result.is_finite() or result < 0:
        raise BudgetConfigurationError(
            f"{field_name} must be a finite non-negative decimal"
        )
    return result


def _require_int(value: object, *, field_name: str, minimum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise BudgetConfigurationError(
            f"{field_name} must be an integer >= {minimum}"
        )
    return value


def _require_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BudgetConfigurationError(
            f"{field_name} must be a non-empty string"
        )
    if any(char.isspace() for char in value):
        raise BudgetConfigurationError(
            f"{field_name} must not contain whitespace"
        )
    return value


def _optional_nonnegative_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _first_present_int(
    usage: Mapping[str, object], keys: tuple[str, ...]
) -> int | None:
    for key in keys:
        value = usage.get(key)
        if value is not None:
            return _optional_nonnegative_int(value)
    return None


def _optional_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return _to_decimal(value, field_name="reported_cost_usd")
    except BudgetConfigurationError:
        return None


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered in _SECRET_KEY_NAMES
        or lowered == "token"
        or lowered.endswith("_api_key")
        or lowered.endswith("_token")
        or lowered.endswith("_secret")
        or lowered.endswith("_password")
        or "secret" in lowered
        or "password" in lowered
    )


def _sanitize_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_value(item)
            for key, item in value.items()
            if not _is_secret_key(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    return value


def sanitize_request_kwargs(
    kwargs: Mapping[str, object],
) -> dict[str, object]:
    """Copy request parameters without API keys, auth headers, or secrets."""
    return {
        str(key): _sanitize_value(value)
        for key, value in kwargs.items()
        if not _is_secret_key(str(key))
    }


def _extract_generation_id(response: object) -> str | None:
    if response is None:
        return None
    for key in ("id", "generation_id"):
        value = (
            response.get(key)
            if isinstance(response, Mapping)
            else getattr(response, key, None)
        )
        if isinstance(value, str) and value:
            return value
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, Mapping):
        for key in ("generation_id", "id"):
            value = hidden.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _extract_finish_reason(response: object) -> str | None:
    if response is None:
        return None
    choices = (
        response.get("choices")
        if isinstance(response, Mapping)
        else getattr(response, "choices", None)
    )
    if not choices:
        return None
    first = choices[0]
    value = (
        first.get("finish_reason")
        if isinstance(first, Mapping)
        else getattr(first, "finish_reason", None)
    )
    return value if isinstance(value, str) else None


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Selected-route prices in USD per million tokens."""

    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    request_usd: Decimal = Decimal("0")
    cache_write_usd_per_million: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        for name in (
            "input_usd_per_million",
            "output_usd_per_million",
            "request_usd",
            "cache_write_usd_per_million",
        ):
            object.__setattr__(
                self,
                name,
                _to_decimal(getattr(self, name), field_name=name),
            )

    @classmethod
    def from_usd_per_token(
        cls,
        input_usd_per_token: object,
        output_usd_per_token: object,
        *,
        request_usd: object = Decimal("0"),
        cache_write_usd_per_token: object = Decimal("0"),
    ) -> ModelPricing:
        return cls(
            input_usd_per_million=_to_decimal(
                input_usd_per_token, field_name="input_usd_per_token"
            )
            * _MILLION,
            output_usd_per_million=_to_decimal(
                output_usd_per_token, field_name="output_usd_per_token"
            )
            * _MILLION,
            request_usd=_to_decimal(request_usd, field_name="request_usd"),
            cache_write_usd_per_million=_to_decimal(
                cache_write_usd_per_token,
                field_name="cache_write_usd_per_token",
            )
            * _MILLION,
        )

    def input_usd_per_token(self) -> Decimal:
        return self.input_usd_per_million / _MILLION

    def output_usd_per_token(self) -> Decimal:
        return self.output_usd_per_million / _MILLION

    def reservation(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        cache_write_tokens: int = 0,
    ) -> Decimal:
        _require_int(input_tokens, field_name="input_tokens", minimum=0)
        _require_int(output_tokens, field_name="output_tokens", minimum=0)
        _require_int(
            cache_write_tokens,
            field_name="cache_write_tokens",
            minimum=0,
        )
        return (
            Decimal(input_tokens) * self.input_usd_per_million / _MILLION
            + Decimal(output_tokens)
            * self.output_usd_per_million
            / _MILLION
            + Decimal(cache_write_tokens)
            * self.cache_write_usd_per_million
            / _MILLION
            + self.request_usd
        )


@dataclass(frozen=True, slots=True)
class ModelRoute:
    """A frozen selected route plus its conservative reservation bounds."""

    model: str
    provider: str
    context_limit: int
    max_output_tokens: int
    pricing: ModelPricing
    expected_schema_mode: SchemaMode = "native"

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", validate_model_id(self.model))
        object.__setattr__(
            self,
            "provider",
            _require_text(self.provider, field_name="provider"),
        )
        object.__setattr__(
            self,
            "context_limit",
            _require_int(
                self.context_limit, field_name="context_limit", minimum=1
            ),
        )
        object.__setattr__(
            self,
            "max_output_tokens",
            _require_int(
                self.max_output_tokens,
                field_name="max_output_tokens",
                minimum=1,
            ),
        )
        if self.max_output_tokens > self.context_limit:
            raise BudgetConfigurationError(
                "max_output_tokens must not exceed context_limit"
            )
        if not isinstance(self.pricing, ModelPricing):
            raise BudgetConfigurationError(
                "pricing must be a ModelPricing instance"
            )
        if self.expected_schema_mode not in (
            "native",
            "json_object",
            "none",
        ):
            raise BudgetConfigurationError(
                "expected_schema_mode must be native, json_object, or none"
            )


@dataclass(frozen=True, slots=True)
class Reservation:
    call_id: str
    reserved_cost_usd: Decimal


@dataclass(frozen=True, slots=True)
class RequestRecord:
    """One admitted attempt; in-band costs are estimates until reconciliation."""

    call_id: str
    sequence: int
    phase: str
    model: str
    provider: str | None
    status: RequestStatus
    cost_status: CostStatus
    reserved_cost_usd: Decimal
    estimated_cost_usd: Decimal | None
    reported_cost_usd: Decimal | None
    input_token_bound: int
    output_token_bound: int
    input_tokens: int | None
    output_tokens: int | None
    generation_id: str | None
    response_model: str | None
    finish_reason: str | None
    error: str | None
    released_reason: str | None
    messages: tuple[Mapping[str, object], ...]
    request_kwargs: Mapping[str, object]
    usage: Mapping[str, object]
    started_at: datetime
    finished_at: datetime | None


@dataclass(slots=True)
class _RequestState:
    call_id: str
    sequence: int
    phase: str
    model: str
    provider: str | None
    status: RequestStatus
    cost_status: CostStatus
    reserved_cost_usd: Decimal
    estimated_cost_usd: Decimal | None
    reported_cost_usd: Decimal | None
    input_token_bound: int
    output_token_bound: int
    input_tokens: int | None
    output_tokens: int | None
    generation_id: str | None
    response_model: str | None
    finish_reason: str | None
    error: str | None
    released_reason: str | None
    messages: list[dict[str, object]]
    request_kwargs: dict[str, object]
    usage: dict[str, object]
    started_at: datetime
    finished_at: datetime | None

    def freeze(self) -> RequestRecord:
        return RequestRecord(
            call_id=self.call_id,
            sequence=self.sequence,
            phase=self.phase,
            model=self.model,
            provider=self.provider,
            status=self.status,
            cost_status=self.cost_status,
            reserved_cost_usd=self.reserved_cost_usd,
            estimated_cost_usd=self.estimated_cost_usd,
            reported_cost_usd=self.reported_cost_usd,
            input_token_bound=self.input_token_bound,
            output_token_bound=self.output_token_bound,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            generation_id=self.generation_id,
            response_model=self.response_model,
            finish_reason=self.finish_reason,
            error=self.error,
            released_reason=self.released_reason,
            messages=tuple(
                copy.deepcopy(message) for message in self.messages
            ),
            request_kwargs=copy.deepcopy(self.request_kwargs),
            usage=copy.deepcopy(self.usage),
            started_at=self.started_at,
            finished_at=self.finished_at,
        )


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """A frozen view of one ledger.

    `failed_calls` counts dispatch exceptions and `pending_cost_calls` counts
    unknown-cost records, so a failed call appears in both. `succeeded_calls`
    counts completions with an in-band estimated cost; `response_calls` counts
    all observed responses, including pending-cost responses. `delegated_calls`
    counts calls that reached the LM boundary; `rejected_calls` counts
    pre-dispatch refusals that have no request record.
    """

    max_calls: int
    max_cost_usd: Decimal
    admitted_calls: int
    delegated_calls: int
    response_calls: int
    succeeded_calls: int
    failed_calls: int
    pending_cost_calls: int
    rejected_calls: int
    cancelled_before_dispatch_calls: int
    estimated_cost_usd: Decimal
    actual_cost_usd: Decimal
    committed_cost_usd: Decimal
    unresolved_cost_usd: Decimal
    total_cost_usd: Decimal
    remaining_calls: int
    remaining_cost_usd: Decimal
    stop_reason: StopReason | None
    cancelled: bool
    exhausted: bool
    overrun: bool


class BudgetLedger:
    """One authoritative, thread-safe budget shared by all LMs in a job."""

    def __init__(
        self,
        *,
        max_calls: int,
        max_cost_usd: object,
    ) -> None:
        self._max_calls = _require_int(
            max_calls, field_name="max_calls", minimum=1
        )
        max_cost = _to_decimal(max_cost_usd, field_name="max_cost_usd")
        if max_cost <= 0:
            raise BudgetConfigurationError(
                "max_cost_usd must be greater than zero"
            )
        self._max_cost = max_cost
        self._lock = threading.Lock()
        self._records: list[_RequestState] = []
        self._next_sequence = 1
        self._admitted = 0
        self._delegated = 0
        self._responses = 0
        self._succeeded = 0
        self._failed = 0
        self._pending = 0
        self._rejected = 0
        self._cancelled_before_dispatch = 0
        self._estimated = Decimal("0")
        self._actual = Decimal("0")
        self._unresolved = Decimal("0")
        self._stop_reason: StopReason | None = None
        self._cancelled = False
        self._overrun = False

    @classmethod
    def from_limits(cls, limits: BudgetLimits) -> BudgetLedger:
        return cls(
            max_calls=limits.max_calls,
            max_cost_usd=limits.max_cost_usd,
        )

    @property
    def max_calls(self) -> int:
        return self._max_calls

    @property
    def max_cost_usd(self) -> Decimal:
        return self._max_cost

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    @property
    def stop_reason(self) -> StopReason | None:
        with self._lock:
            return self._stop_reason

    def admit(
        self,
        *,
        phase: str,
        model: str,
        provider: str | None,
        reservation_usd: object,
        input_token_bound: int,
        output_token_bound: int,
        messages: Sequence[Mapping[str, object]] = (),
        request_kwargs: Mapping[str, object] | None = None,
    ) -> Reservation:
        phase = _require_text(phase, field_name="phase")
        model = _require_text(model, field_name="model")
        if provider is not None:
            provider = _require_text(provider, field_name="provider")
        reservation = _to_decimal(
            reservation_usd, field_name="reservation_usd"
        )
        input_bound = _require_int(
            input_token_bound, field_name="input_token_bound", minimum=0
        )
        output_bound = _require_int(
            output_token_bound,
            field_name="output_token_bound",
            minimum=0,
        )
        with self._lock:
            if self._cancelled:
                self._rejected += 1
                raise BudgetCancelled(
                    "budget is cancelled", self._snapshot_locked()
                )
            if self._stop_reason is not None:
                self._rejected += 1
                raise BudgetExceeded(
                    self._stop_reason, self._snapshot_locked()
                )
            if self._admitted >= self._max_calls:
                self._stop_reason = "call_cap"
                self._rejected += 1
                raise BudgetExceeded(
                    "call_cap", self._snapshot_locked()
                )
            pending_total = (
                self._estimated
                + self._actual
                + self._unresolved
                + reservation
            )
            if pending_total > self._max_cost:
                self._stop_reason = "cost_cap"
                self._rejected += 1
                raise BudgetExceeded(
                    "cost_cap", self._snapshot_locked()
                )
            self._admitted += 1
            self._unresolved += reservation
            state = _RequestState(
                call_id=uuid4().hex,
                sequence=self._next_sequence,
                phase=phase,
                model=model,
                provider=provider,
                status="admitted",
                cost_status="none",
                reserved_cost_usd=reservation,
                estimated_cost_usd=None,
                reported_cost_usd=None,
                input_token_bound=input_bound,
                output_token_bound=output_bound,
                input_tokens=None,
                output_tokens=None,
                generation_id=None,
                response_model=None,
                finish_reason=None,
                error=None,
                released_reason=None,
                messages=[
                    copy.deepcopy(dict(message)) for message in messages
                ],
                request_kwargs=copy.deepcopy(dict(request_kwargs or {})),
                usage={},
                started_at=datetime.now(timezone.utc),
                finished_at=None,
            )
            self._next_sequence += 1
            self._records.append(state)
            return Reservation(state.call_id, reservation)

    def mark_in_flight(self, call_id: str) -> None:
        with self._lock:
            state = self._find_locked(call_id)
            if state.status != "admitted":
                raise BudgetLedgerError(
                    f"{call_id}: cannot dispatch from {state.status}"
                )
            state.status = "in_flight"
            self._delegated += 1

    def succeed(
        self,
        call_id: str,
        *,
        reported_cost_usd: object,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        generation_id: str | None = None,
        response_model: str | None = None,
        finish_reason: str | None = None,
        usage: Mapping[str, object] | None = None,
    ) -> None:
        cost = _optional_decimal(reported_cost_usd)
        with self._lock:
            state = self._find_locked(call_id)
            if state.status not in ("admitted", "in_flight"):
                raise BudgetLedgerError(
                    f"{call_id}: cannot complete from {state.status}"
                )
            self._responses += 1
            self._attach_response_locked(
                state,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_id=generation_id,
                response_model=response_model,
                finish_reason=finish_reason,
                usage=usage,
            )
            if cost is None:
                state.status = "pending"
                state.cost_status = "pending"
                self._pending += 1
                return
            self._unresolved -= state.reserved_cost_usd
            self._estimated += cost
            state.status = "succeeded"
            state.cost_status = "estimated"
            state.estimated_cost_usd = cost
            state.reported_cost_usd = cost
            self._succeeded += 1
            self._check_overrun_locked()

    def fail(self, call_id: str, *, error: str) -> None:
        with self._lock:
            state = self._find_locked(call_id)
            if state.status not in ("admitted", "in_flight"):
                raise BudgetLedgerError(
                    f"{call_id}: cannot fail from {state.status}"
                )
            state.status = "pending"
            state.cost_status = "pending"
            state.error = error
            state.finished_at = datetime.now(timezone.utc)
            self._failed += 1
            self._pending += 1

    def mark_pending(self, call_id: str, *, error: str) -> None:
        with self._lock:
            state = self._find_locked(call_id)
            if state.status not in ("admitted", "in_flight"):
                raise BudgetLedgerError(
                    f"{call_id}: cannot mark pending from {state.status}"
                )
            state.status = "pending"
            state.cost_status = "pending"
            state.error = error
            state.finished_at = datetime.now(timezone.utc)
            self._responses += 1
            self._pending += 1

    def release(self, call_id: str, *, reason: str) -> None:
        with self._lock:
            state = self._find_locked(call_id)
            if state.status != "admitted":
                raise BudgetLedgerError(
                    f"{call_id}: cannot release from {state.status}"
                )
            self._unresolved -= state.reserved_cost_usd
            state.status = "released"
            state.cost_status = "none"
            state.released_reason = reason
            state.finished_at = datetime.now(timezone.utc)
            self._cancelled_before_dispatch += 1

    def reconcile(
        self,
        call_id: str,
        *,
        actual_cost_usd: object,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        generation_id: str | None = None,
        finish_reason: str | None = None,
        usage: Mapping[str, object] | None = None,
    ) -> None:
        cost = _to_decimal(
            actual_cost_usd, field_name="actual_cost_usd"
        )
        with self._lock:
            state = self._find_locked(call_id)
            if state.cost_status == "pending":
                self._unresolved -= state.reserved_cost_usd
                self._pending -= 1
            elif state.cost_status == "estimated":
                if state.reported_cost_usd is not None:
                    self._estimated -= state.reported_cost_usd
            else:
                raise BudgetLedgerError(
                    f"{call_id}: cannot reconcile from "
                    f"{state.cost_status}"
                )
            self._actual += cost
            state.status = "reconciled"
            state.cost_status = "actual"
            state.reported_cost_usd = cost
            self._attach_response_locked(
                state,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_id=generation_id,
                response_model=None,
                finish_reason=finish_reason,
                usage=usage,
            )
            if state.finished_at is None:
                state.finished_at = datetime.now(timezone.utc)
            self._check_overrun_locked()

    def cancel(self, reason: str = "cancelled") -> None:
        _require_text(reason, field_name="reason")
        with self._lock:
            self._cancelled = True
            if self._stop_reason is None:
                self._stop_reason = "cancelled"

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return self._snapshot_locked()

    @property
    def records(self) -> tuple[RequestRecord, ...]:
        with self._lock:
            return tuple(state.freeze() for state in self._records)

    def record(self, call_id: str) -> RequestRecord:
        with self._lock:
            return self._find_locked(call_id).freeze()

    def _find_locked(self, call_id: str) -> _RequestState:
        for state in self._records:
            if state.call_id == call_id:
                return state
        raise BudgetLedgerError(f"unknown call_id {call_id!r}")

    def _attach_response_locked(
        self,
        state: _RequestState,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        generation_id: str | None,
        response_model: str | None,
        finish_reason: str | None,
        usage: Mapping[str, object] | None,
    ) -> None:
        if input_tokens is not None:
            state.input_tokens = _optional_nonnegative_int(input_tokens)
        if output_tokens is not None:
            state.output_tokens = _optional_nonnegative_int(output_tokens)
        if generation_id is not None:
            state.generation_id = str(generation_id)
        if response_model is not None:
            state.response_model = str(response_model)
        if finish_reason is not None:
            state.finish_reason = str(finish_reason)
        if usage is not None:
            state.usage = copy.deepcopy(dict(usage))
        state.finished_at = datetime.now(timezone.utc)

    def _check_overrun_locked(self) -> None:
        total = self._estimated + self._actual + self._unresolved
        if total > self._max_cost:
            self._overrun = True
            if self._stop_reason is None:
                self._stop_reason = "overrun"

    def _snapshot_locked(self) -> BudgetSnapshot:
        committed = self._estimated + self._actual
        total = committed + self._unresolved
        exhausted = (
            self._stop_reason in ("call_cap", "cost_cap", "overrun")
            or self._admitted >= self._max_calls
            or total >= self._max_cost
        )
        return BudgetSnapshot(
            max_calls=self._max_calls,
            max_cost_usd=self._max_cost,
            admitted_calls=self._admitted,
            delegated_calls=self._delegated,
            response_calls=self._responses,
            succeeded_calls=self._succeeded,
            failed_calls=self._failed,
            pending_cost_calls=self._pending,
            rejected_calls=self._rejected,
            cancelled_before_dispatch_calls=self._cancelled_before_dispatch,
            estimated_cost_usd=self._estimated,
            actual_cost_usd=self._actual,
            committed_cost_usd=committed,
            unresolved_cost_usd=self._unresolved,
            total_cost_usd=total,
            remaining_calls=max(0, self._max_calls - self._admitted),
            remaining_cost_usd=max(
                Decimal("0"), self._max_cost - total
            ),
            stop_reason=self._stop_reason,
            cancelled=self._cancelled,
            exhausted=exhausted,
            overrun=self._overrun,
        )

    def __repr__(self) -> str:
        snapshot = self.snapshot()
        return (
            "BudgetLedger("
            f"max_calls={snapshot.max_calls}, "
            f"max_cost_usd={snapshot.max_cost_usd}, "
            f"admitted={snapshot.admitted_calls}, "
            f"committed={snapshot.committed_cost_usd}, "
            f"unresolved={snapshot.unresolved_cost_usd}, "
            f"stop_reason={snapshot.stop_reason!r})"
        )


class GuardedLM(dspy.LM):
    """A genuine LiteLLM-backed DSPy LM with pre-dispatch budget enforcement."""

    def __init__(
        self,
        model: str,
        *,
        ledger: BudgetLedger,
        pricing: ModelPricing,
        context_limit: int,
        max_output_tokens: int,
        expected_schema_mode: SchemaMode = "native",
        provider_label: str | None = None,
        phase: str = "task",
        token_counter: Callable[..., int] | None = None,
        formatting_overhead_tokens: int = 0,
        temperature: float | None = 0.0,
        **kwargs: Any,
    ) -> None:
        if not isinstance(model, str) or not model.startswith("openrouter/"):
            raise BudgetConfigurationError(
                "model must be the LiteLLM-prefixed OpenRouter ID"
            )
        if not isinstance(ledger, BudgetLedger):
            raise BudgetConfigurationError(
                "ledger must be a BudgetLedger instance"
            )
        if not isinstance(pricing, ModelPricing):
            raise BudgetConfigurationError(
                "pricing must be a ModelPricing instance"
            )
        context_limit = _require_int(
            context_limit, field_name="context_limit", minimum=1
        )
        max_output_tokens = _require_int(
            max_output_tokens,
            field_name="max_output_tokens",
            minimum=1,
        )
        if max_output_tokens > context_limit:
            raise BudgetConfigurationError(
                "max_output_tokens must not exceed context_limit"
            )
        if expected_schema_mode not in (
            "native",
            "json_object",
            "none",
        ):
            raise BudgetConfigurationError(
                "expected_schema_mode must be native, json_object, or none"
            )
        phase = _require_text(phase, field_name="phase")
        if provider_label is not None:
            provider_label = _require_text(
                provider_label, field_name="provider_label"
            )
        formatting_overhead_tokens = _require_int(
            formatting_overhead_tokens,
            field_name="formatting_overhead_tokens",
            minimum=0,
        )
        if token_counter is not None and not callable(token_counter):
            raise BudgetConfigurationError(
                "token_counter must be callable or None"
            )
        if "engine" in kwargs:
            engine = kwargs.pop("engine")
            if engine != "litellm":
                raise BudgetConfigurationError(
                    "GuardedLM requires engine='litellm'"
                )
        if "cache" in kwargs:
            cache = kwargs.pop("cache")
            if cache is not False:
                raise BudgetConfigurationError(
                    "GuardedLM requires cache=False"
                )
        if "num_retries" in kwargs:
            retries = kwargs.pop("num_retries")
            if (
                isinstance(retries, bool)
                or not isinstance(retries, int)
                or retries != 0
            ):
                raise BudgetConfigurationError(
                    "GuardedLM requires num_retries=0"
                )
        if "prompt_cache" in kwargs and kwargs["prompt_cache"] is not None:
            raise BudgetConfigurationError(
                "provider prompt caching is not allowed"
            )
        if "max_completion_tokens" in kwargs:
            raise BudgetConfigurationError(
                "GuardedLM supports max_tokens only; "
                "max_completion_tokens is not supported"
            )
        if "max_tokens" in kwargs:
            requested_max = kwargs.pop("max_tokens")
            _require_int(
                requested_max,
                field_name="max_tokens",
                minimum=1,
            )
            if requested_max > max_output_tokens:
                raise BudgetConfigurationError(
                    "max_tokens exceeds max_output_tokens"
                )
            super_max_tokens = requested_max
        else:
            super_max_tokens = max_output_tokens
        super().__init__(
            model=model,
            model_type="chat",
            temperature=temperature,
            max_tokens=super_max_tokens,
            cache=False,
            num_retries=0,
            engine="litellm",
            **kwargs,
        )
        self._ledger = ledger
        self._pricing = pricing
        self._context_limit = context_limit
        self._max_output_tokens = max_output_tokens
        self._expected_schema_mode = expected_schema_mode
        self._provider_label = provider_label
        self._phase = phase
        self._token_counter = token_counter
        self._formatting_overhead_tokens = formatting_overhead_tokens
        self._call_lock = threading.Lock()
        self._secret_values = tuple(
            str(value)
            for key, value in self.kwargs.items()
            if _is_secret_key(str(key)) and value
        )

    @classmethod
    def from_route(
        cls,
        route: ModelRoute,
        *,
        ledger: BudgetLedger,
        phase: str,
        **kwargs: Any,
    ) -> GuardedLM:
        if not isinstance(route, ModelRoute):
            raise BudgetConfigurationError(
                "route must be a ModelRoute instance"
            )
        if "max_tokens" in kwargs or "max_completion_tokens" in kwargs:
            raise BudgetConfigurationError(
                "output limits are fixed by the model route"
            )
        return cls(
            model=to_litellm_model_id(route.model),
            ledger=ledger,
            pricing=route.pricing,
            context_limit=route.context_limit,
            max_output_tokens=route.max_output_tokens,
            expected_schema_mode=route.expected_schema_mode,
            provider_label=route.provider,
            phase=phase,
            **kwargs,
        )

    @property
    def ledger(self) -> BudgetLedger:
        return self._ledger

    @property
    def pricing(self) -> ModelPricing:
        return self._pricing

    @property
    def context_limit(self) -> int:
        return self._context_limit

    @property
    def max_output_tokens(self) -> int:
        return self._max_output_tokens

    @property
    def expected_schema_mode(self) -> SchemaMode:
        return self._expected_schema_mode

    @property
    def phase(self) -> str:
        return self._phase

    def __call__(self, prompt=None, *, messages=None, **kwargs):
        delegated = self._validated_kwargs(kwargs)
        input_bound = self._input_token_bound(
            prompt, messages, delegated
        )
        reservation_usd = self._pricing.reservation(
            input_bound, self._max_output_tokens
        )
        reservation = self._ledger.admit(
            phase=self._phase,
            model=self.model,
            provider=self._provider_label,
            reservation_usd=reservation_usd,
            input_token_bound=input_bound,
            output_token_bound=self._max_output_tokens,
            messages=self._recorded_messages(prompt, messages),
            request_kwargs=self._recorded_request_kwargs(delegated),
        )
        if self._ledger.cancelled:
            self._ledger.release(
                reservation.call_id,
                reason="cancelled before dispatch",
            )
            raise BudgetCancelled(
                "budget cancelled before dispatch",
                self._ledger.snapshot(),
            )
        self._ledger.mark_in_flight(reservation.call_id)
        with self._call_lock:
            before = len(self.history)
            try:
                result = super().__call__(
                    prompt=prompt, messages=messages, **delegated
                )
            except BaseException as exc:
                error = self._sanitize_exception(exc)
                self._redact_exception(exc)
                self._ledger.fail(
                    reservation.call_id,
                    error=error,
                )
                raise
            if len(self.history) <= before:
                self._ledger.mark_pending(
                    reservation.call_id,
                    error="no LM history record was produced",
                )
                return result
            record = self.history[-1]
            usage = record.get("usage") or {}
            self._ledger.succeed(
                reservation.call_id,
                reported_cost_usd=_optional_decimal(record.get("cost")),
                input_tokens=_first_present_int(
                    usage, ("prompt_tokens", "input_tokens")
                ),
                output_tokens=_first_present_int(
                    usage, ("completion_tokens", "output_tokens")
                ),
                generation_id=_extract_generation_id(
                    record.get("response")
                ),
                response_model=record.get("response_model"),
                finish_reason=_extract_finish_reason(
                    record.get("response")
                ),
                usage=usage,
            )
            return result

    def _validated_kwargs(
        self, kwargs: Mapping[str, object]
    ) -> dict[str, object]:
        merged = {**self.kwargs, **kwargs}
        n = merged.get("n", 1)
        if isinstance(n, bool) or not isinstance(n, int) or n != 1:
            raise BudgetConfigurationError(
                "budgeted calls require n=1"
            )
        if kwargs.get("cache", self.cache) is not False:
            raise BudgetConfigurationError(
                "budgeted calls require cache=False"
            )
        retries = kwargs.get("num_retries", self.num_retries)
        if (
            isinstance(retries, bool)
            or not isinstance(retries, int)
            or retries != 0
        ):
            raise BudgetConfigurationError(
                "budgeted calls require num_retries=0"
            )
        if "max_completion_tokens" in merged:
            raise BudgetConfigurationError(
                "GuardedLM supports max_tokens only; "
                "max_completion_tokens is not supported"
            )
        if merged.get("prompt_cache") is not None:
            raise BudgetConfigurationError(
                "provider prompt caching is not allowed"
            )
        if merged.get("cache_control") is not None:
            raise BudgetConfigurationError(
                "provider cache-control parameters are not allowed"
            )
        self._check_schema_mode(merged.get("response_format"))
        requested_max = merged.get("max_tokens")
        if requested_max is not None:
            _require_int(
                requested_max, field_name="max_tokens", minimum=1
            )
            if requested_max > self._max_output_tokens:
                raise BudgetConfigurationError(
                    "max_tokens exceeds max_output_tokens"
                )
        delegated = dict(kwargs)
        delegated.pop("num_retries", None)
        delegated["cache"] = False
        delegated["n"] = 1
        delegated["max_tokens"] = (
            requested_max
            if requested_max is not None
            else self._max_output_tokens
        )
        return delegated

    def _check_schema_mode(self, response_format: object) -> None:
        if self._expected_schema_mode == "none":
            return
        if self._expected_schema_mode == "native":
            if isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                return
            if (
                isinstance(response_format, Mapping)
                and response_format.get("type") == "json_schema"
            ):
                return
            raise BudgetConfigurationError(
                "native schema mode was not preserved before dispatch"
            )
        if response_format != {"type": "json_object"}:
            raise BudgetConfigurationError(
                "JSON-object schema mode was not preserved before dispatch"
            )

    def _input_token_bound(
        self,
        prompt: object,
        messages: object,
        request_kwargs: Mapping[str, object],
    ) -> int:
        if self._token_counter is None:
            return self._context_limit
        count = self._token_counter(
            prompt=prompt,
            messages=messages,
            request_kwargs=request_kwargs,
        )
        count = _require_int(
            count, field_name="token_counter result", minimum=0
        )
        bound = count + self._formatting_overhead_tokens
        if bound > self._context_limit:
            raise BudgetConfigurationError(
                "input token bound exceeds context_limit"
            )
        return bound

    def _recorded_messages(
        self, prompt: object, messages: object
    ) -> tuple[dict[str, object], ...]:
        if messages is not None:
            return tuple(
                dict(message) for message in messages  # type: ignore[arg-type]
            )
        if prompt is not None:
            return ({"role": "user", "content": prompt},)
        return ()

    def _recorded_request_kwargs(
        self, delegated: Mapping[str, object]
    ) -> dict[str, object]:
        return sanitize_request_kwargs(
            {**self.kwargs, **delegated}
        )

    def _sanitize_text(self, text: str) -> str:
        for secret in self._secret_values:
            if secret:
                text = text.replace(secret, "***")
        return text

    def _sanitize_exception(self, exc: BaseException) -> str:
        return self._sanitize_text(f"{type(exc).__name__}: {exc}")

    def _redact_exception(self, exc: BaseException) -> None:
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            message = getattr(current, "message", None)
            if isinstance(message, str):
                current.message = self._sanitize_text(message)
            if current.args:
                current.args = tuple(
                    self._sanitize_text(arg)
                    if isinstance(arg, str)
                    else arg
                    for arg in current.args
                )
            current = current.__cause__ or current.__context__

    def __deepcopy__(self, memo: dict[int, object]) -> GuardedLM:
        clone = self.copy()
        memo[id(self)] = clone
        return clone

    def dump_state(self) -> dict[str, Any]:
        raise TypeError(
            "GuardedLM is runtime-only and cannot be serialized"
        )

    def __repr__(self) -> str:
        return (
            "GuardedLM("
            f"model={self.model!r}, "
            f"phase={self._phase!r}, "
            f"ledger={self._ledger!r})"
        )


__all__ = [
    "BudgetCancelled",
    "BudgetConfigurationError",
    "BudgetExceeded",
    "BudgetLedger",
    "BudgetLedgerError",
    "BudgetSnapshot",
    "BudgetStop",
    "CostStatus",
    "GuardedLM",
    "ModelPricing",
    "ModelRoute",
    "RequestRecord",
    "Reservation",
    "SchemaMode",
    "StopReason",
    "sanitize_request_kwargs",
]
