"""Explicit, budgeted OpenRouter compatibility preflight.

S8 verifies the frozen task and reflection routes with at most two task calls
and one bounded reflection call. It snapshots the public endpoint catalog,
fails closed on route or schema drift, and writes one local JSON record. No
preflight request is made by importing this module or by the default test
suite.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal
from uuid import uuid4

import dspy
from pydantic import BaseModel
from dspy.utils.exceptions import AdapterParseError

from .config import (
    PREFLIGHT_CALLS,
    PREFLIGHT_OUTPUT_DIR,
    PROJECT_ROOT,
    ConfigurationError,
    ProviderRouteProfile,
    Settings,
    load_settings,
)
from .data import load_cases
from .llm import (
    BudgetLedger,
    ModelRoute,
    RequestRecord,
    build_guarded_lm,
    model_route_for,
    openrouter_provider_options,
)
from .program import (
    EditorialQA,
    StrictJSONAdapter,
    case_inputs,
    load_editorial_rules,
)
from .schemas import CaseRecord, Finding

PREFLIGHT_SCHEMA_VERSION = "preflight_v1"
PREFLIGHT_REFLECTION_MAX_TOKENS = 256
CATALOG_URL_TEMPLATE = "https://openrouter.ai/api/v1/models/{model}/endpoints"
CATALOG_TIMEOUT_SECONDS = 30.0
CATALOG_MAX_BYTES = 2_000_000
_MILLION = Decimal(1_000_000)


class PreflightError(RuntimeError):
    """Raised when an authorized preflight cannot prove compatibility."""


class PreflightNotAuthorized(PreflightError):
    """Raised when a preflight is attempted without explicit confirmation."""


class PreflightConfigurationError(PreflightError):
    """Raised when a preflight route or budget is not enforceable."""


@dataclass(frozen=True, slots=True)
class EndpointSnapshot:
    """The selected public endpoint metadata observed at preflight time."""

    model: str
    provider_tag: str
    provider_name: str
    context_length: int
    max_completion_tokens: int
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    cache_write_usd_per_million: Decimal
    supported_parameters: tuple[str, ...]
    service_tier: str | None
    quantization: str | None
    supports_implicit_caching: bool
    raw: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "provider_tag": self.provider_tag,
            "provider_name": self.provider_name,
            "context_length": self.context_length,
            "max_completion_tokens": self.max_completion_tokens,
            "input_usd_per_million": str(self.input_usd_per_million),
            "output_usd_per_million": str(self.output_usd_per_million),
            "cache_write_usd_per_million": str(
                self.cache_write_usd_per_million
            ),
            "supported_parameters": list(self.supported_parameters),
            "service_tier": self.service_tier,
            "quantization": self.quantization,
            "supports_implicit_caching": self.supports_implicit_caching,
            "raw": _json_safe(self.raw),
        }


@dataclass(frozen=True, slots=True)
class PreflightResult:
    run_id: str
    status: Literal["passed", "failed"]
    record_path: Path
    record: Mapping[str, object]


class PreflightReflectionSignature(dspy.Signature):
    brief: str = dspy.InputField(
        desc="Short description of the editorial instruction to propose."
    )
    instruction: str = dspy.OutputField(
        desc="One concise instruction for a future reflection request."
    )


def fetch_endpoint_catalog(
    model: str, *, timeout: float = CATALOG_TIMEOUT_SECONDS
) -> Mapping[str, object]:
    """Fetch one public OpenRouter endpoint catalog without an API key."""
    url = CATALOG_URL_TEMPLATE.format(
        model=urllib.parse.quote(model, safe="/")
    )
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "dspy-prompt-optimization-lab/0.1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", None)
            if status != 200:
                raise PreflightError(
                    f"OpenRouter catalog returned HTTP {status} for {url}"
                )
            body = response.read(CATALOG_MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise PreflightError(
            f"OpenRouter catalog returned HTTP {exc.code} for {url}"
        ) from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise PreflightError(
            f"OpenRouter catalog request failed for {url}: "
            f"{type(exc).__name__}"
        ) from exc
    if len(body) > CATALOG_MAX_BYTES:
        raise PreflightError("OpenRouter catalog response is too large")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(
            "OpenRouter catalog response is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise PreflightError("OpenRouter catalog response is not an object")
    return payload


def parse_endpoint_catalog(
    payload: Mapping[str, object],
    *,
    profile: ProviderRouteProfile,
    configured_output_tokens: int,
) -> EndpointSnapshot:
    """Select the frozen provider tag and validate its observed metadata."""
    if not isinstance(payload, Mapping):
        raise PreflightError("catalog payload must be an object")
    data = payload.get("data", payload)
    if not isinstance(data, Mapping):
        raise PreflightError("catalog data must be an object")
    endpoints = data.get("endpoints")
    if not isinstance(endpoints, list):
        raise PreflightError("catalog data.endpoints must be a list")
    matches = [
        endpoint
        for endpoint in endpoints
        if isinstance(endpoint, Mapping)
        and endpoint.get("tag") == profile.provider
    ]
    if len(matches) != 1:
        raise PreflightError(
            f"expected exactly one endpoint tagged {profile.provider!r} for "
            f"{profile.model!r}; found {len(matches)}"
        )
    endpoint = matches[0]
    if endpoint.get("model_id") not in (None, profile.model):
        raise PreflightError(
            "catalog endpoint model_id does not match the configured model"
        )
    provider_name = endpoint.get("provider_name")
    if not isinstance(provider_name, str) or not provider_name.strip():
        raise PreflightError("catalog endpoint provider_name is missing")
    context_length = _require_catalog_int(
        endpoint.get("context_length"), "context_length"
    )
    max_completion_tokens = _require_catalog_int(
        endpoint.get("max_completion_tokens"), "max_completion_tokens"
    )
    if context_length < profile.context_limit:
        raise PreflightError(
            f"catalog context {context_length} is below the frozen "
            f"reservation bound {profile.context_limit}"
        )
    if max_completion_tokens < configured_output_tokens:
        raise PreflightError(
            f"catalog max_completion_tokens {max_completion_tokens} is below "
            f"the configured output cap {configured_output_tokens}"
        )
    supported = endpoint.get("supported_parameters")
    if not isinstance(supported, list) or not all(
        isinstance(item, str) for item in supported
    ):
        raise PreflightError(
            "catalog supported_parameters must be a string list"
        )
    supported_parameters = tuple(supported)
    if "response_format" not in supported_parameters:
        raise PreflightError(
            "catalog endpoint does not advertise response_format"
        )
    if profile.expected_schema_mode == "native" and (
        "structured_outputs" not in supported_parameters
    ):
        raise PreflightError(
            "catalog endpoint does not advertise structured_outputs for "
            "native schema mode"
        )
    pricing = endpoint.get("pricing")
    if not isinstance(pricing, Mapping):
        raise PreflightError("catalog endpoint pricing is missing")
    input_price = _catalog_price(pricing, "prompt")
    output_price = _catalog_price(pricing, "completion")
    if input_price is None or output_price is None:
        raise PreflightError(
            "catalog endpoint must report prompt and completion prices"
        )
    cache_write = _max_catalog_rate(
        pricing,
        ("input_cache_write", "input_cache_write_1h"),
    )
    overrides = pricing.get("overrides")
    if overrides is not None:
        if not isinstance(overrides, list):
            raise PreflightError("catalog pricing.overrides must be a list")
        for override in overrides:
            if not isinstance(override, Mapping):
                raise PreflightError(
                    "catalog pricing override must be an object"
                )
            override_input = _catalog_price(override, "prompt")
            override_output = _catalog_price(override, "completion")
            if override_input is not None:
                input_price = max(input_price, override_input)
            if override_output is not None:
                output_price = max(output_price, override_output)
            cache_write = max(
                cache_write,
                _max_catalog_rate(
                    override,
                    ("input_cache_write", "input_cache_write_1h"),
                ),
            )
    if input_price > profile.input_usd_per_million:
        raise PreflightError(
            "catalog input price (including overrides) exceeds the frozen "
            "ceiling"
        )
    if output_price > profile.output_usd_per_million:
        raise PreflightError(
            "catalog output price (including overrides) exceeds the frozen "
            "ceiling"
        )
    if cache_write > profile.cache_write_usd_per_million:
        raise PreflightError(
            f"catalog cache-write price {cache_write} exceeds the frozen "
            f"ceiling {profile.cache_write_usd_per_million}"
        )
    service_tier = endpoint.get("service_tier")
    if service_tier is not None:
        raise PreflightError(
            "catalog endpoint uses a service tier; Flex/Priority routes are "
            "not allowed"
        )
    return EndpointSnapshot(
        model=profile.model,
        provider_tag=profile.provider,
        provider_name=provider_name,
        context_length=context_length,
        max_completion_tokens=max_completion_tokens,
        input_usd_per_million=input_price,
        output_usd_per_million=output_price,
        cache_write_usd_per_million=cache_write,
        supported_parameters=supported_parameters,
        service_tier=service_tier,
        quantization=(
            endpoint.get("quantization")
            if isinstance(endpoint.get("quantization"), str)
            else None
        ),
        supports_implicit_caching=bool(
            endpoint.get("supports_implicit_caching", False)
        ),
        raw=dict(endpoint),
    )


def select_preflight_cases(
    cases: Sequence[CaseRecord],
) -> tuple[tuple[CaseRecord, str], tuple[CaseRecord, str]]:
    """Choose one clean and one multi-issue development case deterministically."""
    clean = next(
        (case for case in cases if not case.gold_findings), None
    )
    multi = next(
        (case for case in cases if len(case.gold_findings) >= 2), None
    )
    if clean is None or multi is None:
        raise PreflightConfigurationError(
            "the training set must contain one clean and one multi-issue case"
        )
    if clean.case_id == multi.case_id:
        raise PreflightConfigurationError(
            "clean and multi-issue preflight cases must be distinct"
        )
    return (clean, "clean"), (multi, "multi_issue")


def run_preflight(
    settings: Settings,
    *,
    confirm: bool = False,
    output_dir: Path | None = None,
) -> PreflightResult:
    """Run one explicitly authorized, fixed-cost compatibility preflight."""
    if not confirm:
        raise PreflightNotAuthorized(
            "preflight requires an explicit budget confirmation"
        )
    if settings.mode != "live":
        raise PreflightConfigurationError(
            "preflight requires Live settings"
        )
    if not settings.openrouter_api_key:
        raise PreflightConfigurationError(
            "preflight requires OPENROUTER_API_KEY"
        )
    destination = Path(output_dir) if output_dir else PREFLIGHT_OUTPUT_DIR
    started_at = _utcnow()
    run_id = (
        f"preflight-{started_at.strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid4().hex[:8]}"
    )
    record_path = destination / f"{run_id}.json"
    checks: list[dict[str, object]] = []
    warnings: list[str] = []
    record: dict[str, object] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "run_id": run_id,
        "mode": "live",
        "status": "running",
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "settings": settings.redacted(),
        "budget": {
            "phase": "preflight",
            "max_calls": PREFLIGHT_CALLS,
            "max_cost_usd": str(settings.max_preflight_cost_usd),
        },
        "reservation_total_usd": None,
        "routes": {"task": None, "reflection": None},
        "cases": [],
        "task_results": [],
        "checks": checks,
        "warnings": warnings,
        "calls": [],
        "ledger": None,
        "error": None,
    }
    ledger: BudgetLedger | None = None
    failure: BaseException | None = None

    def add_check(name: str, passed: bool, detail: object) -> None:
        check = {
            "name": name,
            "passed": bool(passed),
            "detail": str(detail),
        }
        checks.append(check)
        if not passed:
            raise PreflightError(
                f"preflight check failed: {name}: {detail}"
            )

    try:
        task_route = model_route_for(settings, role="task")
        reflection_route = model_route_for(settings, role="reflection")
        budget = settings.preflight_budget()
        ledger = BudgetLedger.from_limits(budget)
        required_reservation = (
            2
            * task_route.pricing.reservation(
                task_route.context_limit, task_route.max_output_tokens
            )
            + reflection_route.pricing.reservation(
                reflection_route.context_limit,
                reflection_route.max_output_tokens,
            )
        )
        record["reservation_total_usd"] = str(required_reservation)
        if required_reservation > budget.max_cost_usd:
            raise PreflightConfigurationError(
                "conservative preflight reservation "
                f"{required_reservation} USD exceeds MAX_PREFLIGHT_COST_USD "
                f"{budget.max_cost_usd}; raise the confirmed cap deliberately"
            )

        task_payload = fetch_endpoint_catalog(task_route.model)
        task_snapshot = parse_endpoint_catalog(
            task_payload,
            profile=settings.task_route(),
            configured_output_tokens=task_route.max_output_tokens,
        )
        add_check(
            "catalog.task",
            True,
            f"{task_snapshot.provider_tag} "
            f"{task_snapshot.input_usd_per_million}/"
            f"{task_snapshot.output_usd_per_million} USD per 1M",
        )
        reflection_payload = fetch_endpoint_catalog(reflection_route.model)
        reflection_snapshot = parse_endpoint_catalog(
            reflection_payload,
            profile=settings.reflection_route(),
            configured_output_tokens=reflection_route.max_output_tokens,
        )
        add_check(
            "catalog.reflection",
            True,
            f"{reflection_snapshot.provider_tag} "
            f"{reflection_snapshot.input_usd_per_million}/"
            f"{reflection_snapshot.output_usd_per_million} USD per 1M",
        )
        routes = record["routes"]
        assert isinstance(routes, dict)
        routes["task"] = {
            "profile": _profile_dict(settings.task_route()),
            "snapshot": task_snapshot.to_dict(),
        }
        routes["reflection"] = {
            "profile": _profile_dict(settings.reflection_route()),
            "snapshot": reflection_snapshot.to_dict(),
        }

        add_check(
            "task.schema_mode",
            task_route.expected_schema_mode == "native",
            task_route.expected_schema_mode,
        )
        add_check(
            "reflection.schema_mode",
            reflection_route.expected_schema_mode
            in ("native", "json_object"),
            reflection_route.expected_schema_mode,
        )

        cases = load_cases(PROJECT_ROOT / "data/v1/train.jsonl")
        selected = select_preflight_cases(cases)
        rules = load_editorial_rules()
        record["cases"] = [
            {
                "case_id": case.case_id,
                "kind": kind,
                "gold_findings": len(case.gold_findings),
            }
            for case, kind in selected
        ]

        task_lm = build_guarded_lm(
            settings, role="task", ledger=ledger
        )
        reflection_lm = build_guarded_lm(
            settings, role="reflection", ledger=ledger
        )
        add_check(
            "ledger.shared",
            task_lm.ledger is ledger and reflection_lm.ledger is ledger,
            "task and reflection LMs share one ledger",
        )

        program = EditorialQA()
        task_results: list[dict[str, object]] = []
        with dspy.context(lm=task_lm, adapter=StrictJSONAdapter()):
            for case, kind in selected:
                before = len(ledger.records)
                try:
                    prediction = program(**case_inputs(case, rules))
                except AdapterParseError as exc:
                    raise PreflightError(
                        f"{case.case_id}: strict task parse failed: {exc}"
                    ) from exc
                new_records = ledger.records[before:]
                add_check(
                    f"{case.case_id}.exactly_one_request",
                    len(new_records) == 1,
                    f"{len(new_records)} request record(s)",
                )
                request_record = new_records[0]
                for name, passed, detail in _request_checks(
                    request_record,
                    phase="task",
                    route=task_route,
                    max_tokens=task_route.max_output_tokens,
                ):
                    add_check(f"{case.case_id}.{name}", passed, detail)
                findings = prediction.findings
                add_check(
                    f"{case.case_id}.typed_findings",
                    isinstance(findings, list)
                    and all(
                        isinstance(finding, Finding)
                        for finding in findings
                    ),
                    f"{len(findings) if isinstance(findings, list) else '?'} "
                    "typed finding(s)",
                )
                leaks = _gold_leaks(request_record, case)
                add_check(
                    f"{case.case_id}.no_gold_leakage",
                    not leaks,
                    "no forbidden metadata"
                    if not leaks
                    else f"found {leaks}",
                )
                if kind == "multi_issue" and not findings:
                    warnings.append(
                        f"{case.case_id}: multi-issue input returned no "
                        "findings; schema is valid but nested output was empty"
                    )
                task_results.append(
                    {
                        "case_id": case.case_id,
                        "kind": kind,
                        "findings_count": len(findings),
                        "codes": [finding.code for finding in findings],
                    }
                )
                record["task_results"] = list(task_results)

        before = len(ledger.records)
        with dspy.context(lm=reflection_lm, adapter=dspy.JSONAdapter()):
            reflection_prediction = dspy.Predict(
                PreflightReflectionSignature
            )(
                brief=(
                    "Propose one short instruction for reviewing Czech "
                    "precious-metals articles against supplied facts."
                ),
                config={
                    "max_tokens": PREFLIGHT_REFLECTION_MAX_TOKENS,
                },
            )
        new_records = ledger.records[before:]
        add_check(
            "reflection.exactly_one_request",
            len(new_records) == 1,
            f"{len(new_records)} request record(s)",
        )
        reflection_record = new_records[0]
        for name, passed, detail in _request_checks(
            reflection_record,
            phase="reflection",
            route=reflection_route,
            max_tokens=PREFLIGHT_REFLECTION_MAX_TOKENS,
        ):
            add_check(f"reflection.{name}", passed, detail)
        instruction = reflection_prediction.instruction
        add_check(
            "reflection.instruction",
            isinstance(instruction, str) and bool(instruction.strip()),
            "non-empty instruction"
            if isinstance(instruction, str)
            else type(instruction).__name__,
        )

        snapshot = ledger.snapshot()
        add_check(
            "ledger.admitted_calls",
            snapshot.admitted_calls == PREFLIGHT_CALLS,
            f"{snapshot.admitted_calls} of {PREFLIGHT_CALLS}",
        )
        add_check(
            "ledger.delegated_calls",
            snapshot.delegated_calls == PREFLIGHT_CALLS,
            f"{snapshot.delegated_calls} of {PREFLIGHT_CALLS}",
        )
        add_check(
            "ledger.rejected_calls",
            snapshot.rejected_calls == 0,
            f"{snapshot.rejected_calls} rejected",
        )
        add_check(
            "ledger.stop_reason",
            snapshot.stop_reason is None,
            repr(snapshot.stop_reason),
        )
        add_check(
            "ledger.phases",
            [item.phase for item in ledger.records]
            == ["task", "task", "reflection"],
            [item.phase for item in ledger.records],
        )
    except BaseException as exc:  # noqa: BLE001 - recorded and re-raised
        failure = exc
    finally:
        finished_at = _utcnow()
        record["finished_at"] = finished_at.isoformat()
        record["status"] = "failed" if failure else "passed"
        if failure is not None:
            record["error"] = _redact_text(
                f"{type(failure).__name__}: {failure}",
                settings.openrouter_api_key,
            )
        if ledger is not None:
            record["calls"] = [
                _request_record_dict(item) for item in ledger.records
            ]
            record["ledger"] = _json_safe(asdict(ledger.snapshot()))
        safe_record = _json_safe(record)
        try:
            _write_record(record_path, safe_record)
        except OSError as exc:
            persistence_error = PreflightError(
                f"could not persist preflight record: {exc}"
            )
            if failure is None:
                failure = persistence_error
            elif isinstance(failure, (KeyboardInterrupt, SystemExit)):
                failure.add_note(str(persistence_error))
            else:
                failure = PreflightError(
                    f"{failure}; additionally {persistence_error}"
                )
        if isinstance(safe_record, Mapping):
            record = dict(safe_record)

    if failure is not None:
        if isinstance(failure, (KeyboardInterrupt, SystemExit)):
            raise failure
        if isinstance(failure, PreflightError):
            raise failure
        raise PreflightError(
            f"preflight failed: {type(failure).__name__}: {failure}"
        ) from failure
    return PreflightResult(
        run_id=run_id,
        status="passed",
        record_path=record_path,
        record=record,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one explicitly confirmed OpenRouter preflight."
    )
    parser.add_argument(
        "--confirm-preflight-budget",
        action="store_true",
        help="confirm the separate preflight USD cap",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="override the local preflight record directory",
    )
    args = parser.parse_args(argv)
    if not args.confirm_preflight_budget:
        parser.error(
            "--confirm-preflight-budget is required; no request was made"
        )
    try:
        settings = load_settings("live")
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    try:
        result = run_preflight(
            settings,
            confirm=True,
            output_dir=args.output_dir,
        )
    except PreflightError as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "status": result.status,
                "record_path": str(result.record_path),
                "admitted_calls": result.record["ledger"]["admitted_calls"],
                "total_cost_usd": result.record["ledger"]["total_cost_usd"],
                "unresolved_cost_usd": result.record["ledger"][
                    "unresolved_cost_usd"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _request_checks(
    record: RequestRecord,
    *,
    phase: str,
    route: ModelRoute,
    max_tokens: int,
) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, passed: bool, detail: object) -> None:
        checks.append((name, bool(passed), str(detail)))

    add("phase", record.phase == phase, record.phase)
    add(
        "status",
        record.status in ("succeeded", "pending") and record.error is None,
        f"{record.status}: {record.error}" if record.error else record.status,
    )
    add(
        "request_messages",
        bool(record.messages),
        f"{len(record.messages)} message(s)",
    )
    add(
        "max_tokens",
        record.request_kwargs.get("max_tokens") == max_tokens,
        record.request_kwargs.get("max_tokens"),
    )
    add(
        "schema_mode",
        _schema_mode(record) == route.expected_schema_mode,
        _schema_mode(record),
    )
    if route.expected_schema_mode == "native":
        response_format = record.request_kwargs.get("response_format")
        wire = (
            response_format.get("wire")
            if isinstance(response_format, Mapping)
            else None
        )
        wire_schema = (
            wire.get("json_schema")
            if isinstance(wire, Mapping)
            else None
        )
        add(
            "strict_schema",
            isinstance(response_format, Mapping)
            and response_format.get("strict") is True
            and isinstance(wire_schema, Mapping)
            and wire_schema.get("strict") is True,
            "strict=True on schema and wire body",
        )
    add(
        "provider_routing",
        record.request_kwargs.get("extra_body")
        == openrouter_provider_options(route.provider),
        record.request_kwargs.get("extra_body"),
    )
    add(
        "generation_id",
        isinstance(record.generation_id, str)
        and bool(record.generation_id),
        record.generation_id,
    )
    add(
        "usage",
        record.input_tokens is not None
        and record.output_tokens is not None,
        f"input={record.input_tokens}, output={record.output_tokens}",
    )
    add(
        "output_within_cap",
        record.output_tokens is not None
        and record.output_tokens <= max_tokens,
        f"{record.output_tokens} <= {max_tokens}",
    )
    add(
        "finish_reason",
        record.finish_reason == "stop",
        record.finish_reason,
    )
    add(
        "response_model",
        isinstance(record.response_model, str)
        and route.model in record.response_model,
        record.response_model,
    )
    add(
        "cost",
        record.cost_status in ("estimated", "actual")
        and record.reported_cost_usd is not None,
        f"{record.cost_status}: {record.reported_cost_usd}",
    )
    add(
        "reasoning_disabled",
        _reasoning_tokens(record.usage) == 0,
        _reasoning_tokens(record.usage),
    )
    return checks


def _schema_mode(record: RequestRecord) -> str:
    response_format = record.request_kwargs.get("response_format")
    if isinstance(response_format, Mapping):
        if response_format.get("type") == "json_schema":
            return "native"
        if response_format == {"type": "json_object"}:
            return "json_object"
    if response_format is None:
        return "none"
    return "unknown"


def _reasoning_tokens(usage: Mapping[str, object]) -> int:
    candidates: list[object] = [usage.get("reasoning_tokens")]
    details = usage.get("completion_tokens_details")
    if isinstance(details, Mapping):
        candidates.append(details.get("reasoning_tokens"))
    for value in candidates:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return -1
        return value
    return 0


def _gold_leaks(record: RequestRecord, case: CaseRecord) -> list[str]:
    text = json.dumps(
        [dict(message) for message in record.messages],
        ensure_ascii=False,
    )
    forbidden = [
        "gold_findings",
        "issue_id",
        "reviewer_id",
        "reviewer_decision",
    ]
    forbidden.extend(
        finding.issue_id for finding in case.gold_findings
    )
    forbidden.append(case.review.reconciliation_notes)
    return [
        str(item) for item in forbidden if item and str(item) in text
    ]


def _request_record_dict(record: RequestRecord) -> dict[str, object]:
    return {
        "call_id": record.call_id,
        "sequence": record.sequence,
        "phase": record.phase,
        "model": record.model,
        "provider": record.provider,
        "status": record.status,
        "cost_status": record.cost_status,
        "reserved_cost_usd": str(record.reserved_cost_usd),
        "estimated_cost_usd": _decimal_string(record.estimated_cost_usd),
        "reported_cost_usd": _decimal_string(record.reported_cost_usd),
        "input_token_bound": record.input_token_bound,
        "output_token_bound": record.output_token_bound,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "generation_id": record.generation_id,
        "response_model": record.response_model,
        "finish_reason": record.finish_reason,
        "error": record.error,
        "released_reason": record.released_reason,
        "messages": _json_safe(record.messages),
        "request_kwargs": _json_safe(record.request_kwargs),
        "usage": _json_safe(record.usage),
        "started_at": record.started_at.isoformat(),
        "finished_at": (
            record.finished_at.isoformat()
            if record.finished_at is not None
            else None
        ),
    }


def _profile_dict(profile: ProviderRouteProfile) -> dict[str, object]:
    return {
        "role": profile.role,
        "model": profile.model,
        "provider": profile.provider,
        "context_limit": profile.context_limit,
        "endpoint_max_output_tokens": profile.endpoint_max_output_tokens,
        "input_usd_per_million": str(profile.input_usd_per_million),
        "output_usd_per_million": str(profile.output_usd_per_million),
        "cache_write_usd_per_million": str(
            profile.cache_write_usd_per_million
        ),
        "expected_schema_mode": profile.expected_schema_mode,
    }


def _write_record(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _json_safe(record),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def _json_safe(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, type) and issubclass(value, BaseModel):
        return {
            "type": "json_schema",
            "name": value.__name__,
            "schema": value.model_json_schema(),
        }
    if isinstance(value, BaseModel):
        return {
            "type": "json_schema",
            "name": type(value).__name__,
            "schema": value.model_json_schema(),
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _decimal_string(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _require_catalog_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PreflightError(f"catalog {name} must be a positive integer")
    return value


def _catalog_price(
    pricing: Mapping[str, object], key: str
) -> Decimal | None:
    raw = pricing.get(key)
    if raw is None:
        return None
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, ValueError) as exc:
        raise PreflightError(
            f"catalog price {key!r} is not a decimal"
        ) from exc
    if not value.is_finite() or value < 0:
        raise PreflightError(
            f"catalog price {key!r} must be finite and non-negative"
        )
    return value * _MILLION


def _max_catalog_rate(
    pricing: Mapping[str, object], keys: tuple[str, ...]
) -> Decimal:
    rates = [
        rate
        for rate in (_catalog_price(pricing, key) for key in keys)
        if rate is not None
    ]
    return max(rates, default=Decimal("0"))


def _redact_text(text: str, secret: str | None) -> str:
    if secret:
        text = text.replace(secret, "***")
    return text


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "EndpointSnapshot",
    "PreflightConfigurationError",
    "PreflightError",
    "PreflightNotAuthorized",
    "PreflightResult",
    "fetch_endpoint_catalog",
    "main",
    "parse_endpoint_catalog",
    "run_preflight",
    "select_preflight_cases",
]


if __name__ == "__main__":
    raise SystemExit(main())
