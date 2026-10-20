"""Pure display formatting helpers shared by the Streamlit screens."""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping
from decimal import Decimal

from .artifacts import ProgramManifest, RecordedOutcome


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator

def _metric_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)

def _metric_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value

def _rate_text(value: object) -> str:
    number = _metric_number(value)
    if number is None:
        return "N/A"
    return f"{number * 100:.1f}%"

def _rate_delta_text(baseline: object, optimized: object) -> str:
    left = _metric_number(baseline)
    right = _metric_number(optimized)
    if left is None or right is None:
        return "N/A"
    return f"{(right - left) * 100:+.1f} pp"

def _percent_text(value: object) -> str:
    """Format an already 0-100 DSPy percentage without scaling it again."""
    number = _metric_number(value)
    if number is None:
        return "N/A"
    return f"{number:.1f}%"

def _percent_delta_text(baseline: object, optimized: object) -> str:
    left = _metric_number(baseline)
    right = _metric_number(optimized)
    if left is None or right is None:
        return "N/A"
    return f"{right - left:+.1f} pp"

def _decimal_number(value: object) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, (int, float)):
        result = Decimal(str(value))
    elif isinstance(value, str):
        try:
            result = Decimal(value)
        except Exception:  # noqa: BLE001 - display helper fails closed
            return None
    else:
        return None
    return result if result.is_finite() else None

def _money_delta_text(baseline: object, optimized: object) -> str:
    left = _decimal_number(baseline)
    right = _decimal_number(optimized)
    if left is None or right is None:
        return "N/A"
    delta = right - left
    if delta == 0:
        return "0"
    text = format(delta, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if delta > 0:
        text = "+" + text
    return text

def _count_delta_text(baseline: object, optimized: object) -> str:
    left = _metric_int(baseline)
    right = _metric_int(optimized)
    if left is None or right is None:
        return "N/A"
    return f"{right - left:+d}"

def _number_delta_text(baseline: object, optimized: object) -> str:
    left = _metric_number(baseline)
    right = _metric_number(optimized)
    if left is None or right is None:
        return "N/A"
    return f"{right - left:+.4f}"

def _money_text(value: object, status: object) -> str:
    number = _decimal_number(value)
    if number is None:
        return f"unavailable ({status or 'unknown'})"
    return f"{format(number, 'f')} USD ({status or 'unknown status'})"

def _comparison_effective(
    stored_arm: Mapping[str, object] | None,
    recomputed_arm: Mapping[str, object],
    key: str,
) -> object:
    if isinstance(stored_arm, Mapping) and key in stored_arm:
        return stored_arm.get(key)
    return recomputed_arm.get(key)

def _same_schema(
    left: ProgramManifest, right: ProgramManifest
) -> bool:
    return (
        left.signature_schema == right.signature_schema
        and left.signature_fields == right.signature_fields
    )

def _unified_diff(
    before: str,
    after: str,
    *,
    fromfile: str,
    tofile: str,
) -> str:
    return "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=fromfile,
            tofile=tofile,
            lineterm="",
        )
    )

def _cost_value(value: object, status: object) -> str:
    if value is None:
        return f"unavailable ({status or 'unknown status'})"
    return f"{value} USD ({status or 'unknown status'})"

def _cost_status_from_payload(payload: Mapping[str, object]) -> str:
    if "actual_cost_usd" in payload or "estimated_cost_usd" in payload:
        unresolved = _decimal_or_none(payload.get("unresolved_cost_usd"))
        if payload.get("pending_cost_calls"):
            return "pending"
        actual = _decimal_or_none(payload.get("actual_cost_usd"))
        estimated = _decimal_or_none(payload.get("estimated_cost_usd"))
        committed = _decimal_or_none(payload.get("committed_cost_usd"))
        if actual is not None and actual > 0:
            return "actual"
        if estimated is not None and estimated > 0:
            return "estimated"
        if committed is not None and committed > 0:
            return "committed"
        if unresolved is not None and unresolved > 0:
            return "pending"
        return "none"
    return str(payload.get("cost_status", "unavailable"))

def _decimal_or_none(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        decimal = Decimal(str(value))
    except Exception:  # noqa: BLE001 - display fallback
        return None
    return decimal if decimal.is_finite() else None

def _outcome_cost(outcome: RecordedOutcome) -> str:
    if outcome.cost_usd is None:
        return f"unavailable ({outcome.cost_status})"
    return f"{outcome.cost_usd} USD ({outcome.cost_status})"

def _percent(value: object) -> str:
    if value is None or isinstance(value, bool):
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.1f}%"
    return "N/A"

def _number(value: object) -> str:
    if value is None or isinstance(value, bool):
        return "N/A"
    if isinstance(value, (int, float)):
        number = float(value)
        text = f"{number:.4f}".rstrip("0").rstrip(".")
        if number != 0.0 and text in ("0", "-0"):
            return f"{number:.6g}"
        return text
    if isinstance(value, Decimal):
        return format(value, "f")
    return str(value)

def _int_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value

def _latency(value: float | None) -> str:
    if value is None:
        return "not recorded"
    return f"{value:.1f} ms"

def _message_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping):
                text = part.get("text")
                parts.append(str(text) if text is not None else str(part))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False, indent=2, default=str)
