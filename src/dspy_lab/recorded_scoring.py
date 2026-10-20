"""Deterministic re-scoring of archived raw outcomes with the frozen metric."""

from __future__ import annotations

from collections.abc import Mapping

from .artifacts import RecordedOutcome
from .metrics import CaseScore, EditorialQAOutput, InvalidPrediction, score_case
from .schemas import CaseRecord

_INVALID_KINDS = frozenset({"parse_error", "request_error", "missing_output"})


class RecordedOutcomeScoringError(ValueError):
    """Raised when an archived invalid outcome has no recognized kind."""


def rescore_recorded_outcome(
    case: CaseRecord,
    outcome: RecordedOutcome,
    *,
    strict_error_kind: bool = False,
) -> CaseScore:
    """Re-score archived findings or an explicit invalid output.

    Replay and release validation share this exact algorithm. ``strict``
    rejects an unrecognized archived ``error_kind`` instead of treating it as
    a parse error.
    """
    if outcome.status == "ok" and outcome.findings is not None:
        return score_case(
            case, EditorialQAOutput(findings=list(outcome.findings))
        )
    metric = outcome.metric if isinstance(outcome.metric, Mapping) else {}
    kind = metric.get("error_kind")
    if kind not in _INVALID_KINDS:
        if strict_error_kind:
            raise RecordedOutcomeScoringError(
                f"invalid archived outcome {outcome.phase}/{outcome.case_id} "
                "has no recognized error_kind"
            )
        kind = "parse_error"
    reason = (
        outcome.error
        or metric.get("error")
        or "recorded invalid output"
    )
    return score_case(case, InvalidPrediction(str(reason), kind=kind))


__all__ = [
    "RecordedOutcomeScoringError",
    "rescore_recorded_outcome",
]
