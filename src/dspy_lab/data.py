"""Canonicalization, offset validation, and JSONL case loading."""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

from pydantic import ValidationError

from .schemas import CaseRecord, GoldFinding, MatchOption

TITLE_BODY_SEPARATOR = "\n\n"


class DatasetValidationError(ValueError):
    """Raised when a case or dataset violates the frozen data contract."""


def canonicalize_text(text: str) -> str:
    """Apply NFC, LF line endings, and ordinary spaces for NBSP variants."""
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace("\u00a0", " ").replace("\u202f", " ")


def canonicalize_document(title: str, article_text: str) -> str:
    """Build the canonical title + blank line + body scoring text."""
    return (
        f"{canonicalize_text(title)}{TITLE_BODY_SEPARATOR}"
        f"{canonicalize_text(article_text)}"
    )


def canonical_document(case: CaseRecord) -> str:
    return canonicalize_document(case.title, case.article_text)


def canonical_document_sha256(case: CaseRecord) -> str:
    canonical = canonical_document(case)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def find_literal_occurrences(text: str, quote: str) -> list[int]:
    """Return every literal start offset, including overlapping matches."""
    if not quote:
        return []
    occurrences: list[int] = []
    start = 0
    while True:
        index = text.find(quote, start)
        if index < 0:
            return occurrences
        occurrences.append(index)
        start = index + 1


def _crosses_separator(
    start: int, end: int, separator_start: int, separator_end: int
) -> bool:
    return start < separator_end and end > separator_start


def validate_case_offsets(case: CaseRecord) -> None:
    """Validate gold quotes, anchors, and context windows against the text."""
    canonical = canonical_document(case)
    title = canonicalize_text(case.title)
    separator_start = len(title)
    separator_end = separator_start + len(TITLE_BODY_SEPARATOR)
    if canonical[separator_start:separator_end] != TITLE_BODY_SEPARATOR:
        raise DatasetValidationError(
            f"{case.case_id}: canonical title/body separator is not at the "
            "expected offset"
        )
    for issue in case.gold_findings:
        _validate_gold_finding(
            case.case_id,
            issue,
            canonical,
            separator_start,
            separator_end,
        )


def _validate_gold_finding(
    case_id: str,
    issue: GoldFinding,
    canonical: str,
    separator_start: int,
    separator_end: int,
) -> None:
    occurrences = find_literal_occurrences(canonical, issue.evidence)
    if len(occurrences) != 1:
        raise DatasetValidationError(
            f"{case_id}/{issue.issue_id}: evidence must occur exactly once; "
            f"found {len(occurrences)} occurrences"
        )
    evidence_start = occurrences[0]
    evidence_end = evidence_start + len(issue.evidence)
    if _crosses_separator(
        evidence_start, evidence_end, separator_start, separator_end
    ):
        raise DatasetValidationError(
            f"{case_id}/{issue.issue_id}: evidence crosses the title/body "
            "separator"
        )
    matched = False
    for option in issue.match_options:
        _validate_match_option_bounds(
            case_id,
            issue.issue_id,
            option,
            len(canonical),
            separator_start,
            separator_end,
        )
        if not (
            option.context_start <= evidence_start
            and evidence_end <= option.context_end
        ):
            continue
        if not (
            evidence_start <= option.anchor_start
            and option.anchor_end <= evidence_end
        ):
            continue
        matched = True
        break
    if not matched:
        raise DatasetValidationError(
            f"{case_id}/{issue.issue_id}: no match option contains the unique "
            "evidence quote and its mandatory anchor"
        )


def _validate_match_option_bounds(
    case_id: str,
    issue_id: str,
    option: MatchOption,
    canonical_length: int,
    separator_start: int,
    separator_end: int,
) -> None:
    if option.context_end > canonical_length:
        raise DatasetValidationError(
            f"{case_id}/{issue_id}: context_end exceeds canonical text length"
        )
    for label, start, end in (
        ("context", option.context_start, option.context_end),
        ("anchor", option.anchor_start, option.anchor_end),
    ):
        if _crosses_separator(start, end, separator_start, separator_end):
            raise DatasetValidationError(
                f"{case_id}/{issue_id}: {label} window crosses the "
                "title/body separator"
            )


def validate_case(case: CaseRecord) -> None:
    validate_case_offsets(case)


def load_cases(path: Path) -> list[CaseRecord]:
    """Load, validate, and de-duplicate JSONL cases."""
    cases: list[CaseRecord] = []
    seen_case_ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            case = CaseRecord.model_validate_json(line)
        except ValidationError as exc:
            raise DatasetValidationError(
                f"{path}:{line_number}: invalid case record: {exc}"
            ) from exc
        if case.case_id in seen_case_ids:
            raise DatasetValidationError(
                f"{path}:{line_number}: duplicate case_id {case.case_id!r}"
            )
        seen_case_ids.add(case.case_id)
        validate_case(case)
        cases.append(case)
    return cases


__all__ = [
    "DatasetValidationError",
    "TITLE_BODY_SEPARATOR",
    "canonical_document",
    "canonical_document_sha256",
    "canonicalize_document",
    "canonicalize_text",
    "find_literal_occurrences",
    "load_cases",
    "validate_case",
    "validate_case_offsets",
]
