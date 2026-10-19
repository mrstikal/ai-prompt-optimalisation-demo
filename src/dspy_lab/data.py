"""Canonicalization, offset validation, and JSONL case loading."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from .schemas import (
    ALLOWED_CODES,
    RULES_VERSION,
    CaseRecord,
    GoldFinding,
    MatchOption,
    ReviewAuditEntry,
)

TITLE_BODY_SEPARATOR = "\n\n"
TRAINING_ARTICLE_COUNT = 30
TRAINING_FINDING_COUNT = 36
TRAINING_CODE_COUNT = 6
TRAINING_COMPOSITION = {"clean": 6, "one_issue": 12, "two_issues": 12}
CITATION_PATTERN = re.compile(r"\[([A-Za-z][A-Za-z0-9_-]*)\]")


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


def case_composition_bucket(case: CaseRecord) -> str:
    count = len(case.gold_findings)
    if count == 0:
        return "clean"
    if count == 1:
        return "one_issue"
    if count == 2:
        return "two_issues"
    return "more_than_two"


def dataset_sha256(cases: Sequence[CaseRecord]) -> str:
    payload = "\n".join(
        canonical_document_sha256(case) for case in cases
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def training_coverage(
    cases: Sequence[CaseRecord],
) -> dict[str, object]:
    composition = {
        "clean": 0,
        "one_issue": 0,
        "two_issues": 0,
        "more_than_two": 0,
    }
    code_counts = {code: 0 for code in ALLOWED_CODES}
    for case in cases:
        composition[case_composition_bucket(case)] += 1
        for finding in case.gold_findings:
            code_counts[finding.code] += 1
    return {
        "article_count": len(cases),
        "composition": composition,
        "finding_count": sum(
            len(case.gold_findings) for case in cases
        ),
        "code_counts": code_counts,
        "clean_cases": [
            case.case_id for case in cases if not case.gold_findings
        ],
        "hard_negative_cases": [
            case.case_id for case in cases if case.is_hard_negative
        ],
        "same_code_multi_issue_cases": [
            case.case_id
            for case in cases
            if len({finding.code for finding in case.gold_findings})
            < len(case.gold_findings)
        ],
        "families": [
            {
                "family_id": case.family_id,
                "description": case.family_description,
            }
            for case in cases
        ],
        "case_hashes": {
            case.case_id: canonical_document_sha256(case)
            for case in cases
        },
        "dataset_sha256": dataset_sha256(cases),
        "rules_version": RULES_VERSION,
    }


def build_coverage_report(
    cases: Sequence[CaseRecord],
) -> dict[str, object]:
    report = training_coverage(cases)
    report["composition_target"] = dict(TRAINING_COMPOSITION)
    report["code_target"] = {
        code: TRAINING_CODE_COUNT for code in ALLOWED_CODES
    }
    return report


def validate_references(case: CaseRecord) -> None:
    supplied = {source.source_id for source in case.sources}
    cited = set(CITATION_PATTERN.findall(case.article_text))
    invalid_findings = [
        finding
        for finding in case.gold_findings
        if finding.code == "INVALID_CITATION"
    ]
    for source_id in sorted(cited - supplied):
        marker = f"[{source_id}]"
        if not any(
            marker in finding.evidence
            for finding in invalid_findings
        ):
            raise DatasetValidationError(
                f"{case.case_id}: cited source {source_id!r} is missing "
                "but has no INVALID_CITATION gold finding"
            )
    for finding in invalid_findings:
        markers = CITATION_PATTERN.findall(finding.evidence)
        if not markers:
            raise DatasetValidationError(
                f"{case.case_id}/{finding.issue_id}: INVALID_CITATION "
                "evidence must contain a citation marker"
            )


def validate_training_dataset(
    cases: Sequence[CaseRecord],
    *,
    require_approved_review: bool = True,
) -> None:
    if len(cases) != TRAINING_ARTICLE_COUNT:
        raise DatasetValidationError(
            f"training set must contain {TRAINING_ARTICLE_COUNT} cases; "
            f"found {len(cases)}"
        )
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise DatasetValidationError("training case_id values must be unique")
    hashes = [canonical_document_sha256(case) for case in cases]
    if len(hashes) != len(set(hashes)):
        raise DatasetValidationError(
            "training canonical document hashes must be unique"
        )
    for case in cases:
        validate_case(case)
        validate_references(case)
        if not case.family_description.strip():
            raise DatasetValidationError(
                f"{case.case_id}: family_description is required for training"
            )
        if require_approved_review:
            review = case.review
            if (
                not review.reconciled
                or review.author_decision != "approved"
                or review.reviewer_decision != "approved"
            ):
                raise DatasetValidationError(
                    f"{case.case_id}: training review is not approved"
                )
    coverage = training_coverage(cases)
    composition = coverage["composition"]
    for bucket, expected in TRAINING_COMPOSITION.items():
        actual = composition[bucket]
        if actual != expected:
            raise DatasetValidationError(
                f"training composition {bucket} must be {expected}; "
                f"found {actual}"
            )
    if composition["more_than_two"]:
        raise DatasetValidationError(
            "training cases must have at most two gold issues"
        )
    if coverage["finding_count"] != TRAINING_FINDING_COUNT:
        raise DatasetValidationError(
            f"training must contain {TRAINING_FINDING_COUNT} gold findings; "
            f"found {coverage['finding_count']}"
        )
    for code, count in coverage["code_counts"].items():
        if count != TRAINING_CODE_COUNT:
            raise DatasetValidationError(
                f"training code {code} must occur {TRAINING_CODE_COUNT} "
                f"times; found {count}"
            )
    if len(coverage["hard_negative_cases"]) < 3:
        raise DatasetValidationError(
            "training must contain at least three hard negatives"
        )
    if not coverage["same_code_multi_issue_cases"]:
        raise DatasetValidationError(
            "training must contain a same-code multi-issue case"
        )


def load_review_audit(path: Path) -> list[ReviewAuditEntry]:
    entries: list[ReviewAuditEntry] = []
    seen_case_ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            entry = ReviewAuditEntry.model_validate_json(line)
        except ValidationError as exc:
            raise DatasetValidationError(
                f"{path}:{line_number}: invalid review record: {exc}"
            ) from exc
        if entry.case_id in seen_case_ids:
            raise DatasetValidationError(
                f"{path}:{line_number}: duplicate review case_id "
                f"{entry.case_id!r}"
            )
        seen_case_ids.add(entry.case_id)
        entries.append(entry)
    return entries


def validate_review_audit(
    cases: Sequence[CaseRecord],
    entries: Sequence[ReviewAuditEntry],
    *,
    require_approved_review: bool = True,
) -> None:
    cases_by_id = {case.case_id: case for case in cases}
    entries_by_id = {entry.case_id: entry for entry in entries}
    if set(cases_by_id) != set(entries_by_id):
        raise DatasetValidationError(
            "reviews.jsonl must contain exactly one entry per training case"
        )
    for case_id, case in cases_by_id.items():
        entry = entries_by_id[case_id]
        if entry.model_dump(exclude={"case_id"}) != case.review.model_dump():
            raise DatasetValidationError(
                f"{case_id}: reviews.jsonl does not match the case review"
            )
        if require_approved_review and (
            not entry.reconciled
            or entry.author_decision != "approved"
            or entry.reviewer_decision != "approved"
        ):
            raise DatasetValidationError(
                f"{case_id}: training review is not approved"
            )


__all__ = [
    "DatasetValidationError",
    "TITLE_BODY_SEPARATOR",
    "TRAINING_ARTICLE_COUNT",
    "TRAINING_CODE_COUNT",
    "TRAINING_COMPOSITION",
    "TRAINING_FINDING_COUNT",
    "build_coverage_report",
    "canonical_document",
    "canonical_document_sha256",
    "canonicalize_document",
    "canonicalize_text",
    "case_composition_bucket",
    "dataset_sha256",
    "find_literal_occurrences",
    "load_cases",
    "load_review_audit",
    "training_coverage",
    "validate_case",
    "validate_case_offsets",
    "validate_references",
    "validate_review_audit",
    "validate_training_dataset",
]
