"""Canonicalization, offset validation, and JSONL case loading."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import ValidationError

from .schemas import (
    ALLOWED_CODES,
    ISO_UTC_PATTERN,
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
VALIDATION_ARTICLE_COUNT = 10
TEST_ARTICLE_COUNT = 10
TOTAL_ARTICLE_COUNT = 50
TOTAL_FINDING_COUNT = 60
EVALUATION_FINDING_COUNT = 12
EVALUATION_CODE_COUNT = 2
EVALUATION_COMPOSITION = {"clean": 2, "one_issue": 4, "two_issues": 4}
SPLIT_NAMES = ("train", "validation", "test")
NEAR_DUPLICATE_THRESHOLD = 0.65
SHINGLE_SIZE = 5
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


def split_coverage(
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
    }


def training_coverage(
    cases: Sequence[CaseRecord],
) -> dict[str, object]:
    report = split_coverage(cases)
    report["rules_version"] = RULES_VERSION
    return report


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
            "reviews.jsonl must contain exactly one entry per case"
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
                f"{case_id}: review is not approved"
            )


def _word_tokens(text: str) -> list[str]:
    return re.findall(
        r"[^\W_]+",
        canonicalize_text(text).lower(),
        flags=re.UNICODE,
    )


def five_word_shingles(
    text: str, size: int = SHINGLE_SIZE
) -> set[tuple[str, ...]]:
    tokens = _word_tokens(text)
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {
        tuple(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    }


def jaccard_similarity(
    left: set[tuple[str, ...]], right: set[tuple[str, ...]]
) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def word_jaccard(left: str, right: str) -> float:
    left_tokens = set(_word_tokens(left))
    right_tokens = set(_word_tokens(right))
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def masked_template(text: str) -> str:
    normalized = canonicalize_text(text).lower()
    normalized = re.sub(r"\d+", "#", normalized)
    normalized = re.sub(r"[^\w#]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def near_duplicate_pairs(
    cases: Sequence[CaseRecord],
    *,
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> list[dict[str, object]]:
    shingles = {
        case.case_id: five_word_shingles(canonical_document(case))
        for case in cases
    }
    templates = {
        case.case_id: masked_template(canonical_document(case))
        for case in cases
    }
    evidence = {
        case.case_id: [
            finding.evidence for finding in case.gold_findings
        ]
        for case in cases
    }
    pairs: list[dict[str, object]] = []
    for index, left in enumerate(cases):
        for right in cases[index + 1 :]:
            similarity = jaccard_similarity(
                shingles[left.case_id], shingles[right.case_id]
            )
            same_template = (
                templates[left.case_id] == templates[right.case_id]
            )
            same_evidence = any(
                masked_template(left_evidence)
                == masked_template(right_evidence)
                or word_jaccard(left_evidence, right_evidence) >= 0.75
                for left_evidence in evidence[left.case_id]
                for right_evidence in evidence[right.case_id]
            )
            if (
                similarity >= threshold
                or same_template
                or same_evidence
            ):
                pairs.append(
                    {
                        "case_a": left.case_id,
                        "case_b": right.case_id,
                        "jaccard": round(similarity, 6),
                        "same_family": left.family_id == right.family_id,
                        "same_template": same_template,
                        "same_evidence": same_evidence,
                    }
                )
    return pairs


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_split_manifest(
    cases_by_split: Mapping[str, Sequence[CaseRecord]],
    *,
    frozen_at: str,
    file_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    if set(cases_by_split) != set(SPLIT_NAMES):
        raise DatasetValidationError(
            f"splits must be exactly {SPLIT_NAMES}"
        )
    if ISO_UTC_PATTERN.fullmatch(frozen_at) is None:
        raise DatasetValidationError(
            "frozen_at must be a UTC ISO-8601 timestamp ending in Z"
        )
    if file_hashes is not None and set(file_hashes) != set(SPLIT_NAMES):
        raise DatasetValidationError(
            "file_hashes must contain all three split names"
        )
    split_of: dict[str, str] = {}
    family_splits: dict[str, str] = {}
    all_cases: list[CaseRecord] = []
    for split_name in SPLIT_NAMES:
        for case in cases_by_split[split_name]:
            if case.case_id in split_of:
                raise DatasetValidationError(
                    f"duplicate case_id {case.case_id!r} across splits"
                )
            split_of[case.case_id] = split_name
            previous = family_splits.get(case.family_id)
            if previous is not None and previous != split_name:
                raise DatasetValidationError(
                    f"family {case.family_id!r} crosses splits "
                    f"{previous!r} and {split_name!r}"
                )
            family_splits[case.family_id] = split_name
            all_cases.append(case)
    pairs = near_duplicate_pairs(all_cases)
    for pair in pairs:
        pair["split_a"] = split_of[str(pair["case_a"])]
        pair["split_b"] = split_of[str(pair["case_b"])]
        pair["cross_split"] = pair["split_a"] != pair["split_b"]
        if pair["same_family"]:
            pair["status"] = "same_family"
        elif pair["cross_split"]:
            pair["status"] = "unresolved"
        else:
            pair["status"] = "same_split"
    splits: dict[str, object] = {}
    for split_name in SPLIT_NAMES:
        split_cases = list(cases_by_split[split_name])
        coverage = split_coverage(split_cases)
        coverage["file"] = f"data/v1/{split_name}.jsonl"
        if file_hashes is not None:
            coverage["file_sha256"] = file_hashes[split_name]
        coverage["case_ids"] = [
            case.case_id for case in split_cases
        ]
        coverage["family_ids"] = [
            case.family_id for case in split_cases
        ]
        splits[split_name] = coverage
    return {
        "manifest_version": 1,
        "rules_version": RULES_VERSION,
        "frozen_at": frozen_at,
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "splits": splits,
        "family_assignments": family_splits,
        "near_duplicate_report": pairs,
        "total_article_count": len(all_cases),
        "total_finding_count": sum(
            len(case.gold_findings) for case in all_cases
        ),
        "combined_dataset_sha256": dataset_sha256(all_cases),
        "test_sealed": True,
    }


def load_split_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_split_manifest(
    cases_by_split: Mapping[str, Sequence[CaseRecord]],
    manifest: Mapping[str, object],
) -> None:
    frozen_at = manifest.get("frozen_at")
    if not isinstance(frozen_at, str):
        raise DatasetValidationError("manifest frozen_at is missing")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise DatasetValidationError("manifest splits are missing")
    file_hashes: dict[str, str] = {}
    for split_name in SPLIT_NAMES:
        split_data = splits.get(split_name)
        if not isinstance(split_data, Mapping):
            raise DatasetValidationError(
                f"manifest split {split_name!r} is missing"
            )
        value = split_data.get("file_sha256")
        if value is not None:
            if not isinstance(value, str):
                raise DatasetValidationError(
                    f"manifest {split_name} file_sha256 must be a string"
                )
            file_hashes[split_name] = value
    if file_hashes and set(file_hashes) != set(SPLIT_NAMES):
        raise DatasetValidationError(
            "manifest file hashes are incomplete"
        )
    expected = build_split_manifest(
        cases_by_split,
        frozen_at=frozen_at,
        file_hashes=file_hashes or None,
    )
    if dict(manifest) != expected:
        raise DatasetValidationError(
            "split manifest does not match the loaded cases"
        )


def _validate_evaluation_split(
    split_name: str,
    cases: Sequence[CaseRecord],
    *,
    require_approved_review: bool,
) -> None:
    expected_count = (
        VALIDATION_ARTICLE_COUNT
        if split_name == "validation"
        else TEST_ARTICLE_COUNT
    )
    if len(cases) != expected_count:
        raise DatasetValidationError(
            f"{split_name} must contain {expected_count} cases; "
            f"found {len(cases)}"
        )
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise DatasetValidationError(
            f"{split_name} case_id values must be unique"
        )
    hashes = [canonical_document_sha256(case) for case in cases]
    if len(hashes) != len(set(hashes)):
        raise DatasetValidationError(
            f"{split_name} canonical document hashes must be unique"
        )
    for case in cases:
        validate_case(case)
        validate_references(case)
        if not case.family_description.strip():
            raise DatasetValidationError(
                f"{case.case_id}: family_description is required"
            )
        if require_approved_review:
            review = case.review
            if (
                not review.reconciled
                or review.author_decision != "approved"
                or review.reviewer_decision != "approved"
            ):
                raise DatasetValidationError(
                    f"{case.case_id}: review is not approved"
                )
    coverage = split_coverage(cases)
    composition = coverage["composition"]
    for bucket, expected in EVALUATION_COMPOSITION.items():
        actual = composition[bucket]
        if actual != expected:
            raise DatasetValidationError(
                f"{split_name} composition {bucket} must be {expected}; "
                f"found {actual}"
            )
    if composition["more_than_two"]:
        raise DatasetValidationError(
            f"{split_name} cases must have at most two gold issues"
        )
    if coverage["finding_count"] != EVALUATION_FINDING_COUNT:
        raise DatasetValidationError(
            f"{split_name} must contain {EVALUATION_FINDING_COUNT} gold "
            f"findings; found {coverage['finding_count']}"
        )
    for code, count in coverage["code_counts"].items():
        if count != EVALUATION_CODE_COUNT:
            raise DatasetValidationError(
                f"{split_name} code {code} must occur "
                f"{EVALUATION_CODE_COUNT} times; found {count}"
            )
    if len(coverage["hard_negative_cases"]) < 1:
        raise DatasetValidationError(
            f"{split_name} must contain at least one hard negative"
        )


def validate_all_splits(
    cases_by_split: Mapping[str, Sequence[CaseRecord]],
    *,
    require_approved_review: bool = True,
) -> None:
    if set(cases_by_split) != set(SPLIT_NAMES):
        raise DatasetValidationError(
            f"splits must be exactly {SPLIT_NAMES}"
        )
    train = list(cases_by_split["train"])
    validation = list(cases_by_split["validation"])
    test = list(cases_by_split["test"])
    validate_training_dataset(
        train, require_approved_review=require_approved_review
    )
    _validate_evaluation_split(
        "validation",
        validation,
        require_approved_review=require_approved_review,
    )
    _validate_evaluation_split(
        "test",
        test,
        require_approved_review=require_approved_review,
    )
    all_cases = train + validation + test
    if len(all_cases) != TOTAL_ARTICLE_COUNT:
        raise DatasetValidationError(
            f"dataset must contain {TOTAL_ARTICLE_COUNT} cases; "
            f"found {len(all_cases)}"
        )
    if len({case.case_id for case in all_cases}) != len(all_cases):
        raise DatasetValidationError(
            "case_id values must be unique across all splits"
        )
    if len(
        {canonical_document_sha256(case) for case in all_cases}
    ) != len(all_cases):
        raise DatasetValidationError(
            "canonical document hashes must be unique across all splits"
        )
    family_splits: dict[str, str] = {}
    split_of: dict[str, str] = {}
    for split_name, cases in cases_by_split.items():
        for case in cases:
            split_of[case.case_id] = split_name
            previous = family_splits.get(case.family_id)
            if previous is not None and previous != split_name:
                raise DatasetValidationError(
                    f"family {case.family_id!r} crosses splits "
                    f"{previous!r} and {split_name!r}"
                )
            family_splits[case.family_id] = split_name
    for pair in near_duplicate_pairs(all_cases):
        if (
            split_of[str(pair["case_a"])]
            != split_of[str(pair["case_b"])]
            and not pair["same_family"]
        ):
            raise DatasetValidationError(
                "unresolved cross-split near duplicate: "
                f"{pair['case_a']} / {pair['case_b']} "
                f"(Jaccard {pair['jaccard']})"
            )
    if sum(
        len(case.gold_findings) for case in all_cases
    ) != TOTAL_FINDING_COUNT:
        raise DatasetValidationError(
            f"dataset must contain {TOTAL_FINDING_COUNT} gold findings"
        )


def load_all_splits(
    base_dir: Path,
) -> dict[str, list[CaseRecord]]:
    """Load all splits for validation/reporting, never for optimization."""
    return {
        split_name: load_cases(base_dir / f"{split_name}.jsonl")
        for split_name in SPLIT_NAMES
    }


def load_benchmark_cases(base_dir: Path) -> list[CaseRecord]:
    """Load the sealed test split for an explicitly authorized benchmark.

    This is the only test-data loader. Optimization code must use
    ``load_optimization_cases`` and never call this function.
    """
    base = Path(base_dir)
    manifest_path = base / "split_manifest.json"
    test_path = base / "test.jsonl"
    if not manifest_path.is_file():
        raise DatasetValidationError(
            "benchmark loader requires the frozen split manifest"
        )
    if not test_path.is_file():
        raise DatasetValidationError("benchmark loader requires test.jsonl")
    manifest = load_split_manifest(manifest_path)
    if manifest.get("test_sealed") is not True:
        raise DatasetValidationError("split manifest does not seal the test split")
    if manifest.get("rules_version") != RULES_VERSION:
        raise DatasetValidationError(
            "split manifest uses an unsupported rules version"
        )
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise DatasetValidationError("split manifest has no test metadata")
    test_meta = splits.get("test")
    if not isinstance(test_meta, Mapping):
        raise DatasetValidationError("split manifest has no test entry")
    expected_hash = test_meta.get("file_sha256")
    if not isinstance(expected_hash, str) or file_sha256(test_path) != expected_hash:
        raise DatasetValidationError(
            "test.jsonl does not match the frozen split manifest hash"
        )
    cases = load_cases(test_path)
    expected_ids = test_meta.get("case_ids")
    expected_families = test_meta.get("family_ids")
    if expected_ids is not None and list(expected_ids) != [
        case.case_id for case in cases
    ]:
        raise DatasetValidationError(
            "test.jsonl case IDs do not match the frozen split manifest"
        )
    if expected_families is not None and list(expected_families) != [
        case.family_id for case in cases
    ]:
        raise DatasetValidationError(
            "test.jsonl families do not match the frozen split manifest"
        )
    if any(case.rules_version != RULES_VERSION for case in cases):
        raise DatasetValidationError("test cases use an unsupported rules version")
    return cases


def load_optimization_cases(
    train_path: Path, validation_path: Path
) -> dict[str, list[CaseRecord]]:
    """Load only train and validation; the sealed test path is rejected."""
    for path in (train_path, validation_path):
        if path.name == "test.jsonl" or path.stem == "test":
            raise DatasetValidationError(
                "the test split is sealed and cannot be loaded for "
                "optimization"
            )
    return {
        "train": load_cases(train_path),
        "validation": load_cases(validation_path),
    }


__all__ = [
    "DatasetValidationError",
    "EVALUATION_CODE_COUNT",
    "EVALUATION_COMPOSITION",
    "EVALUATION_FINDING_COUNT",
    "NEAR_DUPLICATE_THRESHOLD",
    "SHINGLE_SIZE",
    "SPLIT_NAMES",
    "TEST_ARTICLE_COUNT",
    "TITLE_BODY_SEPARATOR",
    "TOTAL_ARTICLE_COUNT",
    "TOTAL_FINDING_COUNT",
    "TRAINING_ARTICLE_COUNT",
    "TRAINING_CODE_COUNT",
    "TRAINING_COMPOSITION",
    "TRAINING_FINDING_COUNT",
    "VALIDATION_ARTICLE_COUNT",
    "build_coverage_report",
    "build_split_manifest",
    "canonical_document",
    "canonical_document_sha256",
    "canonicalize_document",
    "canonicalize_text",
    "case_composition_bucket",
    "dataset_sha256",
    "file_sha256",
    "find_literal_occurrences",
    "five_word_shingles",
    "jaccard_similarity",
    "load_all_splits",
    "load_benchmark_cases",
    "load_cases",
    "load_optimization_cases",
    "load_review_audit",
    "load_split_manifest",
    "masked_template",
    "near_duplicate_pairs",
    "split_coverage",
    "training_coverage",
    "validate_all_splits",
    "validate_case",
    "validate_case_offsets",
    "validate_references",
    "validate_review_audit",
    "validate_split_manifest",
    "validate_training_dataset",
    "word_jaccard",
]
