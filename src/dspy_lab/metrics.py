"""Deterministic evidence matching, scoring, and GEPA feedback.

The scoring unit is one reviewed issue occurrence. A prediction is a true
positive only when its literal evidence quote contains a reviewed problem
anchor and lies inside that gold issue's allowed context window. Matching is a
deterministic maximum-cardinality one-to-one assignment, so one prediction can
never satisfy two gold issues and repeated predictions remain visible as false
positives.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import dspy
from pydantic import ValidationError

from .data import (
    TITLE_BODY_SEPARATOR,
    canonical_document,
    canonicalize_text,
    find_literal_occurrences,
)
from .schemas import (
    ALLOWED_CODES,
    CaseRecord,
    EditorialQAOutput,
    ErrorCode,
    GoldFinding,
)

InvalidKind = Literal["parse_error", "request_error", "missing_output"]

_MISSING = object()

_RULE_HINTS: dict[ErrorCode, str] = {
    "UNSUPPORTED_NUMERIC_CLAIM": (
        "Check whether the numerical claim is supported by the supplied facts "
        "or excerpts; do not report list numbering, hypotheticals, or a "
        "contradicted product attribute."
    ),
    "INVESTMENT_ADVICE_AS_FACT": (
        "Flag only an unqualified universal recommendation or an objective "
        "suitability claim; bounded, attributed opinions are allowed."
    ),
    "UNWARRANTED_CERTAINTY": (
        "Flag unconditional guarantees of future price, return, or absence of "
        "risk; negation, rejected quotations, and conditional scenarios are "
        "allowed."
    ),
    "UNRESOLVED_PLACEHOLDER": (
        "Quote the exact production placeholder left in publishable content; "
        "do not flag an explicitly quoted instructional example."
    ),
    "INVALID_CITATION": (
        "Check that the cited source ID is supplied and that its excerpt "
        "supports the attributed claim; support elsewhere does not repair a "
        "wrong attribution."
    ),
    "PRODUCT_FACT_MISMATCH": (
        "Compare stated product attributes with the explicit reference facts; "
        "a missing reference attribute is not a contradiction."
    ),
}

_EVIDENCE_HINT = (
    "Quote an exact, unique passage from the canonical article text and extend "
    "it with nearby context when the same phrase occurs more than once."
)
_ANCHOR_HINT = (
    "Quote the problem-bearing passage itself, not an unrelated substring that "
    "happens to sit inside the reviewed context window."
)
_CONTEXT_HINT = (
    "Keep the evidence quote inside the reviewed context window; do not quote "
    "the whole article or a passage that extends far beyond the violation."
)
_DUPLICATE_HINT = (
    "Report each violation once; later predictions with the same code and "
    "evidence are false positives even when the explanation differs."
)
_PRECEDENCE_HINT = (
    "Apply the category precedence in the editorial rules; do not label one "
    "atomic claim with two codes."
)
_ONE_TO_ONE_HINT = (
    "Do not report the same reviewed violation twice with different evidence "
    "quotes; each gold issue can be matched by only one prediction."
)
_CLEAN_HINT = (
    "Do not invent findings on a clean article; return an empty findings list "
    "when the supplied bundle contains no violation."
)
_SCHEMA_HINT = (
    "Return one JSON object with exactly one required findings list; every "
    "finding must contain only code, evidence, and explanation."
)
_UNSUPPORTED_FINDING_HINT = (
    "Do not invent findings for claims that have no reviewed gold issue; "
    "report only violations supported by a reviewed anchor."
)


class ScoringConfigurationError(ValueError):
    """Raised when gold annotations do not match the scoring text."""


class ScoringInvariantError(RuntimeError):
    """Raised when the deterministic matcher violates its own invariants."""


@dataclass(frozen=True)
class InvalidPrediction:
    """An explicit invalid whole response; never treated as an empty list."""

    reason: str
    kind: InvalidKind = "parse_error"

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("invalid-prediction reason must be non-empty")
        if self.kind not in ("parse_error", "request_error", "missing_output"):
            raise ValueError(f"unsupported invalid-prediction kind {self.kind!r}")


@dataclass(frozen=True)
class ScoringCase:
    """Gold annotations and the exact canonical text used for scoring."""

    case_id: str
    document_text: str
    gold_findings: tuple[GoldFinding, ...]
    separator_start: int | None = None
    separator_end: int | None = None


@dataclass(frozen=True)
class MatchedPair:
    prediction_index: int
    issue_id: str
    code: ErrorCode


@dataclass(frozen=True)
class FalsePositive:
    prediction_index: int
    code: ErrorCode
    evidence: str
    reason: str
    occurrences: int
    duplicate_of: int | None = None
    matching_gold_issue_id: str | None = None
    matching_gold_code: ErrorCode | None = None
    evidence_failure: bool = False


@dataclass(frozen=True)
class MissedGold:
    issue_id: str
    code: ErrorCode
    evidence: str


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    valid: bool
    error_kind: InvalidKind | None
    error: str | None
    reward: float
    exact_match: bool
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None
    matched_pairs: tuple[MatchedPair, ...]
    false_positives: tuple[FalsePositive, ...]
    missed_gold: tuple[MissedGold, ...]
    duplicate_count: int
    evidence_failure_count: int
    clean_case: bool
    prediction_count: int
    gold_count: int

    @property
    def matched_issue_ids(self) -> tuple[str, ...]:
        return tuple(sorted(pair.issue_id for pair in self.matched_pairs))


@dataclass(frozen=True)
class CategoryScore:
    code: ErrorCode
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class CorpusScore:
    article_count: int
    valid_count: int
    invalid_count: int
    parse_error_count: int
    request_error_count: int
    missing_output_count: int
    evidence_failure_count: int
    duplicate_prediction_count: int
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    micro_f1: float | None
    macro_f1: float | None
    exact_match_count: int
    exact_match_rate: float | None
    parsing_success: float | None
    clean_article_count: int
    clean_articles_falsely_flagged: int
    mean_reward: float | None
    per_category: tuple[CategoryScore, ...]
    cases: tuple[CaseScore, ...]


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _f1(tp: int, fp: int, fn: int) -> float | None:
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return None
    return 2 * tp / denominator


def _crosses_separator(
    start: int, end: int, separator_start: int, separator_end: int
) -> bool:
    return start < separator_end and end > separator_start


def scoring_case_from_case(case: CaseRecord) -> ScoringCase:
    """Build a scoring case from a validated dataset record."""
    document = canonical_document(case)
    title = canonicalize_text(case.title)
    separator_start = len(title)
    return ScoringCase(
        case_id=case.case_id,
        document_text=document,
        gold_findings=tuple(case.gold_findings),
        separator_start=separator_start,
        separator_end=separator_start + len(TITLE_BODY_SEPARATOR),
    )


def _coerce_gold_finding(value: object) -> GoldFinding:
    if isinstance(value, GoldFinding):
        return value
    if isinstance(value, Mapping):
        try:
            return GoldFinding.model_validate(value)
        except ValidationError as exc:
            raise ScoringConfigurationError(
                f"invalid gold finding: {exc}"
            ) from exc
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return GoldFinding.model_validate(model_dump())
        except ValidationError as exc:
            raise ScoringConfigurationError(
                f"invalid gold finding: {exc}"
            ) from exc
    raise ScoringConfigurationError(
        f"unsupported gold finding type {type(value).__name__}"
    )


def _coerce_gold_findings(value: object) -> tuple[GoldFinding, ...]:
    if value is None:
        raise ScoringConfigurationError("gold_findings must be a sequence")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ScoringConfigurationError("gold_findings must be a sequence")
    return tuple(_coerce_gold_finding(item) for item in value)


def _field(
    source: object, mapping: Mapping[str, object] | None, name: str
) -> Any:
    if mapping is not None:
        return mapping.get(name, _MISSING)
    return getattr(source, name, _MISSING)


def _coerce_scoring_case(gold: object) -> ScoringCase:
    if isinstance(gold, ScoringCase):
        return gold
    if isinstance(gold, CaseRecord):
        return scoring_case_from_case(gold)
    mapping = gold if isinstance(gold, Mapping) else None
    case_id = _field(gold, mapping, "case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ScoringConfigurationError(
            "gold must expose a non-empty case_id"
        )
    document_text = _field(gold, mapping, "document_text")
    title = _field(gold, mapping, "title")
    if not isinstance(title, str):
        title = None
    if document_text is _MISSING or document_text is None:
        article_text = _field(gold, mapping, "article_text")
        if not isinstance(title, str) or not isinstance(article_text, str):
            raise ScoringConfigurationError(
                "gold must expose document_text or title/article_text"
            )
        document_text = canonicalize_text(title) + TITLE_BODY_SEPARATOR
        document_text += canonicalize_text(article_text)
    if not isinstance(document_text, str) or not document_text:
        raise ScoringConfigurationError(
            "gold document_text must be a non-empty string"
        )
    gold_findings_raw = _field(gold, mapping, "gold_findings")
    if gold_findings_raw is _MISSING:
        raise ScoringConfigurationError("gold must expose gold_findings")
    gold_findings = _coerce_gold_findings(gold_findings_raw)
    separator_start = _field(gold, mapping, "separator_start")
    separator_end = _field(gold, mapping, "separator_end")
    if separator_start is _MISSING:
        separator_start = None
    if separator_end is _MISSING:
        separator_end = None
    if title is not None:
        canonical_title = canonicalize_text(title)
        if not document_text.startswith(canonical_title):
            raise ScoringConfigurationError(
                "gold document_text must start with the canonical title when "
                "title is supplied"
            )
        title_length = len(canonical_title)
        separator_start = title_length
        separator_end = title_length + len(TITLE_BODY_SEPARATOR)
    elif separator_start is None and separator_end is None:
        separator_start = document_text.find(TITLE_BODY_SEPARATOR)
        if separator_start < 0:
            raise ScoringConfigurationError(
                "gold with document_text must include the canonical "
                "title/body separator or explicit separator offsets"
            )
        separator_end = separator_start + len(TITLE_BODY_SEPARATOR)
    if (separator_start is None) != (separator_end is None):
        raise ScoringConfigurationError(
            "separator_start and separator_end must be provided together"
        )
    return ScoringCase(
        case_id=case_id,
        document_text=document_text,
        gold_findings=gold_findings,
        separator_start=separator_start,
        separator_end=separator_end,
    )


def validate_scoring_case(case: ScoringCase) -> None:
    """Verify that every gold issue self-matches the exact scoring text."""
    if not case.case_id:
        raise ScoringConfigurationError("scoring case_id must be non-empty")
    if not case.document_text:
        raise ScoringConfigurationError(
            f"{case.case_id}: document_text must be non-empty"
        )
    if (case.separator_start is None) != (case.separator_end is None):
        raise ScoringConfigurationError(
            f"{case.case_id}: separator offsets must be provided together"
        )
    if case.separator_start is not None and case.separator_end is not None:
        if not (
            0 <= case.separator_start < case.separator_end
            <= len(case.document_text)
        ):
            raise ScoringConfigurationError(
                f"{case.case_id}: separator offsets are out of bounds"
            )
        if (
            case.document_text[
                case.separator_start : case.separator_end
            ]
            != TITLE_BODY_SEPARATOR
        ):
            raise ScoringConfigurationError(
                f"{case.case_id}: scoring text does not contain the canonical "
                "title/body separator at the declared offset"
            )
    issue_ids = [issue.issue_id for issue in case.gold_findings]
    if len(issue_ids) != len(set(issue_ids)):
        raise ScoringConfigurationError(
            f"{case.case_id}: gold issue_id values must be unique"
        )
    evidence_quotes = [issue.evidence for issue in case.gold_findings]
    if len(evidence_quotes) != len(set(evidence_quotes)):
        raise ScoringConfigurationError(
            f"{case.case_id}: gold evidence quotes must be unique"
        )
    for issue in case.gold_findings:
        _validate_gold_finding(case, issue)


def _validate_gold_finding(case: ScoringCase, issue: GoldFinding) -> None:
    occurrences = find_literal_occurrences(
        case.document_text, issue.evidence
    )
    if len(occurrences) != 1:
        raise ScoringConfigurationError(
            f"{case.case_id}/{issue.issue_id}: gold evidence must occur "
            f"exactly once; found {len(occurrences)} occurrences"
        )
    evidence_start = occurrences[0]
    evidence_end = evidence_start + len(issue.evidence)
    if case.separator_start is not None and case.separator_end is not None:
        if _crosses_separator(
            evidence_start,
            evidence_end,
            case.separator_start,
            case.separator_end,
        ):
            raise ScoringConfigurationError(
                f"{case.case_id}/{issue.issue_id}: gold evidence crosses the "
                "title/body separator"
            )
    matched = False
    for option in issue.match_options:
        if option.context_end > len(case.document_text):
            raise ScoringConfigurationError(
                f"{case.case_id}/{issue.issue_id}: context_end exceeds the "
                "scoring text length"
            )
        if case.separator_start is not None and case.separator_end is not None:
            for label, start, end in (
                ("context", option.context_start, option.context_end),
                ("anchor", option.anchor_start, option.anchor_end),
            ):
                if _crosses_separator(
                    start,
                    end,
                    case.separator_start,
                    case.separator_end,
                ):
                    raise ScoringConfigurationError(
                        f"{case.case_id}/{issue.issue_id}: {label} window "
                        "crosses the title/body separator"
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
    if not matched:
        raise ScoringConfigurationError(
            f"{case.case_id}/{issue.issue_id}: no match option contains the "
            "unique gold evidence and its mandatory anchor"
        )


def _compact_error(exc: Exception) -> str:
    return " ".join(str(exc).split())


def parse_prediction(
    raw: object, *, kind: InvalidKind = "parse_error"
) -> EditorialQAOutput | InvalidPrediction:
    """Validate a complete response without repair, coercion, or salvaging."""
    if isinstance(raw, InvalidPrediction):
        return raw
    if isinstance(raw, EditorialQAOutput):
        return raw
    if raw is None:
        return InvalidPrediction("prediction is missing", "missing_output")
    if isinstance(raw, str):
        try:
            return EditorialQAOutput.model_validate_json(raw)
        except (ValidationError, ValueError) as exc:
            return InvalidPrediction(
                f"invalid JSON output: {_compact_error(exc)}", kind
            )
    if isinstance(raw, Mapping):
        payload: object = raw
    else:
        findings = getattr(raw, "findings", _MISSING)
        if findings is _MISSING:
            model_dump = getattr(raw, "model_dump", None)
            if callable(model_dump):
                payload = model_dump()
            else:
                return InvalidPrediction(
                    "prediction does not expose a findings list",
                    "missing_output",
                )
        elif findings is None:
            return InvalidPrediction(
                "prediction findings are missing", "missing_output"
            )
        else:
            payload = {"findings": findings}
    try:
        return EditorialQAOutput.model_validate(payload)
    except ValidationError as exc:
        return InvalidPrediction(
            f"invalid structured output: {_compact_error(exc)}", kind
        )


def _all_gold_edges(
    evidence_start: int,
    evidence_end: int,
    gold_findings: Sequence[GoldFinding],
) -> tuple[int, ...]:
    edges: list[int] = []
    for index, issue in enumerate(gold_findings):
        for option in issue.match_options:
            if (
                option.context_start <= evidence_start
                and evidence_end <= option.context_end
                and evidence_start <= option.anchor_start
                and option.anchor_end <= evidence_end
            ):
                edges.append(index)
                break
    return tuple(edges)


def _maximum_matching(
    adjacency: Sequence[Sequence[int]],
    gold_count: int,
) -> tuple[list[int], list[int]]:
    """Deterministic maximum-cardinality one-to-one matching.

    The caller supplies adjacency in prediction order and sorts each
    candidate list by issue ID and then by gold position. This makes the
    tie-break stable without changing the maximum-cardinality objective.
    """
    match_gold = [-1] * gold_count
    match_prediction = [-1] * len(adjacency)

    def augment(prediction_index: int, visited: list[bool]) -> bool:
        for gold_index in adjacency[prediction_index]:
            if visited[gold_index]:
                continue
            visited[gold_index] = True
            current = match_gold[gold_index]
            if current == -1 or augment(current, visited):
                match_gold[gold_index] = prediction_index
                match_prediction[prediction_index] = gold_index
                return True
        return False

    for prediction_index in range(len(adjacency)):
        augment(prediction_index, [False] * gold_count)
    return match_prediction, match_gold


def _false_positive_reason(
    *,
    prediction_code: ErrorCode,
    prediction_index: int,
    occurrences: int,
    duplicate_of: int | None,
    same_code_edges: Sequence[int],
    other_code_edges: Sequence[int],
    matched_gold_indices: set[int],
    gold_findings: Sequence[GoldFinding],
    evidence_start: int | None,
    evidence_end: int | None,
) -> tuple[str, str | None, ErrorCode | None, bool]:
    evidence_failure = occurrences != 1
    if duplicate_of is not None:
        return (
            f"duplicate of prediction #{duplicate_of} for the same "
            "(code, evidence) pair; only the first identical prediction is "
            "eligible for matching",
            None,
            None,
            evidence_failure,
        )
    if occurrences == 0:
        return (
            "evidence does not occur literally in the canonical article text",
            None,
            None,
            True,
        )
    if occurrences > 1:
        return (
            f"evidence occurs {occurrences} times in the canonical article "
            "text; extend the quote with nearby context to identify exactly "
            "one location",
            None,
            None,
            True,
        )
    if same_code_edges:
        unmatched = [
            index
            for index in same_code_edges
            if index not in matched_gold_indices
        ]
        if unmatched:
            raise ScoringInvariantError(
                f"prediction #{prediction_index}: same-code gold issue "
                f"{gold_findings[unmatched[0]].issue_id} remained unmatched "
                "despite a valid matching edge"
            )
        issue = gold_findings[same_code_edges[0]]
        return (
            f"gold issue {issue.issue_id} is already matched by another "
            "prediction; matching is one-to-one",
            issue.issue_id,
            issue.code,
            False,
        )
    if other_code_edges:
        issue = gold_findings[other_code_edges[0]]
        return (
            f"wrong code; this claim is annotated as {issue.code} "
            f"({issue.issue_id})",
            issue.issue_id,
            issue.code,
            False,
        )
    if evidence_start is not None and evidence_end is not None:
        anchor_candidates = [
            issue
            for issue in gold_findings
            if issue.code == prediction_code
            and any(
                evidence_start <= option.anchor_start
                and option.anchor_end <= evidence_end
                for option in issue.match_options
            )
        ]
        if anchor_candidates:
            issue = anchor_candidates[0]
            return (
                f"quote contains the reviewed problem anchor for issue "
                f"{issue.issue_id} but extends outside its allowed context "
                "window",
                issue.issue_id,
                issue.code,
                False,
            )
    same_code_gold = [
        issue for issue in gold_findings if issue.code == prediction_code
    ]
    if same_code_gold:
        return (
            f"quote does not contain a reviewed problem anchor for code "
            f"{prediction_code}",
            None,
            None,
            False,
        )
    if not gold_findings:
        return (
            "the article has no gold issues; this finding is a false positive",
            None,
            None,
            False,
        )
    return (
        f"no reviewed gold issue matches this quote for code "
        f"{prediction_code}",
        None,
        None,
        False,
    )


_SCORING_CASE_CACHE_LIMIT = 256
_SCORING_CASE_CACHE: dict[int, tuple[object, ScoringCase]] = {}
_SCORING_CASE_CACHE_LOCK = threading.Lock()


def _validated_scoring_case(gold: object) -> ScoringCase:
    """Validate a reused gold object once and cache its scoring case.

    GEPA and benchmark evaluation reuse the same immutable ``CaseRecord`` or
    ``dspy.Example`` objects for many predictions. Caching by object identity
    avoids repeating every gold evidence location scan while keeping arbitrary
    mappings on the strict validate-every-call path.
    """
    if not isinstance(gold, (CaseRecord, ScoringCase, dspy.Example)):
        scoring_case = _coerce_scoring_case(gold)
        validate_scoring_case(scoring_case)
        return scoring_case
    key = id(gold)
    with _SCORING_CASE_CACHE_LOCK:
        cached = _SCORING_CASE_CACHE.get(key)
        if cached is not None and cached[0] is gold:
            return cached[1]
    scoring_case = (
        gold
        if isinstance(gold, ScoringCase)
        else (
            scoring_case_from_case(gold)
            if isinstance(gold, CaseRecord)
            else _coerce_scoring_case(gold)
        )
    )
    validate_scoring_case(scoring_case)
    with _SCORING_CASE_CACHE_LOCK:
        if len(_SCORING_CASE_CACHE) >= _SCORING_CASE_CACHE_LIMIT:
            _SCORING_CASE_CACHE.clear()
        _SCORING_CASE_CACHE[key] = (gold, scoring_case)
    return scoring_case


def score_case(case: object, prediction: object) -> CaseScore:
    """Score one article against its reviewed gold issues."""
    scoring_case = _validated_scoring_case(case)
    parsed = parse_prediction(prediction)
    gold_findings = scoring_case.gold_findings
    clean_case = not gold_findings
    if isinstance(parsed, InvalidPrediction):
        return CaseScore(
            case_id=scoring_case.case_id,
            valid=False,
            error_kind=parsed.kind,
            error=parsed.reason,
            reward=0.0,
            exact_match=False,
            tp=0,
            fp=0,
            fn=len(gold_findings),
            precision=None,
            recall=0.0 if gold_findings else None,
            f1=0.0 if gold_findings else None,
            matched_pairs=(),
            false_positives=(),
            missed_gold=tuple(
                MissedGold(
                    issue_id=issue.issue_id,
                    code=issue.code,
                    evidence=issue.evidence,
                )
                for issue in sorted(
                    gold_findings, key=lambda item: item.issue_id
                )
            ),
            duplicate_count=0,
            evidence_failure_count=0,
            clean_case=clean_case,
            prediction_count=0,
            gold_count=len(gold_findings),
        )
    predictions = parsed.findings
    occurrences_by_prediction: list[list[int]] = []
    duplicate_of: list[int | None] = []
    first_index_by_key: dict[tuple[ErrorCode, str], int] = {}
    all_edges: list[tuple[int, ...]] = []
    same_code_edges: list[tuple[int, ...]] = []
    gold_order = sorted(
        range(len(gold_findings)),
        key=lambda index: (gold_findings[index].issue_id, index),
    )
    for prediction_index, finding in enumerate(predictions):
        key = (finding.code, finding.evidence)
        duplicate = first_index_by_key.get(key)
        if duplicate is None:
            first_index_by_key[key] = prediction_index
        duplicate_of.append(duplicate)
        occurrences = find_literal_occurrences(
            scoring_case.document_text, finding.evidence
        )
        occurrences_by_prediction.append(occurrences)
        if duplicate is not None or len(occurrences) != 1:
            all_edges.append(())
            same_code_edges.append(())
            continue
        evidence_start = occurrences[0]
        evidence_end = evidence_start + len(finding.evidence)
        edges = _all_gold_edges(
            evidence_start, evidence_end, gold_findings
        )
        all_edges.append(edges)
        same_code_edges.append(
            tuple(
                index
                for index in edges
                if gold_findings[index].code == finding.code
            )
        )
    adjacency = [
        tuple(
            sorted(
                same_code_edges[index],
                key=lambda gold_index: (
                    gold_findings[gold_index].issue_id,
                    gold_index,
                ),
            )
        )
        for index in range(len(predictions))
    ]
    match_prediction, _ = _maximum_matching(
        adjacency, len(gold_findings)
    )
    matched_gold_indices = {
        gold_index for gold_index in match_prediction if gold_index >= 0
    }
    matched_pairs = tuple(
        MatchedPair(
            prediction_index=prediction_index,
            issue_id=gold_findings[gold_index].issue_id,
            code=gold_findings[gold_index].code,
        )
        for prediction_index, gold_index in enumerate(match_prediction)
        if gold_index >= 0
    )
    false_positives: list[FalsePositive] = []
    duplicate_count = 0
    evidence_failure_count = 0
    for prediction_index, finding in enumerate(predictions):
        if match_prediction[prediction_index] >= 0:
            continue
        occurrences = occurrences_by_prediction[prediction_index]
        duplicate = duplicate_of[prediction_index]
        if duplicate is not None:
            duplicate_count += 1
        evidence_start = occurrences[0] if len(occurrences) == 1 else None
        evidence_end = (
            evidence_start + len(finding.evidence)
            if evidence_start is not None
            else None
        )
        reason, issue_id, issue_code, evidence_failure = (
            _false_positive_reason(
                prediction_code=finding.code,
                prediction_index=prediction_index,
                occurrences=len(occurrences),
                duplicate_of=duplicate,
                same_code_edges=same_code_edges[prediction_index],
                other_code_edges=tuple(
                    index
                    for index in all_edges[prediction_index]
                    if gold_findings[index].code != finding.code
                ),
                matched_gold_indices=matched_gold_indices,
                gold_findings=gold_findings,
                evidence_start=evidence_start,
                evidence_end=evidence_end,
            )
        )
        if evidence_failure:
            evidence_failure_count += 1
        false_positives.append(
            FalsePositive(
                prediction_index=prediction_index,
                code=finding.code,
                evidence=finding.evidence,
                reason=reason,
                occurrences=len(occurrences),
                duplicate_of=duplicate,
                matching_gold_issue_id=issue_id,
                matching_gold_code=issue_code,
                evidence_failure=evidence_failure,
            )
        )
    tp = len(matched_pairs)
    fp = len(false_positives)
    fn = len(gold_findings) - tp
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _f1(tp, fp, fn)
    exact_match = tp == len(gold_findings) and fp == 0
    if clean_case and tp == 0 and fp == 0:
        reward = 1.0
    else:
        reward = f1 if f1 is not None else 0.0
    missed_gold = tuple(
        MissedGold(
            issue_id=gold_findings[gold_index].issue_id,
            code=gold_findings[gold_index].code,
            evidence=gold_findings[gold_index].evidence,
        )
        for gold_index in gold_order
        if gold_index not in matched_gold_indices
    )
    return CaseScore(
        case_id=scoring_case.case_id,
        valid=True,
        error_kind=None,
        error=None,
        reward=reward,
        exact_match=exact_match,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        matched_pairs=matched_pairs,
        false_positives=tuple(false_positives),
        missed_gold=missed_gold,
        duplicate_count=duplicate_count,
        evidence_failure_count=evidence_failure_count,
        clean_case=clean_case,
        prediction_count=len(predictions),
        gold_count=len(gold_findings),
    )


def _hint_lines(score: CaseScore) -> list[str]:
    hints: list[str] = []
    if not score.valid:
        hints.append(_SCHEMA_HINT)
    for missed in score.missed_gold:
        hints.append(_RULE_HINTS[missed.code])
    for false_positive in score.false_positives:
        if false_positive.duplicate_of is not None:
            hints.append(_DUPLICATE_HINT)
        if false_positive.evidence_failure:
            hints.append(_EVIDENCE_HINT)
        if false_positive.matching_gold_code is not None:
            if false_positive.matching_gold_code != false_positive.code:
                hints.append(_PRECEDENCE_HINT)
            elif "context window" in false_positive.reason:
                hints.append(_CONTEXT_HINT)
            elif "already matched" in false_positive.reason:
                hints.append(_ONE_TO_ONE_HINT)
        if (
            false_positive.matching_gold_issue_id is None
            and not false_positive.evidence_failure
            and "anchor" in false_positive.reason
        ):
            hints.append(_ANCHOR_HINT)
        if score.clean_case:
            hints.append(_CLEAN_HINT)
    if score.false_positives and not hints:
        hints.append(_UNSUPPORTED_FINDING_HINT)
    return list(dict.fromkeys(hints))


def build_feedback(score: CaseScore) -> str:
    """Build deterministic, rule-specific GEPA feedback from a case score."""
    lines = [f"Reward: {score.reward:.4f}"]
    lines.append(
        f"Gold issues credited: {score.tp} of {score.gold_count}."
    )
    if not score.valid:
        lines.append(
            f"Invalid output ({score.error_kind}): {score.error}"
        )
    if score.matched_pairs:
        lines.append(
            "Matched gold issues: "
            + ", ".join(score.matched_issue_ids)
        )
    else:
        lines.append("Matched gold issues: none.")
    if score.missed_gold:
        lines.append("Missed gold issues:")
        for missed in score.missed_gold:
            lines.append(
                f"- {missed.issue_id} ({missed.code}): expected evidence "
                f"{missed.evidence!r}"
            )
    else:
        lines.append("Missed gold issues: none.")
    if score.false_positives:
        lines.append("False positives:")
        for false_positive in score.false_positives:
            lines.append(
                f"- prediction #{false_positive.prediction_index} "
                f"({false_positive.code}) evidence "
                f"{false_positive.evidence!r}: {false_positive.reason}"
            )
    else:
        lines.append("False positives: none.")
    lines.append(f"Duplicate predictions: {score.duplicate_count}")
    lines.append(
        f"Evidence validation failures: {score.evidence_failure_count}"
    )
    if score.valid and score.clean_case and not score.false_positives:
        lines.append("No findings were expected and none were returned.")
        return "\n".join(lines)
    if score.valid and score.exact_match:
        lines.append(
            "All gold issues were matched with no false positives."
        )
        return "\n".join(lines)
    hints = _hint_lines(score)
    if hints:
        lines.append("Corrective hints:")
        for hint in hints:
            lines.append(f"- {hint}")
    return "\n".join(lines)


def aggregate_scores(scores: Sequence[CaseScore]) -> CorpusScore:
    """Aggregate per-article scores into corpus and per-category metrics."""
    score_list = tuple(scores)
    article_count = len(score_list)
    valid_count = sum(1 for score in score_list if score.valid)
    invalid_count = article_count - valid_count
    parse_error_count = sum(
        1 for score in score_list if score.error_kind == "parse_error"
    )
    request_error_count = sum(
        1 for score in score_list if score.error_kind == "request_error"
    )
    missing_output_count = sum(
        1 for score in score_list if score.error_kind == "missing_output"
    )
    evidence_failure_count = sum(
        score.evidence_failure_count for score in score_list
    )
    duplicate_prediction_count = sum(
        score.duplicate_count for score in score_list
    )
    tp = sum(score.tp for score in score_list)
    fp = sum(score.fp for score in score_list)
    fn = sum(score.fn for score in score_list)
    category_counts = {code: [0, 0, 0] for code in ALLOWED_CODES}
    for score in score_list:
        for pair in score.matched_pairs:
            category_counts[pair.code][0] += 1
        for false_positive in score.false_positives:
            category_counts[false_positive.code][1] += 1
        for missed in score.missed_gold:
            category_counts[missed.code][2] += 1
    per_category: list[CategoryScore] = []
    for code in ALLOWED_CODES:
        category_tp, category_fp, category_fn = category_counts[code]
        per_category.append(
            CategoryScore(
                code=code,
                tp=category_tp,
                fp=category_fp,
                fn=category_fn,
                precision=_ratio(
                    category_tp, category_tp + category_fp
                ),
                recall=_ratio(
                    category_tp, category_tp + category_fn
                ),
                f1=_f1(category_tp, category_fp, category_fn),
            )
        )
    category_f1_values = [category.f1 for category in per_category]
    macro_f1 = (
        sum(value for value in category_f1_values if value is not None)
        / len(ALLOWED_CODES)
        if all(value is not None for value in category_f1_values)
        else None
    )
    exact_match_count = sum(
        1 for score in score_list if score.exact_match
    )
    clean_article_count = sum(
        1 for score in score_list if score.clean_case
    )
    clean_articles_falsely_flagged = sum(
        1
        for score in score_list
        if score.clean_case and score.valid and score.fp > 0
    )
    return CorpusScore(
        article_count=article_count,
        valid_count=valid_count,
        invalid_count=invalid_count,
        parse_error_count=parse_error_count,
        request_error_count=request_error_count,
        missing_output_count=missing_output_count,
        evidence_failure_count=evidence_failure_count,
        duplicate_prediction_count=duplicate_prediction_count,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=_ratio(tp, tp + fp),
        recall=_ratio(tp, tp + fn),
        micro_f1=_f1(tp, fp, fn),
        macro_f1=macro_f1,
        exact_match_count=exact_match_count,
        exact_match_rate=_ratio(exact_match_count, article_count),
        parsing_success=_ratio(valid_count, article_count),
        clean_article_count=clean_article_count,
        clean_articles_falsely_flagged=clean_articles_falsely_flagged,
        mean_reward=(
            sum(score.reward for score in score_list) / article_count
            if article_count
            else None
        ),
        per_category=tuple(per_category),
        cases=score_list,
    )


def score_corpus(
    cases: Sequence[object], predictions: Sequence[object]
) -> CorpusScore:
    """Score paired cases and predictions; lengths must match exactly."""
    case_list = tuple(cases)
    prediction_list = tuple(predictions)
    if len(case_list) != len(prediction_list):
        raise ValueError(
            "cases and predictions must have the same length; "
            f"found {len(case_list)} and {len(prediction_list)}"
        )
    return aggregate_scores(
        tuple(
            score_case(case, prediction)
            for case, prediction in zip(case_list, prediction_list)
        )
    )


def scalar_metric(
    gold: object,
    pred: object,
    trace: object = None,
    pred_name: object = None,
    pred_trace: object = None,
) -> float:
    """Scalar 0-1 article reward for ordinary DSPy evaluation."""
    del trace, pred_name, pred_trace
    return score_case(gold, pred).reward


def feedback_metric(
    gold: object,
    pred: object,
    trace: object = None,
    pred_name: object = None,
    pred_trace: object = None,
) -> Any:
    """GEPA-compatible metric returning score and deterministic feedback."""
    del trace, pred_name, pred_trace
    score = score_case(gold, pred)
    return dspy.Prediction(
        score=score.reward, feedback=build_feedback(score)
    )


__all__ = [
    "CaseScore",
    "CategoryScore",
    "CorpusScore",
    "FalsePositive",
    "InvalidKind",
    "InvalidPrediction",
    "MatchedPair",
    "MissedGold",
    "ScoringCase",
    "ScoringConfigurationError",
    "ScoringInvariantError",
    "aggregate_scores",
    "build_feedback",
    "feedback_metric",
    "parse_prediction",
    "scalar_metric",
    "score_case",
    "score_corpus",
    "scoring_case_from_case",
    "validate_scoring_case",
]
