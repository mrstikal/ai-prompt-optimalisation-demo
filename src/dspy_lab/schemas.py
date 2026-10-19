"""Strict schemas for findings, cases, gold annotations, and reviews."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

ErrorCode = Literal[
    "UNSUPPORTED_NUMERIC_CLAIM",
    "INVESTMENT_ADVICE_AS_FACT",
    "UNWARRANTED_CERTAINTY",
    "UNRESOLVED_PLACEHOLDER",
    "INVALID_CITATION",
    "PRODUCT_FACT_MISMATCH",
]

ALLOWED_CODES: tuple[ErrorCode, ...] = (
    "UNSUPPORTED_NUMERIC_CLAIM",
    "INVESTMENT_ADVICE_AS_FACT",
    "UNWARRANTED_CERTAINTY",
    "UNRESOLVED_PLACEHOLDER",
    "INVALID_CITATION",
    "PRODUCT_FACT_MISMATCH",
)

RULES_VERSION = "editorial_rules_v1"
EVIDENCE_MAX_LENGTH = 320
EXPLANATION_MAX_LENGTH = 400

CASE_ID_PATTERN = r"^[a-z0-9][a-z0-9-]*$"
ISSUE_ID_PATTERN = r"^[a-z0-9][a-z0-9-]*$"
SOURCE_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]*$"
PURITY_PATTERN = r"^\d{1,4}(\.\d+)?/\d{1,4}$"
ISO_UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
ISO_DATE_OR_UTC_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}Z)?$"
)
ATTRIBUTE_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _require_non_whitespace(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must contain non-whitespace content")
    return value


class Finding(BaseModel):
    """One model-returned editorial finding; no IDs, scores, or offsets."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    code: ErrorCode
    evidence: str = Field(min_length=1, max_length=EVIDENCE_MAX_LENGTH)
    explanation: str = Field(min_length=1, max_length=EXPLANATION_MAX_LENGTH)

    @field_validator("evidence")
    @classmethod
    def _evidence_non_whitespace(cls, value: str) -> str:
        return _require_non_whitespace(value, "evidence")

    @field_validator("explanation")
    @classmethod
    def _explanation_non_whitespace(cls, value: str) -> str:
        return _require_non_whitespace(value, "explanation")


class EditorialQAOutput(BaseModel):
    """Strict task output: exactly one required findings list."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    findings: list[Finding]


class ProductFacts(BaseModel):
    """Optional typed product reference attributes."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    metal: str | None = Field(default=None, min_length=1, max_length=64)
    mass_g: Decimal | None = Field(default=None, gt=0)
    purity: str | None = Field(default=None, pattern=PURITY_PATTERN)
    denomination: str | None = Field(default=None, min_length=1, max_length=128)
    origin: str | None = Field(default=None, min_length=1, max_length=128)
    dimensions_mm: str | None = Field(default=None, min_length=1, max_length=128)
    reference_id: str | None = Field(default=None, min_length=1, max_length=128)
    attributes: dict[str, str] = Field(default_factory=dict)

    @field_validator("mass_g", mode="before")
    @classmethod
    def _parse_mass(cls, value: object) -> object:
        if value is None or isinstance(value, Decimal):
            return value
        if isinstance(value, bool):
            raise ValueError("mass_g must be a positive JSON number")
        if isinstance(value, int):
            return Decimal(value)
        if isinstance(value, float):
            return Decimal(str(value))
        raise ValueError("mass_g must be a positive JSON number")

    @field_validator("attributes")
    @classmethod
    def _validate_attributes(cls, value: dict[str, str]) -> dict[str, str]:
        for key, item in value.items():
            if ATTRIBUTE_KEY_PATTERN.fullmatch(key) is None:
                raise ValueError(
                    f"attribute key {key!r} must match [a-z][a-z0-9_]*"
                )
            if not item.strip():
                raise ValueError(
                    f"attribute {key!r} must contain non-whitespace content"
                )
        return value


class SourceRecord(BaseModel):
    """One local source supplied with a case."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    source_id: str = Field(
        min_length=1, max_length=32, pattern=SOURCE_ID_PATTERN
    )
    title: str = Field(default="", max_length=300)
    excerpt: str = Field(min_length=1, max_length=4_000)
    provenance: str = Field(min_length=1, max_length=500)
    published_at: str | None = None

    @field_validator("excerpt")
    @classmethod
    def _excerpt_non_whitespace(cls, value: str) -> str:
        return _require_non_whitespace(value, "source excerpt")

    @field_validator("provenance")
    @classmethod
    def _provenance_non_whitespace(cls, value: str) -> str:
        return _require_non_whitespace(value, "source provenance")

    @field_validator("title")
    @classmethod
    def _title_empty_or_non_whitespace(cls, value: str) -> str:
        if value and not value.strip():
            raise ValueError("source title must be empty or non-whitespace")
        return value

    @field_validator("published_at")
    @classmethod
    def _published_at_format(cls, value: str | None) -> str | None:
        if value is not None and ISO_DATE_OR_UTC_PATTERN.fullmatch(value) is None:
            raise ValueError(
                "published_at must be YYYY-MM-DD or UTC ISO-8601 ending in Z"
            )
        return value


class MatchOption(BaseModel):
    """One reviewed anchor and context window for a gold issue."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    anchor_start: int = Field(ge=0)
    anchor_end: int = Field(gt=0)
    context_start: int = Field(ge=0)
    context_end: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_order(self) -> "MatchOption":
        if not (
            self.context_start
            <= self.anchor_start
            < self.anchor_end
            <= self.context_end
        ):
            raise ValueError(
                "require context_start <= anchor_start < anchor_end "
                "<= context_end"
            )
        return self


class GoldFinding(BaseModel):
    """One independently reviewed issue occurrence."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    issue_id: str = Field(
        min_length=1, max_length=64, pattern=ISSUE_ID_PATTERN
    )
    code: ErrorCode
    evidence: str = Field(min_length=1, max_length=EVIDENCE_MAX_LENGTH)
    explanation: str = Field(min_length=1, max_length=EXPLANATION_MAX_LENGTH)
    match_options: list[MatchOption] = Field(min_length=1)

    @field_validator("evidence")
    @classmethod
    def _evidence_non_whitespace(cls, value: str) -> str:
        return _require_non_whitespace(value, "gold evidence")

    @field_validator("explanation")
    @classmethod
    def _explanation_non_whitespace(cls, value: str) -> str:
        return _require_non_whitespace(value, "gold explanation")


class ReviewRecord(BaseModel):
    """Author/reviewer decisions and reconciliation for one case."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    author_id: str = Field(min_length=1, max_length=100)
    reviewer_id: str = Field(min_length=1, max_length=100)
    reviewer_kind: Literal["human", "automated"]
    author_decision: Literal["pending", "approved", "changes_requested"]
    reviewer_decision: Literal["pending", "approved", "changes_requested"]
    reconciled: bool
    reconciliation_notes: str = Field(min_length=1, max_length=2_000)
    reviewed_at: str | None = None

    @model_validator(mode="after")
    def _validate_review(self) -> "ReviewRecord":
        if self.reconciled and (
            self.author_decision != "approved"
            or self.reviewer_decision != "approved"
        ):
            raise ValueError(
                "reconciled review requires both decisions to be approved"
            )
        if self.reviewer_decision != "pending" and self.reviewed_at is None:
            raise ValueError(
                "reviewed_at is required once a reviewer decision exists"
            )
        if (
            self.reviewed_at is not None
            and ISO_UTC_PATTERN.fullmatch(self.reviewed_at) is None
        ):
            raise ValueError(
                "reviewed_at must be a UTC ISO-8601 timestamp ending in Z"
            )
        return self


class CaseRecord(BaseModel):
    """One JSONL case with its gold annotations and review record."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    case_id: str = Field(
        min_length=1, max_length=64, pattern=CASE_ID_PATTERN
    )
    family_id: str = Field(
        min_length=1, max_length=64, pattern=CASE_ID_PATTERN
    )
    title: str = Field(min_length=1, max_length=300)
    article_text: str = Field(min_length=1, max_length=20_000)
    product_facts: ProductFacts = Field(default_factory=ProductFacts)
    sources: list[SourceRecord] = Field(default_factory=list)
    rules_version: Literal["editorial_rules_v1"]
    gold_findings: list[GoldFinding] = Field(default_factory=list)
    review: ReviewRecord

    @field_validator("title")
    @classmethod
    def _title_non_whitespace_and_single_line(cls, value: str) -> str:
        _require_non_whitespace(value, "title")
        if "\n" in value or "\r" in value:
            raise ValueError("title must be a single line")
        return value

    @field_validator("article_text")
    @classmethod
    def _article_non_whitespace(cls, value: str) -> str:
        return _require_non_whitespace(value, "article_text")

    @model_validator(mode="after")
    def _validate_case(self) -> "CaseRecord":
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_id values must be unique within a case")
        issue_ids = [finding.issue_id for finding in self.gold_findings]
        if len(issue_ids) != len(set(issue_ids)):
            raise ValueError("issue_id values must be unique within a case")
        evidence = [finding.evidence for finding in self.gold_findings]
        if len(evidence) != len(set(evidence)):
            raise ValueError(
                "gold evidence quotes must be unique within a case"
            )
        return self


__all__ = [
    "ALLOWED_CODES",
    "CASE_ID_PATTERN",
    "EVIDENCE_MAX_LENGTH",
    "EXPLANATION_MAX_LENGTH",
    "EditorialQAOutput",
    "ErrorCode",
    "Finding",
    "GoldFinding",
    "ISSUE_ID_PATTERN",
    "MatchOption",
    "ProductFacts",
    "RULES_VERSION",
    "ReviewRecord",
    "SOURCE_ID_PATTERN",
    "SourceRecord",
]
