from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from dspy_lab.schemas import (
    ALLOWED_CODES,
    EVIDENCE_MAX_LENGTH,
    EXPLANATION_MAX_LENGTH,
    CaseRecord,
    EditorialQAOutput,
    Finding,
    GoldFinding,
    MatchOption,
    ProductFacts,
    ReviewRecord,
    SourceRecord,
)


def valid_review() -> dict[str, object]:
    return {
        "author_id": "author-1",
        "reviewer_id": "reviewer-1",
        "reviewer_kind": "human",
        "author_decision": "pending",
        "reviewer_decision": "pending",
        "reconciled": False,
        "reconciliation_notes": "Review pending.",
        "reviewed_at": None,
    }


def valid_match_option() -> dict[str, int]:
    return {
        "anchor_start": 5,
        "anchor_end": 10,
        "context_start": 0,
        "context_end": 20,
    }


def valid_gold_finding() -> dict[str, object]:
    return {
        "issue_id": "issue-1",
        "code": "UNRESOLVED_PLACEHOLDER",
        "evidence": "[DOPLNIT CENU]",
        "explanation": "V textu zůstala redakční značka.",
        "match_options": [valid_match_option()],
    }


def valid_case() -> dict[str, object]:
    return {
        "case_id": "train-001",
        "family_id": "family-train-001",
        "title": "Testovací titulek",
        "article_text": "Testovací text článku.",
        "product_facts": {},
        "sources": [],
        "rules_version": "editorial_rules_v1",
        "gold_findings": [],
        "review": valid_review(),
    }


def test_editorial_output_accepts_empty_findings() -> None:
    output = EditorialQAOutput.model_validate({"findings": []})
    assert output.findings == []


@pytest.mark.parametrize("code", ALLOWED_CODES)
def test_editorial_output_accepts_each_allowed_code(code: str) -> None:
    output = EditorialQAOutput.model_validate(
        {
            "findings": [
                {
                    "code": code,
                    "evidence": "přesná ukázka",
                    "explanation": "Krátké české vysvětlení.",
                }
            ]
        }
    )
    assert output.findings[0].code == code


def test_editorial_output_rejects_unknown_code() -> None:
    with pytest.raises(ValidationError):
        EditorialQAOutput.model_validate(
            {
                "findings": [
                    {
                        "code": "UNKNOWN_CODE",
                        "evidence": "přesná ukázka",
                        "explanation": "Krátké české vysvětlení.",
                    }
                ]
            }
        )


def test_editorial_output_requires_findings_list() -> None:
    with pytest.raises(ValidationError):
        EditorialQAOutput.model_validate({})
    with pytest.raises(ValidationError):
        EditorialQAOutput.model_validate({"findings": "not-a-list"})


def test_editorial_output_rejects_extra_root_field() -> None:
    with pytest.raises(ValidationError):
        EditorialQAOutput.model_validate({"findings": [], "score": 1.0})


def test_finding_rejects_extra_missing_and_wrong_typed_fields() -> None:
    base = {
        "code": "UNRESOLVED_PLACEHOLDER",
        "evidence": "[DOPLNIT CENU]",
        "explanation": "V textu zůstala redakční značka.",
    }
    for payload in (
        {**base, "extra": "forbidden"},
        {"code": base["code"], "evidence": base["evidence"]},
        {**base, "code": 123},
        {**base, "evidence": 123},
        {**base, "explanation": 123},
    ):
        with pytest.raises(ValidationError):
            Finding.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "code": "UNRESOLVED_PLACEHOLDER",
            "evidence": "",
            "explanation": "Vysvětlení.",
        },
        {
            "code": "UNRESOLVED_PLACEHOLDER",
            "evidence": "   ",
            "explanation": "Vysvětlení.",
        },
        {
            "code": "UNRESOLVED_PLACEHOLDER",
            "evidence": "[DOPLNIT CENU]",
            "explanation": "   ",
        },
        {
            "code": "UNRESOLVED_PLACEHOLDER",
            "evidence": "x" * (EVIDENCE_MAX_LENGTH + 1),
            "explanation": "Vysvětlení.",
        },
        {
            "code": "UNRESOLVED_PLACEHOLDER",
            "evidence": "[DOPLNIT CENU]",
            "explanation": "x" * (EXPLANATION_MAX_LENGTH + 1),
        },
    ],
)
def test_finding_rejects_invalid_text(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Finding.model_validate(payload)


def test_finding_accepts_maximum_lengths_and_does_not_deduplicate() -> None:
    output = EditorialQAOutput.model_validate(
        {
            "findings": [
                {
                    "code": "UNRESOLVED_PLACEHOLDER",
                    "evidence": "x" * EVIDENCE_MAX_LENGTH,
                    "explanation": "y" * EXPLANATION_MAX_LENGTH,
                },
                {
                    "code": "UNRESOLVED_PLACEHOLDER",
                    "evidence": "x" * EVIDENCE_MAX_LENGTH,
                    "explanation": "y" * EXPLANATION_MAX_LENGTH,
                },
            ]
        }
    )
    assert len(output.findings) == 2


def test_product_facts_accepts_empty_and_numeric_mass() -> None:
    empty = ProductFacts.model_validate({})
    assert empty.mass_g is None
    facts = ProductFacts.model_validate(
        {"metal": "gold", "mass_g": 1, "purity": "999.9/1000"}
    )
    assert facts.mass_g == Decimal("1")
    assert ProductFacts.model_validate({"mass_g": 1.5}).mass_g == Decimal("1.5")


@pytest.mark.parametrize(
    "payload",
    [
        {"mass_g": 0},
        {"mass_g": -1},
        {"mass_g": "1"},
        {"mass_g": True},
        {"purity": "not-a-purity"},
        {"extra_field": "forbidden"},
        {"attributes": {"Bad-Key": "value"}},
        {"attributes": {"key": "   "}},
    ],
)
def test_product_facts_rejects_invalid_values(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ProductFacts.model_validate(payload)


def test_source_record_defaults_and_validation() -> None:
    source = SourceRecord.model_validate(
        {
            "source_id": "S1",
            "excerpt": "Referenční text.",
            "provenance": "Authored fixture",
        }
    )
    assert source.title == ""
    assert source.published_at is None
    assert SourceRecord.model_validate(
        {
            "source_id": "S2",
            "excerpt": "Referenční text.",
            "provenance": "Authored fixture",
            "published_at": "2026-10-19",
        }
    ).published_at == "2026-10-19"
    for payload in (
        {"source_id": "S1", "excerpt": "   ", "provenance": "x"},
        {"source_id": "S1", "excerpt": "x", "provenance": "   "},
        {"source_id": "S1", "excerpt": "x", "provenance": "x", "title": "   "},
        {
            "source_id": "S1",
            "excerpt": "x",
            "provenance": "x",
            "published_at": "19.10.2026",
        },
        {"source_id": "1S", "excerpt": "x", "provenance": "x"},
    ):
        with pytest.raises(ValidationError):
            SourceRecord.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "anchor_start": 5,
            "anchor_end": 10,
            "context_start": 6,
            "context_end": 20,
        },
        {
            "anchor_start": 5,
            "anchor_end": 5,
            "context_start": 0,
            "context_end": 20,
        },
        {
            "anchor_start": 5,
            "anchor_end": 21,
            "context_start": 0,
            "context_end": 20,
        },
        {
            "anchor_start": -1,
            "anchor_end": 10,
            "context_start": 0,
            "context_end": 20,
        },
    ],
)
def test_match_option_rejects_invalid_order(payload: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        MatchOption.model_validate(payload)


def test_gold_finding_requires_match_options_and_valid_text() -> None:
    GoldFinding.model_validate(valid_gold_finding())
    for payload in (
        {**valid_gold_finding(), "match_options": []},
        {**valid_gold_finding(), "evidence": "   "},
        {**valid_gold_finding(), "explanation": "   "},
        {**valid_gold_finding(), "extra": "forbidden"},
    ):
        with pytest.raises(ValidationError):
            GoldFinding.model_validate(payload)


def test_review_record_pending_and_reconciled_states() -> None:
    ReviewRecord.model_validate(valid_review())
    approved = {
        **valid_review(),
        "author_decision": "approved",
        "reviewer_decision": "approved",
        "reconciled": True,
        "reviewed_at": "2026-10-19T00:00:00Z",
    }
    ReviewRecord.model_validate(approved)
    for payload in (
        {**approved, "reviewer_decision": "pending"},
        {**approved, "reviewed_at": None},
        {**approved, "reviewed_at": "2026-10-19 00:00:00"},
        {**approved, "reviewer_kind": "robot"},
        {**approved, "extra": "forbidden"},
    ):
        with pytest.raises(ValidationError):
            ReviewRecord.model_validate(payload)


def test_case_record_accepts_minimal_case() -> None:
    case = CaseRecord.model_validate(valid_case())
    assert case.gold_findings == []
    assert case.product_facts.mass_g is None
    assert case.sources == []


def test_case_record_rejects_duplicate_ids_and_evidence() -> None:
    duplicate_source = {
        **valid_case(),
        "sources": [
            {"source_id": "S1", "excerpt": "x", "provenance": "p"},
            {"source_id": "S1", "excerpt": "y", "provenance": "p"},
        ],
    }
    duplicate_issue = {
        **valid_case(),
        "gold_findings": [
            valid_gold_finding(),
            {**valid_gold_finding(), "evidence": "jiná ukázka"},
        ],
    }
    duplicate_evidence = {
        **valid_case(),
        "gold_findings": [
            valid_gold_finding(),
            {**valid_gold_finding(), "issue_id": "issue-2"},
        ],
    }
    for payload in (duplicate_source, duplicate_issue, duplicate_evidence):
        with pytest.raises(ValidationError):
            CaseRecord.model_validate(payload)


def test_case_record_rejects_invalid_metadata() -> None:
    for payload in (
        {**valid_case(), "title": "line\nbreak"},
        {**valid_case(), "title": "   "},
        {**valid_case(), "article_text": "   "},
        {**valid_case(), "rules_version": "editorial_rules_v2"},
        {**valid_case(), "case_id": "Train-001"},
        {**valid_case(), "family_id": "Family Train"},
        {**valid_case(), "extra": "forbidden"},
    ):
        with pytest.raises(ValidationError):
            CaseRecord.model_validate(payload)
    payload_without_review = valid_case()
    del payload_without_review["review"]
    with pytest.raises(ValidationError):
        CaseRecord.model_validate(payload_without_review)
