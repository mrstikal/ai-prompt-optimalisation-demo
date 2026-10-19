from __future__ import annotations

from pathlib import Path

import pytest

from dspy_lab.config import PROJECT_ROOT
from dspy_lab.data import (
    TITLE_BODY_SEPARATOR,
    DatasetValidationError,
    canonical_document,
    canonical_document_sha256,
    canonicalize_document,
    canonicalize_text,
    find_literal_occurrences,
    load_cases,
    validate_case_offsets,
)
from dspy_lab.schemas import ALLOWED_CODES, CaseRecord

TRAIN_PATH = PROJECT_ROOT / "data/v1/train.jsonl"
RULES_PATH = PROJECT_ROOT / "data/editorial_rules_v1.md"


def test_canonicalize_text_applies_nfc_lf_and_space_normalization() -> None:
    assert canonicalize_text("e\u0301") == "é"
    assert canonicalize_text("a\r\nb\rc") == "a\nb\nc"
    assert canonicalize_text("\u00a0x\u202fy") == " x y"
    assert canonicalize_text("Žluťoučký kůň, „test“!") == (
        "Žluťoučký kůň, „test“!"
    )


def test_canonicalize_document_uses_a_blank_line_separator() -> None:
    assert canonicalize_document("Titulek", "Text\u00a0článku.") == (
        "Titulek\n\nText článku."
    )


def test_find_literal_occurrences_is_literal_and_includes_overlaps() -> None:
    assert find_literal_occurrences("aaaa", "aa") == [0, 1, 2]
    assert find_literal_occurrences("abc", "z") == []
    assert find_literal_occurrences("abc", "") == []


def test_five_training_specimens_load_and_cover_all_codes() -> None:
    cases = load_cases(TRAIN_PATH)
    assert len(cases) == 5
    assert [case.case_id for case in cases] == [
        "train-001",
        "train-002",
        "train-003",
        "train-004",
        "train-005",
    ]
    assert [len(case.gold_findings) for case in cases] == [0, 2, 2, 2, 1]
    assert sum(len(case.gold_findings) for case in cases) == 7
    assert {
        finding.code for case in cases for finding in case.gold_findings
    } == set(ALLOWED_CODES)
    assert cases[0].gold_findings == []


def test_specimen_codes_reflect_precedence_and_non_violations() -> None:
    cases = {case.case_id: case for case in load_cases(TRAIN_PATH)}
    assert cases["train-001"].gold_findings == []
    assert {finding.code for finding in cases["train-002"].gold_findings} == {
        "PRODUCT_FACT_MISMATCH",
        "UNRESOLVED_PLACEHOLDER",
    }
    assert {finding.code for finding in cases["train-003"].gold_findings} == {
        "INVESTMENT_ADVICE_AS_FACT",
        "UNWARRANTED_CERTAINTY",
    }
    assert {finding.code for finding in cases["train-004"].gold_findings} == {
        "UNSUPPORTED_NUMERIC_CLAIM",
        "INVALID_CITATION",
    }
    assert {finding.code for finding in cases["train-005"].gold_findings} == {
        "INVALID_CITATION",
    }


def test_every_gold_offset_points_at_the_reviewed_evidence() -> None:
    for case in load_cases(TRAIN_PATH):
        canonical = canonical_document(case)
        title = canonicalize_text(case.title)
        separator_start = len(title)
        separator_end = separator_start + len(TITLE_BODY_SEPARATOR)
        for finding in case.gold_findings:
            occurrences = find_literal_occurrences(canonical, finding.evidence)
            assert len(occurrences) == 1
            for option in finding.match_options:
                anchor = canonical[option.anchor_start : option.anchor_end]
                context = canonical[option.context_start : option.context_end]
                assert anchor in finding.evidence
                assert finding.evidence in context
                assert not (
                    option.context_start < separator_end
                    and option.context_end > separator_start
                )
                assert not (
                    option.anchor_start < separator_end
                    and option.anchor_end > separator_start
                )


def test_canonical_hashes_are_stable_and_text_sensitive() -> None:
    cases = load_cases(TRAIN_PATH)
    hashes = {canonical_document_sha256(case) for case in cases}
    assert len(hashes) == 5
    assert all(len(value) == 64 for value in hashes)
    modified = cases[0].model_copy(
        update={"article_text": cases[0].article_text + " Navíc."}
    )
    assert canonical_document_sha256(cases[0]) != canonical_document_sha256(
        modified
    )


def test_ambiguous_evidence_is_rejected() -> None:
    case = load_cases(TRAIN_PATH)[1]
    payload = case.model_dump()
    payload["article_text"] = (
        "Tento stříbrný slitek má hmotnost 10 g. hmotnost 10 g."
    )
    mutated = CaseRecord.model_validate(payload)
    with pytest.raises(DatasetValidationError, match="exactly once"):
        validate_case_offsets(mutated)


def test_anchor_outside_evidence_is_rejected() -> None:
    case = load_cases(TRAIN_PATH)[1]
    payload = case.model_dump()
    option = payload["gold_findings"][0]["match_options"][0]
    option["anchor_start"] = option["anchor_end"]
    option["anchor_end"] = option["anchor_end"] + 1
    mutated = CaseRecord.model_validate(payload)
    with pytest.raises(DatasetValidationError, match="no match option"):
        validate_case_offsets(mutated)


def test_context_too_small_for_evidence_is_rejected() -> None:
    case = load_cases(TRAIN_PATH)[1]
    payload = case.model_dump()
    option = payload["gold_findings"][0]["match_options"][0]
    option["anchor_end"] = option["anchor_start"] + 1
    option["context_start"] = option["anchor_start"]
    option["context_end"] = option["anchor_end"]
    mutated = CaseRecord.model_validate(payload)
    with pytest.raises(DatasetValidationError, match="no match option"):
        validate_case_offsets(mutated)


def _title_only_case(context_end: int) -> CaseRecord:
    title = "Zaručený výnos"
    return CaseRecord.model_validate(
        {
            "case_id": "train-999",
            "family_id": "family-train-999",
            "title": title,
            "article_text": "Text.",
            "product_facts": {},
            "sources": [],
            "rules_version": "editorial_rules_v1",
            "gold_findings": [
                {
                    "issue_id": "train-999-certainty",
                    "code": "UNWARRANTED_CERTAINTY",
                    "evidence": title,
                    "explanation": "Testovací vysvětlení.",
                    "match_options": [
                        {
                            "anchor_start": 0,
                            "anchor_end": len(title),
                            "context_start": 0,
                            "context_end": context_end,
                        }
                    ],
                }
            ],
            "review": {
                "author_id": "author-1",
                "reviewer_id": "reviewer-1",
                "reviewer_kind": "human",
                "author_decision": "pending",
                "reviewer_decision": "pending",
                "reconciled": False,
                "reconciliation_notes": "Review pending.",
                "reviewed_at": None,
            },
        }
    )


def test_title_only_evidence_is_allowed_when_it_stays_in_the_title() -> None:
    validate_case_offsets(_title_only_case(len("Zaručený výnos")))


def test_window_crossing_the_title_body_separator_is_rejected() -> None:
    with pytest.raises(DatasetValidationError, match="separator"):
        validate_case_offsets(_title_only_case(len("Zaručený výnos") + 3))


def test_rules_file_documents_the_closed_taxonomy_and_boundaries() -> None:
    rules = RULES_PATH.read_text(encoding="utf-8")
    assert "editorial_rules_v1" in rules
    for code in ALLOWED_CODES:
        assert code in rules
    lowered = rules.lower()
    for phrase in (
        "precedence",
        "hypothetical",
        "negation",
        "quotation",
        "title/body separator",
        "320",
        "400",
        "disclaimer",
        "absence of a citation marker",
        "universal, unbounded recommendation",
        "out of scope in this version",
    ):
        assert phrase in lowered


def test_specimens_live_in_training_not_a_secret_test_set() -> None:
    assert TRAIN_PATH.is_file()
    assert not (PROJECT_ROOT / "data/v1/test.jsonl").exists()
    assert not (PROJECT_ROOT / "data/v1/validation.jsonl").exists()


def test_every_specimen_has_a_review_record() -> None:
    for case in load_cases(TRAIN_PATH):
        assert case.review.author_id
        assert case.review.reviewer_id
        assert case.review.reviewer_kind in ("human", "automated")
        assert isinstance(case.review.reconciled, bool)


def test_five_specimens_are_reconciled_by_an_independent_review() -> None:
    for case in load_cases(TRAIN_PATH):
        assert case.review.reconciled is True
        assert case.review.author_decision == "approved"
        assert case.review.reviewer_decision == "approved"
        assert case.review.reviewer_kind == "automated"
        assert case.review.reviewer_id == "decision-reviewer"
        assert case.review.reviewed_at is not None
        assert "Human sign-off remains required" in (
            case.review.reconciliation_notes
        )
