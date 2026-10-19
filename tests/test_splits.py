from __future__ import annotations

import pytest

from dspy_lab.config import PROJECT_ROOT
from dspy_lab.data import (
    EVALUATION_CODE_COUNT,
    EVALUATION_COMPOSITION,
    EVALUATION_FINDING_COUNT,
    SPLIT_NAMES,
    TEST_ARTICLE_COUNT,
    TOTAL_ARTICLE_COUNT,
    TOTAL_FINDING_COUNT,
    TRAINING_ARTICLE_COUNT,
    TRAINING_CODE_COUNT,
    TRAINING_COMPOSITION,
    TRAINING_FINDING_COUNT,
    VALIDATION_ARTICLE_COUNT,
    DatasetValidationError,
    canonical_document_sha256,
    dataset_sha256,
    file_sha256,
    five_word_shingles,
    jaccard_similarity,
    load_all_splits,
    load_optimization_cases,
    load_review_audit,
    load_split_manifest,
    masked_template,
    near_duplicate_pairs,
    split_coverage,
    validate_all_splits,
    validate_review_audit,
    validate_split_manifest,
    word_jaccard,
)
from dspy_lab.schemas import ALLOWED_CODES

DATA_DIR = PROJECT_ROOT / "data/v1"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VALIDATION_PATH = DATA_DIR / "validation.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"
REVIEWS_PATH = DATA_DIR / "reviews.jsonl"
MANIFEST_PATH = DATA_DIR / "split_manifest.json"


def test_all_splits_load_and_validate() -> None:
    cases_by_split = load_all_splits(DATA_DIR)
    validate_all_splits(cases_by_split)
    assert [len(cases_by_split[name]) for name in SPLIT_NAMES] == [
        TRAINING_ARTICLE_COUNT,
        VALIDATION_ARTICLE_COUNT,
        TEST_ARTICLE_COUNT,
    ]
    assert sum(len(cases) for cases in cases_by_split.values()) == (
        TOTAL_ARTICLE_COUNT
    )
    assert sum(
        len(case.gold_findings)
        for cases in cases_by_split.values()
        for case in cases
    ) == TOTAL_FINDING_COUNT


def test_split_composition_and_code_coverage() -> None:
    cases_by_split = load_all_splits(DATA_DIR)
    train = split_coverage(cases_by_split["train"])
    assert train["composition"] == {
        **TRAINING_COMPOSITION,
        "more_than_two": 0,
    }
    assert train["finding_count"] == TRAINING_FINDING_COUNT
    assert train["code_counts"] == {
        code: TRAINING_CODE_COUNT for code in ALLOWED_CODES
    }
    assert len(train["clean_cases"]) == 6
    assert len(train["hard_negative_cases"]) >= 3
    for split_name in ("validation", "test"):
        coverage = split_coverage(cases_by_split[split_name])
        assert coverage["composition"] == {
            **EVALUATION_COMPOSITION,
            "more_than_two": 0,
        }
        assert coverage["finding_count"] == EVALUATION_FINDING_COUNT
        assert coverage["code_counts"] == {
            code: EVALUATION_CODE_COUNT for code in ALLOWED_CODES
        }
        assert len(coverage["clean_cases"]) == 2
        assert len(coverage["hard_negative_cases"]) >= 1


def test_split_ids_families_and_hashes_are_disjoint() -> None:
    cases_by_split = load_all_splits(DATA_DIR)
    case_ids = {
        name: {case.case_id for case in cases}
        for name, cases in cases_by_split.items()
    }
    family_ids = {
        name: {case.family_id for case in cases}
        for name, cases in cases_by_split.items()
    }
    hashes = {
        name: {canonical_document_sha256(case) for case in cases}
        for name, cases in cases_by_split.items()
    }
    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            assert case_ids[left].isdisjoint(case_ids[right])
            assert family_ids[left].isdisjoint(family_ids[right])
            assert hashes[left].isdisjoint(hashes[right])


def test_near_duplicate_helpers() -> None:
    assert five_word_shingles("a b c d e f", 5) == {
        ("a", "b", "c", "d", "e"),
        ("b", "c", "d", "e", "f"),
    }
    assert jaccard_similarity({("a", "b")}, {("a", "b")}) == 1.0
    assert jaccard_similarity({("a", "b")}, {("c", "d")}) == 0.0
    assert masked_template("Cena 100 Kč") == masked_template("Cena 200 Kč")
    assert word_jaccard(
        "Hodnota palladia nikdy neklesne a zaručuje stálý výnos.",
        "Hodnota platiny nikdy neklesne a zaručuje stálý výnos.",
    ) >= 0.75


def test_near_duplicate_pairs_flags_identical_cases() -> None:
    case = load_all_splits(DATA_DIR)["train"][1]
    left = case.model_copy(
        update={"case_id": "synthetic-a", "family_id": "family-synthetic-a"}
    )
    right = case.model_copy(
        update={"case_id": "synthetic-b", "family_id": "family-synthetic-b"}
    )
    pairs = near_duplicate_pairs([left, right])
    assert len(pairs) == 1
    assert pairs[0]["jaccard"] == 1.0
    assert pairs[0]["same_template"] is True
    assert pairs[0]["same_evidence"] is True
    assert pairs[0]["same_family"] is False


def test_near_duplicate_pairs_flags_shared_evidence_with_different_text() -> None:
    case = load_all_splits(DATA_DIR)["train"][1]
    left = case.model_copy(
        update={
            "case_id": "synthetic-a",
            "family_id": "family-synthetic-a",
            "title": "Syntetický text A",
            "article_text": "Jiný text A.",
        }
    )
    right = case.model_copy(
        update={
            "case_id": "synthetic-b",
            "family_id": "family-synthetic-b",
            "title": "Syntetický text B",
            "article_text": "Jiný text B.",
        }
    )
    pairs = near_duplicate_pairs([left, right])
    assert len(pairs) == 1
    assert pairs[0]["same_evidence"] is True
    assert pairs[0]["same_template"] is False


def test_no_unresolved_cross_split_near_duplicates() -> None:
    cases_by_split = load_all_splits(DATA_DIR)
    all_cases = [
        case for name in SPLIT_NAMES for case in cases_by_split[name]
    ]
    split_of = {
        case.case_id: name
        for name, cases in cases_by_split.items()
        for case in cases
    }
    for pair in near_duplicate_pairs(all_cases):
        assert not (
            split_of[str(pair["case_a"])]
            != split_of[str(pair["case_b"])]
            and not pair["same_family"]
        )


def test_review_audit_covers_all_50_cases() -> None:
    cases_by_split = load_all_splits(DATA_DIR)
    all_cases = [
        case for name in SPLIT_NAMES for case in cases_by_split[name]
    ]
    entries = load_review_audit(REVIEWS_PATH)
    assert len(entries) == TOTAL_ARTICLE_COUNT
    validate_review_audit(all_cases, entries)
    for case in all_cases:
        assert case.review.reconciled is True
        assert case.review.author_decision == "approved"
        assert case.review.reviewer_decision == "approved"
        assert case.review.reviewer_kind == "automated"
        assert case.review.reviewed_at is not None
        assert "Human sign-off remains required" in (
            case.review.reconciliation_notes
        )


def test_manifest_matches_data_and_file_hashes() -> None:
    cases_by_split = load_all_splits(DATA_DIR)
    manifest = load_split_manifest(MANIFEST_PATH)
    validate_split_manifest(cases_by_split, manifest)
    all_cases = [
        case for name in SPLIT_NAMES for case in cases_by_split[name]
    ]
    assert manifest["total_article_count"] == TOTAL_ARTICLE_COUNT
    assert manifest["total_finding_count"] == TOTAL_FINDING_COUNT
    assert manifest["combined_dataset_sha256"] == dataset_sha256(all_cases)
    assert manifest["test_sealed"] is True
    assert manifest["rules_version"] == "editorial_rules_v1"
    for split_name in SPLIT_NAMES:
        assert manifest["splits"][split_name]["file_sha256"] == (
            file_sha256(DATA_DIR / f"{split_name}.jsonl")
        )
        assert manifest["splits"][split_name]["article_count"] == len(
            cases_by_split[split_name]
        )
    assert not any(
        pair["status"] == "unresolved"
        for pair in manifest["near_duplicate_report"]
    )
    assert not any(
        pair["cross_split"]
        for pair in manifest["near_duplicate_report"]
    )


def test_optimization_loader_rejects_the_test_split() -> None:
    loaded = load_optimization_cases(TRAIN_PATH, VALIDATION_PATH)
    assert set(loaded) == {"train", "validation"}
    assert len(loaded["train"]) == TRAINING_ARTICLE_COUNT
    assert len(loaded["validation"]) == VALIDATION_ARTICLE_COUNT
    with pytest.raises(DatasetValidationError, match="sealed"):
        load_optimization_cases(TRAIN_PATH, TEST_PATH)
    with pytest.raises(DatasetValidationError, match="sealed"):
        load_optimization_cases(TEST_PATH, VALIDATION_PATH)


def test_evaluation_articles_are_nontrivial() -> None:
    cases_by_split = load_all_splits(DATA_DIR)
    for split_name in ("validation", "test"):
        for case in cases_by_split[split_name]:
            words = len(case.article_text.split())
            assert words >= 150
            assert words <= 450
