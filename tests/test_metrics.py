from __future__ import annotations

import json

import dspy
import pytest

from dspy_lab.config import PROJECT_ROOT
from dspy_lab.data import (
    TITLE_BODY_SEPARATOR,
    canonicalize_document,
    canonicalize_text,
    find_literal_occurrences,
    load_all_splits,
)
from dspy_lab.metrics import (
    ALLOWED_CODES,
    InvalidPrediction,
    ScoringCase,
    ScoringConfigurationError,
    aggregate_scores,
    build_feedback,
    feedback_metric,
    parse_prediction,
    scalar_metric,
    score_case,
    score_corpus,
    scoring_case_from_case,
)
from dspy_lab.schemas import (
    CaseRecord,
    EditorialQAOutput,
    GoldFinding,
    MatchOption,
)

DATA_DIR = PROJECT_ROOT / "data/v1"


def make_scoring_case(
    document: str,
    specs: list[dict[str, object]],
    *,
    case_id: str = "fixture-001",
    separator_start: int | None = None,
    separator_end: int | None = None,
) -> ScoringCase:
    findings: list[GoldFinding] = []
    for spec in specs:
        evidence = str(spec["evidence"])
        occurrences = find_literal_occurrences(document, evidence)
        assert len(occurrences) == 1, (
            f"fixture gold evidence must be unique: {evidence!r}"
        )
        evidence_start = occurrences[0]
        evidence_end = evidence_start + len(evidence)
        anchor = str(spec.get("anchor", evidence))
        anchor_start = document.find(anchor, evidence_start)
        assert anchor_start >= evidence_start
        anchor_end = anchor_start + len(anchor)
        assert anchor_end <= evidence_end
        context_start = int(spec.get("context_start", evidence_start))
        context_end = int(spec.get("context_end", evidence_end))
        assert context_start <= evidence_start
        assert evidence_end <= context_end
        findings.append(
            GoldFinding(
                issue_id=str(spec["issue_id"]),
                code=spec["code"],
                evidence=evidence,
                explanation=str(
                    spec.get("explanation", "Testovací vysvětlení.")
                ),
                match_options=[
                    MatchOption(
                        anchor_start=anchor_start,
                        anchor_end=anchor_end,
                        context_start=context_start,
                        context_end=context_end,
                    )
                ],
            )
        )
    return ScoringCase(
        case_id=case_id,
        document_text=document,
        gold_findings=tuple(findings),
        separator_start=separator_start,
        separator_end=separator_end,
    )


def make_prediction(*findings: dict[str, object]) -> EditorialQAOutput:
    return EditorialQAOutput.model_validate({"findings": list(findings)})


def finding(
    code: str,
    evidence: str,
    explanation: str = "Vysvětlení.",
) -> dict[str, object]:
    return {
        "code": code,
        "evidence": evidence,
        "explanation": explanation,
    }


def single_issue_case() -> ScoringCase:
    document = "Tento stříbrný slitek má hmotnost 10 g."
    return make_scoring_case(
        document,
        [
            {
                "issue_id": "fixture-mass",
                "code": "PRODUCT_FACT_MISMATCH",
                "evidence": "hmotnost 10 g",
                "anchor": "10 g",
                "context_start": 0,
                "context_end": len(document),
            }
        ],
    )


def test_perfect_match_scores_one_and_exact_match() -> None:
    case = single_issue_case()
    score = score_case(
        case,
        make_prediction(finding("PRODUCT_FACT_MISMATCH", "hmotnost 10 g")),
    )
    assert score.valid is True
    assert score.tp == 1
    assert score.fp == 0
    assert score.fn == 0
    assert score.reward == 1.0
    assert score.exact_match is True
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.f1 == 1.0
    assert score.matched_issue_ids == ("fixture-mass",)
    assert score.missed_gold == ()
    assert score.false_positives == ()


def test_clean_case_with_empty_findings_scores_one() -> None:
    case = make_scoring_case("Čistý text bez chyby.", [])
    score = score_case(case, make_prediction())
    assert score.valid is True
    assert score.clean_case is True
    assert score.reward == 1.0
    assert score.exact_match is True
    assert score.tp == score.fp == score.fn == 0
    assert score.precision is None
    assert score.recall is None
    assert score.f1 is None


def test_clean_case_with_false_positive_scores_zero() -> None:
    case = make_scoring_case("Čistý text bez chyby.", [])
    score = score_case(
        case, make_prediction(finding("UNRESOLVED_PLACEHOLDER", "Čistý"))
    )
    assert score.valid is True
    assert score.tp == 0
    assert score.fp == 1
    assert score.fn == 0
    assert score.reward == 0.0
    assert score.exact_match is False
    assert score.precision == 0.0
    assert score.recall is None
    assert score.f1 == 0.0
    assert score.evidence_failure_count == 0


def test_invalid_output_on_clean_case_scores_zero() -> None:
    case = make_scoring_case("Čistý text bez chyby.", [])
    score = score_case(
        case, InvalidPrediction("malformed JSON", "parse_error")
    )
    assert score.valid is False
    assert score.error_kind == "parse_error"
    assert score.reward == 0.0
    assert score.exact_match is False
    assert score.tp == score.fp == score.fn == 0
    assert score.precision is None
    assert score.recall is None
    assert score.f1 is None
    corpus = aggregate_scores([score])
    assert corpus.invalid_count == 1
    assert corpus.parse_error_count == 1
    assert corpus.request_error_count == 0
    assert corpus.parsing_success == 0.0
    assert corpus.exact_match_rate == 0.0
    assert corpus.mean_reward == 0.0
    assert corpus.clean_article_count == 1
    assert corpus.clean_articles_falsely_flagged == 0
    assert corpus.precision is None
    assert corpus.recall is None
    assert corpus.micro_f1 is None
    assert corpus.macro_f1 is None


def test_invalid_output_on_gold_case_counts_all_gold_as_fn() -> None:
    case = single_issue_case()
    score = score_case(
        case, InvalidPrediction("provider timeout", "request_error")
    )
    assert score.valid is False
    assert score.error_kind == "request_error"
    assert score.tp == 0
    assert score.fp == 0
    assert score.fn == 1
    assert score.precision is None
    assert score.recall == 0.0
    assert score.f1 == 0.0
    assert score.reward == 0.0
    assert score.exact_match is False
    assert score.missed_gold[0].issue_id == "fixture-mass"
    corpus = aggregate_scores([score])
    assert corpus.request_error_count == 1
    assert corpus.parse_error_count == 0
    assert corpus.evidence_failure_count == 0


def test_missing_gold_prediction_is_fn() -> None:
    case = single_issue_case()
    score = score_case(case, make_prediction())
    assert score.valid is True
    assert score.tp == 0
    assert score.fp == 0
    assert score.fn == 1
    assert score.precision is None
    assert score.recall == 0.0
    assert score.f1 == 0.0
    assert score.reward == 0.0
    assert score.exact_match is False
    assert score.missed_gold[0].code == "PRODUCT_FACT_MISMATCH"


def test_wrong_code_is_fp_and_fn() -> None:
    case = single_issue_case()
    score = score_case(
        case,
        make_prediction(
            finding("UNSUPPORTED_NUMERIC_CLAIM", "hmotnost 10 g")
        ),
    )
    assert score.tp == 0
    assert score.fp == 1
    assert score.fn == 1
    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.f1 == 0.0
    assert score.reward == 0.0
    false_positive = score.false_positives[0]
    assert false_positive.code == "UNSUPPORTED_NUMERIC_CLAIM"
    assert false_positive.matching_gold_code == "PRODUCT_FACT_MISMATCH"
    assert "wrong code" in false_positive.reason
    corpus = aggregate_scores([score])
    by_code = {category.code: category for category in corpus.per_category}
    assert by_code["PRODUCT_FACT_MISMATCH"].fn == 1
    assert by_code["PRODUCT_FACT_MISMATCH"].f1 == 0.0
    assert by_code["UNSUPPORTED_NUMERIC_CLAIM"].fp == 1
    assert by_code["UNSUPPORTED_NUMERIC_CLAIM"].precision == 0.0
    assert by_code["UNSUPPORTED_NUMERIC_CLAIM"].recall is None
    assert by_code["UNSUPPORTED_NUMERIC_CLAIM"].f1 == 0.0
    assert corpus.macro_f1 is None


def test_same_evidence_with_two_codes_adds_one_fp() -> None:
    case = single_issue_case()
    score = score_case(
        case,
        make_prediction(
            finding("PRODUCT_FACT_MISMATCH", "hmotnost 10 g"),
            finding("UNSUPPORTED_NUMERIC_CLAIM", "hmotnost 10 g"),
        ),
    )
    assert score.tp == 1
    assert score.fp == 1
    assert score.fn == 0
    assert score.reward == pytest.approx(2 / 3)
    assert score.duplicate_count == 0
    assert score.false_positives[0].matching_gold_code == (
        "PRODUCT_FACT_MISMATCH"
    )


def test_one_prediction_cannot_satisfy_two_gold_issues() -> None:
    document = "Chyba A a chyba B."
    case = make_scoring_case(
        document,
        [
            {
                "issue_id": "fixture-a",
                "code": "UNSUPPORTED_NUMERIC_CLAIM",
                "evidence": "Chyba A",
                "anchor": "Chyba A",
                "context_start": 0,
                "context_end": len(document),
            },
            {
                "issue_id": "fixture-b",
                "code": "UNSUPPORTED_NUMERIC_CLAIM",
                "evidence": "chyba B",
                "anchor": "chyba B",
                "context_start": 0,
                "context_end": len(document),
            },
        ],
    )
    score = score_case(
        case,
        make_prediction(
            finding("UNSUPPORTED_NUMERIC_CLAIM", "Chyba A a chyba B")
        ),
    )
    assert score.tp == 1
    assert score.fp == 0
    assert score.fn == 1
    assert score.reward == pytest.approx(2 / 3)
    assert score.matched_issue_ids == ("fixture-a",)
    assert [missed.issue_id for missed in score.missed_gold] == ["fixture-b"]


def test_two_predictions_match_two_same_code_issues() -> None:
    document = "Chyba A a chyba B."
    case = make_scoring_case(
        document,
        [
            {
                "issue_id": "fixture-a",
                "code": "UNSUPPORTED_NUMERIC_CLAIM",
                "evidence": "Chyba A",
                "anchor": "Chyba A",
                "context_start": 0,
                "context_end": len(document),
            },
            {
                "issue_id": "fixture-b",
                "code": "UNSUPPORTED_NUMERIC_CLAIM",
                "evidence": "chyba B",
                "anchor": "chyba B",
                "context_start": 0,
                "context_end": len(document),
            },
        ],
    )
    score = score_case(
        case,
        make_prediction(
            finding("UNSUPPORTED_NUMERIC_CLAIM", "Chyba A"),
            finding("UNSUPPORTED_NUMERIC_CLAIM", "chyba B"),
        ),
    )
    assert score.tp == 2
    assert score.fp == 0
    assert score.fn == 0
    assert score.reward == 1.0
    assert score.exact_match is True
    assert score.matched_issue_ids == ("fixture-a", "fixture-b")


def test_duplicate_prediction_is_false_positive() -> None:
    case = single_issue_case()
    score = score_case(
        case,
        make_prediction(
            finding("PRODUCT_FACT_MISMATCH", "hmotnost 10 g"),
            finding(
                "PRODUCT_FACT_MISMATCH",
                "hmotnost 10 g",
                "Jiné vysvětlení.",
            ),
        ),
    )
    assert score.tp == 1
    assert score.fp == 1
    assert score.fn == 0
    assert score.reward == pytest.approx(2 / 3)
    assert score.duplicate_count == 1
    assert score.false_positives[0].duplicate_of == 0
    assert score.evidence_failure_count == 0
    feedback = build_feedback(score)
    assert "duplicate of prediction #0" in feedback
    assert "Duplicate predictions: 1" in feedback


def test_nonexistent_evidence_is_false_positive() -> None:
    case = single_issue_case()
    score = score_case(
        case,
        make_prediction(finding("PRODUCT_FACT_MISMATCH", "hmotnost 20 g")),
    )
    assert score.tp == 0
    assert score.fp == 1
    assert score.fn == 1
    assert score.evidence_failure_count == 1
    false_positive = score.false_positives[0]
    assert false_positive.occurrences == 0
    assert "does not occur literally" in false_positive.reason
    feedback = build_feedback(score)
    assert "Evidence validation failures: 1" in feedback
    assert "does not occur literally" in feedback


def test_ambiguous_evidence_is_false_positive() -> None:
    document = (
        "Tento stříbrný slitek má hmotnost 10 g. "
        "Jiný text uvádí hmotnost 10 g."
    )
    first_sentence = "Tento stříbrný slitek má hmotnost 10 g."
    case = make_scoring_case(
        document,
        [
            {
                "issue_id": "fixture-mass",
                "code": "PRODUCT_FACT_MISMATCH",
                "evidence": first_sentence,
                "anchor": "10 g",
            }
        ],
    )
    score = score_case(
        case,
        make_prediction(
            finding("PRODUCT_FACT_MISMATCH", "hmotnost 10 g")
        ),
    )
    assert score.tp == 0
    assert score.fp == 1
    assert score.fn == 1
    assert score.evidence_failure_count == 1
    false_positive = score.false_positives[0]
    assert false_positive.occurrences == 2
    assert "occurs 2 times" in false_positive.reason


def test_unrelated_substring_inside_context_is_false_positive() -> None:
    document = "Tento stříbrný slitek má hmotnost 10 g."
    case = make_scoring_case(
        document,
        [
            {
                "issue_id": "fixture-mass",
                "code": "PRODUCT_FACT_MISMATCH",
                "evidence": "hmotnost 10 g",
                "anchor": "10 g",
                "context_start": 0,
                "context_end": len(document),
            }
        ],
    )
    score = score_case(
        case,
        make_prediction(finding("PRODUCT_FACT_MISMATCH", "stříbrný")),
    )
    assert score.tp == 0
    assert score.fp == 1
    assert score.fn == 1
    assert score.evidence_failure_count == 0
    false_positive = score.false_positives[0]
    assert "anchor" in false_positive.reason
    assert false_positive.matching_gold_issue_id is None


def test_quote_outside_context_window_is_false_positive() -> None:
    document = "Tento stříbrný slitek má hmotnost 10 g a je těžký."
    anchor_end = document.index("10 g") + len("10 g")
    case = make_scoring_case(
        document,
        [
            {
                "issue_id": "fixture-mass",
                "code": "PRODUCT_FACT_MISMATCH",
                "evidence": "hmotnost 10 g",
                "anchor": "10 g",
                "context_start": 0,
                "context_end": anchor_end,
            }
        ],
    )
    score = score_case(
        case,
        make_prediction(
            finding("PRODUCT_FACT_MISMATCH", document)
        ),
    )
    assert score.tp == 0
    assert score.fp == 1
    assert score.fn == 1
    assert score.evidence_failure_count == 0
    false_positive = score.false_positives[0]
    assert "context window" in false_positive.reason
    assert false_positive.matching_gold_issue_id == "fixture-mass"


def test_prediction_quote_crossing_title_body_is_false_positive() -> None:
    title = "Titulek"
    body = "Tělo s hmotnost 10 g."
    document = canonicalize_document(title, body)
    body_start = len(title) + len(TITLE_BODY_SEPARATOR)
    case = make_scoring_case(
        document,
        [
            {
                "issue_id": "fixture-mass",
                "code": "PRODUCT_FACT_MISMATCH",
                "evidence": "hmotnost 10 g",
                "anchor": "10 g",
                "context_start": body_start,
                "context_end": len(document),
            }
        ],
        separator_start=len(title),
        separator_end=len(title) + len(TITLE_BODY_SEPARATOR),
    )
    score = score_case(
        case,
        make_prediction(finding("PRODUCT_FACT_MISMATCH", document)),
    )
    assert score.tp == 0
    assert score.fp == 1
    assert score.fn == 1
    assert "context window" in score.false_positives[0].reason


def test_title_only_gold_issue_matches_title_quote() -> None:
    title = "Zaručený výnos"
    document = canonicalize_document(title, "Text.")
    case = make_scoring_case(
        document,
        [
            {
                "issue_id": "fixture-certainty",
                "code": "UNWARRANTED_CERTAINTY",
                "evidence": title,
                "anchor": title,
                "context_start": 0,
                "context_end": len(title),
            }
        ],
        separator_start=len(title),
        separator_end=len(title) + len(TITLE_BODY_SEPARATOR),
    )
    score = score_case(
        case,
        make_prediction(finding("UNWARRANTED_CERTAINTY", title)),
    )
    assert score.tp == 1
    assert score.reward == 1.0
    assert score.exact_match is True


def test_per_category_and_macro_f1_are_hand_computed() -> None:
    document = (
        "Chyba A. Chyba B. Chyba C. Chyba D. Chyba E. Chyba F."
    )
    specs = [
        {
            "issue_id": f"fixture-{index}",
            "code": code,
            "evidence": f"Chyba {letter}.",
            "anchor": f"Chyba {letter}.",
        }
        for index, code, letter in (
            (1, "UNSUPPORTED_NUMERIC_CLAIM", "A"),
            (2, "INVESTMENT_ADVICE_AS_FACT", "B"),
            (3, "UNWARRANTED_CERTAINTY", "C"),
            (4, "UNRESOLVED_PLACEHOLDER", "D"),
            (5, "INVALID_CITATION", "E"),
            (6, "PRODUCT_FACT_MISMATCH", "F"),
        )
    ]
    case = make_scoring_case(document, specs)
    score = score_case(
        case,
        make_prediction(
            finding("UNSUPPORTED_NUMERIC_CLAIM", "Chyba A."),
            finding("INVESTMENT_ADVICE_AS_FACT", "Chyba B."),
            finding("UNWARRANTED_CERTAINTY", "Chyba C."),
            finding("UNRESOLVED_PLACEHOLDER", "Chyba"),
            finding("PRODUCT_FACT_MISMATCH", "Chyba F."),
        ),
    )
    assert score.tp == 4
    assert score.fp == 1
    assert score.fn == 2
    assert score.reward == pytest.approx(8 / 11)
    assert score.exact_match is False
    corpus = aggregate_scores([score])
    assert corpus.tp == 4
    assert corpus.fp == 1
    assert corpus.fn == 2
    assert corpus.precision == pytest.approx(4 / 5)
    assert corpus.recall == pytest.approx(4 / 6)
    assert corpus.micro_f1 == pytest.approx(8 / 11)
    assert corpus.macro_f1 == pytest.approx(4 / 6)
    assert corpus.evidence_failure_count == 1
    assert corpus.duplicate_prediction_count == 0
    assert corpus.parsing_success == 1.0
    assert corpus.exact_match_rate == 0.0
    assert corpus.mean_reward == pytest.approx(8 / 11)
    by_code = {category.code: category for category in corpus.per_category}
    assert by_code["UNSUPPORTED_NUMERIC_CLAIM"].f1 == 1.0
    assert by_code["INVESTMENT_ADVICE_AS_FACT"].f1 == 1.0
    assert by_code["UNWARRANTED_CERTAINTY"].f1 == 1.0
    assert by_code["PRODUCT_FACT_MISMATCH"].f1 == 1.0
    assert by_code["UNRESOLVED_PLACEHOLDER"].tp == 0
    assert by_code["UNRESOLVED_PLACEHOLDER"].fp == 1
    assert by_code["UNRESOLVED_PLACEHOLDER"].fn == 1
    assert by_code["UNRESOLVED_PLACEHOLDER"].f1 == 0.0
    assert by_code["INVALID_CITATION"].tp == 0
    assert by_code["INVALID_CITATION"].fp == 0
    assert by_code["INVALID_CITATION"].fn == 1
    assert by_code["INVALID_CITATION"].f1 == 0.0


def test_corpus_aggregation_across_cases() -> None:
    case_a = make_scoring_case(
        "Chyba A.",
        [
            {
                "issue_id": "fixture-a",
                "code": "UNSUPPORTED_NUMERIC_CLAIM",
                "evidence": "Chyba A.",
                "anchor": "Chyba A.",
            }
        ],
        case_id="fixture-a",
    )
    case_b = make_scoring_case(
        "Chyba B.",
        [
            {
                "issue_id": "fixture-b",
                "code": "INVESTMENT_ADVICE_AS_FACT",
                "evidence": "Chyba B.",
                "anchor": "Chyba B.",
            }
        ],
        case_id="fixture-b",
    )
    corpus = score_corpus(
        [case_a, case_b],
        [
            make_prediction(
                finding("UNSUPPORTED_NUMERIC_CLAIM", "Chyba A.")
            ),
            make_prediction(
                finding("UNSUPPORTED_NUMERIC_CLAIM", "Chyba B.")
            ),
        ],
    )
    assert corpus.article_count == 2
    assert corpus.valid_count == 2
    assert corpus.tp == 1
    assert corpus.fp == 1
    assert corpus.fn == 1
    assert corpus.precision == 0.5
    assert corpus.recall == 0.5
    assert corpus.micro_f1 == 0.5
    assert corpus.macro_f1 is None
    assert corpus.exact_match_count == 1
    assert corpus.exact_match_rate == 0.5
    assert corpus.parsing_success == 1.0
    assert corpus.mean_reward == 0.5
    by_code = {category.code: category for category in corpus.per_category}
    assert by_code["UNSUPPORTED_NUMERIC_CLAIM"].tp == 1
    assert by_code["UNSUPPORTED_NUMERIC_CLAIM"].fp == 1
    assert by_code["UNSUPPORTED_NUMERIC_CLAIM"].fn == 0
    assert by_code["UNSUPPORTED_NUMERIC_CLAIM"].f1 == pytest.approx(2 / 3)
    assert by_code["INVESTMENT_ADVICE_AS_FACT"].tp == 0
    assert by_code["INVESTMENT_ADVICE_AS_FACT"].fp == 0
    assert by_code["INVESTMENT_ADVICE_AS_FACT"].fn == 1
    assert by_code["INVESTMENT_ADVICE_AS_FACT"].f1 == 0.0


def test_undefined_ratios_are_none() -> None:
    corpus = aggregate_scores([])
    assert corpus.article_count == 0
    assert corpus.precision is None
    assert corpus.recall is None
    assert corpus.micro_f1 is None
    assert corpus.macro_f1 is None
    assert corpus.exact_match_rate is None
    assert corpus.parsing_success is None
    assert corpus.mean_reward is None
    assert corpus.clean_article_count == 0
    assert all(category.f1 is None for category in corpus.per_category)
    clean_case = make_scoring_case("Čistý text.", [])
    clean_score = score_case(clean_case, make_prediction())
    assert clean_score.precision is None
    assert clean_score.recall is None
    assert clean_score.f1 is None
    clean_corpus = aggregate_scores([clean_score])
    assert clean_corpus.precision is None
    assert clean_corpus.recall is None
    assert clean_corpus.micro_f1 is None
    assert clean_corpus.macro_f1 is None
    assert clean_corpus.exact_match_rate == 1.0
    assert clean_corpus.parsing_success == 1.0
    assert clean_corpus.mean_reward == 1.0
    assert clean_corpus.clean_article_count == 1
    assert clean_corpus.clean_articles_falsely_flagged == 0


def test_scalar_and_feedback_metrics_agree_exactly() -> None:
    perfect_case = single_issue_case()
    clean_case = make_scoring_case("Čistý text.", [])
    fixtures = [
        (
            perfect_case,
            make_prediction(
                finding("PRODUCT_FACT_MISMATCH", "hmotnost 10 g")
            ),
        ),
        (clean_case, make_prediction()),
        (
            perfect_case,
            InvalidPrediction("malformed JSON", "parse_error"),
        ),
        (
            perfect_case,
            make_prediction(
                finding("UNSUPPORTED_NUMERIC_CLAIM", "hmotnost 10 g")
            ),
        ),
        (
            perfect_case,
            make_prediction(
                finding("PRODUCT_FACT_MISMATCH", "hmotnost 10 g"),
                finding("PRODUCT_FACT_MISMATCH", "hmotnost 10 g"),
            ),
        ),
    ]
    for case, prediction in fixtures:
        scalar = scalar_metric(case, prediction)
        feedback = feedback_metric(case, prediction)
        assert isinstance(feedback, dspy.Prediction)
        assert feedback.score == scalar
        assert feedback.feedback == build_feedback(
            score_case(case, prediction)
        )
        assert 0.0 <= feedback.score <= 1.0


def test_feedback_metric_accepts_five_positional_arguments() -> None:
    case = single_issue_case()
    prediction = make_prediction(
        finding("PRODUCT_FACT_MISMATCH", "hmotnost 10 g")
    )
    result = feedback_metric(case, prediction, None, "predictor", None)
    assert result.score == 1.0


def test_feedback_contains_required_details() -> None:
    document = "Chyba A. Chyba B."
    case = make_scoring_case(
        document,
        [
            {
                "issue_id": "fixture-a",
                "code": "UNSUPPORTED_NUMERIC_CLAIM",
                "evidence": "Chyba A.",
                "anchor": "Chyba A.",
            },
            {
                "issue_id": "fixture-b",
                "code": "INVESTMENT_ADVICE_AS_FACT",
                "evidence": "Chyba B.",
                "anchor": "Chyba B.",
            },
        ],
    )
    score = score_case(
        case,
        make_prediction(
            finding("UNSUPPORTED_NUMERIC_CLAIM", "Chyba B.")
        ),
    )
    feedback = build_feedback(score)
    assert "Reward: 0.0000" in feedback
    assert "Missed gold issues:" in feedback
    assert "expected evidence" in feedback
    assert "False positives:" in feedback
    assert "wrong code" in feedback
    assert "Corrective hints:" in feedback
    assert "improve accuracy" not in feedback.lower()
    assert "fixture-a" in feedback
    assert "fixture-b" in feedback


def test_invalid_prediction_feedback_is_deterministic() -> None:
    case = single_issue_case()
    score = score_case(
        case, InvalidPrediction("truncated response", "parse_error")
    )
    first = build_feedback(score)
    second = build_feedback(score)
    assert first == second
    assert "Invalid output (parse_error): truncated response" in first
    assert "Return one JSON object" in first
    assert "Gold issues credited: 0 of 1." in first


def test_perfect_and_clean_feedback_are_explicit() -> None:
    perfect_case = single_issue_case()
    perfect = build_feedback(
        score_case(
            perfect_case,
            make_prediction(
                finding("PRODUCT_FACT_MISMATCH", "hmotnost 10 g")
            ),
        )
    )
    assert "Reward: 1.0000" in perfect
    assert (
        "All gold issues were matched with no false positives." in perfect
    )
    clean = build_feedback(
        score_case(
            make_scoring_case("Čistý text.", []), make_prediction()
        )
    )
    assert "Reward: 1.0000" in clean
    assert (
        "No findings were expected and none were returned." in clean
    )


def test_parse_prediction_rejects_invalid_whole_outputs() -> None:
    valid_finding = {
        "code": "UNRESOLVED_PLACEHOLDER",
        "evidence": "[DOPLNIT CENU]",
        "explanation": "V textu zůstala redakční značka.",
    }
    valid_output = parse_prediction({"findings": [valid_finding]})
    assert isinstance(valid_output, EditorialQAOutput)
    assert len(valid_output.findings) == 1
    valid_json = parse_prediction(
        json.dumps({"findings": [valid_finding]})
    )
    assert isinstance(valid_json, EditorialQAOutput)
    invalid_payloads = [
        {"findings": [], "score": 1.0},
        {},
        {"findings": "not-a-list"},
        {"findings": [{**valid_finding, "code": "UNKNOWN_CODE"}]},
        {
            "findings": [
                {
                    "code": valid_finding["code"],
                    "evidence": valid_finding["evidence"],
                }
            ]
        },
        {"findings": [{**valid_finding, "explanation": 123}]},
        {"findings": [{**valid_finding, "evidence": ""}]},
        {"findings": [{**valid_finding, "evidence": "x" * 321}]},
        {"findings": [{**valid_finding, "explanation": "x" * 401}]},
        {"findings": [{**valid_finding, "extra": "forbidden"}]},
    ]
    for payload in invalid_payloads:
        parsed = parse_prediction(payload)
        assert isinstance(parsed, InvalidPrediction)
        assert parsed.kind == "parse_error"
    malformed_json = parse_prediction("{not valid json")
    assert isinstance(malformed_json, InvalidPrediction)
    assert malformed_json.kind == "parse_error"
    missing = parse_prediction(None)
    assert isinstance(missing, InvalidPrediction)
    assert missing.kind == "missing_output"
    unsupported = parse_prediction(object())
    assert isinstance(unsupported, InvalidPrediction)
    assert unsupported.kind == "missing_output"
    list_payload = parse_prediction([valid_finding])
    assert isinstance(list_payload, InvalidPrediction)
    assert list_payload.kind == "missing_output"
    explicit = InvalidPrediction("timeout", "request_error")
    assert parse_prediction(explicit) is explicit
    dspy_output = parse_prediction(dspy.Prediction(findings=[]))
    assert isinstance(dspy_output, EditorialQAOutput)
    dspy_none = parse_prediction(dspy.Prediction(findings=None))
    assert isinstance(dspy_none, InvalidPrediction)
    assert dspy_none.kind == "missing_output"
    dspy_empty = parse_prediction(dspy.Prediction())
    assert isinstance(dspy_empty, InvalidPrediction)
    assert dspy_empty.kind == "missing_output"


def test_scoring_configuration_fails_closed_on_mismatched_text() -> None:
    case = single_issue_case()
    mutated = ScoringCase(
        case_id=case.case_id,
        document_text="Jiný text bez hledané ukázky.",
        gold_findings=case.gold_findings,
    )
    with pytest.raises(ScoringConfigurationError):
        score_case(
            mutated,
            make_prediction(
                finding("PRODUCT_FACT_MISMATCH", "hmotnost 10 g")
            ),
        )
    with pytest.raises(ScoringConfigurationError):
        score_case(
            {"case_id": "fixture-001", "document_text": "Text."},
            make_prediction(),
        )
    with pytest.raises(ScoringConfigurationError):
        score_case(
            {
                "case_id": "fixture-001",
                "document_text": "Text.",
                "gold_findings": None,
            },
            make_prediction(),
        )
    with pytest.raises(ScoringConfigurationError):
        score_case(
            {
                "case_id": "fixture-001",
                "document_text": "Text.",
                "gold_findings": "not-a-list",
            },
            make_prediction(),
        )


def test_score_corpus_rejects_length_mismatch() -> None:
    case = single_issue_case()
    with pytest.raises(ValueError):
        score_corpus([case], [])


def test_scoring_case_from_real_record_preserves_canonical_text() -> None:
    cases = load_all_splits(DATA_DIR)
    case: CaseRecord = cases["train"][1]
    scoring = scoring_case_from_case(case)
    assert scoring.case_id == case.case_id
    assert scoring.document_text == canonicalize_document(
        case.title, case.article_text
    )
    title_length = len(canonicalize_text(case.title))
    assert scoring.separator_start == title_length
    assert scoring.separator_end == title_length + len(
        TITLE_BODY_SEPARATOR
    )
    assert scoring.gold_findings == tuple(case.gold_findings)


def test_every_gold_canonical_prediction_scores_one() -> None:
    cases = load_all_splits(DATA_DIR)
    for split_cases in cases.values():
        for case in split_cases:
            prediction = EditorialQAOutput.model_validate(
                {
                    "findings": [
                        {
                            "code": issue.code,
                            "evidence": issue.evidence,
                            "explanation": issue.explanation,
                        }
                        for issue in case.gold_findings
                    ]
                }
            )
            score = score_case(case, prediction)
            assert score.valid is True
            assert score.tp == len(case.gold_findings)
            assert score.fp == 0
            assert score.fn == 0
            assert score.reward == 1.0
            assert score.exact_match is True


def test_always_empty_valid_predictor_reference_on_frozen_dataset() -> None:
    cases = [
        case
        for split_cases in load_all_splits(DATA_DIR).values()
        for case in split_cases
    ]
    predictions = [
        EditorialQAOutput(findings=[]) for _ in cases
    ]
    corpus = score_corpus(cases, predictions)
    assert corpus.article_count == 50
    assert corpus.valid_count == 50
    assert corpus.invalid_count == 0
    assert corpus.clean_article_count == 10
    assert corpus.clean_articles_falsely_flagged == 0
    assert corpus.tp == 0
    assert corpus.fp == 0
    assert corpus.fn == 60
    assert corpus.precision is None
    assert corpus.recall == 0.0
    assert corpus.micro_f1 == 0.0
    assert corpus.macro_f1 == 0.0
    assert corpus.exact_match_count == 10
    assert corpus.exact_match_rate == pytest.approx(0.2)
    assert corpus.parsing_success == 1.0
    assert corpus.mean_reward == pytest.approx(0.2)
    assert all(
        category.f1 == 0.0 for category in corpus.per_category
    )
    assert [category.code for category in corpus.per_category] == list(
        ALLOWED_CODES
    )


def test_second_match_option_can_accept_the_quote() -> None:
    document = "Chyba A. Chyba B."
    issue = GoldFinding(
        issue_id="fixture-a",
        code="UNSUPPORTED_NUMERIC_CLAIM",
        evidence="Chyba A.",
        explanation="Testovací vysvětlení.",
        match_options=[
            MatchOption(
                anchor_start=0,
                anchor_end=8,
                context_start=0,
                context_end=8,
            ),
            MatchOption(
                anchor_start=0,
                anchor_end=8,
                context_start=0,
                context_end=len(document),
            ),
        ],
    )
    case = ScoringCase(
        case_id="fixture-options",
        document_text=document,
        gold_findings=(issue,),
    )
    score = score_case(
        case,
        make_prediction(
            finding("UNSUPPORTED_NUMERIC_CLAIM", document)
        ),
    )
    assert score.tp == 1
    assert score.fp == 0
    assert score.fn == 0
    assert score.reward == 1.0
    assert score.exact_match is True


def test_maximum_matching_reroutes_to_cover_both_gold_issues() -> None:
    document = "Chyba A a chyba B."
    case = make_scoring_case(
        document,
        [
            {
                "issue_id": "fixture-a",
                "code": "UNSUPPORTED_NUMERIC_CLAIM",
                "evidence": "Chyba A",
                "anchor": "Chyba A",
                "context_start": 0,
                "context_end": len(document),
            },
            {
                "issue_id": "fixture-b",
                "code": "UNSUPPORTED_NUMERIC_CLAIM",
                "evidence": "chyba B",
                "anchor": "chyba B",
                "context_start": 0,
                "context_end": len(document),
            },
        ],
    )
    score = score_case(
        case,
        make_prediction(
            finding(
                "UNSUPPORTED_NUMERIC_CLAIM", "Chyba A a chyba B"
            ),
            finding("UNSUPPORTED_NUMERIC_CLAIM", "Chyba A"),
        ),
    )
    assert score.tp == 2
    assert score.fp == 0
    assert score.fn == 0
    assert score.reward == 1.0
    assert score.exact_match is True
    assert score.matched_issue_ids == ("fixture-a", "fixture-b")


def test_feedback_adds_hint_for_unsupported_extra_finding() -> None:
    document = "Chyba A. Chyba B."
    case = make_scoring_case(
        document,
        [
            {
                "issue_id": "fixture-a",
                "code": "UNSUPPORTED_NUMERIC_CLAIM",
                "evidence": "Chyba A.",
                "anchor": "Chyba A.",
            }
        ],
    )
    score = score_case(
        case,
        make_prediction(
            finding("UNSUPPORTED_NUMERIC_CLAIM", "Chyba A."),
            finding("INVESTMENT_ADVICE_AS_FACT", "Chyba B."),
        ),
    )
    assert score.tp == 1
    assert score.fp == 1
    assert score.fn == 0
    feedback = build_feedback(score)
    assert "Corrective hints:" in feedback
    assert "Do not invent findings" in feedback


def test_mapping_gold_derives_separator_or_fails_closed() -> None:
    title = "Titulek"
    body = "Tělo s hmotnost 10 g."
    document = canonicalize_document(title, body)
    body_start = len(title) + len(TITLE_BODY_SEPARATOR)
    scoring = make_scoring_case(
        document,
        [
            {
                "issue_id": "fixture-mass",
                "code": "PRODUCT_FACT_MISMATCH",
                "evidence": "hmotnost 10 g",
                "anchor": "10 g",
                "context_start": body_start,
                "context_end": len(document),
            }
        ],
        separator_start=len(title),
        separator_end=body_start,
    )
    mapping = {
        "case_id": "fixture-mapping",
        "document_text": document,
        "gold_findings": [
            issue.model_dump() for issue in scoring.gold_findings
        ],
    }
    score = score_case(
        mapping,
        make_prediction(
            finding("PRODUCT_FACT_MISMATCH", "hmotnost 10 g")
        ),
    )
    assert score.reward == 1.0
    with pytest.raises(ScoringConfigurationError):
        score_case(
            {
                "case_id": "fixture-no-separator",
                "document_text": "Text bez oddělovače.",
                "gold_findings": [],
            },
            make_prediction(),
        )


def test_invalid_output_missed_gold_is_sorted_by_issue_id() -> None:
    document = "Chyba A. Chyba B."
    case = make_scoring_case(
        document,
        [
            {
                "issue_id": "fixture-b",
                "code": "INVESTMENT_ADVICE_AS_FACT",
                "evidence": "Chyba B.",
                "anchor": "Chyba B.",
            },
            {
                "issue_id": "fixture-a",
                "code": "UNSUPPORTED_NUMERIC_CLAIM",
                "evidence": "Chyba A.",
                "anchor": "Chyba A.",
            },
        ],
    )
    score = score_case(
        case, InvalidPrediction("truncated", "parse_error")
    )
    assert [missed.issue_id for missed in score.missed_gold] == [
        "fixture-a",
        "fixture-b",
    ]


def test_parse_prediction_rejects_nonfinite_json_constants() -> None:
    for constant in ("NaN", "Infinity", "-Infinity"):
        raw = (
            '{"findings": [{"code": "UNRESOLVED_PLACEHOLDER", '
            f'"evidence": {constant}, "explanation": "x"}}]}}'
        )
        parsed = parse_prediction(raw)
        assert isinstance(parsed, InvalidPrediction)
        assert parsed.kind == "parse_error"


def test_scalar_metric_accepts_five_positional_arguments() -> None:
    case = single_issue_case()
    prediction = make_prediction(
        finding("PRODUCT_FACT_MISMATCH", "hmotnost 10 g")
    )
    assert scalar_metric(case, prediction, None, "predictor", None) == 1.0


def test_request_error_feedback_names_the_error_kind() -> None:
    case = single_issue_case()
    score = score_case(
        case, InvalidPrediction("provider timeout", "request_error")
    )
    feedback = build_feedback(score)
    assert "Invalid output (request_error): provider timeout" in feedback


def test_mapping_gold_crossing_derived_separator_fails_closed() -> None:
    title = "Titulek"
    body = "Tělo."
    document = canonicalize_document(title, body)
    anchor_start = document.index("Tělo")
    cross = GoldFinding(
        issue_id="fixture-cross",
        code="PRODUCT_FACT_MISMATCH",
        evidence=document,
        explanation="Testovací vysvětlení.",
        match_options=[
            MatchOption(
                anchor_start=anchor_start,
                anchor_end=anchor_start + len("Tělo"),
                context_start=0,
                context_end=len(document),
            )
        ],
    )
    with pytest.raises(ScoringConfigurationError):
        score_case(
            {
                "case_id": "fixture-cross",
                "document_text": document,
                "gold_findings": [cross.model_dump()],
            },
            make_prediction(),
        )


def test_two_separators_fail_closed() -> None:
    with pytest.raises(ScoringConfigurationError):
        score_case(
            {
                "case_id": "fixture-two-separators",
                "document_text": "T\n\nB\n\nC",
                "gold_findings": [],
            },
            make_prediction(),
        )
