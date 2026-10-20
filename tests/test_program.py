from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import get_args, get_origin

import dspy
import pytest
from dspy.utils.exceptions import AdapterParseError

from dspy_lab import program as program_module
from dspy_lab.config import PROJECT_ROOT
from dspy_lab.data import (
    canonical_document,
    canonicalize_text,
    load_cases,
)
from dspy_lab.metrics import feedback_metric, scalar_metric
from dspy_lab.program import (
    BASELINE_PROMPT_PATH,
    EDITORIAL_RULES_PATH,
    EditorialQA,
    EditorialQASignature,
    StrictJSONAdapter,
    StrictOutputError,
    baseline_fingerprint,
    baseline_manifest,
    baseline_schema,
    program_source_sha256,
    case_inputs,
    fingerprint_payload,
    load_baseline_instructions,
    load_editorial_rules,
    parse_strict_output,
)
from dspy_lab.schemas import (
    ALLOWED_CODES,
    CaseRecord,
    Finding,
    ProductFacts,
    SourceRecord,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:Implementing custom LMs through BaseLM.forward.*:DeprecationWarning"
)

DATA_DIR = PROJECT_ROOT / "data/v1"
RULES = load_editorial_rules()
CASES = {
    case.case_id: case for case in load_cases(DATA_DIR / "train.jsonl")
}


def case(case_id: str = "train-002") -> CaseRecord:
    return CASES[case_id]


def gold_response(record: CaseRecord) -> str:
    return json.dumps(
        {
            "findings": [
                {
                    "code": finding.code,
                    "evidence": finding.evidence,
                    "explanation": finding.explanation,
                }
                for finding in record.gold_findings
            ]
        },
        ensure_ascii=False,
    )


class ScriptedLM(dspy.BaseLM):
    """Offline transport returning one scripted raw completion per call."""

    def __init__(
        self,
        responses: list[str],
        *,
        supports_schema: bool = False,
        supports_json_mode: bool = False,
    ) -> None:
        super().__init__(
            "scripted",
            "chat",
            0.0,
            1000,
            cache=False,
            num_retries=0,
        )
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []
        self._supports_schema = supports_schema
        self._supports_json_mode = supports_json_mode

    @property
    def supported_params(self) -> set[str]:
        if self._supports_schema or self._supports_json_mode:
            return {"response_format"}
        return set()

    @property
    def supports_response_schema(self) -> bool:
        return self._supports_schema

    def forward(self, prompt=None, messages=None, **kwargs):
        if not self._responses:
            raise AssertionError(
                "ScriptedLM received an unexpected extra request"
            )
        content = self._responses.pop(0)
        self.calls.append(
            {
                "prompt": prompt,
                "messages": messages,
                "kwargs": dict(kwargs),
            }
        )
        return {
            "model": "scripted",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }


def run_program(
    record: CaseRecord,
    responses: list[str],
    *,
    module: EditorialQA | None = None,
    supports_schema: bool = False,
    supports_json_mode: bool = False,
) -> tuple[dspy.Prediction, ScriptedLM]:
    lm = ScriptedLM(
        responses,
        supports_schema=supports_schema,
        supports_json_mode=supports_json_mode,
    )
    program = module if module is not None else EditorialQA()
    with dspy.context(lm=lm, adapter=StrictJSONAdapter()):
        prediction = program(**case_inputs(record, RULES))
    return prediction, lm


def run_program_error(
    record: CaseRecord,
    responses: list[str],
) -> tuple[AdapterParseError, ScriptedLM]:
    lm = ScriptedLM(responses)
    program = EditorialQA()
    with dspy.context(lm=lm, adapter=StrictJSONAdapter()):
        with pytest.raises(AdapterParseError) as exc_info:
            program(**case_inputs(record, RULES))
    return exc_info.value, lm


def prompt_text(lm: ScriptedLM) -> str:
    messages = lm.calls[-1]["messages"]
    if not messages:
        return ""
    return "\n".join(
        str(message.get("content", "")) for message in messages
    )


def test_signature_input_and_output_whitelist() -> None:
    assert set(EditorialQASignature.input_fields) == {
        "document_text",
        "product_facts",
        "sources",
        "editorial_rules",
    }
    assert set(EditorialQASignature.output_fields) == {"findings"}
    assert (
        EditorialQASignature.input_fields["product_facts"].annotation
        is ProductFacts
    )
    sources_annotation = EditorialQASignature.input_fields[
        "sources"
    ].annotation
    assert get_origin(sources_annotation) is list
    assert get_args(sources_annotation) == (SourceRecord,)
    findings_annotation = EditorialQASignature.output_fields[
        "findings"
    ].annotation
    assert get_origin(findings_annotation) is list
    assert get_args(findings_annotation) == (Finding,)
    parameters = inspect.signature(EditorialQA.forward).parameters
    assert list(parameters) == [
        "self",
        "document_text",
        "product_facts",
        "sources",
        "editorial_rules",
    ]
    inputs = case_inputs(case("train-002"), RULES)
    assert set(inputs) == {
        "document_text",
        "product_facts",
        "sources",
        "editorial_rules",
    }
    assert inputs["document_text"] == canonical_document(case("train-002"))
    assert inputs["product_facts"] == case("train-002").product_facts
    assert inputs["sources"] == case("train-002").sources
    assert inputs["editorial_rules"] == RULES
    assert "gold_findings" not in inputs
    assert "case_id" not in inputs
    assert "review" not in inputs


def test_module_has_one_predictor_zero_demos_and_no_lm() -> None:
    module = EditorialQA()
    predictors = list(module.named_predictors())
    assert len(predictors) == 1
    _name, predictor = predictors[0]
    assert isinstance(predictor, dspy.Predict)
    assert module.predict is predictor
    assert predictor.demos == []
    assert predictor.lm is None
    assert (
        predictor.signature.input_fields.keys()
        == EditorialQASignature.input_fields.keys()
    )
    assert (
        predictor.signature.output_fields.keys()
        == EditorialQASignature.output_fields.keys()
    )


def test_valid_gold_response_returns_typed_findings_and_scores_one() -> None:
    record = case("train-002")
    prediction, lm = run_program(record, [gold_response(record)])
    assert isinstance(prediction, dspy.Prediction)
    assert len(prediction.findings) == 2
    assert all(
        isinstance(finding, Finding)
        for finding in prediction.findings
    )
    assert [finding.code for finding in prediction.findings] == [
        finding.code for finding in record.gold_findings
    ]
    assert prediction.findings == [
        Finding(
            code=finding.code,
            evidence=finding.evidence,
            explanation=finding.explanation,
        )
        for finding in record.gold_findings
    ]
    assert len(lm.calls) == 1
    assert scalar_metric(record, prediction) == 1.0
    feedback = feedback_metric(record, prediction)
    assert feedback.score == 1.0
    assert (
        "All gold issues were matched with no false positives."
        in feedback.feedback
    )


def test_empty_findings_list_is_valid_and_scores_one_on_clean_case() -> None:
    record = case("train-001")
    prediction, lm = run_program(record, ['{"findings": []}'])
    assert prediction.findings == []
    assert isinstance(prediction.findings, list)
    assert len(lm.calls) == 1
    assert scalar_metric(record, prediction) == 1.0


def test_prompt_includes_four_inputs_and_excludes_gold() -> None:
    record = case("train-002")
    prediction, lm = run_program(record, ['{"findings": []}'])
    assert prediction.findings == []
    prompt = prompt_text(lm)
    messages = lm.calls[-1]["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
    ]
    assert EditorialQASignature.instructions in prompt
    for field_name in (
        "document_text",
        "product_facts",
        "sources",
        "editorial_rules",
    ):
        assert f"[[ ## {field_name} ## ]]" in prompt
    assert canonicalize_text(record.title) in prompt
    assert canonicalize_text(record.article_text) in prompt
    assert "This is an example of the task" not in prompt
    assert record.product_facts.reference_id is not None
    assert record.product_facts.reference_id in prompt
    assert record.product_facts.metal is not None
    assert record.product_facts.metal in prompt
    assert record.case_id not in prompt
    assert record.family_id not in prompt
    assert "gold_findings" not in prompt
    assert "issue_id" not in prompt
    assert "reviewer_id" not in prompt
    assert record.gold_findings[0].explanation not in prompt


def test_prompt_includes_sources() -> None:
    record = case("train-004")
    prediction, lm = run_program(record, ['{"findings": []}'])
    assert prediction.findings == []
    prompt = prompt_text(lm)
    source = record.sources[0]
    assert source.source_id in prompt
    assert source.excerpt in prompt
    assert source.provenance in prompt


def test_module_forward_rejects_extra_gold_argument() -> None:
    record = case("train-001")
    inputs = case_inputs(record, RULES)
    inputs["gold_findings"] = record.gold_findings
    with pytest.raises(TypeError):
        EditorialQA()(**inputs)


def test_strict_adapter_accepts_valid_json_and_returns_typed_findings() -> None:
    adapter = StrictJSONAdapter()
    parsed = adapter.parse(EditorialQASignature, '{"findings": []}')
    assert parsed == {"findings": []}
    completion = json.dumps(
        {
            "findings": [
                {
                    "code": "UNRESOLVED_PLACEHOLDER",
                    "evidence": "[DOPLNIT CENU]",
                    "explanation": "V textu zůstala značka.",
                }
            ]
        },
        ensure_ascii=False,
    )
    parsed = adapter.parse(EditorialQASignature, completion)
    assert len(parsed["findings"]) == 1
    assert isinstance(parsed["findings"][0], Finding)
    assert parsed["findings"][0].code == "UNRESOLVED_PLACEHOLDER"


def test_parse_strict_output_accepts_surrounding_whitespace() -> None:
    output = parse_strict_output(' \n\t {"findings": []} \n ')
    assert output.findings == []


def test_parse_strict_output_rejects_non_string() -> None:
    with pytest.raises(StrictOutputError):
        parse_strict_output(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "completion",
    [
        "",
        "not json",
        '```json\n{"findings": []}\n```',
        'Here is the JSON: {"findings": []}',
        '{"findings": []} trailing',
        "[]",
        '"{}"',
        "null",
        '{"findings": [',
    ],
)
def test_strict_adapter_rejects_non_object_and_repair_attempts(
    completion: str,
) -> None:
    with pytest.raises(AdapterParseError) as exc_info:
        StrictJSONAdapter().parse(EditorialQASignature, completion)
    error = exc_info.value
    assert error.adapter_name == "StrictJSONAdapter"
    assert error.lm_response == completion


@pytest.mark.parametrize(
    "completion",
    [
        '{"findings": [], "findings": []}',
        (
            '{"findings": [{"code": "UNRESOLVED_PLACEHOLDER", '
            '"code": "UNRESOLVED_PLACEHOLDER", "evidence": "x", '
            '"explanation": "y"}]}'
        ),
        (
            '{"findings": [], "findings": [{"code": '
            '"UNRESOLVED_PLACEHOLDER", "evidence": "x", '
            '"explanation": "y"}]}'
        ),
    ],
)
def test_strict_adapter_rejects_duplicate_keys(
    completion: str,
) -> None:
    with pytest.raises(AdapterParseError) as exc_info:
        StrictJSONAdapter().parse(EditorialQASignature, completion)
    assert "duplicate JSON key" in str(exc_info.value)


@pytest.mark.parametrize(
    "completion",
    [
        '{"findings": [], "extra": 1}',
        "{}",
        '{"other": []}',
        '{"findings": null}',
        '{"findings": {}}',
        '{"findings": "x"}',
    ],
)
def test_strict_adapter_rejects_unknown_root_fields_and_missing_findings(
    completion: str,
) -> None:
    with pytest.raises(AdapterParseError) as exc_info:
        StrictJSONAdapter().parse(EditorialQASignature, completion)
    assert "strict findings schema" in str(exc_info.value)


@pytest.mark.parametrize(
    "completion",
    [
        (
            '{"findings": [{"code": "UNKNOWN_CODE", "evidence": "x", '
            '"explanation": "y"}]}'
        ),
        (
            '{"findings": [{"code": "UNRESOLVED_PLACEHOLDER", '
            '"evidence": "x", "explanation": "y", "extra": 1}]}'
        ),
    ],
)
def test_strict_adapter_rejects_unknown_codes_and_extra_finding_fields(
    completion: str,
) -> None:
    with pytest.raises(AdapterParseError) as exc_info:
        StrictJSONAdapter().parse(EditorialQASignature, completion)
    assert "strict findings schema" in str(exc_info.value)


@pytest.mark.parametrize(
    "completion",
    [
        (
            '{"findings": [{"code": "UNRESOLVED_PLACEHOLDER", '
            '"evidence": "x"}]}'
        ),
        (
            '{"findings": [{"code": "UNRESOLVED_PLACEHOLDER", '
            '"evidence": 123, "explanation": "y"}]}'
        ),
        (
            '{"findings": [{"code": "UNRESOLVED_PLACEHOLDER", '
            '"evidence": "x", "explanation": 123}]}'
        ),
        (
            '{"findings": [{"code": "UNRESOLVED_PLACEHOLDER", '
            '"evidence": true, "explanation": "y"}]}'
        ),
    ],
)
def test_strict_adapter_rejects_default_insertion_and_coercion(
    completion: str,
) -> None:
    with pytest.raises(AdapterParseError) as exc_info:
        StrictJSONAdapter().parse(EditorialQASignature, completion)
    assert "strict findings schema" in str(exc_info.value)


@pytest.mark.parametrize(
    "constant",
    ["NaN", "Infinity", "-Infinity", "1e999"],
)
def test_strict_adapter_rejects_nonfinite_constants(
    constant: str,
) -> None:
    completion = (
        '{"findings": [{"code": "UNRESOLVED_PLACEHOLDER", '
        f'"evidence": {constant}, "explanation": "x"}}]}}'
    )
    with pytest.raises(AdapterParseError) as exc_info:
        StrictJSONAdapter().parse(EditorialQASignature, completion)
    assert "non-finite" in str(exc_info.value)


@pytest.mark.parametrize(
    "completion",
    [
        (
            '{"findings": [{"code": "UNRESOLVED_PLACEHOLDER", '
            '"evidence": "   ", "explanation": "y"}]}'
        ),
        (
            '{"findings": [{"code": "UNRESOLVED_PLACEHOLDER", '
            '"evidence": "x", "explanation": "   "}]}'
        ),
        json.dumps(
            {
                "findings": [
                    {
                        "code": "UNRESOLVED_PLACEHOLDER",
                        "evidence": "x" * 321,
                        "explanation": "y",
                    }
                ]
            }
        ),
        json.dumps(
            {
                "findings": [
                    {
                        "code": "UNRESOLVED_PLACEHOLDER",
                        "evidence": "x",
                        "explanation": "y" * 401,
                    }
                ]
            }
        ),
    ],
)
def test_strict_adapter_rejects_empty_or_oversized_strings(
    completion: str,
) -> None:
    with pytest.raises(AdapterParseError) as exc_info:
        StrictJSONAdapter().parse(EditorialQASignature, completion)
    assert "strict findings schema" in str(exc_info.value)


def test_strict_adapter_rejects_wrong_signature() -> None:
    class OtherSignature(dspy.Signature):
        answer: str = dspy.OutputField()

    with pytest.raises(AdapterParseError) as exc_info:
        StrictJSONAdapter().parse(OtherSignature, '{"findings": []}')
    assert "exactly one output field" in str(exc_info.value)

    class WrongFindingsType(dspy.Signature):
        findings: str = dspy.OutputField()

    with pytest.raises(AdapterParseError) as exc_info:
        StrictJSONAdapter().parse(WrongFindingsType, '{"findings": []}')
    assert "typed as list[Finding]" in str(exc_info.value)


def test_inherited_parse_failure_makes_exactly_one_request() -> None:
    record = case("train-001")
    lm = ScriptedLM(['```json\n{"findings": []}\n```'])
    program = EditorialQA()
    with dspy.context(lm=lm, adapter=StrictJSONAdapter()):
        with pytest.raises(AdapterParseError):
            program(**case_inputs(record, RULES))
    assert len(lm.calls) == 1


def test_invalid_json_never_returns_empty_findings() -> None:
    error, lm = run_program_error(case("train-001"), ['{"findings": ['])
    assert isinstance(error, AdapterParseError)
    assert len(lm.calls) == 1


def test_native_schema_mode_uses_strict_finding_schema() -> None:
    record = case("train-001")
    prediction, lm = run_program(
        record,
        ['{"findings": []}'],
        supports_schema=True,
    )
    assert prediction.findings == []
    response_format = lm.calls[0]["kwargs"]["response_format"]
    assert hasattr(response_format, "model_json_schema")
    schema = response_format.model_json_schema()
    assert schema["properties"]["findings"]["type"] == "array"
    assert (
        schema["properties"]["findings"]["items"]["$ref"]
        == "#/$defs/Finding"
    )
    finding_schema = schema["$defs"]["Finding"]
    assert finding_schema["additionalProperties"] is False
    assert set(finding_schema["properties"]["code"]["enum"]) == set(
        ALLOWED_CODES
    )
    assert set(finding_schema["required"]) == {
        "code",
        "evidence",
        "explanation",
    }


def test_json_mode_fallback_when_schema_unsupported() -> None:
    record = case("train-001")
    prediction, lm = run_program(
        record,
        ['{"findings": []}'],
        supports_json_mode=True,
    )
    assert prediction.findings == []
    assert lm.calls[0]["kwargs"]["response_format"] == {
        "type": "json_object"
    }


def test_no_response_format_when_unsupported() -> None:
    record = case("train-001")
    prediction, lm = run_program(record, ['{"findings": []}'])
    assert prediction.findings == []
    assert "response_format" not in lm.calls[0]["kwargs"]


def test_baseline_manifest_and_fingerprint_are_stable() -> None:
    manifest = baseline_manifest()
    assert manifest["program_name"] == "EditorialQA"
    assert manifest["rules_version"] == "editorial_rules_v1"
    assert manifest["rules_sha256"] == hashlib.sha256(
        EDITORIAL_RULES_PATH.read_bytes()
    ).hexdigest()
    assert manifest["predictor_count"] == len(
        list(EditorialQA().named_predictors())
    )
    assert manifest["demos"] == []
    assert manifest["lm"] is None
    assert manifest["adapter"] == "StrictJSONAdapter"
    assert manifest["prompt_version"] == "baseline_v1"
    instructions = str(manifest["instructions"])
    assert instructions == EditorialQASignature.instructions
    assert instructions == load_baseline_instructions()
    assert manifest["instructions_sha256"] == hashlib.sha256(
        instructions.encode("utf-8")
    ).hexdigest()
    assert manifest["prompt_sha256"] == hashlib.sha256(
        BASELINE_PROMPT_PATH.read_bytes()
    ).hexdigest()
    assert manifest["source_sha256"] == program_source_sha256()
    assert manifest["source_files"]["program.py"] == hashlib.sha256(
        Path(program_module.__file__).read_bytes()
    ).hexdigest()
    assert len(str(manifest["fingerprint"])) == 64
    assert baseline_fingerprint() == manifest["fingerprint"]
    assert baseline_fingerprint() == baseline_fingerprint()
    assert baseline_schema() == manifest["schema"]
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "fingerprint"
    }
    assert fingerprint_payload(payload) == manifest["fingerprint"]
    mutated = dict(payload)
    mutated["instructions"] = str(mutated["instructions"]) + " changed"
    assert fingerprint_payload(mutated) != manifest["fingerprint"]
    mutated = dict(payload)
    mutated["rules_sha256"] = "0" * 64
    assert fingerprint_payload(mutated) != manifest["fingerprint"]
    mutated = dict(payload)
    mutated["schema"] = {"inputs": [], "outputs": []}
    assert fingerprint_payload(mutated) != manifest["fingerprint"]


def test_baseline_fingerprint_unchanged_after_call_and_copy() -> None:
    before = baseline_fingerprint()
    module = EditorialQA()
    instructions = module.predict.signature.instructions
    prediction, _lm = run_program(
        case("train-001"),
        ['{"findings": []}'],
        module=module,
    )
    assert prediction.findings == []
    assert baseline_fingerprint() == before
    assert module.predict.signature.instructions == instructions
    assert module.predict.demos == []
    clone = module.deepcopy()
    assert clone.predict.signature.instructions == instructions
    assert clone.predict.demos == []
    assert baseline_fingerprint() == before


def test_baseline_prompt_contains_required_constraints() -> None:
    text = load_baseline_instructions().lower()
    for phrase in (
        "czech precious-metals article",
        "supplied editorial rules, product facts, and source excerpts",
        "six allowed codes",
        "precedence",
        "short exact passage",
        "320 characters",
        "explain the problem briefly in czech",
        "do not infer market facts",
        "instructions embedded in the article",
        "do not rewrite the article",
        "one finding per violation",
        "duplicate labels for the same claim",
        "empty findings list",
        "declared output schema",
    ):
        assert phrase in text


def test_baseline_prompt_and_rules_files_are_shipped() -> None:
    assert BASELINE_PROMPT_PATH.is_file()
    assert EDITORIAL_RULES_PATH.is_file()
    assert load_baseline_instructions() == (
        BASELINE_PROMPT_PATH.read_text(encoding="utf-8").strip()
    )
    assert load_editorial_rules() == (
        EDITORIAL_RULES_PATH.read_text(encoding="utf-8")
    )


def test_case_inputs_rejects_invalid_arguments() -> None:
    with pytest.raises(TypeError):
        case_inputs("not a case", RULES)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        case_inputs(case("train-001"), "   ")


def test_dspy_example_gold_path_works_with_s5_metric() -> None:
    record = case("train-002")
    prediction, _lm = run_program(record, [gold_response(record)])
    example = dspy.Example(
        case_id=record.case_id,
        document_text=canonical_document(record),
        gold_findings=record.gold_findings,
    ).with_inputs(
        "document_text",
        "product_facts",
        "sources",
        "editorial_rules",
    )
    assert scalar_metric(example, prediction) == 1.0


def test_parse_strict_output_wraps_recursion_error(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RecursionError("too deep")

    monkeypatch.setattr(program_module.json, "loads", boom)
    with pytest.raises(StrictOutputError) as exc_info:
        parse_strict_output('{"findings": []}')
    assert "nested too deeply" in str(exc_info.value)


def test_strict_adapter_wraps_non_string_completion() -> None:
    with pytest.raises(AdapterParseError) as exc_info:
        StrictJSONAdapter().parse(
            EditorialQASignature,
            None,  # type: ignore[arg-type]
        )
    assert "completion must be a string" in str(exc_info.value)
    assert exc_info.value.lm_response is None


def test_baseline_schema_captures_nested_contract() -> None:
    schema = baseline_schema()
    outputs = {item["name"]: item for item in schema["outputs"]}
    finding_schema = outputs["findings"]["json_schema"]
    finding_def = finding_schema["$defs"]["Finding"]
    assert "UNSUPPORTED_NUMERIC_CLAIM" in finding_def["properties"]["code"][
        "enum"
    ]
    assert finding_def["properties"]["evidence"]["maxLength"] == 320
    assert finding_def["required"] == [
        "code",
        "evidence",
        "explanation",
    ]
    assert finding_def["additionalProperties"] is False
    inputs = {item["name"]: item for item in schema["inputs"]}
    assert inputs["sources"]["json_schema"]["$defs"]["SourceRecord"][
        "required"
    ] == ["source_id", "excerpt", "provenance"]
