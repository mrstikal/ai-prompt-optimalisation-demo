"""One-predictor DSPy baseline with a fail-closed task output parser.

S6 defines the single ``EditorialQASignature``, the one-``Predict``
``EditorialQA`` module, and a small ``StrictJSONAdapter`` subclass. The adapter
reuses DSPy's formatting and native-schema request generation but replaces
stock ``JSONAdapter`` repair and filtering with one strict parse of the
complete response.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, get_args, get_origin

import dspy
from dspy.utils.exceptions import AdapterParseError
from pydantic import ValidationError

from .config import PROJECT_ROOT
from .data import canonical_document
from .schemas import (
    RULES_VERSION,
    CaseRecord,
    EditorialQAOutput,
    Finding,
    ProductFacts,
    SourceRecord,
)

BASELINE_PROMPT_PATH = PROJECT_ROOT / "prompts" / "baseline.txt"
EDITORIAL_RULES_PATH = PROJECT_ROOT / "data" / "editorial_rules_v1.md"
PROGRAM_SOURCE_PATH = Path(__file__).resolve()
BASELINE_PROMPT_VERSION = "baseline_v1"
STRICT_ADAPTER_VERSION = 1


def load_baseline_instructions() -> str:
    """Load the frozen manual baseline instructions."""
    text = BASELINE_PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("baseline prompt must contain non-whitespace text")
    return text


def load_editorial_rules() -> str:
    """Load the frozen editorial-rules input exactly as shipped."""
    text = EDITORIAL_RULES_PATH.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("editorial rules file must not be empty")
    return text


_BASELINE_INSTRUCTIONS = load_baseline_instructions()


class EditorialQASignature(dspy.Signature):
    document_text: str = dspy.InputField(
        desc=(
            "Canonical Czech article text: title, one blank line, then body. "
            "Quote evidence exactly from this text."
        )
    )
    product_facts: ProductFacts = dspy.InputField(
        desc=(
            "Structured reference attributes supplied with the article; an "
            "empty object means no product reference was supplied."
        )
    )
    sources: list[SourceRecord] = dspy.InputField(
        desc=(
            "Supplied source records only; each excerpt is the full allowed "
            "evidence for citation checks."
        )
    )
    editorial_rules: str = dspy.InputField(
        desc=(
            "Frozen six-code editorial policy, including category precedence "
            "and non-violation examples."
        )
    )
    findings: list[Finding] = dspy.OutputField(
        desc=(
            "One independently actionable violation per finding; return an "
            "empty list when the article is clean."
        )
    )


EditorialQASignature.instructions = _BASELINE_INSTRUCTIONS


class EditorialQA(dspy.Module):
    """The frozen one-predictor baseline; no demos, retrieval, or repair."""

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(EditorialQASignature)

    def forward(
        self,
        document_text: str,
        product_facts: ProductFacts,
        sources: list[SourceRecord],
        editorial_rules: str,
    ) -> dspy.Prediction:
        return self.predict(
            document_text=document_text,
            product_facts=product_facts,
            sources=sources,
            editorial_rules=editorial_rules,
        )


class StrictOutputError(ValueError):
    """Internal fail-closed error for the strict task parser."""


class _DuplicateJSONKeyError(StrictOutputError):
    pass


class _NonFiniteJSONConstantError(StrictOutputError):
    pass


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise _NonFiniteJSONConstantError(
        f"non-finite JSON constant {value!r}"
    )


def _reject_nonfinite_values(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise _NonFiniteJSONConstantError("non-finite JSON number")
    if isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite_values(item)
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite_values(item)


def _compact_error(exc: Exception) -> str:
    return " ".join(str(exc).split())


def parse_strict_output(completion: str) -> EditorialQAOutput:
    """Parse one complete JSON object and validate the strict task schema."""
    if not isinstance(completion, str):
        raise StrictOutputError("completion must be a string")
    try:
        raw = json.loads(
            completion,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except StrictOutputError:
        raise
    except json.JSONDecodeError as exc:
        raise StrictOutputError(
            "response is not one complete JSON object: "
            f"{exc.msg} at line {exc.lineno} column {exc.colno}"
        ) from exc
    except RecursionError as exc:
        raise StrictOutputError("response JSON is nested too deeply") from exc
    if not isinstance(raw, dict):
        raise StrictOutputError(
            "response must be one JSON object; "
            f"got {type(raw).__name__}"
        )
    _reject_nonfinite_values(raw)
    try:
        return EditorialQAOutput.model_validate(raw)
    except ValidationError as exc:
        raise StrictOutputError(
            "response does not match the strict findings schema: "
            f"{_compact_error(exc)}"
        ) from exc


class StrictJSONAdapter(dspy.JSONAdapter):
    """JSONAdapter with one strict parse and no repair or filtering."""

    def parse(
        self,
        signature: type[dspy.Signature],
        completion: str,
    ) -> dict[str, Any]:
        output_fields = signature.output_fields
        if set(output_fields) != {"findings"}:
            raise AdapterParseError(
                adapter_name=type(self).__name__,
                signature=signature,
                lm_response=completion,
                message=(
                    "StrictJSONAdapter requires exactly one output field "
                    "named 'findings'."
                ),
            )
        annotation = output_fields["findings"].annotation
        if (
            get_origin(annotation) is not list
            or get_args(annotation) != (Finding,)
        ):
            raise AdapterParseError(
                adapter_name=type(self).__name__,
                signature=signature,
                lm_response=completion,
                message=(
                    "StrictJSONAdapter requires the findings output to be "
                    "typed as list[Finding]."
                ),
            )
        try:
            output = parse_strict_output(completion)
        except StrictOutputError as exc:
            raise AdapterParseError(
                adapter_name=type(self).__name__,
                signature=signature,
                lm_response=completion,
                message=str(exc),
            ) from exc
        return {"findings": output.findings}


def case_inputs(
    case: CaseRecord,
    editorial_rules: str,
) -> dict[str, object]:
    """Build exactly the four task inputs; gold labels are never included."""
    if not isinstance(case, CaseRecord):
        raise TypeError("case must be a CaseRecord")
    if not isinstance(editorial_rules, str) or not editorial_rules.strip():
        raise ValueError("editorial_rules must be non-empty text")
    return {
        "document_text": canonical_document(case),
        "product_facts": case.product_facts,
        "sources": case.sources,
        "editorial_rules": editorial_rules,
    }


def _annotation_name(annotation: object) -> str:
    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        if args:
            return f"list[{_annotation_name(args[0])}]"
    if isinstance(annotation, type):
        return f"{annotation.__module__}.{annotation.__qualname__}"
    return repr(annotation)


def baseline_schema() -> dict[str, object]:
    """Return the frozen input/output field schema identity."""
    return {
        "inputs": [
            {"name": name, "annotation": _annotation_name(field.annotation)}
            for name, field in EditorialQASignature.input_fields.items()
        ],
        "outputs": [
            {"name": name, "annotation": _annotation_name(field.annotation)}
            for name, field in EditorialQASignature.output_fields.items()
        ],
    }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def fingerprint_payload(payload: Mapping[str, object]) -> str:
    """Return the deterministic checksum of a manifest payload."""
    return _sha256_text(_canonical_json(payload))


def _baseline_runtime_identity() -> tuple[int, list[object]]:
    module = EditorialQA()
    predictor_count = len(list(module.named_predictors()))
    demos = list(module.predict.demos)
    if predictor_count != 1:
        raise AssertionError(
            "the baseline must contain exactly one predictor; "
            f"found {predictor_count}"
        )
    if demos:
        raise AssertionError(
            "the baseline must contain zero demonstrations; "
            f"found {len(demos)}"
        )
    if module.predict.lm is not None:
        raise AssertionError(
            "the baseline must not persist an LM on its predictor"
        )
    return predictor_count, demos


def baseline_manifest() -> dict[str, object]:
    """Return the frozen baseline identity used for checksum comparisons."""
    instructions = EditorialQASignature.instructions
    schema = baseline_schema()
    predictor_count, demos = _baseline_runtime_identity()
    payload: dict[str, object] = {
        "program_name": "EditorialQA",
        "rules_version": RULES_VERSION,
        "rules_sha256": _sha256_file(EDITORIAL_RULES_PATH),
        "predictor_count": predictor_count,
        "demos": demos,
        "lm": None,
        "adapter": "StrictJSONAdapter",
        "adapter_version": STRICT_ADAPTER_VERSION,
        "prompt_version": BASELINE_PROMPT_VERSION,
        "instructions": instructions,
        "instructions_sha256": _sha256_text(instructions),
        "schema": schema,
        "schema_sha256": _sha256_text(_canonical_json(schema)),
        "prompt_sha256": _sha256_file(BASELINE_PROMPT_PATH),
        "source_sha256": _sha256_file(PROGRAM_SOURCE_PATH),
    }
    return {
        **payload,
        "fingerprint": fingerprint_payload(payload),
    }


def baseline_fingerprint() -> str:
    """Return the stable checksum of the frozen baseline definition."""
    return str(baseline_manifest()["fingerprint"])


__all__ = [
    "BASELINE_PROMPT_PATH",
    "BASELINE_PROMPT_VERSION",
    "EDITORIAL_RULES_PATH",
    "EditorialQA",
    "EditorialQASignature",
    "StrictJSONAdapter",
    "StrictOutputError",
    "baseline_fingerprint",
    "baseline_manifest",
    "baseline_schema",
    "case_inputs",
    "fingerprint_payload",
    "load_baseline_instructions",
    "load_editorial_rules",
    "parse_strict_output",
]
