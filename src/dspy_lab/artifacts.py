"""Safe program persistence, linked run records, and offline Replay reads.

Live program state is persisted as trusted JSON only and reloaded into the
known ``EditorialQA`` class after manifest, hash, schema, rules, and runtime
compatibility checks. Replay reads validated JSON/JSONL/text snapshots and
never constructs an LM or loads an executable program.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Literal

import dspy
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .config import PROJECT_ROOT, collect_runtime_versions
from .program import (
    BASELINE_PROMPT_PATH,
    BASELINE_PROMPT_VERSION,
    EDITORIAL_RULES_PATH,
    PROGRAM_SOURCE_PATH,
    STRICT_ADAPTER_VERSION,
    EditorialQA,
    baseline_schema,
    case_inputs,
    load_baseline_instructions,
)
from .schemas import CaseRecord, Finding, RULES_VERSION

PROGRAM_MANIFEST_SCHEMA_VERSION = "program_manifest_v1"
RUN_MANIFEST_SCHEMA_VERSION = "run_manifest_v1"
SNAPSHOT_SCHEMA_VERSION = "snapshot_v1"
OUTCOME_SCHEMA_VERSION = "outcome_v1"
PROGRAMS_ROOT = PROJECT_ROOT / "artifacts" / "programs"
RUNS_ROOT = PROJECT_ROOT / "runs"
PROGRAM_STATE_FILENAME = "program.json"
PROGRAM_MANIFEST_FILENAME = "manifest.json"
PROGRAM_INSTRUCTIONS_FILENAME = "instructions.txt"
SNAPSHOT_CASES_FILENAME = "inputs_snapshot.jsonl"
SNAPSHOT_RULES_FILENAME = "editorial_rules_snapshot.md"
RUN_MANIFEST_FILENAME = "manifest.json"
RUN_OUTCOMES_FILENAME = "outcomes.jsonl"
RUN_REQUESTS_FILENAME = "requests.jsonl"
RUN_GEPA_STATS_FILENAME = "gepa_stats.json"

RunOrigin = Literal["live", "mock_test"]
RunKind = Literal["development", "optimization", "benchmark", "playground"]
RunStatus = Literal[
    "running", "completed", "stopped", "failed", "interrupted"
]
OutcomePhase = Literal["development", "baseline", "optimized"]
OutcomeStatus = Literal["ok", "error"]
SplitName = Literal["train", "validation", "test", "train_validation"]
RequestStatus = Literal[
    "admitted", "in_flight", "succeeded", "pending", "released", "reconciled"
]
CostStatus = Literal["none", "estimated", "actual", "pending"]

_PATTERN_RUN_ID = r"^[a-z0-9][a-z0-9-]{0,63}$"
_PATTERN_PROGRAM_ID = r"^program-[0-9a-f]{16}$"
_PATTERN_CANDIDATE_ID = r"^[a-z0-9][a-z0-9._-]{0,63}$"
_PATTERN_SHA256 = r"^[0-9a-f]{64}$"
_PATTERN_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_CORE_VERSION_KEYS = (
    "python",
    "application",
    "dspy",
    "gepa",
    "litellm",
    "pydantic",
)
_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "api_base",
        "base_url",
        "model_list",
        "headers",
    }
)


class ArtifactError(RuntimeError):
    """Base class for persisted artifact failures."""


class ArtifactNotFoundError(ArtifactError):
    """Raised when a required artifact or file is absent."""


class ArtifactCorruptError(ArtifactError):
    """Raised when hashes, JSON structure, or state consistency fail."""


class ArtifactSchemaError(ArtifactError):
    """Raised when an artifact does not match its frozen schema."""


class ArtifactCompatibilityError(ArtifactError):
    """Raised when an artifact cannot run in the current environment."""


class ArtifactSecretError(ArtifactError):
    """Raised when an artifact appears to contain credential material."""


class ArtifactIncompleteError(ArtifactError):
    """Raised when a Replay archive is not complete enough to present."""


class ReplayError(ArtifactError):
    """Raised when linked Replay records are inconsistent or incomplete."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_text(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_bytes(payload: object) -> bytes:
    return canonical_json_text(payload).encode("utf-8")


def _pretty_json_text(payload: object) -> str:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utc_string(value: datetime | None = None) -> str:
    current = value or _utcnow()
    return current.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact file path must be a non-empty string")
    if "\\" in value:
        raise ValueError("artifact file paths must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError("artifact file path must be relative without '..'")
    return value


def _safe_join(root: Path, relative: str) -> Path:
    relative = _validate_relative_path(relative)
    root_resolved = root.resolve()
    target = (root / relative).resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise ArtifactError(f"artifact path escapes its root: {relative}")
    return target


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_bytes(
    path: Path, data: bytes, *, mode: int = 0o600
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def atomic_write_text(
    path: Path, text: str, *, mode: int = 0o600
) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), mode=mode)


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(path, _pretty_json_text(payload))


def atomic_write_jsonl(
    path: Path, records: Iterable[object]
) -> None:
    lines = "".join(
        canonical_json_text(record) + "\n" for record in records
    )
    atomic_write_text(path, lines)


def append_jsonl(path: Path, record: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (canonical_json_text(record) + "\n").encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short append_jsonl write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_regular_file(path: Path) -> None:
    if path.is_symlink():
        raise ArtifactError(f"artifact files must not be symlinks: {path}")
    if not path.is_file():
        raise ArtifactNotFoundError(f"artifact file is missing: {path}")


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json_object(path: Path) -> dict[str, object]:
    _require_regular_file(path)
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ArtifactCorruptError(
            f"artifact file is not valid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ArtifactCorruptError(
            f"artifact file must contain one JSON object: {path}"
        )
    return payload


def _split_jsonl_lines(text: str) -> list[str]:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    _require_regular_file(path)
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(
        _split_jsonl_lines(path.read_text(encoding="utf-8")), start=1
    ):
        if not line.strip():
            raise ArtifactCorruptError(
                f"{path}:{line_number}: blank JSONL line"
            )
        try:
            payload = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ArtifactCorruptError(
                f"{path}:{line_number}: invalid JSONL record: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ArtifactCorruptError(
                f"{path}:{line_number}: JSONL record must be an object"
            )
        records.append(payload)
    return records


def _validate_model(
    model: type[BaseModel],
    payload: Mapping[str, object],
    *,
    context: str,
) -> BaseModel:
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise ArtifactSchemaError(
            f"{context} does not match {model.__name__}: {exc}"
        ) from exc


def _effective_secret_values(
    secret_values: Sequence[str],
) -> tuple[str, ...]:
    values = [value for value in secret_values if value]
    environment_key = (
        os.environ.get("OPENROUTER_API_KEY") or ""
    ).strip()
    if environment_key and environment_key not in values:
        values.append(environment_key)
    return tuple(values)


def _assert_no_secret_values(
    text: str,
    secrets: Sequence[str],
    *,
    context: str,
) -> None:
    for secret in secrets:
        if secret and secret in text:
            raise ArtifactSecretError(
                f"{context} contains a configured secret value"
            )


def _assert_no_secret_keys(
    payload: object, *, context: str
) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = str(key).strip().lower()
            if normalized in _SECRET_KEYS:
                raise ArtifactSecretError(
                    f"{context} contains a credential-bearing key "
                    f"{key!r}"
                )
            _assert_no_secret_keys(value, context=context)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            _assert_no_secret_keys(item, context=context)


def _file_contains_secret(path: Path, secret: str) -> bool:
    needle = secret.encode("utf-8")
    if not needle:
        return False
    overlap = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            data = overlap + chunk
            if needle in data:
                return True
            overlap = data[-(len(needle) - 1) :] if len(needle) > 1 else b""
    return False


def scan_tree_for_secrets(
    root: Path, secrets: Sequence[str]
) -> list[Path]:
    hits: list[Path] = []
    if not root.exists():
        return hits
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        for secret in secrets:
            if secret and secret.encode("utf-8") in data:
                hits.append(path)
                break
    return hits


def file_index(root: Path, names: Sequence[str]) -> dict[str, str]:
    index: dict[str, str] = {}
    for name in names:
        name = _validate_relative_path(name)
        path = _safe_join(root, name)
        _require_regular_file(path)
        index[name] = sha256_file(path)
    return index


def verify_file_index(
    root: Path, expected: Mapping[str, str]
) -> None:
    for name, expected_hash in expected.items():
        path = _safe_join(root, name)
        _require_regular_file(path)
        actual = sha256_file(path)
        if actual != expected_hash:
            raise ArtifactCorruptError(
                f"artifact hash mismatch for {name}: "
                f"expected {expected_hash}, got {actual}"
            )


def _core_versions(
    versions: Mapping[str, str] | None = None,
) -> dict[str, str]:
    source = versions or collect_runtime_versions()
    missing = [key for key in _CORE_VERSION_KEYS if key not in source]
    if missing:
        raise ArtifactCompatibilityError(
            f"runtime version report is missing {missing}"
        )
    return {key: source[key] for key in _CORE_VERSION_KEYS}


def _program_identity_payload(
    *,
    program_name: str,
    instructions: str,
    schema: Mapping[str, object],
    signature_fields: Sequence[Mapping[str, object]],
    demos: Sequence[object],
    rules_version: str,
    rules_sha256: str,
    prompt_version: str,
    prompt_sha256: str,
    source_sha256: str,
    adapter: str,
    adapter_version: int,
    versions: Mapping[str, str],
) -> dict[str, object]:
    return {
        "program_name": program_name,
        "instructions": instructions,
        "schema": schema,
        "signature_fields": [dict(field) for field in signature_fields],
        "demos": list(demos),
        "rules_version": rules_version,
        "rules_sha256": rules_sha256,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256,
        "source_sha256": source_sha256,
        "adapter": adapter,
        "adapter_version": adapter_version,
        "versions": _core_versions(versions),
    }


def _program_identity_sha256(
    *,
    program_name: str,
    instructions: str,
    schema: Mapping[str, object],
    signature_fields: Sequence[Mapping[str, object]],
    demos: Sequence[object],
    rules_version: str,
    rules_sha256: str,
    prompt_version: str,
    prompt_sha256: str,
    source_sha256: str,
    adapter: str,
    adapter_version: int,
    versions: Mapping[str, str],
) -> str:
    payload = _program_identity_payload(
        program_name=program_name,
        instructions=instructions,
        schema=schema,
        signature_fields=signature_fields,
        demos=demos,
        rules_version=rules_version,
        rules_sha256=rules_sha256,
        prompt_version=prompt_version,
        prompt_sha256=prompt_sha256,
        source_sha256=source_sha256,
        adapter=adapter,
        adapter_version=adapter_version,
        versions=versions,
    )
    return sha256_text(canonical_json_text(payload))


class ProgramManifest(BaseModel):
    """One immutable, state-only program artifact manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["program_manifest_v1"] = (
        PROGRAM_MANIFEST_SCHEMA_VERSION
    )
    program_id: str = Field(pattern=_PATTERN_PROGRAM_ID)
    program_name: Literal["EditorialQA"] = "EditorialQA"
    created_at: str
    origin: RunOrigin
    parent_program_id: str | None = Field(
        default=None, pattern=_PATTERN_PROGRAM_ID
    )
    baseline: bool
    rules_version: Literal["editorial_rules_v1"]
    rules_sha256: str = Field(pattern=_PATTERN_SHA256)
    prompt_version: str = Field(min_length=1, max_length=64)
    prompt_sha256: str = Field(pattern=_PATTERN_SHA256)
    source_sha256: str = Field(pattern=_PATTERN_SHA256)
    adapter: Literal["StrictJSONAdapter"] = "StrictJSONAdapter"
    adapter_version: int = Field(ge=1)
    instructions: str = Field(min_length=1)
    instructions_sha256: str = Field(pattern=_PATTERN_SHA256)
    signature_schema: dict[str, object]
    signature_fields: list[dict[str, object]] = Field(min_length=1)
    schema_sha256: str = Field(pattern=_PATTERN_SHA256)
    demos: list[object] = Field(default_factory=list)
    predictor_count: Literal[1] = 1
    lm: None = None
    state_sha256: str = Field(pattern=_PATTERN_SHA256)
    state_bytes: int = Field(ge=1)
    versions: dict[str, str]
    identity_sha256: str = Field(pattern=_PATTERN_SHA256)

    @field_validator("created_at")
    @classmethod
    def _created_at_utc(cls, value: str) -> str:
        if _PATTERN_UTC.fullmatch(value) is None:
            raise ValueError("created_at must be a UTC timestamp ending in Z")
        return value

    @field_validator("instructions")
    @classmethod
    def _instructions_non_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instructions must contain non-whitespace text")
        return value

    @model_validator(mode="after")
    def _validate_manifest(self) -> "ProgramManifest":
        if self.demos:
            raise ValueError("program artifacts must contain zero demos")
        if self.lm is not None:
            raise ValueError("program artifacts must not persist an LM")
        if self.parent_program_id == self.program_id:
            raise ValueError("a program cannot be its own parent")
        if self.instructions_sha256 != sha256_text(self.instructions):
            raise ValueError("instructions_sha256 does not match instructions")
        if self.schema_sha256 != sha256_text(
            canonical_json_text(self.signature_schema)
        ):
            raise ValueError(
                "schema_sha256 does not match signature_schema"
            )
        identity = _program_identity_sha256(
            program_name=self.program_name,
            instructions=self.instructions,
            schema=self.signature_schema,
            signature_fields=self.signature_fields,
            demos=self.demos,
            rules_version=self.rules_version,
            rules_sha256=self.rules_sha256,
            prompt_version=self.prompt_version,
            prompt_sha256=self.prompt_sha256,
            source_sha256=self.source_sha256,
            adapter=self.adapter,
            adapter_version=self.adapter_version,
            versions=self.versions,
        )
        if identity != self.identity_sha256:
            raise ValueError("identity_sha256 does not match program identity")
        if self.program_id != f"program-{identity[:16]}":
            raise ValueError("program_id does not match program identity")
        return self


class SnapshotManifest(BaseModel):
    """Exact cases and rules snapshot referenced by one run."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["snapshot_v1"] = SNAPSHOT_SCHEMA_VERSION
    split: SplitName
    dataset_version: str = Field(min_length=1, max_length=64)
    case_count: int = Field(ge=1)
    case_ids: list[str] = Field(min_length=1)
    cases_file: Literal["inputs_snapshot.jsonl"] = (
        SNAPSHOT_CASES_FILENAME
    )
    cases_sha256: str = Field(pattern=_PATTERN_SHA256)
    rules_file: Literal["editorial_rules_snapshot.md"] = (
        SNAPSHOT_RULES_FILENAME
    )
    rules_version: Literal["editorial_rules_v1"]
    rules_sha256: str = Field(pattern=_PATTERN_SHA256)

    @model_validator(mode="after")
    def _validate_snapshot(self) -> "SnapshotManifest":
        if self.case_count != len(self.case_ids):
            raise ValueError("case_count does not match case_ids")
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("snapshot case_ids must be unique")
        return self


class ProgramReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    program_id: str = Field(pattern=_PATTERN_PROGRAM_ID)
    manifest_sha256: str = Field(pattern=_PATTERN_SHA256)
    state_sha256: str = Field(pattern=_PATTERN_SHA256)


class RunLinks(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    baseline_development_run_id: str | None = Field(
        default=None, pattern=_PATTERN_RUN_ID
    )
    optimization_run_id: str | None = Field(
        default=None, pattern=_PATTERN_RUN_ID
    )
    benchmark_run_id: str | None = Field(
        default=None, pattern=_PATTERN_RUN_ID
    )
    baseline_program_id: str | None = Field(
        default=None, pattern=_PATTERN_PROGRAM_ID
    )
    optimized_program_id: str | None = Field(
        default=None, pattern=_PATTERN_PROGRAM_ID
    )


class BenchmarkProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    exposure_index: int = Field(ge=0)
    paired_order: list[Literal["baseline", "optimized"]]
    case_orders: dict[str, list[Literal["baseline", "optimized"]]] = (
        Field(default_factory=dict)
    )
    seed: int | None = None
    protocol_sha256: str = Field(pattern=_PATTERN_SHA256)
    released: bool
    comparability_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_protocol(self) -> "BenchmarkProtocol":
        if len(self.paired_order) != 2 or set(self.paired_order) != {
            "baseline",
            "optimized",
        }:
            raise ValueError(
                "paired_order must contain baseline and optimized exactly once"
            )
        for case_id, order in self.case_orders.items():
            if re.fullmatch(r"^[a-z0-9][a-z0-9-]*$", case_id) is None:
                raise ValueError(
                    f"case_orders contains an invalid case ID {case_id!r}"
                )
            if len(order) != 2 or set(order) != {
                "baseline",
                "optimized",
            }:
                raise ValueError(
                    "every case order must contain baseline and optimized "
                    "exactly once"
                )
        return self


class RunManifest(BaseModel):
    """One linked development, optimization, benchmark, or playground run."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["run_manifest_v1"] = RUN_MANIFEST_SCHEMA_VERSION
    run_id: str = Field(pattern=_PATTERN_RUN_ID)
    kind: RunKind
    origin: RunOrigin
    status: RunStatus
    created_at: str
    finished_at: str | None = None
    versions: dict[str, str]
    snapshot: SnapshotManifest
    programs: list[ProgramReference] = Field(min_length=1)
    links: RunLinks = Field(default_factory=RunLinks)
    benchmark: BenchmarkProtocol | None = None
    files: dict[str, str] = Field(default_factory=dict)
    stop_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("created_at", "finished_at")
    @classmethod
    def _timestamp_utc(cls, value: str | None) -> str | None:
        if value is not None and _PATTERN_UTC.fullmatch(value) is None:
            raise ValueError(
                "run timestamps must be UTC ISO-8601 ending in Z"
            )
        return value

    @field_validator("files")
    @classmethod
    def _files_are_safe(
        cls, value: dict[str, str]
    ) -> dict[str, str]:
        for name, digest in value.items():
            _validate_relative_path(name)
            if name == RUN_MANIFEST_FILENAME:
                raise ValueError("files must not include manifest.json")
            if re.fullmatch(_PATTERN_SHA256, digest) is None:
                raise ValueError(f"invalid sha256 for {name!r}")
        return value

    @model_validator(mode="after")
    def _validate_run(self) -> "RunManifest":
        if self.status == "running":
            if self.finished_at is not None:
                raise ValueError("running runs must not have finished_at")
        elif self.finished_at is None:
            raise ValueError(
                "finished_at is required once a run is no longer running"
            )
        if self.created_at > (
            self.finished_at or self.created_at
        ):
            raise ValueError("finished_at must not precede created_at")
        program_ids = [ref.program_id for ref in self.programs]
        references_by_id: dict[str, ProgramReference] = {}
        for reference in self.programs:
            previous = references_by_id.get(reference.program_id)
            if previous is not None and previous != reference:
                raise ValueError(
                    "duplicate program references have conflicting hashes"
                )
            references_by_id[reference.program_id] = reference
        missing_versions = [
            key for key in _CORE_VERSION_KEYS if key not in self.versions
        ]
        if missing_versions:
            raise ValueError(
                f"run versions are missing {missing_versions}"
            )
        links = self.links
        if self.kind == "development":
            if links.baseline_program_id is None:
                raise ValueError(
                    "development runs require links.baseline_program_id"
                )
            if program_ids != [links.baseline_program_id]:
                raise ValueError(
                    "development runs must reference exactly the baseline"
                )
            if links.optimization_run_id is not None:
                raise ValueError(
                    "development runs cannot link an optimization run"
                )
            if links.optimized_program_id is not None:
                raise ValueError(
                    "development runs cannot link an optimized program"
                )
            if self.snapshot.split != "validation":
                raise ValueError(
                    "development runs require a validation snapshot"
                )
        elif self.kind == "optimization":
            required = (
                links.baseline_development_run_id,
                links.baseline_program_id,
                links.optimized_program_id,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "optimization runs require baseline development, "
                    "baseline program, and optimized program links"
                )
            if set(program_ids) != {
                links.baseline_program_id,
                links.optimized_program_id,
            }:
                raise ValueError(
                    "optimization runs must reference baseline and "
                    "optimized programs"
                )
            if self.snapshot.split != "train_validation":
                raise ValueError(
                    "optimization runs require a train_validation snapshot"
                )
        elif self.kind == "benchmark":
            required = (
                links.baseline_development_run_id,
                links.optimization_run_id,
                links.baseline_program_id,
                links.optimized_program_id,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "benchmark runs require development, optimization, "
                    "baseline program, and optimized program links"
                )
            if set(program_ids) != {
                links.baseline_program_id,
                links.optimized_program_id,
            }:
                raise ValueError(
                    "benchmark runs must reference exactly the baseline "
                    "and optimized programs"
                )
            if self.snapshot.split != "test":
                raise ValueError(
                    "benchmark runs require a test snapshot"
                )
            if self.benchmark is None:
                raise ValueError(
                    "benchmark runs require benchmark protocol metadata"
                )
            if self.origin == "mock_test" and self.benchmark.released:
                raise ValueError(
                    "mock/test archives cannot be marked released"
                )
        elif self.kind == "playground":
            if self.benchmark is not None:
                raise ValueError(
                    "playground runs cannot contain benchmark protocol"
                )
        if self.kind != "benchmark" and self.benchmark is not None:
            raise ValueError(
                "benchmark protocol metadata is only valid for benchmarks"
            )
        return self


class RecordedRequest(BaseModel):
    """One archived guarded request; no credential fields are allowed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    phase: str = Field(min_length=1)
    model: str = Field(min_length=1)
    provider: str | None = None
    status: RequestStatus
    cost_status: CostStatus
    reserved_cost_usd: Decimal
    estimated_cost_usd: Decimal | None = None
    reported_cost_usd: Decimal | None = None
    input_token_bound: int = Field(ge=0)
    output_token_bound: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    generation_id: str | None = None
    response_model: str | None = None
    finish_reason: str | None = None
    error: str | None = None
    released_reason: str | None = None
    messages: list[dict[str, object]] = Field(min_length=1)
    request_kwargs: dict[str, object]
    usage: dict[str, object]
    started_at: datetime
    finished_at: datetime | None = None
    response_text: str | None = None

    @field_validator(
        "reserved_cost_usd",
        "estimated_cost_usd",
        "reported_cost_usd",
        mode="before",
    )
    @classmethod
    def _nonnegative_decimal(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("cost must be a non-negative decimal")
        try:
            decimal = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("cost must be a decimal") from exc
        if not decimal.is_finite() or decimal < 0:
            raise ValueError("cost must be finite and non-negative")
        return decimal

    @model_validator(mode="after")
    def _validate_request(self) -> "RecordedRequest":
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        return self


class RecordedOutcome(BaseModel):
    """One archived prediction/error associated with a case and arm."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["outcome_v1"] = OUTCOME_SCHEMA_VERSION
    run_id: str = Field(pattern=_PATTERN_RUN_ID)
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    program_id: str = Field(pattern=_PATTERN_PROGRAM_ID)
    candidate_id: str = Field(pattern=_PATTERN_CANDIDATE_ID)
    phase: OutcomePhase
    split: Literal["train", "validation", "test"]
    status: OutcomeStatus
    findings: list[Finding] | None = None
    error: str | None = None
    metric: dict[str, object] | None = None
    request_ids: list[str] = Field(default_factory=list)
    cost_usd: Decimal | None = None
    cost_status: CostStatus
    latency_ms: float | None = None
    input_sha256: str = Field(pattern=_PATTERN_SHA256)
    created_at: str | None = None

    @field_validator("cost_usd", mode="before")
    @classmethod
    def _outcome_cost(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("cost_usd must be a non-negative decimal")
        try:
            decimal = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("cost_usd must be a decimal") from exc
        if not decimal.is_finite() or decimal < 0:
            raise ValueError("cost_usd must be finite and non-negative")
        return decimal

    @field_validator("latency_ms")
    @classmethod
    def _latency(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("latency_ms must be non-negative")
        return value

    @field_validator("created_at")
    @classmethod
    def _created_at(cls, value: str | None) -> str | None:
        if value is not None and _PATTERN_UTC.fullmatch(value) is None:
            raise ValueError(
                "created_at must be a UTC timestamp ending in Z"
            )
        return value

    @model_validator(mode="after")
    def _validate_outcome(self) -> "RecordedOutcome":
        if len(self.request_ids) != len(set(self.request_ids)):
            raise ValueError("request_ids must be unique")
        if self.status == "ok":
            if self.findings is None:
                raise ValueError("ok outcomes require a findings list")
            if self.error is not None:
                raise ValueError("ok outcomes must not contain an error")
        else:
            if self.findings is not None:
                raise ValueError("error outcomes must not contain findings")
            if self.error is None or not self.error.strip():
                raise ValueError("error outcomes require a non-empty error")
        return self


class SentTrace(BaseModel):
    """Exactly the messages and metadata recorded at the LM boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["sent"] = "sent"
    sent: Literal[True] = True
    phase: str = Field(min_length=1)
    model: str = Field(min_length=1)
    provider: str | None
    messages: list[dict[str, object]] = Field(min_length=1)
    request_kwargs: dict[str, object]
    generation_id: str | None
    usage: dict[str, object]
    estimated_cost_usd: str | None
    reported_cost_usd: str | None = None
    cost_status: str | None = None
    status: str
    finish_reason: str | None = None
    response_text: str | None = None


class PreviewTrace(BaseModel):
    """A prompt preview that was not sent and has no provider metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["preview"] = "preview"
    sent: Literal[False] = False
    notice: str = (
        "Preview only; this prompt was not sent and has no provider usage, "
        "generation ID, or cost."
    )
    messages: list[dict[str, object]] = Field(min_length=1)
    request_kwargs: dict[str, object] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LoadedProgram:
    program: EditorialQA
    manifest: ProgramManifest
    artifact_dir: Path


@dataclass(frozen=True, slots=True)
class ProgramSnapshot:
    program_id: str
    manifest: ProgramManifest
    manifest_sha256: str
    state: Mapping[str, object]
    state_sha256: str
    instructions: str
    schema: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ReplayArchive:
    benchmark_dir: Path
    benchmark: RunManifest
    development: RunManifest
    optimization: RunManifest
    snapshot: SnapshotManifest
    cases: tuple[CaseRecord, ...]
    rules_text: str
    programs: Mapping[str, ProgramSnapshot]
    development_outcomes: tuple[RecordedOutcome, ...]
    outcomes: tuple[RecordedOutcome, ...]
    requests: Mapping[str, RecordedRequest]
    candidate_stats: Mapping[str, object] | None
    is_measured: bool
    warnings: tuple[str, ...]

    @property
    def additional_api_calls(self) -> int:
        return 0

    @property
    def additional_api_cost_usd(self) -> Decimal:
        return Decimal("0.00")

    def outcome_pairs(
        self, case_id: str
    ) -> dict[Literal["baseline", "optimized"], RecordedOutcome]:
        selected = {
            outcome.phase: outcome
            for outcome in self.outcomes
            if outcome.case_id == case_id
        }
        if set(selected) != {"baseline", "optimized"}:
            raise ReplayError(
                f"case {case_id!r} does not have one baseline and one "
                "optimized outcome"
            )
        return selected  # type: ignore[return-value]

    def sent_trace(self, call_id: str) -> SentTrace:
        try:
            request = self.requests[call_id]
        except KeyError as exc:
            raise ReplayError(
                f"archived request {call_id!r} does not exist"
            ) from exc
        return format_sent_trace(request)


def save_program_artifact(
    program: EditorialQA,
    *,
    origin: RunOrigin,
    programs_root: Path = PROGRAMS_ROOT,
    parent_program_id: str | None = None,
    created_at: str | None = None,
    secret_values: Sequence[str] = (),
) -> ProgramManifest:
    """Atomically save one state-only program, manifest, and instructions."""
    if not isinstance(program, EditorialQA):
        raise ArtifactError("program must be an EditorialQA instance")
    predictors = list(program.named_predictors())
    if len(predictors) != 1 or predictors[0][0] != "predict":
        raise ArtifactError(
            "EditorialQA must contain exactly the named predictor 'predict'"
        )
    predictor = program.predict
    if predictor.lm is not None:
        raise ArtifactError("the program predictor must not persist an LM")
    if predictor.demos:
        raise ArtifactError("program artifacts require zero demos")
    instructions = predictor.signature.instructions
    if not isinstance(instructions, str) or not instructions.strip():
        raise ArtifactError("program instructions must be non-empty")
    schema = baseline_schema()
    signature_fields = _signature_fields_snapshot(predictor.signature)
    rules_sha256 = sha256_file(EDITORIAL_RULES_PATH)
    prompt_sha256 = sha256_file(BASELINE_PROMPT_PATH)
    source_sha256 = sha256_file(PROGRAM_SOURCE_PATH)
    versions = collect_runtime_versions()
    identity_payload = _program_identity_payload(
        program_name="EditorialQA",
        instructions=instructions,
        schema=schema,
        signature_fields=signature_fields,
        demos=[],
        rules_version=RULES_VERSION,
        rules_sha256=rules_sha256,
        prompt_version=BASELINE_PROMPT_VERSION,
        prompt_sha256=prompt_sha256,
        source_sha256=source_sha256,
        adapter="StrictJSONAdapter",
        adapter_version=STRICT_ADAPTER_VERSION,
        versions=versions,
    )
    identity_sha256 = sha256_text(
        canonical_json_text(identity_payload)
    )
    baseline_identity = _program_identity_sha256(
        program_name="EditorialQA",
        instructions=load_baseline_instructions(),
        schema=schema,
        signature_fields=signature_fields,
        demos=[],
        rules_version=RULES_VERSION,
        rules_sha256=rules_sha256,
        prompt_version=BASELINE_PROMPT_VERSION,
        prompt_sha256=prompt_sha256,
        source_sha256=source_sha256,
        adapter="StrictJSONAdapter",
        adapter_version=STRICT_ADAPTER_VERSION,
        versions=versions,
    )
    program_id = f"program-{identity_sha256[:16]}"
    if parent_program_id is not None:
        if re.fullmatch(_PATTERN_PROGRAM_ID, parent_program_id) is None:
            raise ArtifactError("parent_program_id is not a program ID")
        parent_dir = programs_root / parent_program_id
        if not parent_dir.is_dir():
            raise ArtifactNotFoundError(
                f"parent program artifact is missing: {parent_program_id}"
            )
        parent_manifest = read_program_manifest(parent_dir)
        if parent_manifest.origin != origin:
            raise ArtifactError(
                "parent and child program artifacts must share one origin"
            )
    final_dir = programs_root / program_id
    if final_dir.exists():
        existing = read_program_snapshot(final_dir).manifest
        if (
            existing.identity_sha256 == identity_sha256
            and existing.origin == origin
            and existing.parent_program_id == parent_program_id
        ):
            return existing
        raise ArtifactError(
            f"program artifact {program_id} already exists with "
            "different lineage or origin"
        )
    programs_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            dir=programs_root,
            prefix=f".{program_id}.",
            suffix=".tmp",
        )
    )
    effective_secrets = _effective_secret_values(secret_values)
    try:
        state_path = temporary_dir / PROGRAM_STATE_FILENAME
        program.save(str(state_path), save_program=False)
        os.chmod(state_path, 0o600)
        state = _read_json_object(state_path)
        _validate_program_state(
            state,
            instructions=instructions,
            signature_fields=signature_fields,
        )
        state_bytes = state_path.read_bytes()
        state_sha256 = sha256_bytes(state_bytes)
        if effective_secrets:
            _assert_no_secret_values(
                state_bytes.decode("utf-8"),
                effective_secrets,
                context="program state",
            )
        _assert_no_secret_keys(state, context="program state")
        manifest = ProgramManifest(
            program_id=program_id,
            created_at=created_at or _utc_string(),
            origin=origin,
            parent_program_id=parent_program_id,
            baseline=identity_sha256 == baseline_identity,
            rules_version=RULES_VERSION,
            rules_sha256=rules_sha256,
            prompt_version=BASELINE_PROMPT_VERSION,
            prompt_sha256=prompt_sha256,
            source_sha256=source_sha256,
            adapter_version=STRICT_ADAPTER_VERSION,
            instructions=instructions,
            instructions_sha256=sha256_text(instructions),
            signature_schema=schema,
            signature_fields=signature_fields,
            schema_sha256=sha256_text(canonical_json_text(schema)),
            state_sha256=state_sha256,
            state_bytes=len(state_bytes),
            versions=versions,
            identity_sha256=identity_sha256,
        )
        manifest_text = _pretty_json_text(
            manifest.model_dump(mode="json")
        )
        instructions_text = instructions
        if effective_secrets:
            _assert_no_secret_values(
                manifest_text,
                effective_secrets,
                context="program manifest",
            )
            _assert_no_secret_values(
                instructions_text,
                effective_secrets,
                context="program instructions",
            )
        atomic_write_text(
            temporary_dir / PROGRAM_INSTRUCTIONS_FILENAME,
            instructions_text,
        )
        atomic_write_text(
            temporary_dir / PROGRAM_MANIFEST_FILENAME,
            manifest_text,
        )
        if effective_secrets:
            hits = scan_tree_for_secrets(
                temporary_dir, effective_secrets
            )
            if hits:
                raise ArtifactSecretError(
                    f"program artifact contains a secret in {hits[0].name}"
                )
        os.replace(temporary_dir, final_dir)
        _fsync_directory(programs_root)
        return manifest
    except Exception:
        for path in sorted(
            temporary_dir.rglob("*"), reverse=True
        ):
            if path.is_file():
                path.unlink(missing_ok=True)
            else:
                path.rmdir()
        temporary_dir.rmdir()
        raise


def read_program_manifest(artifact_dir: Path) -> ProgramManifest:
    artifact_dir = Path(artifact_dir)
    if artifact_dir.is_symlink():
        raise ArtifactError(
            f"program artifact directory must not be a symlink: {artifact_dir}"
        )
    manifest_path = artifact_dir / PROGRAM_MANIFEST_FILENAME
    payload = _read_json_object(manifest_path)
    manifest = _validate_model(
        ProgramManifest,
        payload,
        context=f"program manifest {manifest_path}",
    )
    assert isinstance(manifest, ProgramManifest)
    if artifact_dir.name != manifest.program_id:
        raise ArtifactCorruptError(
            "program artifact directory does not match program_id"
        )
    state_path = artifact_dir / PROGRAM_STATE_FILENAME
    instructions_path = artifact_dir / PROGRAM_INSTRUCTIONS_FILENAME
    _require_regular_file(state_path)
    _require_regular_file(instructions_path)
    state_bytes = state_path.read_bytes()
    if len(state_bytes) != manifest.state_bytes:
        raise ArtifactCorruptError("program state size does not match manifest")
    if sha256_bytes(state_bytes) != manifest.state_sha256:
        raise ArtifactCorruptError("program state hash does not match manifest")
    instructions_text = instructions_path.read_text(encoding="utf-8")
    if instructions_text != manifest.instructions:
        raise ArtifactCorruptError(
            "instructions.txt does not match manifest instructions"
        )
    return manifest


def load_program_artifact(
    artifact_dir: Path, *, allow_mock: bool = False
) -> LoadedProgram:
    """Load one trusted state-only program after compatibility checks."""
    artifact_dir = Path(artifact_dir)
    manifest = read_program_manifest(artifact_dir)
    if manifest.origin != "live" and not allow_mock:
        raise ArtifactCompatibilityError(
            "mock/test program artifacts cannot be loaded in Live mode"
        )
    mismatches = _compatibility_mismatches(manifest)
    if mismatches:
        raise ArtifactCompatibilityError(
            "program artifact is incompatible: " + "; ".join(mismatches)
        )
    state_path = artifact_dir / PROGRAM_STATE_FILENAME
    state = _read_json_object(state_path)
    _validate_program_state(
        state,
        instructions=manifest.instructions,
        signature_fields=manifest.signature_fields,
    )
    _assert_no_secret_keys(state, context="program state")
    program = EditorialQA()
    try:
        program.load(
            str(state_path),
            allow_pickle=False,
            allow_unsafe_lm_state=False,
        )
    except Exception as exc:
        raise ArtifactCorruptError(
            f"trusted DSPy state failed to load: {exc}"
        ) from exc
    predictors = list(program.named_predictors())
    if len(predictors) != 1 or predictors[0][0] != "predict":
        raise ArtifactCorruptError(
            "loaded program does not have exactly one predictor"
        )
    if program.predict.demos:
        raise ArtifactCorruptError("loaded program contains demos")
    if program.predict.lm is not None:
        raise ArtifactCorruptError("loaded program persisted an LM")
    if program.predict.signature.instructions != manifest.instructions:
        raise ArtifactCorruptError(
            "loaded instructions do not match the manifest"
        )
    return LoadedProgram(
        program=program,
        manifest=manifest,
        artifact_dir=artifact_dir,
    )


def read_program_snapshot(
    artifact_dir: Path,
) -> ProgramSnapshot:
    """Inspect a program artifact for Replay without native loading."""
    artifact_dir = Path(artifact_dir)
    manifest = read_program_manifest(artifact_dir)
    state_path = artifact_dir / PROGRAM_STATE_FILENAME
    state = _read_json_object(state_path)
    _validate_program_state(
        state,
        instructions=manifest.instructions,
        signature_fields=manifest.signature_fields,
    )
    _assert_no_secret_keys(state, context="program state")
    manifest_path = artifact_dir / PROGRAM_MANIFEST_FILENAME
    return ProgramSnapshot(
        program_id=manifest.program_id,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        state=state,
        state_sha256=sha256_file(state_path),
        instructions=manifest.instructions,
        schema=manifest.signature_schema,
    )


def program_reference(artifact_dir: Path) -> ProgramReference:
    manifest = read_program_manifest(artifact_dir)
    return ProgramReference(
        program_id=manifest.program_id,
        manifest_sha256=sha256_file(
            Path(artifact_dir) / PROGRAM_MANIFEST_FILENAME
        ),
        state_sha256=manifest.state_sha256,
    )


def _compatibility_mismatches(
    manifest: ProgramManifest,
) -> list[str]:
    mismatches: list[str] = []
    current_versions = _core_versions(collect_runtime_versions())
    for key, current in current_versions.items():
        saved = manifest.versions.get(key)
        if saved != current:
            mismatches.append(
                f"{key} saved={saved!r} current={current!r}"
            )
    if manifest.rules_version != RULES_VERSION:
        mismatches.append(
            f"rules_version saved={manifest.rules_version!r} "
            f"current={RULES_VERSION!r}"
        )
    if manifest.rules_sha256 != sha256_file(EDITORIAL_RULES_PATH):
        mismatches.append("rules_sha256 changed")
    if manifest.prompt_version != BASELINE_PROMPT_VERSION:
        mismatches.append(
            f"prompt_version saved={manifest.prompt_version!r} "
            f"current={BASELINE_PROMPT_VERSION!r}"
        )
    if manifest.prompt_sha256 != sha256_file(BASELINE_PROMPT_PATH):
        mismatches.append("prompt_sha256 changed")
    if manifest.source_sha256 != sha256_file(PROGRAM_SOURCE_PATH):
        mismatches.append("program source_sha256 changed")
    current_schema = baseline_schema()
    if (
        sha256_text(canonical_json_text(current_schema))
        != manifest.schema_sha256
    ):
        mismatches.append("signature schema changed")
    current_fields = _signature_fields_snapshot(
        EditorialQA().predict.signature
    )
    if current_fields != manifest.signature_fields:
        mismatches.append("signature field descriptions changed")
    if manifest.adapter_version != STRICT_ADAPTER_VERSION:
        mismatches.append(
            f"adapter_version saved={manifest.adapter_version} "
            f"current={STRICT_ADAPTER_VERSION}"
        )
    return mismatches


def _signature_fields_snapshot(
    signature: object,
) -> list[dict[str, object]]:
    fields = getattr(signature, "fields", None)
    if not isinstance(fields, Mapping) or not fields:
        raise ArtifactCorruptError(
            "program signature must contain input/output fields"
        )
    snapshot: list[dict[str, object]] = []
    for name, field in fields.items():
        extra = getattr(field, "json_schema_extra", None)
        if not isinstance(extra, Mapping):
            raise ArtifactCorruptError(
                f"signature field {name!r} has no DSPy metadata"
            )
        prefix = extra.get("prefix")
        description = extra.get("desc")
        field_type = extra.get("__dspy_field_type")
        if not all(
            isinstance(value, str) and value
            for value in (prefix, description, field_type)
        ):
            raise ArtifactCorruptError(
                f"signature field {name!r} metadata is incomplete"
            )
        snapshot.append(
            {
                "name": str(name),
                "prefix": prefix,
                "description": description,
                "field_type": field_type,
            }
        )
    return snapshot


def _validate_program_state(
    state: Mapping[str, object],
    *,
    instructions: str,
    signature_fields: Sequence[Mapping[str, object]],
) -> None:
    if set(state) != {"predict", "metadata"}:
        raise ArtifactCorruptError(
            "program state must contain exactly 'predict' and 'metadata'"
        )
    predict = state.get("predict")
    if not isinstance(predict, Mapping):
        raise ArtifactCorruptError("program state predict must be an object")
    if predict.get("lm") is not None:
        raise ArtifactCorruptError("program state must contain lm: null")
    demos = predict.get("demos")
    if demos != []:
        raise ArtifactCorruptError("program state must contain demos: []")
    signature = predict.get("signature")
    if not isinstance(signature, Mapping):
        raise ArtifactCorruptError(
            "program state signature must be an object"
        )
    if signature.get("instructions") != instructions:
        raise ArtifactCorruptError(
            "program state instructions do not match manifest"
        )
    state_fields = signature.get("fields")
    if not isinstance(state_fields, list):
        raise ArtifactCorruptError(
            "program state signature fields must be a list"
        )
    expected_fields = [
        {
            "prefix": field.get("prefix"),
            "description": field.get("description"),
        }
        for field in signature_fields
    ]
    actual_fields = [
        {
            "prefix": field.get("prefix"),
            "description": field.get("description"),
        }
        for field in state_fields
        if isinstance(field, Mapping)
    ]
    if actual_fields != expected_fields:
        raise ArtifactCorruptError(
            "program state signature fields do not match manifest"
        )
    metadata = state.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ArtifactCorruptError(
            "program state metadata must be an object"
        )


def _json_ready(value: object) -> object:
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _case_json_payload(case: CaseRecord) -> dict[str, object]:
    payload = _json_ready(case.model_dump(mode="python"))
    assert isinstance(payload, dict)
    return payload


def cases_snapshot_sha256(cases: Sequence[CaseRecord]) -> str:
    """Return the exact ordered JSONL hash used by input snapshots."""
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ArtifactError("snapshot case IDs must be unique")
    cases_text = "".join(
        canonical_json_text(_case_json_payload(case)) + "\n"
        for case in cases
    )
    return sha256_bytes(cases_text.encode("utf-8"))


def write_input_snapshot(
    directory: Path,
    *,
    cases: Sequence[CaseRecord],
    rules_text: str,
    split: SplitName,
    dataset_version: str = "v1",
) -> SnapshotManifest:
    """Atomically write exact cases and rules used by a run."""
    directory = Path(directory)
    if not cases:
        raise ArtifactError("input snapshots require at least one case")
    if not isinstance(rules_text, str) or not rules_text.strip():
        raise ArtifactError("rules_text must be non-empty")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ArtifactError("snapshot case IDs must be unique")
    if any(case.rules_version != RULES_VERSION for case in cases):
        raise ArtifactError("all snapshot cases must use the frozen rules")
    cases_text = "".join(
        canonical_json_text(_case_json_payload(case)) + "\n"
        for case in cases
    )
    cases_bytes = cases_text.encode("utf-8")
    if cases_snapshot_sha256(cases) != sha256_bytes(cases_bytes):
        raise ArtifactError("snapshot case hash helper is inconsistent")
    rules_bytes = rules_text.encode("utf-8")
    atomic_write_bytes(
        directory / SNAPSHOT_CASES_FILENAME, cases_bytes
    )
    atomic_write_bytes(
        directory / SNAPSHOT_RULES_FILENAME, rules_bytes
    )
    return SnapshotManifest(
        split=split,
        dataset_version=dataset_version,
        case_count=len(cases),
        case_ids=case_ids,
        cases_sha256=sha256_bytes(cases_bytes),
        rules_version=RULES_VERSION,
        rules_sha256=sha256_bytes(rules_bytes),
    )


def read_input_snapshot(
    directory: Path, snapshot: SnapshotManifest
) -> tuple[list[CaseRecord], str]:
    """Read only the recorded snapshot, never the current dataset."""
    directory = Path(directory)
    cases_path = directory / snapshot.cases_file
    rules_path = directory / snapshot.rules_file
    _require_regular_file(cases_path)
    _require_regular_file(rules_path)
    if sha256_file(cases_path) != snapshot.cases_sha256:
        raise ArtifactCorruptError("snapshot cases hash mismatch")
    if sha256_file(rules_path) != snapshot.rules_sha256:
        raise ArtifactCorruptError("snapshot rules hash mismatch")
    cases: list[CaseRecord] = []
    for line_number, line in enumerate(
        _split_jsonl_lines(cases_path.read_text(encoding="utf-8")),
        start=1,
    ):
        if not line.strip():
            raise ArtifactCorruptError(
                f"snapshot cases line {line_number} is blank"
            )
        try:
            payload = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ArtifactCorruptError(
                f"snapshot cases line {line_number} is invalid: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ArtifactCorruptError(
                f"snapshot cases line {line_number} is not an object"
            )
        case = _validate_model(
            CaseRecord,
            payload,
            context=f"snapshot case line {line_number}",
        )
        assert isinstance(case, CaseRecord)
        cases.append(case)
    if len(cases) != snapshot.case_count:
        raise ArtifactCorruptError(
            "snapshot case_count does not match the cases file"
        )
    if [case.case_id for case in cases] != snapshot.case_ids:
        raise ArtifactCorruptError(
            "snapshot case_ids do not match the cases file"
        )
    rules_text = rules_path.read_text(encoding="utf-8")
    if sha256_text(rules_text) != snapshot.rules_sha256:
        raise ArtifactCorruptError("snapshot rules text hash mismatch")
    return cases, rules_text


def case_input_sha256(case: CaseRecord, rules_text: str) -> str:
    inputs = case_inputs(case, rules_text)
    payload = {
        "document_text": inputs["document_text"],
        "product_facts": inputs["product_facts"].model_dump(mode="json"),
        "sources": [
            source.model_dump(mode="json")
            for source in inputs["sources"]
        ],
        "editorial_rules": inputs["editorial_rules"],
    }
    return sha256_text(canonical_json_text(payload))


def write_run_manifest(
    run_dir: Path,
    manifest: RunManifest,
    *,
    secret_values: Sequence[str] = (),
) -> Path:
    """Verify referenced files, scan for secrets, then replace the manifest."""
    run_dir = Path(run_dir)
    if not isinstance(manifest, RunManifest):
        raise ArtifactError("manifest must be a RunManifest")
    if run_dir.name != manifest.run_id:
        raise ArtifactError(
            "run directory name must match manifest.run_id"
        )
    required_snapshot_files = (
        manifest.snapshot.cases_file,
        manifest.snapshot.rules_file,
    )
    missing_snapshot_files = [
        name
        for name in required_snapshot_files
        if name not in manifest.files
    ]
    if missing_snapshot_files:
        raise ArtifactError(
            "run manifest files must include snapshot files: "
            f"{missing_snapshot_files}"
        )
    read_input_snapshot(run_dir, manifest.snapshot)
    verify_file_index(run_dir, manifest.files)
    manifest_text = _pretty_json_text(
        manifest.model_dump(mode="json")
    )
    effective_secrets = _effective_secret_values(secret_values)
    if effective_secrets:
        _assert_no_secret_values(
            manifest_text,
            effective_secrets,
            context=f"run manifest {manifest.run_id}",
        )
        for name in manifest.files:
            path = _safe_join(run_dir, name)
            for secret in effective_secrets:
                if _file_contains_secret(path, secret):
                    raise ArtifactSecretError(
                        f"run file {name} contains a configured secret"
                    )
    path = run_dir / RUN_MANIFEST_FILENAME
    atomic_write_text(path, manifest_text)
    return path


def read_run_manifest(run_dir: Path) -> RunManifest:
    run_dir = Path(run_dir)
    path = run_dir / RUN_MANIFEST_FILENAME
    payload = _read_json_object(path)
    manifest = _validate_model(
        RunManifest,
        payload,
        context=f"run manifest {path}",
    )
    assert isinstance(manifest, RunManifest)
    if manifest.run_id != run_dir.name:
        raise ArtifactCorruptError(
            "run manifest ID does not match its directory"
        )
    return manifest


def read_recorded_outcomes(
    path: Path, *, run_id: str
) -> list[RecordedOutcome]:
    outcomes: list[RecordedOutcome] = []
    for line_number, payload in enumerate(
        _read_jsonl(path), start=1
    ):
        outcome = _validate_model(
            RecordedOutcome,
            payload,
            context=f"{path}:{line_number}",
        )
        assert isinstance(outcome, RecordedOutcome)
        if outcome.run_id != run_id:
            raise ReplayError(
                f"{path}:{line_number}: outcome run_id does not match "
                f"{run_id!r}"
            )
        outcomes.append(outcome)
    return outcomes


def read_recorded_requests(path: Path) -> dict[str, RecordedRequest]:
    requests: dict[str, RecordedRequest] = {}
    for line_number, payload in enumerate(
        _read_jsonl(path), start=1
    ):
        request = _validate_model(
            RecordedRequest,
            payload,
            context=f"{path}:{line_number}",
        )
        assert isinstance(request, RecordedRequest)
        _assert_no_secret_keys(
            payload, context=f"{path}:{line_number}"
        )
        if request.call_id in requests:
            raise ReplayError(
                f"{path}:{line_number}: duplicate call_id"
            )
        requests[request.call_id] = request
    return requests


def _record_mapping(record: object) -> Mapping[str, object]:
    if isinstance(record, Mapping):
        return record
    if isinstance(record, BaseModel):
        return record.model_dump(mode="python")
    if is_dataclass(record) and not isinstance(record, type):
        return {
            field.name: copy.deepcopy(getattr(record, field.name))
            for field in dataclass_fields(record)
        }
    if hasattr(record, "__dict__"):
        return {
            key: copy.deepcopy(value)
            for key, value in vars(record).items()
        }
    raise ArtifactSchemaError(
        f"unsupported trace record type: {type(record).__name__}"
    )


_SENT_TRACE_REQUIRED_KEYS = frozenset(
    {
        "phase",
        "model",
        "provider",
        "messages",
        "request_kwargs",
        "generation_id",
        "usage",
        "estimated_cost_usd",
        "status",
    }
)


def format_sent_trace(record: object) -> SentTrace:
    """Format one already-sent request for display without changing it."""
    data = _record_mapping(record)
    _assert_no_secret_keys(data, context="sent trace")
    missing = _SENT_TRACE_REQUIRED_KEYS - set(data)
    if missing:
        raise ArtifactSchemaError(
            f"sent trace record is missing {sorted(missing)}"
        )
    if data.get("source") == "preview" or data.get("sent") is False:
        raise ArtifactSchemaError(
            "a preview cannot be labeled as a sent trace"
        )
    messages = data.get("messages")
    request_kwargs = data.get("request_kwargs")
    if (
        not isinstance(messages, (list, tuple))
        or not messages
        or not all(isinstance(item, Mapping) for item in messages)
    ):
        raise ArtifactSchemaError(
            "sent traces require the exact recorded messages"
        )
    if not isinstance(request_kwargs, Mapping):
        raise ArtifactSchemaError(
            "sent traces require recorded request kwargs"
        )
    usage = data.get("usage")
    if not isinstance(usage, Mapping):
        raise ArtifactSchemaError(
            "sent traces require a recorded usage object"
        )
    return SentTrace(
        phase=str(data.get("phase", "")),
        model=str(data.get("model", "")),
        provider=(
            None
            if data.get("provider") is None
            else str(data.get("provider"))
        ),
        messages=copy.deepcopy([dict(item) for item in messages]),
        request_kwargs=copy.deepcopy(dict(request_kwargs)),
        generation_id=(
            None
            if data.get("generation_id") is None
            else str(data.get("generation_id"))
        ),
        usage=copy.deepcopy(dict(usage)),
        estimated_cost_usd=_decimal_text(
            data.get("estimated_cost_usd")
        ),
        reported_cost_usd=_decimal_text(
            data.get("reported_cost_usd")
        ),
        cost_status=(
            None
            if data.get("cost_status") is None
            else str(data.get("cost_status"))
        ),
        status=str(data.get("status", "")),
        finish_reason=(
            None
            if data.get("finish_reason") is None
            else str(data.get("finish_reason"))
        ),
        response_text=(
            None
            if data.get("response_text") is None
            else str(data.get("response_text"))
        ),
    )


def format_preview_trace(
    messages: Sequence[Mapping[str, object]],
    request_kwargs: Mapping[str, object] | None = None,
) -> PreviewTrace:
    """Format a hypothetical prompt without implying provider execution."""
    if not messages:
        raise ArtifactSchemaError(
            "preview traces require at least one message"
        )
    if not all(isinstance(item, Mapping) for item in messages):
        raise ArtifactSchemaError("preview messages must be objects")
    return PreviewTrace(
        messages=copy.deepcopy([dict(item) for item in messages]),
        request_kwargs=copy.deepcopy(dict(request_kwargs or {})),
    )


def _decimal_text(value: object) -> str | None:
    return None if value is None else str(value)


def _run_dir(runs_root: Path, kind: RunKind, run_id: str) -> Path:
    folders = {
        "development": "development",
        "optimization": "optimization",
        "benchmark": "benchmarks",
        "playground": "playground",
    }
    return runs_root / folders[kind] / run_id


def _read_linked_run(
    runs_root: Path,
    kind: RunKind,
    run_id: str,
) -> tuple[Path, RunManifest]:
    run_dir = _run_dir(runs_root, kind, run_id)
    if not run_dir.is_dir():
        raise ArtifactNotFoundError(
            f"linked {kind} run is missing: {run_id}"
        )
    manifest = read_run_manifest(run_dir)
    if manifest.kind != kind:
        raise ReplayError(
            f"linked run {run_id!r} is {manifest.kind!r}, not {kind!r}"
        )
    verify_file_index(run_dir, manifest.files)
    return run_dir, manifest


def is_measured_archive(manifest: RunManifest) -> bool:
    """True only for a completed, released Live benchmark."""
    return bool(
        manifest.kind == "benchmark"
        and manifest.origin == "live"
        and manifest.status == "completed"
        and manifest.benchmark is not None
        and manifest.benchmark.released
    )


def open_replay_archive(
    benchmark_dir: Path,
    *,
    runs_root: Path | None = None,
    programs_root: Path = PROGRAMS_ROOT,
) -> ReplayArchive:
    """Validate and read one benchmark's linked, recorded data offline."""
    benchmark_dir = Path(benchmark_dir)
    if not benchmark_dir.is_dir():
        raise ArtifactNotFoundError(
            f"benchmark run directory is missing: {benchmark_dir}"
        )
    if benchmark_dir.parent.name != "benchmarks":
        raise ReplayError(
            "benchmark directory must live under runs/benchmarks/"
        )
    effective_runs_root = runs_root or benchmark_dir.parent.parent
    benchmark = read_run_manifest(benchmark_dir)
    if benchmark.kind != "benchmark":
        raise ReplayError("Replay entrypoint must be a benchmark run")
    if benchmark.status != "completed":
        raise ArtifactIncompleteError(
            f"benchmark run is {benchmark.status!r}, not completed"
        )
    if benchmark.benchmark is None:
        raise ReplayError("benchmark protocol metadata is missing")
    verify_file_index(benchmark_dir, benchmark.files)
    links = benchmark.links
    assert links.baseline_development_run_id is not None
    assert links.optimization_run_id is not None
    assert links.baseline_program_id is not None
    assert links.optimized_program_id is not None
    development_dir, development = _read_linked_run(
        effective_runs_root,
        "development",
        links.baseline_development_run_id,
    )
    optimization_dir, optimization = _read_linked_run(
        effective_runs_root,
        "optimization",
        links.optimization_run_id,
    )
    read_input_snapshot(
        optimization_dir, optimization.snapshot
    )
    if development.status != "completed":
        raise ArtifactIncompleteError(
            "baseline development run is not completed"
        )
    if optimization.status != "completed":
        raise ArtifactIncompleteError(
            "optimization run is not completed"
        )
    if not (
        development.origin
        == optimization.origin
        == benchmark.origin
    ):
        raise ReplayError("linked runs do not share one origin")
    _validate_run_links(
        benchmark, development, optimization
    )
    _validate_shared_rules(
        benchmark, development, optimization
    )
    if len(
        {
            development.snapshot.dataset_version,
            optimization.snapshot.dataset_version,
            benchmark.snapshot.dataset_version,
        }
    ) != 1:
        raise ReplayError(
            "linked runs used different dataset versions"
        )

    program_snapshots: dict[str, ProgramSnapshot] = {}
    for reference in (
        benchmark.programs + development.programs + optimization.programs
    ):
        snapshot = _read_program_snapshot_for_reference(
            reference, programs_root
        )
        existing = program_snapshots.get(snapshot.program_id)
        if existing is not None and (
            existing.manifest_sha256 != snapshot.manifest_sha256
            or existing.state_sha256 != snapshot.state_sha256
        ):
            raise ReplayError(
                f"program {snapshot.program_id} has conflicting hashes"
            )
        if snapshot.manifest.origin != benchmark.origin:
            raise ReplayError(
                f"program {snapshot.program_id} origin does not match "
                "the benchmark"
            )
        program_snapshots[snapshot.program_id] = snapshot

    if len(
        {
            canonical_json_text(
                snapshot.manifest.signature_schema
            )
            for snapshot in program_snapshots.values()
        }
    ) != 1:
        raise ReplayError(
            "linked programs do not share one signature schema"
        )
    if len(
        {
            canonical_json_text(
                snapshot.manifest.signature_fields
            )
            for snapshot in program_snapshots.values()
        }
    ) != 1:
        raise ReplayError(
            "linked programs do not share one signature field contract"
        )

    snapshot = benchmark.snapshot
    cases, rules_text = read_input_snapshot(
        benchmark_dir, snapshot
    )
    case_by_id = {case.case_id: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ReplayError("snapshot contains duplicate case IDs")

    development_cases, development_rules_text = read_input_snapshot(
        development_dir, development.snapshot
    )
    development_case_by_id = {
        case.case_id: case for case in development_cases
    }
    development_outcomes = _read_and_validate_outcomes(
        development_dir,
        development,
        development_case_by_id,
        development_rules_text,
        phases=("development",),
        program_ids={
            "development": development.links.baseline_program_id
        },
        expected_split="validation",
    )
    outcomes = _read_and_validate_outcomes(
        benchmark_dir,
        benchmark,
        case_by_id,
        rules_text,
        phases=("baseline", "optimized"),
        program_ids={
            "baseline": benchmark.links.baseline_program_id,
            "optimized": benchmark.links.optimized_program_id,
        },
        expected_split="test",
    )

    warnings: list[str] = []
    baseline_program_id = benchmark.links.baseline_program_id
    optimized_program_id = benchmark.links.optimized_program_id
    assert baseline_program_id is not None
    assert optimized_program_id is not None
    optimized_snapshot = program_snapshots[optimized_program_id]
    if optimized_program_id != baseline_program_id:
        if optimized_snapshot.manifest.parent_program_id is None:
            raise ReplayError(
                "optimized program has no recorded parent lineage"
            )
        if (
            optimized_snapshot.manifest.parent_program_id
            != baseline_program_id
        ):
            warnings.append(
                "Optimized program parent is not the archived baseline; "
                "full candidate lineage is not archived."
            )
    if benchmark.origin == "mock_test":
        warnings.append(
            "Mock/test archive: this is not a measured experiment."
        )
    if not benchmark.benchmark.released:
        warnings.append("This archive is not released.")
    requests: dict[str, RecordedRequest] = {}
    if RUN_REQUESTS_FILENAME in benchmark.files:
        requests = read_recorded_requests(
            benchmark_dir / RUN_REQUESTS_FILENAME
        )
        referenced = {
            request_id
            for outcome in outcomes
            for request_id in outcome.request_ids
        }
        missing = sorted(referenced - set(requests))
        if missing:
            raise ReplayError(
                f"outcomes reference missing requests: {missing}"
            )
        owners: dict[str, str] = {}
        for outcome in outcomes:
            for request_id in outcome.request_ids:
                previous = owners.get(request_id)
                if previous is not None:
                    raise ReplayError(
                        f"request {request_id!r} is shared by outcomes "
                        f"{previous} and {outcome.case_id}/{outcome.phase}"
                    )
                owners[request_id] = (
                    f"{outcome.case_id}/{outcome.phase}"
                )
                if requests[request_id].phase != "task":
                    raise ReplayError(
                        f"outcome {outcome.case_id}/{outcome.phase} "
                        f"references non-task request {request_id!r}"
                    )
    else:
        referenced = {
            request_id
            for outcome in outcomes
            for request_id in outcome.request_ids
        }
        if referenced:
            raise ReplayError(
                "outcomes reference requests but requests.jsonl is absent"
            )
        warnings.append(
            "No request trace file is archived; sent traces are unavailable."
        )

    candidate_stats: Mapping[str, object] | None = None
    if RUN_GEPA_STATS_FILENAME in optimization.files:
        stats_payload = _read_json_object(
            optimization_dir / RUN_GEPA_STATS_FILENAME
        )
        candidate_stats = stats_payload
    else:
        warnings.append(
            "Optimization candidate statistics are not archived."
        )

    is_measured = is_measured_archive(benchmark)
    return ReplayArchive(
        benchmark_dir=benchmark_dir,
        benchmark=benchmark,
        development=development,
        optimization=optimization,
        snapshot=snapshot,
        cases=tuple(cases),
        rules_text=rules_text,
        programs=program_snapshots,
        development_outcomes=tuple(development_outcomes),
        outcomes=tuple(outcomes),
        requests=requests,
        candidate_stats=candidate_stats,
        is_measured=is_measured,
        warnings=tuple(warnings),
    )


def _read_program_snapshot_for_reference(
    reference: ProgramReference, programs_root: Path
) -> ProgramSnapshot:
    snapshot = read_program_snapshot(
        programs_root / reference.program_id
    )
    if snapshot.manifest_sha256 != reference.manifest_sha256:
        raise ReplayError(
            f"program {reference.program_id} manifest hash mismatch"
        )
    if snapshot.state_sha256 != reference.state_sha256:
        raise ReplayError(
            f"program {reference.program_id} state hash mismatch"
        )
    return snapshot


def _validate_run_links(
    benchmark: RunManifest,
    development: RunManifest,
    optimization: RunManifest,
) -> None:
    links = benchmark.links
    assert links.baseline_development_run_id is not None
    assert links.optimization_run_id is not None
    assert links.baseline_program_id is not None
    assert links.optimized_program_id is not None
    if development.run_id != links.baseline_development_run_id:
        raise ReplayError("benchmark baseline development link mismatch")
    if optimization.run_id != links.optimization_run_id:
        raise ReplayError("benchmark optimization link mismatch")
    if (
        optimization.links.baseline_development_run_id
        != development.run_id
    ):
        raise ReplayError(
            "optimization does not link the benchmark's development run"
        )
    if development.links.baseline_program_id != links.baseline_program_id:
        raise ReplayError(
            "development baseline program does not match benchmark"
        )
    if optimization.links.baseline_program_id != links.baseline_program_id:
        raise ReplayError(
            "optimization baseline program does not match benchmark"
        )
    if optimization.links.optimized_program_id != links.optimized_program_id:
        raise ReplayError(
            "optimization selected program does not match benchmark"
        )
    for run in (development, optimization):
        if run.links.benchmark_run_id is not None and (
            run.links.benchmark_run_id != benchmark.run_id
        ):
            raise ReplayError(
                f"linked run {run.run_id} points at another benchmark"
            )
    _validate_program_reference_set(
        benchmark.programs,
        {links.baseline_program_id, links.optimized_program_id},
        "benchmark",
    )
    _validate_program_reference_set(
        development.programs,
        {links.baseline_program_id},
        "development",
    )
    _validate_program_reference_set(
        optimization.programs,
        {links.baseline_program_id, links.optimized_program_id},
        "optimization",
    )


def _validate_program_reference_set(
    references: Sequence[ProgramReference],
    expected_ids: set[str],
    context: str,
) -> None:
    if {reference.program_id for reference in references} != expected_ids:
        raise ReplayError(
            f"{context} program references do not match its links"
        )


def _validate_shared_rules(
    *snapshots: RunManifest,
) -> None:
    rules = {
        (
            run.snapshot.rules_version,
            run.snapshot.rules_sha256,
        )
        for run in snapshots
    }
    if len(rules) != 1:
        raise ReplayError(
            "linked runs used different editorial rules snapshots"
        )


def _read_and_validate_outcomes(
    run_dir: Path,
    manifest: RunManifest,
    case_by_id: Mapping[str, CaseRecord],
    rules_text: str,
    *,
    phases: Sequence[OutcomePhase],
    program_ids: Mapping[OutcomePhase, str | None],
    expected_split: Literal["train", "validation", "test"],
) -> list[RecordedOutcome]:
    if RUN_OUTCOMES_FILENAME not in manifest.files:
        raise ArtifactIncompleteError(
            f"run {manifest.run_id} does not archive outcomes.jsonl"
        )
    path = run_dir / RUN_OUTCOMES_FILENAME
    outcomes = read_recorded_outcomes(path, run_id=manifest.run_id)
    expected_phases = set(phases)
    seen: set[tuple[str, str]] = set()
    by_phase: dict[OutcomePhase, set[str]] = {
        phase: set() for phase in phases
    }
    for outcome in outcomes:
        if outcome.phase not in expected_phases:
            raise ReplayError(
                f"unexpected outcome phase {outcome.phase!r} in "
                f"{manifest.run_id}"
            )
        if outcome.split != expected_split:
            raise ReplayError(
                f"outcome {outcome.case_id}/{outcome.phase} has split "
                f"{outcome.split!r}, expected {expected_split!r}"
            )
        case = case_by_id.get(outcome.case_id)
        if case is None:
            raise ReplayError(
                f"outcome references case {outcome.case_id!r} that is not "
                "in the recorded snapshot"
            )
        expected_program = program_ids[outcome.phase]
        if expected_program is None:
            raise ReplayError(
                f"outcome phase {outcome.phase!r} has no linked program"
            )
        if outcome.program_id != expected_program:
            raise ReplayError(
                f"outcome {outcome.case_id}/{outcome.phase} used program "
                f"{outcome.program_id}, expected {expected_program}"
            )
        expected_input = case_input_sha256(case, rules_text)
        if outcome.input_sha256 != expected_input:
            raise ReplayError(
                f"outcome {outcome.case_id}/{outcome.phase} input hash "
                "does not match the recorded case snapshot"
            )
        key = (outcome.case_id, outcome.phase)
        if key in seen:
            raise ReplayError(
                f"duplicate outcome for {outcome.case_id}/{outcome.phase}"
            )
        seen.add(key)
        by_phase[outcome.phase].add(outcome.candidate_id)
    for case in case_by_id.values():
        for phase in phases:
            if (case.case_id, phase) not in seen:
                raise ArtifactIncompleteError(
                    f"run {manifest.run_id} has no {phase} outcome for "
                    f"{case.case_id}"
                )
    for phase, candidates in by_phase.items():
        if len(candidates) > 1:
            raise ReplayError(
                f"phase {phase!r} contains multiple candidate IDs: "
                f"{sorted(candidates)}"
            )
    return outcomes


__all__ = [
    "ArtifactCompatibilityError",
    "ArtifactCorruptError",
    "ArtifactError",
    "ArtifactIncompleteError",
    "ArtifactNotFoundError",
    "ArtifactSchemaError",
    "ArtifactSecretError",
    "BenchmarkProtocol",
    "LoadedProgram",
    "OUTCOME_SCHEMA_VERSION",
    "PROGRAMS_ROOT",
    "PROGRAM_INSTRUCTIONS_FILENAME",
    "PROGRAM_MANIFEST_FILENAME",
    "PROGRAM_MANIFEST_SCHEMA_VERSION",
    "PROGRAM_STATE_FILENAME",
    "PreviewTrace",
    "ProgramManifest",
    "ProgramReference",
    "ProgramSnapshot",
    "RUNS_ROOT",
    "RUN_GEPA_STATS_FILENAME",
    "RUN_MANIFEST_FILENAME",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "RUN_OUTCOMES_FILENAME",
    "RUN_REQUESTS_FILENAME",
    "RecordedOutcome",
    "RecordedRequest",
    "ReplayArchive",
    "ReplayError",
    "RunLinks",
    "RunManifest",
    "SNAPSHOT_CASES_FILENAME",
    "SNAPSHOT_RULES_FILENAME",
    "SNAPSHOT_SCHEMA_VERSION",
    "SentTrace",
    "SnapshotManifest",
    "append_jsonl",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_jsonl",
    "atomic_write_text",
    "canonical_json_bytes",
    "canonical_json_text",
    "case_input_sha256",
    "cases_snapshot_sha256",
    "file_index",
    "format_preview_trace",
    "format_sent_trace",
    "is_measured_archive",
    "load_program_artifact",
    "open_replay_archive",
    "program_reference",
    "read_input_snapshot",
    "read_program_manifest",
    "read_program_snapshot",
    "read_recorded_outcomes",
    "read_recorded_requests",
    "read_run_manifest",
    "save_program_artifact",
    "scan_tree_for_secrets",
    "sha256_bytes",
    "sha256_file",
    "sha256_text",
    "verify_file_index",
    "write_input_snapshot",
    "write_run_manifest",
]
