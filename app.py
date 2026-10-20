"""Live/Replay UI for the DSPy prompt optimization lab.

Replay is the default and reads only local recorded artifacts. Live exposes one
server-side confirmation-gated optimization job backed by a single local worker
thread; the worker never calls Streamlit APIs. Overview, Baseline, and Prompt
Inspector remain read-only in both modes.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
import secrets
import statistics
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import streamlit as st

from dspy_lab.artifacts import (
    PROGRAMS_ROOT,
    RUNS_ROOT,
    ArtifactError,
    ProgramManifest,
    ProgramSnapshot,
    RecordedDevelopmentRun,
    RecordedOutcome,
    RecordedRequest,
    ReplayArchive,
    RunDirectoryEntry,
    RunManifest,
    append_jsonl,
    atomic_write_json,
    format_preview_trace,
    format_sent_trace,
    is_measured_archive,
    list_run_directories,
    open_development_run,
    open_optimization_run,
    open_replay_archive,
    read_run_manifest,
)
from dspy_lab.config import (
    DEFAULT_MAX_OPTIMIZATION_METRIC_CALLS,
    DEFAULT_REFLECTION_MODEL,
    DEFAULT_TASK_MODEL,
    PROJECT_ROOT,
    ConfigurationError,
    Settings,
    load_settings,
)
from dspy_lab.data import (
    canonical_document,
    load_optimization_cases,
    load_review_audit,
    load_split_manifest,
)
from dspy_lab.llm import BudgetLedger, BudgetSnapshot
from dspy_lab.metrics import InvalidPrediction, score_case
from dspy_lab.program import (
    case_inputs,
    load_baseline_instructions,
)
from dspy_lab.schemas import (
    ALLOWED_CODES,
    CaseRecord,
    EditorialQAOutput,
    ProductFacts,
    SourceRecord,
)

DATA_DIR = PROJECT_ROOT / "data" / "v1"
MODE_REPLAY = "Replay Experiment"
MODE_LIVE = "Live Experiment"
SCREENS = (
    "Overview",
    "Baseline",
    "Optimization",
    "Comparison",
    "Prompt Inspector",
    "Playground",
)
CODE_DESCRIPTIONS = {
    "UNSUPPORTED_NUMERIC_CLAIM": (
        "A numerical product, market, or performance claim is not supported "
        "by the supplied facts or excerpts."
    ),
    "INVESTMENT_ADVICE_AS_FACT": (
        "An unqualified universal recommendation or suitability claim is "
        "presented as objective fact."
    ),
    "UNWARRANTED_CERTAINTY": (
        "A future price, return, or absence of risk is guaranteed without "
        "qualification."
    ),
    "UNRESOLVED_PLACEHOLDER": (
        "A production placeholder or unfinished editorial marker remains in "
        "publishable content."
    ),
    "INVALID_CITATION": (
        "A cited local source ID is missing or its supplied excerpt does not "
        "support the attributed claim."
    ),
    "PRODUCT_FACT_MISMATCH": (
        "A stated product attribute conflicts with the explicit reference "
        "facts."
    ),
}


@dataclass(frozen=True, slots=True)
class ReplayContext:
    entries: tuple[RunDirectoryEntry, ...]
    selected: RunDirectoryEntry | None
    archive: ReplayArchive | None
    error: str | None


@dataclass(frozen=True, slots=True)
class LiveContext:
    settings: Settings | None
    config_error: str | None
    development_entries: tuple[RunDirectoryEntry, ...]
    selected_development: RunDirectoryEntry | None
    development: RecordedDevelopmentRun | None
    development_error: str | None
    optimization_entries: tuple[RunDirectoryEntry, ...]
    benchmark_entries: tuple[RunDirectoryEntry, ...]


@dataclass(frozen=True, slots=True)
class BaselineView:
    manifest: ProgramManifest | None
    run_manifest: RunManifest
    summary: Mapping[str, object] | None
    cases: tuple[CaseRecord, ...]
    outcomes: tuple[RecordedOutcome, ...]
    warnings: tuple[str, ...]
    historical: bool


@dataclass(frozen=True, slots=True)
class InspectorView:
    baseline: ProgramSnapshot | None
    optimized: ProgramSnapshot | None
    cases: tuple[CaseRecord, ...]
    outcomes: tuple[RecordedOutcome, ...]
    task_requests: Mapping[str, RecordedRequest]
    optimization_requests: Mapping[str, RecordedRequest]
    candidate_stats: Mapping[str, object] | None
    rules_text: str
    split_label: str
    release_label: str
    measured: bool
    historical: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ComparisonView:
    manifest: RunManifest
    cases: tuple[CaseRecord, ...]
    outcomes: tuple[RecordedOutcome, ...]
    summary: Mapping[str, object] | None
    rules_text: str
    candidate_stats: Mapping[str, object] | None
    baseline: ProgramSnapshot | None
    optimized: ProgramSnapshot | None
    historical: bool
    measured: bool
    warnings: tuple[str, ...]


JOB_SCHEMA_VERSION = "ui_job_v1"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_CANCELLING = "cancelling"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_STOPPED = "stopped"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_INTERRUPTED = "interrupted"
JOB_ACTIVE_STATUSES = frozenset(
    {JOB_STATUS_RUNNING, JOB_STATUS_CANCELLING}
)
JOB_KIND_OPTIMIZATION = "optimization"
JOB_KIND_BENCHMARK = "benchmark"
JOB_KIND_PLAYGROUND = "playground"
JOB_FOLDERS = {
    JOB_KIND_OPTIMIZATION: "optimization",
    JOB_KIND_BENCHMARK: "benchmarks",
    JOB_KIND_PLAYGROUND: "playground",
}
JOB_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
JOB_REGISTRY_DIRNAME = "jobs"
ACTIVE_JOB_LOCK_FILENAME = ".active.lock"
PROCESS_INSTANCE_ID = secrets.token_hex(8)


class JobGateError(RuntimeError):
    """Base error for rejected server-side job submissions."""


class JobBusyError(JobGateError):
    """Raised when another Live job is already active."""


class JobStaleError(JobGateError):
    """Raised for a consumed token, changed mode, or existing run ID."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _redact_secret(text: str, secret: str | None) -> str:
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return re.sub(r"sk-or-[A-Za-z0-9._-]+", "[REDACTED]", text)


def _token_sha256(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _new_job_run_id(prefix: str = "opt") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ").lower()
    return f"{prefix}-{stamp}-{secrets.token_hex(3)}"


def _process_start_token(pid: object) -> str | None:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    marker = stat.rfind(")")
    if marker < 0:
        return None
    fields = stat[marker + 2 :].split()
    return fields[19] if len(fields) > 19 else None


def _pid_alive(pid: object) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_json_file(path: Path) -> Mapping[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def read_active_job_lock(
    runs_root: Path = RUNS_ROOT,
) -> Mapping[str, object] | None:
    """Return the live active-job lock without creating or mutating anything."""
    path = Path(runs_root) / JOB_REGISTRY_DIRNAME / ACTIVE_JOB_LOCK_FILENAME
    payload = _read_json_file(path)
    if payload is None or not _pid_alive(payload.get("pid")):
        return None
    if (
        payload.get("pid") == os.getpid()
        and payload.get("process_instance_id")
        not in (None, PROCESS_INSTANCE_ID)
    ):
        return None
    recorded_start = payload.get("process_start_token")
    current_start = _process_start_token(payload.get("pid"))
    if (
        isinstance(recorded_start, str)
        and current_start is not None
        and recorded_start != current_start
    ):
        return None
    return payload


@dataclass(frozen=True, slots=True)
class OptimizationJobSpec:
    mode: str
    run_id: str
    token: str
    mode_epoch: int
    confirmation_generation: int
    settings: Settings
    baseline_development_run_id: str
    train_cases: tuple[CaseRecord, ...]
    validation_cases: tuple[CaseRecord, ...]
    rules_text: str
    runs_root: Path
    programs_root: Path


@dataclass(frozen=True, slots=True)
class BenchmarkJobSpec:
    mode: str
    run_id: str
    token: str
    mode_epoch: int
    confirmation_generation: int
    settings: Settings
    baseline_development_run_id: str
    optimization_run_id: str
    release_results: bool
    runs_root: Path
    programs_root: Path
    test_cases: tuple[CaseRecord, ...] | None = None


@dataclass(frozen=True, slots=True)
class PlaygroundJobSpec:
    mode: str
    run_id: str
    token: str
    mode_epoch: int
    confirmation_generation: int
    settings: Settings
    optimization_run_id: str
    title: str
    article_text: str
    product_facts: ProductFacts
    sources: tuple[SourceRecord, ...]
    retain_content: bool
    runs_root: Path
    programs_root: Path


@dataclass
class JobHandle:
    """One in-process paid job; the worker never renders Streamlit."""

    run_id: str
    token_sha256: str
    mode_epoch: int
    runs_root: Path
    programs_root: Path
    settings: Settings
    ledger: BudgetLedger
    events_path: Path
    started_at: str
    baseline_development_run_id: str = ""
    train_cases: tuple[CaseRecord, ...] = ()
    validation_cases: tuple[CaseRecord, ...] = ()
    rules_text: str = ""
    job_kind: str = JOB_KIND_OPTIMIZATION
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False
    )
    _status: str = JOB_STATUS_RUNNING
    _error: str | None = None
    _finished_at: str | None = None
    _result: object | None = None
    _thread: threading.Thread | None = field(default=None, repr=False)

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    @property
    def finished_at(self) -> str | None:
        with self._lock:
            return self._finished_at

    @property
    def result(self) -> object | None:
        with self._lock:
            return self._result

    @property
    def active(self) -> bool:
        return self.status in JOB_ACTIVE_STATUSES

    def request_cancel(self) -> bool:
        with self._lock:
            if self._status != JOB_STATUS_RUNNING:
                return False
            self._status = JOB_STATUS_CANCELLING
        self.ledger.cancel("cancelled_by_operator")
        return True

    def finish(
        self,
        status: str,
        *,
        error: str | None = None,
        result: object | None = None,
    ) -> None:
        with self._lock:
            self._status = status
            self._error = error
            self._result = result
            self._finished_at = _utc_now()

    def read_events(
        self, limit: int = 40
    ) -> list[Mapping[str, object]]:
        """Read persisted progress events only; never dispatch a request."""
        try:
            lines = self.events_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        events: list[Mapping[str, object]] = []
        for line in lines[-max(1, limit) :]:
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, Mapping):
                events.append(payload)
        return events

    def budget_snapshot(self) -> BudgetSnapshot:
        return self.ledger.snapshot()


class JobManager:
    """Single-worker coordinator, active-job lock, and consumed-token registry."""

    def __init__(
        self,
        runs_root: Path,
        programs_root: Path,
        *,
        runner: Callable[[object, JobHandle], object] | None = None,
    ) -> None:
        self._runs_root = Path(runs_root)
        self._programs_root = Path(programs_root)
        self._runner = runner or _run_paid_job
        self._lock = threading.RLock()
        self._handle: JobHandle | None = None
        self._consumed_tokens: set[str] = set()
        self._mode_epoch = 0
        self._confirmation_generation = 0
        self._last_mode: str | None = None
        self._records: dict[str, dict[str, object]] = {}
        self._registry_dir = self._runs_root / JOB_REGISTRY_DIRNAME
        self._active_lock_path = (
            self._registry_dir / ACTIVE_JOB_LOCK_FILENAME
        )
        self._registry_dir.mkdir(parents=True, exist_ok=True)
        self._recover()

    @property
    def active_handle(self) -> JobHandle | None:
        with self._lock:
            handle = self._handle
        return handle if handle is not None and handle.active else None

    @property
    def latest_handle(self) -> JobHandle | None:
        with self._lock:
            return self._handle

    def latest_handle_for(self, kind: str) -> JobHandle | None:
        with self._lock:
            handle = self._handle
        if handle is None or handle.job_kind != kind:
            return None
        return handle

    @property
    def mode_epoch(self) -> int:
        with self._lock:
            return self._mode_epoch

    @property
    def confirmation_generation(self) -> int:
        with self._lock:
            return self._confirmation_generation

    def note_mode(self, mode: str) -> int:
        with self._lock:
            if self._last_mode is None:
                self._last_mode = mode
            elif mode != self._last_mode:
                self._last_mode = mode
                self._mode_epoch += 1
            return self._mode_epoch

    def start_job(self, spec: OptimizationJobSpec) -> JobHandle:
        return self._start(
            spec,
            kind=JOB_KIND_OPTIMIZATION,
            limits=spec.settings.optimization_budget(),
            optimization_fields=spec,
        )

    def start_benchmark(self, spec: BenchmarkJobSpec) -> JobHandle:
        return self._start(
            spec,
            kind=JOB_KIND_BENCHMARK,
            limits=spec.settings.benchmark_budget(),
        )

    def start_playground(self, spec: PlaygroundJobSpec) -> JobHandle:
        return self._start(
            spec,
            kind=JOB_KIND_PLAYGROUND,
            limits=spec.settings.playground_budget(),
        )

    def _start(
        self,
        spec: object,
        *,
        kind: str,
        limits: object,
        optimization_fields: OptimizationJobSpec | None = None,
    ) -> JobHandle:
        mode = getattr(spec, "mode", None)
        if mode != MODE_LIVE:
            raise JobGateError(
                f"paid {kind} can start only in Live Experiment"
            )
        settings = getattr(spec, "settings", None)
        if not isinstance(settings, Settings):
            raise JobGateError("a validated Live settings object is required")
        token = getattr(spec, "token", None)
        run_id = getattr(spec, "run_id", None)
        if not isinstance(token, str) or not token.strip():
            raise JobStaleError("a fresh confirmation token is required")
        if (
            not isinstance(run_id, str)
            or JOB_RUN_ID_PATTERN.fullmatch(run_id) is None
        ):
            raise JobStaleError(f"invalid {kind} run ID")
        token_hash = _token_sha256(token)
        with self._lock:
            if self._handle is not None and self._handle.active:
                raise JobBusyError(
                    "another Live job is already active"
                )
            if token_hash in self._consumed_tokens:
                raise JobStaleError(
                    "this confirmation was already consumed; confirm a new "
                    "budget"
                )
            if getattr(spec, "mode_epoch", None) != self._mode_epoch:
                raise JobStaleError(
                    "the experiment mode changed after this form was "
                    "prepared; review and confirm a fresh budget"
                )
            if (
                getattr(spec, "confirmation_generation", None)
                != self._confirmation_generation
            ):
                raise JobStaleError(
                    "another confirmed job consumed this form generation; "
                    "review and confirm a fresh budget"
                )
            if self._run_exists(run_id, kind):
                raise JobStaleError(
                    "this run ID already exists; a stale form cannot start "
                    "another paid job"
                )
            if read_active_job_lock(self._runs_root) is not None:
                raise JobBusyError(
                    "another process holds the single active-job lock"
                )
            ledger = BudgetLedger.from_limits(limits)  # type: ignore[arg-type]
            started_at = _utc_now()
            handle = JobHandle(
                run_id=run_id,
                token_sha256=token_hash,
                mode_epoch=int(getattr(spec, "mode_epoch", 0)),
                runs_root=Path(getattr(spec, "runs_root")),
                programs_root=Path(getattr(spec, "programs_root")),
                settings=settings,
                ledger=ledger,
                events_path=(
                    self._runs_root
                    / JOB_FOLDERS[kind]
                    / run_id
                    / "events.jsonl"
                ),
                started_at=started_at,
                job_kind=kind,
                baseline_development_run_id=(
                    optimization_fields.baseline_development_run_id
                    if optimization_fields is not None
                    else str(
                        getattr(spec, "baseline_development_run_id", "") or ""
                    )
                ),
                train_cases=(
                    optimization_fields.train_cases
                    if optimization_fields is not None
                    else ()
                ),
                validation_cases=(
                    optimization_fields.validation_cases
                    if optimization_fields is not None
                    else ()
                ),
                rules_text=(
                    optimization_fields.rules_text
                    if optimization_fields is not None
                    else ""
                ),
            )
            record = self._record_payload(spec, handle)
            self._consumed_tokens.add(token_hash)
            self._confirmation_generation += 1
            self._handle = handle
            self._records[run_id] = record
            try:
                self._write_record(record)
                self._write_active_lock(run_id)
                thread = threading.Thread(
                    target=self._run_job,
                    args=(spec, handle),
                    name=f"dspy-{kind}-{run_id}",
                    daemon=True,
                )
                handle._thread = thread
                thread.start()
            except JobBusyError:
                handle.finish(
                    JOB_STATUS_FAILED,
                    error="another process holds the active-job lock",
                )
                self._complete_record(handle)
                raise
            except Exception as exc:  # noqa: BLE001 - worker start boundary
                handle.finish(
                    JOB_STATUS_FAILED,
                    error=_redact_secret(
                        f"worker start failed: {exc}",
                        settings.openrouter_api_key,
                    ),
                )
                self._complete_record(handle)
                self._release_active_lock(run_id)
                raise JobGateError(
                    f"the {kind} worker could not start; no provider "
                    "request was dispatched"
                ) from exc
            return handle

    def request_cancel(self, run_id: str) -> bool:
        with self._lock:
            handle = self._handle
        if handle is None or handle.run_id != run_id:
            return False
        return handle.request_cancel()

    def recent_records(self, limit: int = 5) -> list[dict[str, object]]:
        with self._lock:
            records = [dict(record) for record in self._records.values()]
        records.sort(
            key=lambda record: str(
                record.get("started_at") or record.get("created_at") or ""
            ),
            reverse=True,
        )
        return records[: max(1, limit)]

    def _run_job(self, spec: object, handle: JobHandle) -> None:
        try:
            result = self._runner(spec, handle)
        except Exception as exc:  # noqa: BLE001 - worker boundary
            recovered = self._status_from_run(handle.job_kind, handle.run_id)
            if recovered is None and handle.ledger.cancelled:
                recovered = JOB_STATUS_STOPPED
            handle.finish(
                recovered or JOB_STATUS_FAILED,
                error=_redact_secret(
                    f"{type(exc).__name__}: {exc}",
                    handle.settings.openrouter_api_key,
                ),
            )
        else:
            manifest = getattr(result, "manifest", None)
            status = getattr(result, "status", None)
            if status is None:
                status = getattr(manifest, "status", JOB_STATUS_COMPLETED)
            handle.finish(str(status), result=result)
        finally:
            try:
                self._complete_record(handle)
            finally:
                self._release_active_lock(handle.run_id)

    def _run_dir_for(self, kind: str, run_id: str) -> Path:
        return self._runs_root / JOB_FOLDERS.get(kind, kind) / run_id

    def _status_from_run(self, kind: str, run_id: str) -> str | None:
        run_dir = self._run_dir_for(kind, run_id)
        if kind == JOB_KIND_PLAYGROUND:
            payload = _read_json_file(run_dir / "summary.json")
            status = None if payload is None else payload.get("status")
            return status if isinstance(status, str) else None
        try:
            manifest = read_run_manifest(run_dir)
        except Exception:  # noqa: BLE001 - missing manifest is not fatal here
            return None
        return manifest.status

    def _record_payload(
        self, spec: object, handle: JobHandle
    ) -> dict[str, object]:
        settings = getattr(spec, "settings")
        if handle.job_kind == JOB_KIND_OPTIMIZATION:
            limits = settings.optimization_budget()
        elif handle.job_kind == JOB_KIND_BENCHMARK:
            limits = settings.benchmark_budget()
        else:
            limits = settings.playground_budget()
        return {
            "schema_version": JOB_SCHEMA_VERSION,
            "run_id": handle.run_id,
            "job_kind": handle.job_kind,
            "token_sha256": handle.token_sha256,
            "mode": MODE_LIVE,
            "status": handle.status,
            "pid": os.getpid(),
            "process_instance_id": PROCESS_INSTANCE_ID,
            "process_start_token": _process_start_token(os.getpid()),
            "created_at": handle.started_at,
            "started_at": handle.started_at,
            "finished_at": None,
            "error": None,
            "stop_reason": None,
            "baseline_development_run_id": getattr(
                spec, "baseline_development_run_id", None
            ),
            "optimization_run_id": getattr(
                spec, "optimization_run_id", None
            ),
            "max_calls": limits.max_calls,
            "max_cost_usd": str(limits.max_cost_usd),
        }

    def _complete_record(self, handle: JobHandle) -> None:
        with self._lock:
            record = dict(self._records.get(handle.run_id, {}))
            record.update(
                {
                    "status": handle.status,
                    "finished_at": handle.finished_at,
                    "error": handle.error,
                    "stop_reason": handle.ledger.snapshot().stop_reason,
                }
            )
            self._records[handle.run_id] = record
            self._consumed_tokens.add(handle.token_sha256)
            self._write_record(record)

    def _run_exists(self, run_id: str, kind: str) -> bool:
        if run_id in self._records:
            return True
        return self._run_dir_for(kind, run_id).exists()

    def _write_record(self, record: Mapping[str, object]) -> None:
        atomic_write_json(
            self._registry_dir / f"{record['run_id']}.json", record
        )

    def _write_active_lock(self, run_id: str) -> None:
        payload = {
            "schema_version": JOB_SCHEMA_VERSION,
            "run_id": run_id,
            "pid": os.getpid(),
            "process_instance_id": PROCESS_INSTANCE_ID,
            "process_start_token": _process_start_token(os.getpid()),
            "created_at": _utc_now(),
        }
        data = (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        # The app is a single local process; this re-read narrows the
        # stale-cleanup race. Cross-process exclusion is advisory
        # defense-in-depth, backed by the job registry and run-ID checks.
        for _attempt in (1, 2):
            try:
                descriptor = os.open(
                    self._active_lock_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError as exc:
                existing = _read_json_file(self._active_lock_path)
                if existing is None:
                    raise JobBusyError(
                        "the active-job lock is unreadable; inspect it "
                        "before starting another paid job"
                    ) from exc
                if _pid_alive(existing.get("pid")):
                    raise JobBusyError(
                        "another process holds the single active-job lock"
                    ) from exc
                current = _read_json_file(self._active_lock_path)
                if current != existing:
                    continue
                try:
                    self._active_lock_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as unlink_error:
                    raise JobBusyError(
                        "a stale active-job lock could not be cleared"
                    ) from unlink_error
                continue
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            return
        raise JobBusyError(
            "another process holds the single active-job lock"
        )

    def _release_active_lock(self, run_id: str) -> None:
        payload = _read_json_file(self._active_lock_path)
        if payload is not None and payload.get("run_id") == run_id:
            try:
                self._active_lock_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _recover(self) -> None:
        records: dict[str, dict[str, object]] = {}
        for path in sorted(self._registry_dir.glob("*.json")):
            payload = _read_json_file(path)
            if payload is None:
                continue
            run_id = payload.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                continue
            record = dict(payload)
            kind = record.get("job_kind")
            if kind not in JOB_FOLDERS:
                kind = JOB_KIND_OPTIMIZATION
                record["job_kind"] = kind
            records[run_id] = record
            token_hash = record.get("token_sha256")
            if isinstance(token_hash, str):
                self._consumed_tokens.add(token_hash)
            if record.get("status") != JOB_STATUS_RUNNING:
                continue
            pid = record.get("pid")
            same_process_instance = (
                pid == os.getpid()
                and record.get("process_instance_id")
                == PROCESS_INSTANCE_ID
            )
            if same_process_instance:
                continue
            if pid != os.getpid() and _pid_alive(pid):
                recorded_start = record.get("process_start_token")
                current_start = _process_start_token(pid)
                if not (
                    isinstance(recorded_start, str)
                    and current_start is not None
                    and recorded_start != current_start
                ):
                    continue
            # Same PID with a different instance, or a reused PID with a
            # different start token, is a restarted job with no live worker.
            recovered_status = (
                self._status_from_run(str(kind), run_id)
                or JOB_STATUS_INTERRUPTED
            )
            record["status"] = recovered_status
            record["finished_at"] = _utc_now()
            if recovered_status == JOB_STATUS_INTERRUPTED:
                record["error"] = (
                    f"process restarted before the {kind} job finished"
                )
                record["stop_reason"] = "interrupted"
                self._append_interrupted_event(run_id, str(kind))
            self._records[run_id] = record
            self._write_record(record)
        self._records = records
        lock_payload = _read_json_file(self._active_lock_path)
        lock_stale_same_process = (
            lock_payload is not None
            and lock_payload.get("pid") == os.getpid()
            and lock_payload.get("process_instance_id")
            not in (None, PROCESS_INSTANCE_ID)
        )
        lock_process_reused = False
        if lock_payload is not None:
            recorded_start = lock_payload.get("process_start_token")
            current_start = _process_start_token(lock_payload.get("pid"))
            lock_process_reused = bool(
                isinstance(recorded_start, str)
                and current_start is not None
                and recorded_start != current_start
            )
        if lock_payload is not None and (
            not _pid_alive(lock_payload.get("pid"))
            or lock_stale_same_process
            or lock_process_reused
        ):
            try:
                self._active_lock_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _append_interrupted_event(self, run_id: str, kind: str) -> None:
        run_dir = self._run_dir_for(kind, run_id)
        if not run_dir.is_dir():
            return
        try:
            append_jsonl(
                run_dir / "events.jsonl",
                {
                    "at": _utc_now(),
                    "event": "job_interrupted",
                    "reason": "process_restart",
                    "run_id": run_id,
                },
            )
        except OSError:
            pass


def _run_paid_job(spec: object, handle: JobHandle) -> object:
    """Dispatch one confirmed Live worker. Imports stay off the Replay path."""
    if isinstance(spec, BenchmarkJobSpec):
        return _run_benchmark_job(spec, handle)
    if isinstance(spec, PlaygroundJobSpec):
        return _run_playground_job(spec, handle)
    return _run_optimization_job(spec, handle)  # type: ignore[arg-type]


def _run_optimization_job(
    spec: OptimizationJobSpec, handle: JobHandle
) -> object:
    """Default optimization worker body."""
    from dspy_lab.experiments import run_optimization

    return run_optimization(
        spec.settings,
        spec.train_cases,
        spec.validation_cases,
        baseline_development_run_id=spec.baseline_development_run_id,
        confirm_budget=True,
        run_id=spec.run_id,
        origin="live",
        runs_root=spec.runs_root,
        programs_root=spec.programs_root,
        rules_text=spec.rules_text,
        ledger=handle.ledger,
    )


def _run_benchmark_job(
    spec: BenchmarkJobSpec, handle: JobHandle
) -> object:
    """Load the sealed split only inside the authorized benchmark worker."""
    from dspy_lab.experiments import (
        load_benchmark_cases,
        run_benchmark,
    )
    from dspy_lab.program import load_editorial_rules

    test_cases = (
        spec.test_cases
        if spec.test_cases is not None
        else tuple(load_benchmark_cases(DATA_DIR))
    )
    return run_benchmark(
        spec.settings,
        test_cases,
        baseline_development_run_id=spec.baseline_development_run_id,
        optimization_run_id=spec.optimization_run_id,
        confirm_budget=True,
        release_results=spec.release_results,
        run_id=spec.run_id,
        origin="live",
        runs_root=spec.runs_root,
        programs_root=spec.programs_root,
        rules_text=load_editorial_rules(),
        ledger=handle.ledger,
    )


def _run_playground_job(
    spec: PlaygroundJobSpec, handle: JobHandle
) -> object:
    """Run both frozen artifacts without any gold-based scoring."""
    from dspy_lab.experiments import run_playground

    return run_playground(
        spec.settings,
        optimization_run_id=spec.optimization_run_id,
        title=spec.title,
        article_text=spec.article_text,
        product_facts=spec.product_facts,
        sources=spec.sources,
        retain_content=spec.retain_content,
        confirm_budget=True,
        run_id=spec.run_id,
        runs_root=spec.runs_root,
        programs_root=spec.programs_root,
        ledger=handle.ledger,
    )


_JOB_MANAGER_OVERRIDE: JobManager | None = None


@st.cache_resource(show_spinner=False)
def _cached_job_manager() -> JobManager:
    return JobManager(RUNS_ROOT, PROGRAMS_ROOT)


def _get_job_manager() -> JobManager:
    if _JOB_MANAGER_OVERRIDE is not None:
        return _JOB_MANAGER_OVERRIDE
    return _cached_job_manager()


def main() -> None:
    st.set_page_config(
        page_title="DSPy Prompt Optimization Lab",
        layout="wide",
    )
    st.title("DSPy Prompt Optimization Lab")
    st.caption(
        "Replay-default presentation plus separately confirmed Live "
        "optimization, paired benchmark, and unscored Playground jobs. "
        "Replay and archived inspection remain read-only."
    )

    active_lock = read_active_job_lock(RUNS_ROOT)
    if (
        active_lock is not None
        and st.session_state.get("experiment_mode", MODE_REPLAY)
        != MODE_LIVE
    ):
        st.session_state["experiment_mode"] = MODE_LIVE
        st.session_state["_mode_switch_blocked"] = True
        st.rerun()
    mode = _render_mode_selector(disabled=active_lock is not None)
    if st.session_state.pop("_mode_switch_blocked", False):
        st.warning(
            "A Live job is active. Switching to Replay is blocked until "
            "the job finishes or is cancelled."
        )
    job_manager: JobManager | None = None
    previous_mode = st.session_state.get("_ui_previous_mode")
    if mode == MODE_LIVE:
        job_manager = _get_job_manager()
        st.session_state["_job_manager_created"] = True
    if previous_mode is not None and previous_mode != mode:
        if job_manager is None and st.session_state.get(
            "_job_manager_created"
        ):
            job_manager = _get_job_manager()
        if job_manager is not None:
            job_manager.note_mode(mode)
        _clear_confirmation_state()
    elif job_manager is not None:
        job_manager.note_mode(mode)
    st.session_state["_ui_previous_mode"] = mode

    replay: ReplayContext | None = None
    live: LiveContext | None = None
    if mode == MODE_REPLAY:
        replay = _resolve_replay_context()
        _render_replay_mode_status()
    else:
        live = _resolve_live_context()
        _render_live_mode_status(live)

    screen = st.radio(
        "Screen",
        SCREENS,
        index=0,
        horizontal=True,
        key="screen",
    )

    if replay is not None:
        if replay.archive is not None:
            _render_archive_banner(replay.archive)
            _render_replay_screen(screen, replay.archive)
        else:
            _render_replay_empty_state(replay)
            if screen == "Overview":
                _render_static_overview(MODE_REPLAY)
            elif screen == "Playground":
                _render_replay_playground()
            else:
                st.info(
                    "No recorded experiment is available for this screen."
                )
        return

    assert live is not None
    _render_live_screen(screen, live, job_manager)


def _clear_confirmation_state() -> None:
    for key in (
        "opt_confirm_token",
        "opt_confirm_run_id",
        "opt_confirm_key",
        "opt_confirm_epoch",
        "opt_confirm_generation",
        "bench_confirm_token",
        "bench_confirm_run_id",
        "bench_confirm_key",
        "bench_confirm_epoch",
        "bench_confirm_generation",
        "play_confirm_token",
        "play_confirm_run_id",
        "play_confirm_key",
        "play_confirm_epoch",
        "play_confirm_generation",
    ):
        st.session_state.pop(key, None)


def _render_mode_selector(*, disabled: bool = False) -> str:
    return st.radio(
        "Experiment mode",
        (MODE_REPLAY, MODE_LIVE),
        index=0,
        horizontal=True,
        key="experiment_mode",
        disabled=disabled,
        help=(
            "Replay presents a completed recorded benchmark and never calls "
            "a provider. Live adds one explicitly confirmed optimization job; "
            "switching is blocked while that job is active."
        ),
    )


def _resolve_replay_context() -> ReplayContext:
    entries = list_run_directories(RUNS_ROOT, kind="benchmark")
    if not entries:
        return ReplayContext(
            entries=(),
            selected=None,
            archive=None,
            error=None,
        )
    labels = [_run_entry_label(entry) for entry in entries]
    selected_label = st.selectbox(
        "Recorded benchmark archive",
        labels,
        index=_default_entry_index(entries),
        key="replay_archive",
    )
    try:
        selected = entries[labels.index(selected_label)]
    except ValueError:
        selected = entries[0]
    if selected.manifest is None:
        return ReplayContext(
            entries=entries,
            selected=selected,
            archive=None,
            error=selected.error or "archive manifest is unavailable",
        )
    try:
        archive = open_replay_archive(
            selected.run_dir,
            runs_root=RUNS_ROOT,
            programs_root=PROGRAMS_ROOT,
        )
    except ArtifactError as exc:
        return ReplayContext(
            entries=entries,
            selected=selected,
            archive=None,
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - read-only UI must fail visibly
        return ReplayContext(
            entries=entries,
            selected=selected,
            archive=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    return ReplayContext(
        entries=entries,
        selected=selected,
        archive=archive,
        error=None,
    )


def _resolve_live_context() -> LiveContext:
    settings: Settings | None = None
    config_error: str | None = None
    try:
        settings = load_settings("live")
    except ConfigurationError as exc:
        config_error = str(exc)
    development_entries = list_run_directories(
        RUNS_ROOT, kind="development"
    )
    selected_development: RunDirectoryEntry | None = None
    development: RecordedDevelopmentRun | None = None
    development_error: str | None = None
    if development_entries:
        labels = [
            _run_entry_label(entry) for entry in development_entries
        ]
        selected_label = st.selectbox(
            "Recorded baseline development run",
            labels,
            index=_default_entry_index(development_entries),
            key="live_development_run",
        )
        try:
            selected_development = development_entries[
                labels.index(selected_label)
            ]
        except ValueError:
            selected_development = development_entries[0]
        if selected_development.manifest is None:
            development_error = (
                selected_development.error or "run manifest is unavailable"
            )
        else:
            try:
                development = open_development_run(
                    selected_development.run_dir,
                    runs_root=RUNS_ROOT,
                    programs_root=PROGRAMS_ROOT,
                )
            except ArtifactError as exc:
                development_error = str(exc)
            except Exception as exc:  # noqa: BLE001
                development_error = f"{type(exc).__name__}: {exc}"
    optimization_entries = list_run_directories(
        RUNS_ROOT, kind="optimization"
    )
    benchmark_entries = list_run_directories(
        RUNS_ROOT, kind="benchmark"
    )
    return LiveContext(
        settings=settings,
        config_error=config_error,
        development_entries=development_entries,
        selected_development=selected_development,
        development=development,
        development_error=development_error,
        optimization_entries=optimization_entries,
        benchmark_entries=benchmark_entries,
    )


def _run_entry_label(entry: RunDirectoryEntry) -> str:
    if entry.manifest is None:
        return f"{entry.run_id} — unavailable: {entry.error or 'unknown error'}"
    manifest = entry.manifest
    date = manifest.created_at[:10]
    if manifest.kind == "benchmark":
        measured = (
            "measured"
            if is_measured_archive(manifest)
            else "not measured"
        )
        return (
            f"{manifest.run_id} — {date} — {manifest.status} — {measured}"
        )
    return (
        f"{manifest.run_id} — {date} — {manifest.status} — "
        f"{manifest.origin}"
    )


def _default_entry_index(entries: Sequence[RunDirectoryEntry]) -> int:
    for index, entry in enumerate(entries):
        if entry.manifest is not None and entry.manifest.status == "completed":
            return index
    return 0


def _latest_run_label(entries: Sequence[RunDirectoryEntry]) -> str:
    if not entries:
        return "not run"
    manifest = entries[0].manifest
    if manifest is None:
        return "unavailable"
    return f"{manifest.status} ({manifest.origin})"


def _render_replay_mode_status() -> None:
    st.info(
        "Replay Experiment — local recorded artifacts only. No API key is "
        "required; no provider, catalog, or billing request is made. "
        "Additional API calls: 0; additional API cost: 0.00 USD."
    )


def _render_live_mode_status(context: LiveContext) -> None:
    st.warning(
        "Live Experiment — optimization, paired benchmark, and unscored "
        "Playground actions are available only after their own explicit "
        "budget confirmation. No LM or provider request is constructed "
        "until a confirmed job is started."
    )
    if context.settings is not None:
        st.caption(
            "Server-side OPENROUTER_API_KEY: configured. The value is never "
            "displayed, logged, or sent to the browser."
        )
    else:
        st.warning(
            "Live configuration is incomplete: "
            f"{context.config_error or 'OPENROUTER_API_KEY is required'}. "
            "Reading local development artifacts still works."
        )


def _render_replay_empty_state(context: ReplayContext) -> None:
    st.warning("No recorded experiment available")
    if context.error:
        st.error(
            "Recorded archive unavailable. The selected archive cannot be "
            f"presented: {context.error}. No data will be regenerated and "
            "no provider request will be made."
        )
    st.markdown(
        "Switch to **Live Experiment** to prepare an authorized run. "
        "Switching modes does not itself start work, and Replay exposes no "
        "paid action. A full Replay requires a completed, linked "
        "development + optimization + benchmark archive."
    )


def _render_archive_banner(archive: ReplayArchive) -> None:
    benchmark = archive.benchmark
    st.markdown("### Replay Experiment")
    st.caption(
        "Historical presentation of one recorded run. This view does not "
        "re-evaluate, re-optimize, reprice, or contact a provider."
    )
    st.markdown(
        f"- **Benchmark run:** `{benchmark.run_id}`\n"
        f"- **Created / finished (UTC):** {benchmark.created_at} / "
        f"{benchmark.finished_at or 'not finished'}\n"
        f"- **Status / origin:** {benchmark.status} / {benchmark.origin}\n"
        f"- **Baseline development run:** "
        f"`{archive.development.run_id}`\n"
        f"- **Optimization run:** `{archive.optimization.run_id}`\n"
        f"- **Dataset version:** {archive.snapshot.dataset_version}\n"
        f"- **Test snapshot hash:** `{archive.snapshot.cases_sha256}`"
    )
    st.info(
        "Additional API calls: 0; additional API cost: 0.00 USD. "
        "Original historical cost is shown separately and is not spend "
        "caused by this view."
    )
    if archive.is_measured:
        st.success(
            "Measured archive: completed, released Live benchmark. This is "
            "the only kind of archive presented as a measured experiment."
        )
    else:
        st.warning(
            "Not a measured experiment. This archive is mock/test or not "
            "released; its numbers must not be read as a real benchmark "
            "result."
        )
    for warning in archive.warnings:
        st.warning(warning)


def _render_replay_screen(screen: str, archive: ReplayArchive) -> None:
    if screen == "Overview":
        _render_replay_overview(archive)
    elif screen == "Baseline":
        _render_baseline_screen(
            _baseline_view_from_replay(archive)
        )
    elif screen == "Prompt Inspector":
        _render_inspector_screen(
            _inspector_view_from_replay(archive)
        )
    elif screen == "Optimization":
        _render_replay_optimization(archive)
    elif screen == "Comparison":
        _render_comparison_screen(
            _comparison_view_from_archive(archive, historical=True)
        )
    else:
        _render_replay_playground()


def _render_replay_optimization(archive: ReplayArchive) -> None:
    st.subheader("Optimization")
    st.caption(
        "Replay Experiment: recorded optimization history only. This screen "
        "cannot compile, resume, cancel, or start GEPA, and candidate "
        "stepping makes no provider request."
    )
    optimization = archive.optimization
    candidates, candidate_error = _candidate_records(
        archive.candidate_stats
    )
    st.markdown(
        f"- **Optimization run:** `{optimization.run_id}`\n"
        f"- **Status / origin:** {optimization.status} / "
        f"{optimization.origin}\n"
        f"- **Created / finished (UTC):** {optimization.created_at} / "
        f"{optimization.finished_at or 'not finished'}\n"
        f"- **Baseline development run:** "
        f"`{optimization.links.baseline_development_run_id}`\n"
        f"- **Selected program:** "
        f"`{optimization.links.optimized_program_id}`\n"
        f"- **Recorded candidates:** {len(candidates)}"
    )
    if candidate_error:
        st.warning(candidate_error)
    for warning in optimization.warnings:
        st.warning(warning)
    if optimization.stop_reason is not None:
        st.markdown(
            f"- **Stop reason:** {optimization.stop_reason}"
        )
    st.info(
        "Replay exposes no paid execution path. Candidate and reflection "
        "inspection is read-only; start a fresh confirmed optimization in "
        "Live to create a new job."
    )


def _comparison_view_from_archive(
    archive: ReplayArchive, *, historical: bool
) -> ComparisonView:
    baseline_id = archive.benchmark.links.baseline_program_id
    optimized_id = archive.benchmark.links.optimized_program_id
    warnings = list(archive.warnings)
    warnings.extend(archive.benchmark.warnings)
    warnings.extend(archive.optimization.warnings)
    summary = archive.benchmark_summary
    if isinstance(summary, Mapping):
        summary_warnings = summary.get("warnings")
        if isinstance(summary_warnings, list):
            warnings.extend(str(item) for item in summary_warnings)
    return ComparisonView(
        manifest=archive.benchmark,
        cases=archive.cases,
        outcomes=archive.outcomes,
        summary=summary,
        rules_text=archive.rules_text,
        candidate_stats=archive.candidate_stats,
        baseline=(
            archive.programs.get(baseline_id)
            if baseline_id is not None
            else None
        ),
        optimized=(
            archive.programs.get(optimized_id)
            if optimized_id is not None
            else None
        ),
        historical=historical,
        measured=archive.is_measured,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _metric_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _metric_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _rescore_outcome(
    case: CaseRecord, outcome: RecordedOutcome
):
    """Re-score one archived raw prediction with the frozen metric.

    This is deterministic local arithmetic over recorded findings and the
    archived gold case. It constructs no LM and makes no provider call.
    """
    if outcome.status == "ok" and outcome.findings is not None:
        return score_case(
            case, EditorialQAOutput(findings=list(outcome.findings))
        )
    metric = outcome.metric if isinstance(outcome.metric, Mapping) else {}
    kind = metric.get("error_kind")
    if kind not in ("parse_error", "request_error", "missing_output"):
        kind = "parse_error"
    reason = (
        outcome.error
        or metric.get("error")
        or "recorded invalid output"
    )
    return score_case(case, InvalidPrediction(str(reason), kind=kind))


def _comparison_recompute(
    cases: Sequence[CaseRecord],
    outcomes: Sequence[RecordedOutcome],
) -> dict[str, dict[str, object]]:
    case_by_id = {case.case_id: case for case in cases}
    result: dict[str, dict[str, object]] = {}
    for arm in ("baseline", "optimized"):
        arm_outcomes = sorted(
            (item for item in outcomes if item.phase == arm),
            key=lambda item: item.case_id,
        )
        per_code = {
            code: {"tp": 0, "fp": 0, "fn": 0} for code in ALLOWED_CODES
        }
        attempted = len(arm_outcomes)
        valid = 0
        missing_metric = 0
        rescoring_failures = 0
        tp = fp = fn = exact = 0
        evidence_failures = duplicates = 0
        parse_errors = request_errors = missing_outputs = 0
        rewards: list[float] = []
        costs: list[Decimal] = []
        cost_statuses: set[str] = set()
        latencies: list[float] = []
        unknown_cost = False
        for outcome in arm_outcomes:
            case = case_by_id.get(outcome.case_id)
            if case is None:
                missing_metric += 1
                continue
            try:
                score = _rescore_outcome(case, outcome)
            except Exception:  # noqa: BLE001 - fail closed on raw records
                rescoring_failures += 1
                continue
            if score.valid:
                valid += 1
            elif score.error_kind == "parse_error":
                parse_errors += 1
            elif score.error_kind == "request_error":
                request_errors += 1
            elif score.error_kind == "missing_output":
                missing_outputs += 1
            if score.exact_match:
                exact += 1
            tp += score.tp
            fp += score.fp
            fn += score.fn
            evidence_failures += score.evidence_failure_count
            duplicates += score.duplicate_count
            rewards.append(score.reward)
            for pair in score.matched_pairs:
                if pair.code in per_code:
                    per_code[pair.code]["tp"] += 1
            for missed in score.missed_gold:
                if missed.code in per_code:
                    per_code[missed.code]["fn"] += 1
            for false_positive in score.false_positives:
                if false_positive.code in per_code:
                    per_code[false_positive.code]["fp"] += 1
            if outcome.cost_usd is not None:
                costs.append(outcome.cost_usd)
            else:
                unknown_cost = True
            cost_statuses.add(outcome.cost_status)
            if outcome.latency_ms is not None:
                latencies.append(float(outcome.latency_ms))
        if "pending" in cost_statuses:
            cost_status = "pending"
        elif "unavailable" in cost_statuses or unknown_cost:
            cost_status = "unavailable"
        elif "estimated" in cost_statuses:
            cost_status = "estimated"
        elif "actual" in cost_statuses:
            cost_status = "actual"
        elif costs:
            cost_status = "none"
        else:
            cost_status = "none"
        if cost_status in ("pending", "unavailable"):
            known_cost = None
        elif costs:
            known_cost = sum(costs, Decimal("0"))
        else:
            known_cost = Decimal("0") if attempted else None
        per_category = []
        category_f1s: list[float] = []
        for code in ALLOWED_CODES:
            counts = per_code[code]
            f1 = _ratio(
                2 * counts["tp"],
                2 * counts["tp"] + counts["fp"] + counts["fn"],
            )
            if f1 is not None:
                category_f1s.append(f1)
            per_category.append(
                {
                    "code": code,
                    **counts,
                    "f1": f1,
                }
            )
        latencies.sort()
        median = (
            statistics.median(latencies) if latencies else None
        )
        p95 = None
        if latencies:
            index = min(
                len(latencies) - 1,
                max(0, math.ceil(0.95 * len(latencies)) - 1),
            )
            p95 = latencies[index]
        result[arm] = {
            "article_count": attempted,
            "attempted_articles": attempted,
            "valid_count": valid,
            "invalid_count": attempted - valid,
            "parsing_success": _ratio(valid, attempted),
            "exact_match_count": exact,
            "exact_match_rate": _ratio(exact, attempted),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "micro_f1": _ratio(2 * tp, 2 * tp + fp + fn),
            "macro_f1": (
                sum(category_f1s) / len(category_f1s)
                if len(category_f1s) == len(ALLOWED_CODES)
                else None
            ),
            "mean_reward": (
                sum(rewards) / len(rewards) if rewards else None
            ),
            "per_category": per_category,
            "evidence_failure_count": evidence_failures,
            "duplicate_prediction_count": duplicates,
            "parse_error_count": parse_errors,
            "request_error_count": request_errors,
            "missing_output_count": missing_outputs,
            "missing_metric_count": missing_metric,
            "rescoring_failure_count": rescoring_failures,
            "cost_usd": known_cost,
            "cost_status": cost_status,
            "cost_per_article_usd": (
                known_cost / attempted
                if known_cost is not None and attempted
                else None
            ),
            "unknown_cost": unknown_cost,
            "latency_median_ms": median,
            "latency_p95_ms": p95,
            "latency_article_count": len(latencies),
        }
    return result


def _comparison_consistency_warnings(
    stored: Mapping[str, object],
    recomputed: Mapping[str, Mapping[str, object]],
) -> list[str]:
    warnings: list[str] = []
    for arm in ("baseline", "optimized"):
        stored_arm = stored.get(arm)
        if not isinstance(stored_arm, Mapping):
            continue
        actual = recomputed[arm]
        for key in (
            "tp",
            "fp",
            "fn",
            "exact_match_count",
            "valid_count",
            "invalid_count",
            "article_count",
            "parse_error_count",
            "request_error_count",
            "missing_output_count",
            "evidence_failure_count",
            "duplicate_prediction_count",
            "parsing_success",
            "micro_f1",
            "macro_f1",
            "mean_reward",
        ):
            expected = actual.get(key)
            recorded = stored_arm.get(key)
            if key in (
                "tp",
                "fp",
                "fn",
                "exact_match_count",
                "valid_count",
                "invalid_count",
                "article_count",
                "parse_error_count",
                "request_error_count",
                "missing_output_count",
                "evidence_failure_count",
                "duplicate_prediction_count",
            ):
                recorded_number = _metric_int(recorded)
                expected_number = _metric_int(expected)
                equal = recorded_number == expected_number
            else:
                recorded_number = _metric_number(recorded)
                expected_number = _metric_number(expected)
                equal = (
                    recorded_number is None
                    and expected_number is None
                ) or (
                    recorded_number is not None
                    and expected_number is not None
                    and abs(recorded_number - expected_number) <= 1e-9
                )
            if not equal:
                warnings.append(
                    f"Stored {arm} {key} ({recorded if recorded is not None else 'missing'}) "
                    f"does not match the value recomputed from archived raw "
                    f"findings and gold ({expected if expected is not None else 'missing'})."
                )
    return warnings


def _comparison_delta_warnings(
    stored_delta: object,
    baseline: Mapping[str, object],
    optimized: Mapping[str, object],
) -> list[str]:
    if not isinstance(stored_delta, Mapping) or not stored_delta:
        return []
    warnings: list[str] = []
    for key in (
        "precision",
        "recall",
        "micro_f1",
        "macro_f1",
        "exact_match_rate",
        "parsing_success",
        "mean_reward",
    ):
        recorded = _metric_number(stored_delta.get(key))
        left = _metric_number(baseline.get(key))
        right = _metric_number(optimized.get(key))
        expected = None if left is None or right is None else right - left
        equal = (recorded is None and expected is None) or (
            recorded is not None
            and expected is not None
            and abs(recorded - expected) <= 1e-9
        )
        if not equal:
            warnings.append(
                f"Stored {key} delta ({recorded if recorded is not None else 'missing'}) "
                f"does not match the delta recomputed from archived raw "
                f"findings and gold ({expected if expected is not None else 'missing'})."
            )
    return warnings


def _rate_text(value: object) -> str:
    number = _metric_number(value)
    if number is None:
        return "N/A"
    return f"{number * 100:.1f}%"


def _rate_delta_text(baseline: object, optimized: object) -> str:
    left = _metric_number(baseline)
    right = _metric_number(optimized)
    if left is None or right is None:
        return "N/A"
    return f"{(right - left) * 100:+.1f} pp"


def _count_delta_text(baseline: object, optimized: object) -> str:
    left = _metric_int(baseline)
    right = _metric_int(optimized)
    if left is None or right is None:
        return "N/A"
    return f"{right - left:+d}"


def _number_delta_text(baseline: object, optimized: object) -> str:
    left = _metric_number(baseline)
    right = _metric_number(optimized)
    if left is None or right is None:
        return "N/A"
    return f"{right - left:+.4f}"


def _money_text(value: object, status: object) -> str:
    if value is None:
        return f"unavailable ({status or 'unknown'})"
    return f"{value} USD ({status or 'unknown status'})"


def _comparison_effective(
    stored_arm: Mapping[str, object] | None,
    recomputed_arm: Mapping[str, object],
    key: str,
) -> object:
    if isinstance(stored_arm, Mapping) and key in stored_arm:
        return stored_arm.get(key)
    return recomputed_arm.get(key)


def _render_comparison_screen(view: ComparisonView) -> None:
    st.subheader("Comparison")
    mode_label = "Replay Experiment" if view.historical else "Live Experiment"
    st.caption(
        f"{mode_label}: stored paired baseline/optimized results. This "
        "screen deterministically re-scores recorded findings against "
        "archived gold, executes no program, and starts no GEPA."
    )
    manifest = view.manifest
    if manifest.benchmark is None:
        st.error(
            "The benchmark protocol metadata is missing from this archive; "
            "comparison values are not available."
        )
        return
    st.markdown(
        f"- **Benchmark run:** `{manifest.run_id}` — {manifest.status} / "
        f"{manifest.origin}\n"
        f"- **Baseline development run:** "
        f"`{manifest.links.baseline_development_run_id}`\n"
        f"- **Optimization run:** `{manifest.links.optimization_run_id}`\n"
        f"- **Baseline program:** `{manifest.links.baseline_program_id}`\n"
        f"- **Optimized program:** `{manifest.links.optimized_program_id}`\n"
        f"- **Split:** held-out test snapshot "
        f"({len(view.cases)} archived cases)\n"
        f"- **Released:** "
        f"{'yes' if manifest.benchmark.released else 'no'}\n"
        f"- **Measured experiment:** {'yes' if view.measured else 'no'}\n"
        f"- **Exposure index:** {manifest.benchmark.exposure_index}\n"
        f"- **Per-case order seed:** {manifest.benchmark.seed}"
    )
    if view.historical and not view.measured:
        st.warning(
            "This Replay archive is mock/test or not released. Its numbers "
            "are not a measured held-out result."
        )
    if not view.historical:
        st.warning(
            "This Live screen is reading an existing local benchmark "
            "archive. It does not read the current test split for display; "
            "a fresh benchmark is a separate confirmed paid action below."
        )
    summary = view.summary
    if not isinstance(summary, Mapping):
        st.warning(
            "The benchmark summary is not archived. Aggregate metrics and "
            "the verdict are unavailable; raw per-case records are shown "
            "without inventing a summary."
        )
        summary = {}
    for warning in view.warnings:
        st.warning(warning)
    st.markdown("### Paired quality metrics")
    st.caption(
        "Rates are kept on the internal 0-1 scale and displayed as "
        "percentages. Deltas for rates are percentage points (pp), not "
        "relative percent. Micro-F1 is the corpus aggregate; the mean case "
        "reward is a separate metric."
    )
    recomputed = _comparison_recompute(view.cases, view.outcomes)
    stored_metrics = summary.get("metrics")
    if not isinstance(stored_metrics, Mapping):
        stored_metrics = {}
    stored_baseline = stored_metrics.get("baseline")
    stored_optimized = stored_metrics.get("optimized")
    if not isinstance(stored_baseline, Mapping):
        stored_baseline = None
    if not isinstance(stored_optimized, Mapping):
        stored_optimized = None
    consistency = _comparison_consistency_warnings(
        {"baseline": stored_baseline or {}, "optimized": stored_optimized or {}},
        recomputed,
    )
    stored_delta = (
        stored_metrics.get("delta")
        if isinstance(stored_metrics, Mapping)
        else None
    )
    delta_warnings = _comparison_delta_warnings(
        stored_delta,
        stored_baseline or {},
        stored_optimized or {},
    )
    consistency.extend(delta_warnings)
    stored_latency_delta = summary.get("latency_delta")
    if isinstance(stored_latency_delta, Mapping):
        for key, left_key, right_key in (
            ("median_ms", "latency_median_ms", "latency_median_ms"),
            ("p95_ms", "latency_p95_ms", "latency_p95_ms"),
        ):
            recorded = _metric_number(stored_latency_delta.get(key))
            left = _metric_number(recomputed["baseline"].get(left_key))
            right = _metric_number(recomputed["optimized"].get(right_key))
            expected = None if left is None or right is None else right - left
            if not (
                (recorded is None and expected is None)
                or (
                    recorded is not None
                    and expected is not None
                    and abs(recorded - expected) <= 1e-9
                )
            ):
                consistency.append(
                    f"Stored latency delta {key} ({recorded if recorded is not None else 'missing'}) "
                    "does not match the archived outcome timings."
                )
    rescoring_failures = sum(
        _int_value(recomputed[arm].get("rescoring_failure_count"))
        for arm in ("baseline", "optimized")
    )
    if rescoring_failures:
        st.error(
            f"{rescoring_failures} archived raw outcome(s) could not be "
            "re-scored with the frozen metric. The aggregate view is "
            "incomplete and no missing value is invented."
        )
    if stored_baseline is None or stored_optimized is None:
        st.warning(
            "Stored aggregate metrics are incomplete. Values marked below "
            "are recomputed from archived raw findings and gold with the "
            "frozen deterministic metric; no inference is run."
        )
    if consistency:
        st.error(
            "Stored aggregate metrics disagree with the archived raw "
            "outcomes. The stored values are not silently replaced."
        )
        for warning in consistency:
            st.warning(warning)
    else:
        st.caption(
            "Stored aggregate metrics were recomputed from the archived "
            "raw findings and gold and matched."
        )
    base = {
        key: _comparison_effective(
            stored_baseline, recomputed["baseline"], key
        )
        for key in (
            "article_count",
            "valid_count",
            "invalid_count",
            "parsing_success",
            "exact_match_count",
            "exact_match_rate",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "micro_f1",
            "macro_f1",
            "mean_reward",
            "evidence_failure_count",
            "duplicate_prediction_count",
            "parse_error_count",
            "request_error_count",
            "missing_output_count",
            "cost_per_article_usd",
            "cost_status",
            "latency_median_ms",
            "latency_p95_ms",
            "evaluate_percent_mean",
        )
    }
    opt = {
        key: _comparison_effective(
            stored_optimized, recomputed["optimized"], key
        )
        for key in base
    }
    st.markdown(
        "| Metric | Baseline | Optimized | Delta |\n"
        "|---|---|---|---|\n"
        f"| Articles evaluated | {base.get('article_count', 'N/A')} | "
        f"{opt.get('article_count', 'N/A')} | "
        f"{_count_delta_text(base.get('article_count'), opt.get('article_count'))} |\n"
        f"| Valid / invalid | {base.get('valid_count', 'N/A')} / "
        f"{base.get('invalid_count', 'N/A')} | "
        f"{opt.get('valid_count', 'N/A')} / "
        f"{opt.get('invalid_count', 'N/A')} | "
        f"{_count_delta_text(base.get('valid_count'), opt.get('valid_count'))} valid |\n"
        f"| Parsing success | {_rate_text(base.get('parsing_success'))} | "
        f"{_rate_text(opt.get('parsing_success'))} | "
        f"{_rate_delta_text(base.get('parsing_success'), opt.get('parsing_success'))} |\n"
        f"| Exact match | {base.get('exact_match_count', 'N/A')} "
        f"({_rate_text(base.get('exact_match_rate'))}) | "
        f"{opt.get('exact_match_count', 'N/A')} "
        f"({_rate_text(opt.get('exact_match_rate'))}) | "
        f"{_count_delta_text(base.get('exact_match_count'), opt.get('exact_match_count'))} cases, "
        f"{_rate_delta_text(base.get('exact_match_rate'), opt.get('exact_match_rate'))} |\n"
        f"| Precision | {_rate_text(base.get('precision'))} | "
        f"{_rate_text(opt.get('precision'))} | "
        f"{_rate_delta_text(base.get('precision'), opt.get('precision'))} |\n"
        f"| Recall | {_rate_text(base.get('recall'))} | "
        f"{_rate_text(opt.get('recall'))} | "
        f"{_rate_delta_text(base.get('recall'), opt.get('recall'))} |\n"
        f"| Micro-F1 | {_rate_text(base.get('micro_f1'))} | "
        f"{_rate_text(opt.get('micro_f1'))} | "
        f"{_rate_delta_text(base.get('micro_f1'), opt.get('micro_f1'))} |\n"
        f"| Macro-F1 | {_rate_text(base.get('macro_f1'))} | "
        f"{_rate_text(opt.get('macro_f1'))} | "
        f"{_rate_delta_text(base.get('macro_f1'), opt.get('macro_f1'))} |\n"
        f"| TP / FP / FN | {base.get('tp', 'N/A')} / {base.get('fp', 'N/A')} / "
        f"{base.get('fn', 'N/A')} | {opt.get('tp', 'N/A')} / "
        f"{opt.get('fp', 'N/A')} / {opt.get('fn', 'N/A')} | "
        f"FP {_count_delta_text(base.get('fp'), opt.get('fp'))}, "
        f"FN {_count_delta_text(base.get('fn'), opt.get('fn'))} |\n"
        f"| Mean case reward | {_number(base.get('mean_reward'))} | "
        f"{_number(opt.get('mean_reward'))} | "
        f"{_number_delta_text(base.get('mean_reward'), opt.get('mean_reward'))} |\n"
        f"| Evidence failures | {base.get('evidence_failure_count', 'N/A')} | "
        f"{opt.get('evidence_failure_count', 'N/A')} | "
        f"{_count_delta_text(base.get('evidence_failure_count'), opt.get('evidence_failure_count'))} |\n"
        f"| Duplicate predictions | "
        f"{base.get('duplicate_prediction_count', 'N/A')} | "
        f"{opt.get('duplicate_prediction_count', 'N/A')} | "
        f"{_count_delta_text(base.get('duplicate_prediction_count'), opt.get('duplicate_prediction_count'))} |\n"
        f"| Parse / request / missing | "
        f"{base.get('parse_error_count', 'N/A')} / "
        f"{base.get('request_error_count', 'N/A')} / "
        f"{base.get('missing_output_count', 'N/A')} | "
        f"{opt.get('parse_error_count', 'N/A')} / "
        f"{opt.get('request_error_count', 'N/A')} / "
        f"{opt.get('missing_output_count', 'N/A')} | "
        f"{_count_delta_text(base.get('parse_error_count'), opt.get('parse_error_count'))} parse |\n"
        f"| Cost / article | "
        f"{_money_text(base.get('cost_per_article_usd'), base.get('cost_status'))} | "
        f"{_money_text(opt.get('cost_per_article_usd'), opt.get('cost_status'))} | "
        f"{_number_delta_text(_metric_number(base.get('cost_per_article_usd')), _metric_number(opt.get('cost_per_article_usd')))} USD |\n"
        f"| Latency median | {_latency(base.get('latency_median_ms'))} | "
        f"{_latency(opt.get('latency_median_ms'))} | "
        f"{_number_delta_text(_metric_number(base.get('latency_median_ms')), _metric_number(opt.get('latency_median_ms')))} ms |\n"
        f"| Latency p95 (exploratory) | "
        f"{_latency(base.get('latency_p95_ms'))} | "
        f"{_latency(opt.get('latency_p95_ms'))} | "
        f"{_number_delta_text(_metric_number(base.get('latency_p95_ms')), _metric_number(opt.get('latency_p95_ms')))} ms |\n"
        f"| DSPy Evaluate mean | "
        f"{_rate_text(base.get('evaluate_percent_mean'))} | "
        f"{_rate_text(opt.get('evaluate_percent_mean'))} | "
        f"{_rate_delta_text(base.get('evaluate_percent_mean'), opt.get('evaluate_percent_mean'))} |"
    )
    st.caption(
        "The DSPy Evaluate row is the evaluator's rounded per-case mean and "
        "is distinct from corpus micro-F1. Cost values inherit their "
        "recorded actual/estimated/pending status; a pending value is never "
        "presented as settled."
    )
    per_category = stored_baseline.get("per_category") if stored_baseline else None
    if not isinstance(per_category, list) or not per_category:
        per_category = recomputed["baseline"].get("per_category")
    optimized_categories = (
        stored_optimized.get("per_category") if stored_optimized else None
    )
    if not isinstance(optimized_categories, list) or not optimized_categories:
        optimized_categories = recomputed["optimized"].get("per_category")
    if isinstance(per_category, list) and isinstance(optimized_categories, list):
        optimized_by_code = {
            str(item.get("code")): item
            for item in optimized_categories
            if isinstance(item, Mapping)
        }
        st.markdown("#### Per-category F1")
        for item in per_category:
            if not isinstance(item, Mapping):
                continue
            code = str(item.get("code"))
            other = optimized_by_code.get(code, {})
            st.markdown(
                f"- **{code}:** baseline TP/FP/FN "
                f"{item.get('tp', 'N/A')}/{item.get('fp', 'N/A')}/"
                f"{item.get('fn', 'N/A')} F1 {_rate_text(item.get('f1'))}; "
                f"optimized TP/FP/FN {other.get('tp', 'N/A')}/"
                f"{other.get('fp', 'N/A')}/{other.get('fn', 'N/A')} F1 "
                f"{_rate_text(other.get('f1'))}"
            )
    st.markdown("### Validity, verdict, and warnings")
    st.markdown(
        f"- **Complete:** {summary.get('complete', 'not archived')}\n"
        f"- **Complete with errors:** "
        f"{summary.get('complete_with_errors', 'not archived')}\n"
        f"- **Comparable / inputs equal:** "
        f"{summary.get('comparable', 'not archived')} / "
        f"{summary.get('inputs_equal', 'not archived')}\n"
        f"- **Eligible for verdict:** "
        f"{summary.get('eligible_for_verdict', 'not archived')}\n"
        f"- **Quality non-regression gate:** "
        f"{summary.get('quality_non_regression_gate', 'not archived')}\n"
        f"- **Verdict:** {summary.get('verdict', 'not archived')}\n"
        f"- **Verdict reason:** "
        f"{summary.get('verdict_reason', 'not archived')}"
    )
    validity = summary.get("validity")
    if isinstance(validity, Mapping):
        st.markdown(
            f"- **Both arms full validity:** "
            f"{validity.get('both_arms_full_validity', 'N/A')}\n"
            f"- **Baseline invalid cases:** "
            f"{', '.join(validity.get('baseline_invalid_cases') or []) or 'none'}\n"
            f"- **Optimized invalid cases:** "
            f"{', '.join(validity.get('optimized_invalid_cases') or []) or 'none'}"
        )
    corrected = summary.get("corrected_cases")
    regressed = summary.get("regressed_cases")
    st.markdown(
        f"- **Corrected cases (exact match):** "
        f"{', '.join(corrected or []) or 'none'}\n"
        f"- **Regressed cases (exact match):** "
        f"{', '.join(regressed or []) or 'none'}"
    )
    category_regressions = summary.get("category_regressions")
    if isinstance(category_regressions, list) and category_regressions:
        st.warning(
            "Category F1 regressed for: "
            + ", ".join(
                str(item.get("code"))
                for item in category_regressions
                if isinstance(item, Mapping)
            )
        )
    st.markdown("### Historical costs and conditional break-even")
    _render_historical_costs(summary)
    economics = summary.get("economics")
    if isinstance(economics, Mapping):
        st.markdown(
            f"- **Cost-only break-even status:** "
            f"{economics.get('status', 'not archived')}\n"
            f"- **Break-even articles:** "
            f"{economics.get('break_even_articles', 'not finite')}\n"
            f"- **Reason:** {economics.get('reason', 'not archived')}\n"
            f"- **Costs estimated:** "
            f"{economics.get('costs_estimated', 'not archived')}"
        )
    else:
        st.caption(
            "Break-even is not archived for this run. It is not computed "
            "from unrecorded or unsettled costs."
        )
    st.markdown("### Test exposure and reuse")
    st.markdown(
        f"- **First exposure:** "
        f"{manifest.benchmark.exposure_index == 0}\n"
        f"- **Exposure index:** {manifest.benchmark.exposure_index}\n"
        f"- **Paired order (first case):** "
        f"{' → '.join(manifest.benchmark.paired_order)}\n"
        f"- **Comparability warnings:** "
        f"{'; '.join(manifest.benchmark.comparability_warnings) or 'none'}"
    )
    test_reuse = summary.get("test_reuse")
    if isinstance(test_reuse, Mapping):
        st.markdown(
            f"- **Reuse provenance:** "
            f"{test_reuse.get('provenance', 'not archived')}\n"
            f"- **Recorded prior runs:** "
            f"{len(test_reuse.get('prior_runs') or [])}"
        )
        reuse_warnings = test_reuse.get("warnings")
        if isinstance(reuse_warnings, list):
            for warning in reuse_warnings:
                st.warning(str(warning))
    st.markdown("### Per-article paired results")
    st.caption(
        "Corrected and newly missed gold issues use the archived "
        "deterministic matching. If same-code windows were ambiguous, the "
        "tie-break attribution can change without changing corpus counts."
    )
    _render_comparison_cases(view)
    _render_comparison_candidates(view)


def _comparison_outcome_by_phase(
    outcomes: Sequence[RecordedOutcome], case_id: str
) -> dict[str, RecordedOutcome]:
    return {
        outcome.phase: outcome
        for outcome in outcomes
        if outcome.case_id == case_id
    }


def _render_comparison_cases(view: ComparisonView) -> None:
    for case in view.cases:
        pair = _comparison_outcome_by_phase(view.outcomes, case.case_id)
        baseline = pair.get("baseline")
        optimized = pair.get("optimized")
        baseline_status = "missing" if baseline is None else baseline.status
        optimized_status = "missing" if optimized is None else optimized.status
        with st.expander(
            f"{case.case_id} — baseline {baseline_status} / "
            f"optimized {optimized_status}"
        ):
            st.markdown("**Canonical article text**")
            st.text(canonical_document(case))
            st.markdown("**Supplied product facts**")
            st.code(
                json.dumps(
                    case.product_facts.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                language="json",
            )
            st.markdown("**Supplied sources**")
            st.code(
                json.dumps(
                    [source.model_dump(mode="json") for source in case.sources],
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                language="json",
            )
            st.markdown("**Gold findings**")
            _render_gold_findings(case)
            for arm, outcome in (
                ("Baseline", baseline),
                ("Optimized", optimized),
            ):
                st.markdown(f"**{arm} arm**")
                if outcome is None:
                    st.error(f"No recorded {arm.lower()} outcome exists.")
                    continue
                if outcome.status == "ok":
                    _render_findings(outcome.findings or [])
                else:
                    st.text("No parsed findings: this is an invalid output.")
                _render_outcome_metric(outcome)
                st.markdown(
                    f"- **Latency:** {_latency(outcome.latency_ms)}\n"
                    f"- **Recorded cost:** {_outcome_cost(outcome)}\n"
                    f"- **Request IDs:** "
                    f"{', '.join(outcome.request_ids) or 'none'}"
                )
            try:
                baseline_score = (
                    _rescore_outcome(case, baseline)
                    if baseline is not None
                    else None
                )
            except Exception as exc:  # noqa: BLE001 - one archived case
                st.error(
                    f"Baseline raw outcome for {case.case_id} cannot be "
                    f"re-scored: {exc}"
                )
                baseline_score = None
            try:
                optimized_score = (
                    _rescore_outcome(case, optimized)
                    if optimized is not None
                    else None
                )
            except Exception as exc:  # noqa: BLE001 - one archived case
                st.error(
                    f"Optimized raw outcome for {case.case_id} cannot be "
                    f"re-scored: {exc}"
                )
                optimized_score = None
            baseline_missed = (
                {item.issue_id for item in baseline_score.missed_gold}
                if baseline_score is not None
                else set()
            )
            optimized_missed = (
                {item.issue_id for item in optimized_score.missed_gold}
                if optimized_score is not None
                else set()
            )
            corrected = sorted(baseline_missed - optimized_missed)
            newly_missed = sorted(optimized_missed - baseline_missed)
            st.markdown("**Gold-issue changes**")
            st.text(
                f"Corrected gold issues: "
                f"{', '.join(corrected) or 'none'}\n"
                f"Newly missed gold issues: "
                f"{', '.join(newly_missed) or 'none'}\n"
                "Attribution uses deterministic re-scoring of the archived "
                "raw findings; ambiguous same-code windows may change "
                "assignment without changing corpus counts."
            )
            if baseline_score is not None and optimized_score is not None:
                st.text(
                    f"False positives: baseline {baseline_score.fp}, "
                    f"optimized {optimized_score.fp}; cosmetic quotation "
                    "changes are not automatically classified as repaired "
                    "issues."
                )


def _render_comparison_candidates(view: ComparisonView) -> None:
    st.markdown("### Optimization candidate history")
    candidate_records, candidate_error = _candidate_records(
        view.candidate_stats
    )
    baseline = view.baseline
    if baseline is None:
        st.warning(
            "The baseline program artifact is unavailable; candidate "
            "instructions cannot be compared."
        )
        return
    if candidate_error:
        st.warning(candidate_error)
        return
    labels = []
    for record in candidate_records:
        marker = " — selected" if record.get("selected") else ""
        labels.append(
            f"Candidate {record['candidate_index']} — validation "
            f"{_number(record.get('val_aggregate_score'))}{marker}"
        )
    selected_label = next(
        (
            label
            for label, record in zip(labels, candidate_records)
            if record.get("selected")
        ),
        labels[0],
    )
    choice = st.selectbox(
        "Recorded candidate instructions",
        labels,
        index=labels.index(selected_label),
        key="comparison_candidate",
    )
    record = candidate_records[labels.index(choice)]
    st.caption(
        "Candidate validation scores are original optimization results. "
        "Only the selected program was evaluated on the held-out snapshot."
    )
    st.markdown(
        f"- **Candidate index:** {record['candidate_index']}\n"
        f"- **Original validation score:** "
        f"{_number(record.get('val_aggregate_score'))}\n"
        f"- **Recorded selection:** "
        f"{'yes' if record.get('selected') else 'no'}"
    )
    instructions = str(record["instructions"])
    st.code(instructions, language="text")
    if instructions == baseline.manifest.instructions:
        st.info("This candidate is the unchanged baseline instructions.")
    else:
        st.code(
            _unified_diff(
                baseline.manifest.instructions,
                instructions,
                fromfile="baseline",
                tofile=f"candidate-{record['candidate_index']}",
            ),
            language="diff",
        )


def _render_benchmark_confirmation(
    context: LiveContext, manager: JobManager
) -> None:
    settings = context.settings
    development = context.development
    if settings is None:
        st.warning(
            "A confirmed Live benchmark requires a configured "
            "OPENROUTER_API_KEY."
        )
        return
    if development is None:
        st.info(
            "A completed Live baseline development run is required before a "
            "paired benchmark. This stage cannot start one."
        )
        return
    if (
        development.manifest.status != "completed"
        or development.manifest.origin != "live"
    ):
        st.warning(
            "A paired benchmark requires a completed Live development run."
        )
        return
    linked = [
        entry
        for entry in context.optimization_entries
        if entry.manifest is not None
        and entry.manifest.status == "completed"
        and entry.manifest.origin == "live"
        and entry.manifest.links.baseline_development_run_id
        == development.manifest.run_id
    ]
    if not linked:
        st.info(
            "A completed Live optimization linked to the selected "
            "development run is required before the paired benchmark."
        )
        return
    labels = [_run_entry_label(entry) for entry in linked]
    selected_label = st.selectbox(
        "Completed optimization run for the paired benchmark",
        labels,
        index=0,
        key="benchmark_optimization_run",
    )
    try:
        optimization_entry = linked[labels.index(selected_label)]
    except ValueError:
        optimization_entry = linked[0]
    if optimization_entry.manifest is None:
        st.error(
            "The selected optimization archive has no readable manifest; "
            "no benchmark can start."
        )
        return
    limits = settings.benchmark_budget()
    task_route = settings.task_route()
    config_key = "|".join(
        [
            str(manager.mode_epoch),
            str(manager.confirmation_generation),
            development.manifest.run_id,
            optimization_entry.run_id,
            settings.task_model,
            settings.task_provider,
            str(settings.max_benchmark_calls),
            str(settings.max_benchmark_cost_usd),
            str(settings.temperature),
            str(settings.max_output_tokens),
            str(settings.request_timeout_seconds),
        ]
    )
    if (
        st.session_state.get("bench_confirm_key") != config_key
        or not st.session_state.get("bench_confirm_token")
        or not st.session_state.get("bench_confirm_run_id")
        or st.session_state.get("bench_confirm_epoch")
        != manager.mode_epoch
        or st.session_state.get("bench_confirm_generation")
        != manager.confirmation_generation
    ):
        _clear_confirmation_state()
        st.session_state["bench_confirm_key"] = config_key
        st.session_state["bench_confirm_token"] = secrets.token_hex(16)
        st.session_state["bench_confirm_run_id"] = _new_job_run_id("bench")
        st.session_state["bench_confirm_epoch"] = manager.mode_epoch
        st.session_state["bench_confirm_generation"] = (
            manager.confirmation_generation
        )
    token = str(st.session_state["bench_confirm_token"])
    run_id = str(st.session_state["bench_confirm_run_id"])
    epoch = int(st.session_state["bench_confirm_epoch"])
    generation = int(st.session_state["bench_confirm_generation"])
    st.markdown("### Start a fresh paired held-out benchmark")
    st.warning(
        "This action opens the sealed held-out test split for both frozen "
        "programs. It never runs automatically after optimization and is "
        "not restartable from Replay. Local benchmark runs already "
        f"recorded: {len(context.benchmark_entries)}."
    )
    st.markdown(
        f"- **Baseline development run:** `{development.manifest.run_id}`\n"
        f"- **Optimization run:** `{optimization_entry.run_id}`\n"
        f"- **Baseline / optimized program:** "
        f"`{optimization_entry.manifest.links.baseline_program_id}` / "
        f"`{optimization_entry.manifest.links.optimized_program_id}`\n"
        f"- **Task model/provider:** `{task_route.model}` / "
        f"`{task_route.provider}` at "
        f"{task_route.input_usd_per_million}/"
        f"{task_route.output_usd_per_million} USD per 1M tokens\n"
        f"- **Confirmed benchmark budget:** {limits.max_calls} calls / "
        f"{limits.max_cost_usd} USD\n"
        f"- **Fresh job run ID:** `{run_id}`"
    )
    st.caption(
        "A previous exposure can make this run exploratory rather than a "
        "fresh confirmatory result; the archive preserves its original "
        "exposure index and warnings."
    )
    release = st.checkbox(
        "Release the completed benchmark as a measured Replay archive",
        key=f"bench_release_{token}",
        help=(
            "Only a completed Live benchmark can be released. A failed or "
            "stopped run remains incomplete and is never released."
        ),
    )
    confirmed = st.checkbox(
        "I confirm this benchmark budget and that this action opens the "
        "sealed held-out split.",
        key=f"bench_confirm_checkbox_{token}",
    )
    with st.form("benchmark_confirmation_form"):
        submitted = st.form_submit_button("Start confirmed benchmark")
    if not submitted:
        return
    if not confirmed:
        st.error(
            "Explicit benchmark budget confirmation is required; no request "
            "was made."
        )
        return
    spec = BenchmarkJobSpec(
        mode=MODE_LIVE,
        run_id=run_id,
        token=token,
        mode_epoch=epoch,
        confirmation_generation=generation,
        settings=settings,
        baseline_development_run_id=development.manifest.run_id,
        optimization_run_id=optimization_entry.run_id,
        release_results=bool(release),
        runs_root=RUNS_ROOT,
        programs_root=PROGRAMS_ROOT,
    )
    try:
        handle = manager.start_benchmark(spec)
    except JobGateError as exc:
        _clear_confirmation_state()
        st.error(f"Benchmark was not started: {exc}")
        active = manager.active_handle
        if active is not None:
            _render_job_details(active)
        return
    st.session_state["bench_started_run_id"] = handle.run_id
    _clear_confirmation_state()
    st.success(
        "Benchmark started. Progress polls local events and the shared "
        "per-job ledger; polling sends no provider request."
    )
    _render_job_details(handle, cancel_label="Cancel benchmark")


def _render_live_comparison(
    context: LiveContext, manager: JobManager
) -> None:
    st.subheader("Comparison")
    st.caption(
        "Live Experiment: recorded paired results are read-only. A new "
        "paired holdout run requires a separate explicit budget "
        "confirmation."
    )
    active = manager.active_handle
    if active is not None:
        if active.job_kind == JOB_KIND_BENCHMARK:
            _render_active_job_fragment(
                active, cancel_label="Cancel benchmark"
            )
            return
        st.warning(
            "Another Live job is active. This screen will not start a "
            "second paid job."
        )
        _render_job_details(active)
        return
    latest = manager.latest_handle_for(JOB_KIND_BENCHMARK)
    if (
        latest is not None
        and st.session_state.get("bench_started_run_id") == latest.run_id
    ):
        _render_finished_job(latest, cancel_label="Cancel benchmark")
    entries = context.benchmark_entries
    if entries:
        labels = [_run_entry_label(entry) for entry in entries]
        selected_label = st.selectbox(
            "Recorded benchmark archive",
            labels,
            index=_default_entry_index(entries),
            key="live_benchmark_archive",
        )
        try:
            entry = entries[labels.index(selected_label)]
        except ValueError:
            entry = entries[0]
        if entry.manifest is None:
            st.error(
                "The selected benchmark archive cannot be read: "
                f"{entry.error or 'manifest is unavailable'}. Nothing is "
                "regenerated."
            )
        else:
            try:
                archive = open_replay_archive(
                    entry.run_dir,
                    runs_root=RUNS_ROOT,
                    programs_root=PROGRAMS_ROOT,
                )
            except ArtifactError as exc:
                st.error(
                    "The selected benchmark archive cannot be read: "
                    f"{exc}. Nothing is regenerated."
                )
            except Exception as exc:  # noqa: BLE001 - read-only UI
                st.error(
                    "The selected benchmark archive cannot be read: "
                    f"{type(exc).__name__}: {exc}. Nothing is regenerated."
                )
            else:
                _render_comparison_screen(
                    _comparison_view_from_archive(
                        archive, historical=False
                    )
                )
    else:
        st.info(
            "No recorded benchmark archive exists locally. Stored paired "
            "metrics cannot be shown until a completed benchmark is "
            "archived."
        )
    if read_active_job_lock(RUNS_ROOT) is None:
        _render_benchmark_confirmation(context, manager)
    else:
        st.warning(
            "Another app process holds the single active-job lock. This "
            "screen will not start a benchmark."
        )


def _render_replay_playground() -> None:
    st.subheader("Playground")
    st.warning(
        "Replay Experiment: new Playground inference is disabled. Replay "
        "reads recorded artifacts only and cannot send article text to a "
        "provider or create another paid run."
    )
    st.info(
        "Recorded cases remain browsable through Comparison and Prompt "
        "Inspector. Switch to Live Experiment to run both frozen artifacts "
        "on new text after a separate budget confirmation."
    )


def _parse_playground_json(
    facts_text: str, sources_text: str
) -> tuple[ProductFacts | None, tuple[SourceRecord, ...], str | None]:
    try:
        facts_payload = json.loads(facts_text or "{}")
    except ValueError as exc:
        return None, (), f"Product facts are not valid JSON: {exc}"
    if not isinstance(facts_payload, dict):
        return None, (), "Product facts must be one JSON object."
    try:
        facts = ProductFacts.model_validate(facts_payload)
    except Exception as exc:  # noqa: BLE001 - fail closed in the form
        return None, (), f"Product facts do not match the schema: {exc}"
    try:
        sources_payload = json.loads(sources_text or "[]")
    except ValueError as exc:
        return None, (), f"Sources are not valid JSON: {exc}"
    if not isinstance(sources_payload, list):
        return None, (), "Sources must be one JSON array."
    try:
        sources = tuple(
            SourceRecord.model_validate(item) for item in sources_payload
        )
    except Exception as exc:  # noqa: BLE001
        return None, (), f"Sources do not match the schema: {exc}"
    source_ids = [source.source_id for source in sources]
    if len(source_ids) != len(set(source_ids)):
        return None, (), "Source IDs must be unique."
    return facts, sources, None


def _render_live_playground(
    context: LiveContext, manager: JobManager
) -> None:
    st.subheader("Playground")
    st.caption(
        "Live Experiment: an unscored comparison of both frozen artifacts "
        "on supplied Czech text. It shows findings, errors, messages, usage, "
        "cost, and latency; it never computes a gold-based score."
    )
    active = manager.active_handle
    if active is not None:
        if active.job_kind == JOB_KIND_PLAYGROUND:
            _render_active_job_fragment(
                active, cancel_label="Cancel playground"
            )
            return
        st.warning(
            "Another Live job is active. The Playground will not start a "
            "second paid job."
        )
        _render_job_details(active)
        return
    if read_active_job_lock(RUNS_ROOT) is not None:
        st.warning(
            "Another app process holds the single active-job lock. The "
            "Playground will not render a paid confirmation form."
        )
        return
    latest = manager.latest_handle_for(JOB_KIND_PLAYGROUND)
    if (
        latest is not None
        and st.session_state.get("play_started_run_id") == latest.run_id
    ):
        _render_finished_job(latest, cancel_label="Cancel playground")
    settings = context.settings
    if settings is None:
        st.warning(
            "A Live Playground requires a configured OPENROUTER_API_KEY. "
            "No paid action is available."
        )
        return
    st.warning(
        "Live Playground content is sent to the configured LLM provider. "
        "The browser is local, but inference is not local."
    )
    optimization_entries = [
        entry
        for entry in context.optimization_entries
        if entry.manifest is not None
        and entry.manifest.status == "completed"
        and entry.manifest.origin == "live"
    ]
    if not optimization_entries:
        st.info(
            "A completed Live optimization run is required so both frozen "
            "artifacts exist. This stage cannot start optimization from the "
            "Playground."
        )
        return
    labels = [_run_entry_label(entry) for entry in optimization_entries]
    selected_label = st.selectbox(
        "Optimization run supplying both artifacts",
        labels,
        index=0,
        key="playground_optimization_run",
    )
    try:
        optimization_entry = optimization_entries[
            labels.index(selected_label)
        ]
    except ValueError:
        optimization_entry = optimization_entries[0]
    if optimization_entry.manifest is None:
        st.error(
            "The selected optimization archive has no readable manifest; "
            "the Playground cannot start."
        )
        return
    limits = settings.playground_budget()
    st.markdown(
        f"- **Baseline program:** "
        f"`{optimization_entry.manifest.links.baseline_program_id}`\n"
        f"- **Optimized program:** "
        f"`{optimization_entry.manifest.links.optimized_program_id}`\n"
        f"- **Rules:** archived optimization snapshot "
        f"({optimization_entry.manifest.snapshot.rules_version})\n"
        f"- **Playground budget:** {limits.max_calls} calls / "
        f"{limits.max_cost_usd} USD (separate phase)"
    )
    st.caption(
        "Product facts and sources are optional. Leave {} / [] empty when "
        "the article has no supplied references."
    )

    def playground_signature(
        draft_title: object,
        draft_article: object,
        draft_facts: object,
        draft_sources: object,
        draft_retain: object,
    ) -> str:
        payload = json.dumps(
            {
                "optimization_run_id": optimization_entry.run_id,
                "title": str(draft_title or ""),
                "article": str(draft_article or ""),
                "facts_text": str(draft_facts or ""),
                "sources_text": str(draft_sources or ""),
                "retain": bool(draft_retain),
                "mode_epoch": manager.mode_epoch,
                "confirmation_generation": manager.confirmation_generation,
                "task_model": settings.task_model,
                "task_provider": settings.task_provider,
                "max_calls": settings.max_playground_calls,
                "max_cost_usd": str(settings.max_playground_cost_usd),
                "temperature": settings.temperature,
                "max_output_tokens": settings.max_output_tokens,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    config_key = playground_signature(
        st.session_state.get("play_title", ""),
        st.session_state.get("play_article", ""),
        st.session_state.get("play_facts", "{}"),
        st.session_state.get("play_sources", "[]"),
        st.session_state.get("play_retain", False),
    )
    if (
        st.session_state.get("play_confirm_key") != config_key
        or not st.session_state.get("play_confirm_token")
        or not st.session_state.get("play_confirm_run_id")
        or st.session_state.get("play_confirm_epoch")
        != manager.mode_epoch
        or st.session_state.get("play_confirm_generation")
        != manager.confirmation_generation
    ):
        _clear_confirmation_state()
        st.session_state["play_confirm_key"] = config_key
        st.session_state["play_confirm_token"] = secrets.token_hex(16)
        st.session_state["play_confirm_run_id"] = _new_job_run_id("play")
        st.session_state["play_confirm_epoch"] = manager.mode_epoch
        st.session_state["play_confirm_generation"] = (
            manager.confirmation_generation
        )
    token = str(st.session_state["play_confirm_token"])
    run_id = str(st.session_state["play_confirm_run_id"])
    confirmed = st.checkbox(
        "I confirm the separate playground budget and that article "
        "content will be sent to the configured provider.",
        key=f"play_confirm_{token}",
    )
    with st.form("playground_form", clear_on_submit=False):
        title = st.text_input(
            "Article title",
            key="play_title",
            max_chars=300,
        )
        article = st.text_area(
            "Article body",
            key="play_article",
            height=220,
            max_chars=20_000,
        )
        facts_text = st.text_area(
            "Optional product facts (JSON object)",
            value="{}",
            key="play_facts",
            height=120,
        )
        sources_text = st.text_area(
            "Optional sources (JSON array)",
            value="[]",
            key="play_sources",
            height=140,
        )
        retain = st.checkbox(
            "Store this article and request text locally for audit",
            key="play_retain",
            help=(
                "Off by default. When off, the run keeps only hashes, "
                "status, cost, and timing; findings and messages remain in "
                "this app session only."
            ),
        )
        submitted = st.form_submit_button("Run unscored playground")
    if not submitted:
        return
    if not confirmed:
        st.error(
            "Explicit playground budget confirmation is required; no "
            "request was made."
        )
        return
    submitted_key = playground_signature(
        title, article, facts_text, sources_text, retain
    )
    if submitted_key != config_key:
        _clear_confirmation_state()
        st.warning(
            "The playground inputs changed. Review them and confirm the "
            "separate budget again; no request was made."
        )
        return
    facts, sources, parse_error = _parse_playground_json(
        facts_text, sources_text
    )
    if parse_error is not None:
        st.error(parse_error)
        return
    if not title.strip() or not article.strip():
        st.error("A non-empty article title and body are required.")
        return
    assert facts is not None
    spec = PlaygroundJobSpec(
        mode=MODE_LIVE,
        run_id=run_id,
        token=token,
        mode_epoch=int(st.session_state["play_confirm_epoch"]),
        confirmation_generation=int(
            st.session_state["play_confirm_generation"]
        ),
        settings=settings,
        optimization_run_id=optimization_entry.run_id,
        title=title,
        article_text=article,
        product_facts=facts,
        sources=sources,
        retain_content=bool(retain),
        runs_root=RUNS_ROOT,
        programs_root=PROGRAMS_ROOT,
    )
    try:
        handle = manager.start_playground(spec)
    except JobGateError as exc:
        _clear_confirmation_state()
        st.error(f"Playground was not started: {exc}")
        active = manager.active_handle
        if active is not None:
            _render_job_details(active)
        return
    st.session_state["play_started_run_id"] = handle.run_id
    _clear_confirmation_state()
    st.success(
        "Playground started with its own confirmed budget. Polling reads "
        "local events only; no additional provider request is made."
    )
    _render_job_details(handle, cancel_label="Cancel playground")


def _render_playground_result(handle: JobHandle) -> None:
    result = handle.result
    if result is None:
        st.markdown(f"### Playground job `{handle.run_id}` — {handle.status}")
        if handle.status == JOB_STATUS_FAILED:
            st.error(
                f"Playground failed: "
                f"{handle.error or 'unknown worker error'}"
            )
        elif handle.status == JOB_STATUS_STOPPED:
            st.warning(
                "Playground stopped before producing a result. Charges "
                "already incurred are not reversed."
            )
        else:
            st.info("No playground result is available in this process.")
        return
    status = getattr(result, "status", handle.status)
    st.markdown(f"### Playground `{handle.run_id}` — {status}")
    st.info(
        "UNSCORED PLAYGROUND — no gold labels or benchmark metric were "
        "used. These findings are not a quality score and do not enter "
        "candidate selection or the held-out comparison."
    )
    st.markdown(
        f"- **Optimization run:** `{getattr(result, 'optimization_run_id', 'unknown')}`\n"
        f"- **Article and request text retained on disk:** "
        f"{'yes' if getattr(result, 'retained_content', False) else 'no'}\n"
        f"- **Additional provider calls by viewing:** 0"
    )
    snapshot = handle.budget_snapshot()
    st.markdown(
        f"- **Admitted calls:** {snapshot.admitted_calls} / "
        f"{snapshot.max_calls}\n"
        f"- **Cost status:** {_job_cost_status(snapshot)}\n"
        f"- **Estimated cost:** {snapshot.estimated_cost_usd} USD\n"
        f"- **Actual cost:** {snapshot.actual_cost_usd} USD\n"
        f"- **Total known cost:** {snapshot.total_cost_usd} USD"
    )
    for warning in getattr(result, "warnings", ()):
        st.warning(str(warning))
    for arm, label in (
        ("baseline", "Baseline"),
        ("optimized", "Optimized"),
    ):
        arm_result = getattr(result, arm, None)
        st.markdown(f"### {label} arm")
        if arm_result is None:
            st.error("No arm result is available.")
            continue
        st.markdown(
            f"- **Program ID:** `{arm_result.program_id}`\n"
            f"- **Status:** {arm_result.status}\n"
            f"- **Latency:** {_latency(arm_result.latency_ms)}\n"
            f"- **Recorded cost:** {_money_text(arm_result.cost_usd, arm_result.cost_status)}\n"
            f"- **Input / output tokens:** "
            f"{arm_result.input_tokens if arm_result.input_tokens is not None else 'unknown'} / "
            f"{arm_result.output_tokens if arm_result.output_tokens is not None else 'unknown'}\n"
            f"- **Request IDs:** "
            f"{', '.join(arm_result.request_ids) or 'none'}"
        )
        if arm_result.findings is not None:
            _render_findings(arm_result.findings)
        else:
            st.error(
                f"No parsed findings: {arm_result.error or 'invalid output'}"
            )
        for request in arm_result.requests:
            _render_sent_request(request)
    if handle.status == JOB_STATUS_FAILED:
        st.error(f"Playground failed: {handle.error or 'unknown error'}")
    elif handle.status == JOB_STATUS_STOPPED:
        st.warning(
            "Playground stopped at its confirmed budget. Outputs already "
            "recorded remain visible; no request is retried."
        )
    elif handle.status == JOB_STATUS_COMPLETED:
        st.success(
            "Playground completed without a gold-based score. Start a new "
            "run only with a fresh confirmation and budget review."
        )


def _render_optimization_screen(
    context: LiveContext, manager: JobManager
) -> None:
    st.subheader("Optimization")
    st.caption(
        "Live Experiment: one confirmation-gated GEPA optimization job. The "
        "worker uses a separate thread; this screen polls local events and "
        "ledger counters without sending provider requests."
    )
    active = manager.active_handle
    if active is not None:
        if active.job_kind != JOB_KIND_OPTIMIZATION:
            st.warning(
                "Another Live job is active. This screen will not start a "
                "second paid job."
            )
            _render_job_details(active)
            return
        _render_active_job_fragment(
            active, cancel_label="Cancel optimization"
        )
        return
    latest = manager.latest_handle_for(JOB_KIND_OPTIMIZATION)
    if (
        latest is not None
        and st.session_state.get("opt_started_run_id") == latest.run_id
    ):
        _render_finished_job(latest)
    _render_recent_jobs(manager)
    if context.settings is None:
        st.warning(
            "A confirmed Live optimization requires a configured "
            "OPENROUTER_API_KEY. No paid action is available."
        )
        return
    if context.development is None:
        st.info(
            "A completed baseline development run is required before "
            "optimization. This stage does not start baseline development "
            "from the UI."
        )
        return
    if context.development.manifest.status != "completed":
        st.warning(
            "The selected baseline development run is not completed. A paid "
            "optimization requires a completed Live development run."
        )
        return
    if context.development.manifest.origin != "live":
        st.warning(
            "The selected baseline development run is a mock/test record. "
            "A paid Live optimization requires a completed Live development "
            "run."
        )
        return
    if read_active_job_lock(RUNS_ROOT) is not None:
        st.warning(
            "Another app process holds the single active-job lock. This "
            "screen will not start a second optimization."
        )
        return
    _render_optimization_confirmation(context, manager)


def _render_recent_jobs(manager: JobManager) -> None:
    records = manager.recent_records(5)
    if not records:
        return
    with st.expander("Recent Live job records", expanded=False):
        for record in records:
            run_id = record.get("run_id", "unknown")
            status = record.get("status", "unknown")
            kind = record.get("job_kind", JOB_KIND_OPTIMIZATION)
            st.markdown(f"**`{run_id}` — {kind} — {status}**")
            st.text(
                f"Started: {record.get('started_at') or 'unknown'}\n"
                f"Finished: {record.get('finished_at') or 'still active'}\n"
                f"Baseline development run: "
                f"{record.get('baseline_development_run_id') or 'unknown'}\n"
                f"Error: {record.get('error') or 'none'}"
            )


@st.fragment(run_every="1s")
def _render_active_job_fragment(
    handle: JobHandle, *, cancel_label: str = "Cancel job"
) -> None:
    _render_job_details(handle, cancel_label=cancel_label)
    if not handle.active:
        st.rerun(scope="app")


def _render_finished_job(
    handle: JobHandle, *, cancel_label: str = "Cancel job"
) -> None:
    _render_job_details(handle, cancel_label=cancel_label)


def _render_job_details(
    handle: JobHandle, *, cancel_label: str = "Cancel job"
) -> None:
    snapshot = handle.budget_snapshot()
    status = handle.status
    st.markdown(f"### Job `{handle.run_id}` — {status}")
    st.markdown(
        f"- **Admitted calls:** {snapshot.admitted_calls} / "
        f"{snapshot.max_calls}\n"
        f"- **Remaining calls:** {snapshot.remaining_calls}\n"
        f"- **Delegated / succeeded / failed:** "
        f"{snapshot.delegated_calls} / {snapshot.succeeded_calls} / "
        f"{snapshot.failed_calls}\n"
        f"- **Cost status:** {_job_cost_status(snapshot)}\n"
        f"- **Estimated cost:** {snapshot.estimated_cost_usd} USD\n"
        f"- **Actual cost:** {snapshot.actual_cost_usd} USD\n"
        f"- **Unresolved reservation:** "
        f"{snapshot.unresolved_cost_usd} USD\n"
        f"- **Total known cost:** {snapshot.total_cost_usd} USD"
    )
    if snapshot.stop_reason is not None:
        st.markdown(f"- **Stop reason:** {snapshot.stop_reason}")
    events = handle.read_events(limit=25)
    if events:
        with st.expander("Persisted progress events", expanded=False):
            for event in events:
                st.text(_job_event_line(event))
    else:
        st.caption(
            "Waiting for the first persisted progress event. Polling reads "
            "local files only."
        )
    if handle.status == JOB_STATUS_RUNNING:
        st.warning(
            "Cancellation is cooperative before the next dispatch. An "
            "already-sent request can still finish and incur its reserved "
            "charge; cancellation never reverses charges already incurred."
        )
        if st.button(
            cancel_label,
            key=f"cancel-job-{handle.run_id}",
        ):
            if handle.request_cancel():
                st.warning(
                    "Cancellation requested. Waiting for any in-flight "
                    "request to finish before the job can stop."
                )
            else:
                st.info("The job is no longer running.")
    elif handle.status == JOB_STATUS_CANCELLING:
        st.info(
            "Cancellation is in progress. Waiting for any in-flight request "
            "to finish; already incurred charges are not reversed."
        )
    else:
        _render_job_result(handle)


def _render_job_result(handle: JobHandle) -> None:
    if handle.job_kind == JOB_KIND_PLAYGROUND:
        _render_playground_result(handle)
        return
    if handle.job_kind == JOB_KIND_BENCHMARK:
        _render_benchmark_job_result(handle)
        return
    result = handle.result
    manifest = getattr(result, "manifest", None)
    selected = getattr(result, "selected_program", None)
    warnings = getattr(result, "warnings", ())
    snapshot = handle.budget_snapshot()
    if manifest is not None:
        st.markdown(
            f"- **Recorded run ID:** `{manifest.run_id}`\n"
            f"- **Recorded status:** {manifest.status}\n"
            f"- **Recorded origin:** {manifest.origin}"
        )
    if selected is not None:
        st.markdown(
            f"- **Selected program ID:** `{selected.program_id}`"
        )
    for warning in warnings:
        st.warning(str(warning))
    if handle.status == JOB_STATUS_STOPPED and snapshot.stop_reason == "cancelled":
        st.warning(
            "Optimization stopped after cancellation. Charges already "
            "incurred are not reversed; a new run requires fresh "
            "confirmation and a new budget review."
        )
    elif handle.status == JOB_STATUS_INTERRUPTED:
        st.error(
            "This job was interrupted by a process restart. It was not "
            "resumed; start a fresh confirmed run if needed."
        )
    elif handle.status == JOB_STATUS_FAILED:
        st.error(
            f"Optimization failed: {handle.error or 'unknown worker error'}"
        )
    elif handle.status == JOB_STATUS_COMPLETED:
        st.success("Optimization completed with the recorded status above.")


def _render_benchmark_job_result(handle: JobHandle) -> None:
    result = handle.result
    manifest = getattr(result, "manifest", None)
    warnings = getattr(result, "warnings", ())
    summary = getattr(result, "summary", None)
    snapshot = handle.budget_snapshot()
    if manifest is not None:
        st.markdown(
            f"- **Recorded benchmark run:** `{manifest.run_id}`\n"
            f"- **Recorded status:** {manifest.status}\n"
            f"- **Recorded origin:** {manifest.origin}\n"
            f"- **Released as measured:** "
            f"{'yes' if getattr(manifest.benchmark, 'released', False) else 'no'}"
        )
    if isinstance(summary, Mapping):
        st.markdown(
            f"- **Verdict:** {summary.get('verdict', 'not recorded')}\n"
            f"- **Verdict reason:** "
            f"{summary.get('verdict_reason', 'not recorded')}"
        )
    for warning in warnings:
        st.warning(str(warning))
    if handle.status == JOB_STATUS_FAILED:
        st.error(
            f"Benchmark failed: {handle.error or 'unknown worker error'}"
        )
    elif handle.status == JOB_STATUS_STOPPED:
        st.warning(
            "Benchmark stopped before both arms completed. The partial run "
            "is incomplete and has no official comparison verdict."
        )
    elif handle.status == JOB_STATUS_COMPLETED:
        st.success(
            "Benchmark completed. Its recorded archive is selected below "
            "for the stored comparison."
        )
    if snapshot.stop_reason is not None:
        st.caption(f"Ledger stop reason: {snapshot.stop_reason}")


def _job_event_line(event: Mapping[str, object]) -> str:
    parts = [
        str(event.get("at", "")),
        str(event.get("event", "event")),
    ]
    for key in (
        "status",
        "case_id",
        "candidate_index",
        "program_id",
        "stop_reason",
        "admitted_calls",
        "train_cases",
        "validation_cases",
    ):
        if key in event:
            parts.append(f"{key}={event[key]}")
    return " · ".join(part for part in parts if part)


def _job_cost_status(snapshot: BudgetSnapshot) -> str:
    if snapshot.pending_cost_calls or snapshot.unresolved_cost_usd > 0:
        return "pending"
    if snapshot.actual_cost_usd > 0:
        return "actual"
    if snapshot.estimated_cost_usd > 0:
        return "estimated"
    if snapshot.committed_cost_usd > 0:
        return "committed"
    return "none"


def _render_optimization_confirmation(
    context: LiveContext, manager: JobManager
) -> None:
    settings = context.settings
    development = context.development
    assert settings is not None
    assert development is not None
    try:
        split = load_optimization_cases(
            DATA_DIR / "train.jsonl",
            DATA_DIR / "validation.jsonl",
        )
    except Exception as exc:  # noqa: BLE001 - explicit UI gap
        st.error(f"Optimization inputs are unavailable: {exc}")
        return
    train = tuple(split["train"])
    validation = tuple(development.cases)
    task_route = settings.task_route()
    reflection_route = settings.reflection_route()
    limits = settings.optimization_budget()
    headroom = (
        settings.max_optimization_calls
        - settings.max_optimization_metric_calls
    )
    config_key = "|".join(
        [
            str(manager.mode_epoch),
            str(manager.confirmation_generation),
            development.manifest.run_id,
            settings.task_model,
            settings.task_provider,
            settings.reflection_model,
            settings.reflection_provider,
            str(settings.max_optimization_calls),
            str(settings.max_optimization_cost_usd),
            str(settings.max_optimization_metric_calls),
            str(settings.temperature),
            str(settings.max_output_tokens),
            str(settings.reflection_temperature),
            str(settings.reflection_max_output_tokens),
            str(settings.request_timeout_seconds),
            str(len(train)),
            str(len(validation)),
        ]
    )
    if (
        st.session_state.get("opt_confirm_key") != config_key
        or not st.session_state.get("opt_confirm_token")
        or not st.session_state.get("opt_confirm_run_id")
        or st.session_state.get("opt_confirm_epoch")
        != manager.mode_epoch
        or st.session_state.get("opt_confirm_generation")
        != manager.confirmation_generation
    ):
        _clear_confirmation_state()
        st.session_state["opt_confirm_key"] = config_key
        st.session_state["opt_confirm_token"] = secrets.token_hex(16)
        st.session_state["opt_confirm_run_id"] = _new_job_run_id()
        st.session_state["opt_confirm_epoch"] = manager.mode_epoch
        st.session_state["opt_confirm_generation"] = (
            manager.confirmation_generation
        )
    token = str(st.session_state["opt_confirm_token"])
    run_id = str(st.session_state["opt_confirm_run_id"])
    epoch = int(st.session_state["opt_confirm_epoch"])
    generation = int(st.session_state["opt_confirm_generation"])

    st.markdown(
        f"- **Baseline development run:** `{development.manifest.run_id}`\n"
        f"- **Train + validation cases:** {len(train)} + "
        f"{len(validation)} (validation is the development snapshot)\n"
        f"- **Sealed test split:** not read by optimization\n"
        f"- **Task model/provider:** `{task_route.model}` / "
        f"`{task_route.provider}` at "
        f"{task_route.input_usd_per_million}/"
        f"{task_route.output_usd_per_million} USD per 1M tokens; "
        f"temperature {settings.temperature}, output cap "
        f"{settings.max_output_tokens}\n"
        f"- **Reflection model/provider:** `{reflection_route.model}` / "
        f"`{reflection_route.provider}` at "
        f"{reflection_route.input_usd_per_million}/"
        f"{reflection_route.output_usd_per_million} USD per 1M tokens; "
        f"temperature {settings.reflection_temperature}, output cap "
        f"{settings.reflection_max_output_tokens}\n"
        f"- **Request timeout:** {settings.request_timeout_seconds} s\n"
        f"- **Confirmed optimization budget:** {limits.max_calls} calls / "
        f"{limits.max_cost_usd} USD\n"
        f"- **GEPA metric-call target:** "
        f"{settings.max_optimization_metric_calls} "
        f"({headroom} call headroom)\n"
        f"- **Fresh job run ID:** `{run_id}`"
    )
    confirmed = st.checkbox(
        "I confirm this optimization budget and authorize real provider "
        "requests. I understand cancellation cannot reverse charges "
        "already incurred.",
        key=f"opt_confirm_checkbox_{token}",
    )
    with st.form("optimization_confirmation_form", clear_on_submit=False):
        st.markdown("#### Confirm the paid Live optimization")
        st.markdown(
            "This optimization action can make paid provider calls. It "
            "starts from the selected completed development run and the "
            "frozen train/validation inputs. No test case is loaded."
        )
        submitted = st.form_submit_button("Start confirmed optimization")
    if not submitted:
        return
    if not confirmed:
        st.error(
            "Explicit budget confirmation is required; no request was made."
        )
        return
    spec = OptimizationJobSpec(
        mode=MODE_LIVE,
        run_id=run_id,
        token=token,
        mode_epoch=epoch,
        confirmation_generation=generation,
        settings=settings,
        baseline_development_run_id=development.manifest.run_id,
        train_cases=train,
        validation_cases=validation,
        rules_text=development.rules_text,
        runs_root=RUNS_ROOT,
        programs_root=PROGRAMS_ROOT,
    )
    try:
        handle = manager.start_job(spec)
    except JobGateError as exc:
        _clear_confirmation_state()
        st.error(f"Optimization was not started: {exc}")
        active = manager.active_handle
        if active is not None:
            _render_job_details(active)
        return
    st.session_state["opt_started_run_id"] = handle.run_id
    _clear_confirmation_state()
    st.success(
        "Optimization started. Progress polls local events and the shared "
        "per-job ledger; no additional provider request is made by polling."
    )
    _render_job_details(handle)


def _render_live_screen(
    screen: str, context: LiveContext, manager: JobManager | None
) -> None:
    if (
        context.development is not None
        and context.development.manifest.origin == "mock_test"
    ):
        st.warning(
            "The selected Live run is a mock/test record. Its numbers are "
            "not a measured Live experiment."
        )
    if screen == "Overview":
        _render_live_overview(context)
    elif screen == "Baseline":
        if context.development is None:
            st.subheader("Baseline")
            if context.development_error:
                st.error(
                    "The selected development run cannot be read: "
                    f"{context.development_error}"
                )
            else:
                st.info(
                    "No baseline development run recorded. This read-only "
                    "stage cannot start one."
                )
            st.caption(
                "Held-out test split remains sealed and is not displayed in "
                "Live read-only screens."
            )
            return
        _render_baseline_screen(_baseline_view_from_live(context))
    elif screen == "Prompt Inspector":
        if context.development is None:
            st.subheader("Prompt Inspector")
            if context.development_error:
                st.error(
                    "The selected development run cannot be read: "
                    f"{context.development_error}"
                )
            else:
                st.info(
                    "No baseline development run recorded; there is no "
                    "server-local baseline program or request trace to "
                    "inspect."
                )
            return
        _render_inspector_screen(_inspector_view_from_live(context))
    elif screen == "Optimization":
        _render_optimization_screen(
            context, manager or _get_job_manager()
        )
    elif screen == "Comparison":
        _render_live_comparison(
            context, manager or _get_job_manager()
        )
    else:
        _render_live_playground(
            context, manager or _get_job_manager()
        )


def _render_static_overview(mode: str) -> None:
    st.markdown("### What this lab does")
    st.markdown(
        "One DSPy predictor reviews Czech precious-metals articles against "
        "supplied reference facts, source excerpts, and frozen editorial "
        "rules. It returns typed findings; it does not rewrite articles. "
        "A frozen zero-shot baseline is compared with a genuine GEPA "
        "instruction-optimized copy under the same task model, schema, "
        "rules, references, adapter, and inference settings."
    )
    st.markdown("### Six allowed error codes")
    for code in ALLOWED_CODES:
        st.markdown(f"- **{code}** — {CODE_DESCRIPTIONS[code]}")
    st.markdown(
        f"**Experiment mode:** {mode}. Replay is the safe default and makes "
        "no provider request."
    )
    st.caption(
        "The frozen v1 dataset has 50 Czech cases in 30/10/10 "
        "train/validation/test splits. The test split is sealed until a "
        "completed paired benchmark archive is presented."
    )


def _render_replay_overview(archive: ReplayArchive) -> None:
    st.subheader("Overview")
    st.caption(
        "Archived provenance and original measurements. Replay is not a new "
        "experiment."
    )
    summary = archive.benchmark_summary or {}
    baseline_id = archive.benchmark.links.baseline_program_id
    optimized_id = archive.benchmark.links.optimized_program_id
    models = sorted({request.model for request in archive.requests.values()})
    providers = sorted(
        {
            request.provider
            for request in archive.requests.values()
            if request.provider is not None
        }
    )
    candidate_records, candidate_error = _candidate_records(
        archive.candidate_stats
    )
    optimizer_label = (
        candidate_error
        if candidate_error
        else (
            "genuine GEPA; "
            f"{len(candidate_records)} recorded candidate(s)"
        )
    )
    st.markdown(
        f"- **Mode:** Replay Experiment\n"
        f"- **Experiment status:** {archive.benchmark.status}\n"
        f"- **Recorded verdict:** {summary.get('verdict', 'not archived')}\n"
        f"- **Split:** test (sealed benchmark snapshot)\n"
        f"- **Task model(s):** {', '.join(models) or 'not archived'}\n"
        f"- **Provider(s):** {', '.join(providers) or 'not archived'}\n"
        f"- **Optimizer:** {optimizer_label}\n"
        f"- **Baseline program:** `{baseline_id}`\n"
        f"- **Selected program:** `{optimized_id}`\n"
        f"- **Test cases in snapshot:** {archive.snapshot.case_count}\n"
        f"- **Rules version:** {archive.snapshot.rules_version}\n"
        f"- **Additional API calls:** 0\n"
        f"- **Additional API cost:** 0.00 USD"
    )
    if summary.get("verdict_reason"):
        st.text(str(summary["verdict_reason"]))
    st.markdown("### Historical original cost (not additional spend)")
    _render_historical_costs(summary)
    st.markdown(
        f"- **Measured archive:** "
        f"{'yes, released Live benchmark' if archive.is_measured else 'no'}"
    )
    st.markdown("### Six allowed error codes")
    for code in ALLOWED_CODES:
        st.markdown(f"- **{code}** — {CODE_DESCRIPTIONS[code]}")


def _render_live_overview(context: LiveContext) -> None:
    st.subheader("Overview")
    st.caption(
        "Live workspace overview. It shows server-local recorded runs and "
        "frozen configuration; paid actions live on their own screens."
    )
    settings = context.settings
    task_model = settings.task_model if settings else DEFAULT_TASK_MODEL
    reflection_model = (
        settings.reflection_model if settings else DEFAULT_REFLECTION_MODEL
    )
    task_provider = settings.task_provider if settings else "default route"
    metric_calls = (
        settings.max_optimization_metric_calls
        if settings
        else DEFAULT_MAX_OPTIMIZATION_METRIC_CALLS
    )
    active_program = "frozen baseline definition (not yet persisted)"
    if context.development is not None:
        active_program = (
            context.development.manifest.links.baseline_program_id
            or "unavailable"
        )
    latest_development = _latest_run_label(context.development_entries)
    latest_optimization = _latest_run_label(context.optimization_entries)
    latest_benchmark = _latest_run_label(context.benchmark_entries)
    st.markdown(
        f"- **Mode:** Live Experiment (three separately confirmed paid actions)\n"
        f"- **Active baseline program:** `{active_program}`\n"
        f"- **Task model:** `{task_model}`\n"
        f"- **Task provider:** `{task_provider}`\n"
        f"- **Reflection model:** `{reflection_model}`\n"
        f"- **Optimizer:** genuine GEPA; metric-call target {metric_calls}\n"
        f"- **Latest development run:** {latest_development}\n"
        f"- **Latest optimization run:** {latest_optimization}\n"
        f"- **Latest benchmark run:** {latest_benchmark}\n"
        f"- **Provider requests started by this screen:** 0"
    )
    _render_live_dataset_status()
    st.markdown("### Six allowed error codes")
    for code in ALLOWED_CODES:
        st.markdown(f"- **{code}** — {CODE_DESCRIPTIONS[code]}")


def _render_live_dataset_status() -> None:
    st.markdown("### Frozen dataset and review status")
    try:
        manifest = load_split_manifest(DATA_DIR / "split_manifest.json")
    except Exception as exc:  # noqa: BLE001 - display an explicit gap
        st.warning(f"Split manifest is unavailable: {exc}")
        manifest = None
    if isinstance(manifest, Mapping):
        splits = manifest.get("splits")
        if isinstance(splits, Mapping):
            counts = []
            for name in ("train", "validation", "test"):
                split = splits.get(name)
                count = (
                    split.get("article_count")
                    if isinstance(split, Mapping)
                    else None
                )
                counts.append(f"{name}={count}")
            st.markdown("- **Split counts:** " + ", ".join(counts))
        st.markdown(
            f"- **Combined dataset hash:** "
            f"`{manifest.get('combined_dataset_sha256', 'unavailable')}`\n"
            f"- **Test sealed:** {manifest.get('test_sealed', 'unknown')}"
        )
    try:
        reviews = load_review_audit(DATA_DIR / "reviews.jsonl")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Review audit is unavailable: {exc}")
        reviews = []
    if reviews:
        automated = sum(
            1 for review in reviews if review.reviewer_kind == "automated"
        )
        human = len(reviews) - automated
        approved = sum(
            1
            for review in reviews
            if review.author_decision == "approved"
            and review.reviewer_decision == "approved"
            and review.reconciled
        )
        st.markdown(
            f"- **Gold review audit:** {approved}/{len(reviews)} reconciled "
            f"and approved\n"
            f"- **Reviewer kinds:** {automated} automated, {human} human"
        )
        if human == 0:
            st.warning(
                "Human gold sign-off is still required before a final "
                "held-out run."
            )
    st.caption(
        "The held-out test split is sealed. Live read-only screens never "
        "load test cases or expose held-out outputs."
    )


def _baseline_view_from_replay(archive: ReplayArchive) -> BaselineView:
    development = archive.development
    baseline_id = development.links.baseline_program_id
    program = (
        archive.programs.get(baseline_id)
        if baseline_id is not None
        else None
    )
    warnings = list(development.warnings)
    if archive.development_summary is None:
        warnings.append(
            "Baseline development summary is not archived; aggregate "
            "metrics and cost are unavailable."
        )
    return BaselineView(
        manifest=None if program is None else program.manifest,
        run_manifest=development,
        summary=archive.development_summary,
        cases=archive.development_cases,
        outcomes=archive.development_outcomes,
        warnings=tuple(dict.fromkeys(warnings)),
        historical=True,
    )


def _baseline_view_from_live(context: LiveContext) -> BaselineView:
    run = context.development
    assert run is not None
    program_id = run.manifest.links.baseline_program_id
    program: ProgramSnapshot | None = None
    warnings = list(run.warnings)
    if program_id is not None:
        program = run.programs.get(program_id)
        if program is None:
            warnings.append(
                "Baseline program artifact is not part of this run."
            )
    else:
        warnings.append("Baseline program ID is not recorded.")
    return BaselineView(
        manifest=None if program is None else program.manifest,
        run_manifest=run.manifest,
        summary=run.summary,
        cases=run.cases,
        outcomes=run.outcomes,
        warnings=tuple(dict.fromkeys(warnings)),
        historical=False,
    )


def _render_baseline_screen(view: BaselineView) -> None:
    st.subheader("Baseline")
    if view.historical:
        st.caption(
            "Replay Experiment: original recorded baseline development "
            "results. No new inference or scoring is performed."
        )
    else:
        st.caption(
            "Live workspace: server-local recorded development results, "
            "read-only. No paid action is available in this stage."
        )
    run = view.run_manifest
    st.markdown(
        f"- **Recorded run:** `{run.run_id}` — {run.status} / {run.origin}\n"
        "- **Split:** Validation (adaptive development split) — not held-out\n"
        "- **Held-out test:** sealed and not displayed on this screen"
    )
    manifest = view.manifest
    if manifest is not None:
        st.markdown(
            f"- **Program ID:** `{manifest.program_id}`\n"
            f"- **Origin / baseline:** {manifest.origin} / {manifest.baseline}\n"
            f"- **Rules version:** {manifest.rules_version}\n"
            f"- **Demos:** {len(manifest.demos)} (zero)\n"
            f"- **Adapter:** {manifest.adapter} v{manifest.adapter_version}\n"
            f"- **Program identity hash:** `{manifest.identity_sha256}`"
        )
        st.markdown("#### Frozen baseline instructions")
        st.code(manifest.instructions, language="text")
    else:
        st.warning(
            "The baseline program artifact is unavailable; instructions and "
            "identity cannot be displayed."
        )
        st.markdown("#### Frozen baseline instructions")
        st.code(load_baseline_instructions(), language="text")
    for warning in view.warnings:
        st.warning(warning)
    summary = view.summary or {}
    corpus = summary.get("corpus") if isinstance(summary, Mapping) else None
    st.markdown("### Development corpus summary")
    if isinstance(corpus, Mapping):
        _render_corpus_summary(corpus)
    else:
        st.warning(
            "Aggregate corpus metrics are not archived for this run. Raw "
            "per-case counts below are shown without recomputation."
        )
        _render_corpus_counts_from_outcomes(view.outcomes)
    budget = summary.get("budget") if isinstance(summary, Mapping) else None
    if isinstance(budget, Mapping):
        st.markdown("#### Recorded development budget/cost")
        st.markdown(
            f"- **Admitted calls:** {budget.get('admitted_calls', 'N/A')} / "
            f"{budget.get('max_calls', 'N/A')}\n"
            f"- **Cost status:** {_cost_status_from_payload(budget)}\n"
            f"- **Known total cost:** "
            f"{budget.get('total_cost_usd', 'unavailable')} USD\n"
            f"- **Stop reason:** {budget.get('stop_reason', 'none')}"
        )
    st.markdown("### Per-case development outcomes")
    _render_outcomes(view.cases, view.outcomes)


def _render_corpus_summary(corpus: Mapping[str, object]) -> None:
    st.markdown(
        f"- **Articles:** {corpus.get('article_count', 'N/A')} "
        f"(valid {corpus.get('valid_count', 'N/A')}, invalid "
        f"{corpus.get('invalid_count', 'N/A')})\n"
        f"- **Parsing success:** {_percent(corpus.get('parsing_success'))}\n"
        f"- **TP / FP / FN:** {corpus.get('tp', 'N/A')} / "
        f"{corpus.get('fp', 'N/A')} / {corpus.get('fn', 'N/A')}\n"
        f"- **Precision:** {_percent(corpus.get('precision'))}\n"
        f"- **Recall:** {_percent(corpus.get('recall'))}\n"
        f"- **Micro-F1:** {_percent(corpus.get('micro_f1'))}\n"
        f"- **Macro-F1:** {_percent(corpus.get('macro_f1'))}\n"
        f"- **Exact match:** {corpus.get('exact_match_count', 'N/A')} / "
        f"{corpus.get('article_count', 'N/A')} "
        f"({_percent(corpus.get('exact_match_rate'))})\n"
        f"- **Mean case reward:** "
        f"{_number(corpus.get('mean_reward'))} (separate from micro-F1)\n"
        f"- **Clean articles falsely flagged:** "
        f"{corpus.get('clean_articles_falsely_flagged', 'N/A')} / "
        f"{corpus.get('clean_article_count', 'N/A')}\n"
        f"- **Parse / request errors:** "
        f"{corpus.get('parse_error_count', 'N/A')} / "
        f"{corpus.get('request_error_count', 'N/A')}\n"
        f"- **Evidence failures / duplicates:** "
        f"{corpus.get('evidence_failure_count', 'N/A')} / "
        f"{corpus.get('duplicate_prediction_count', 'N/A')}"
    )
    per_category = corpus.get("per_category")
    if isinstance(per_category, list) and per_category:
        st.markdown("#### Per-category counts")
        for item in per_category:
            if not isinstance(item, Mapping):
                continue
            st.markdown(
                f"- **{item.get('code', 'unknown')}:** "
                f"TP {item.get('tp', 'N/A')}, FP {item.get('fp', 'N/A')}, "
                f"FN {item.get('fn', 'N/A')}, F1 "
                f"{_percent(item.get('f1'))}"
            )


def _render_corpus_counts_from_outcomes(
    outcomes: Sequence[RecordedOutcome],
) -> None:
    tp = fp = fn = exact = valid = 0
    for outcome in outcomes:
        metric = outcome.metric or {}
        tp += _int_value(metric.get("tp"))
        fp += _int_value(metric.get("fp"))
        fn += _int_value(metric.get("fn"))
        exact += 1 if metric.get("exact_match") is True else 0
        valid += 1 if metric.get("valid") is True else 0
    st.markdown(
        f"- **Articles:** {len(outcomes)} (valid {valid}, invalid "
        f"{len(outcomes) - valid})\n"
        f"- **TP / FP / FN:** {tp} / {fp} / {fn}\n"
        f"- **Exact match:** {exact} / {len(outcomes)}"
    )


def _render_outcomes(
    cases: Sequence[CaseRecord],
    outcomes: Sequence[RecordedOutcome],
) -> None:
    by_case = {outcome.case_id: outcome for outcome in outcomes}
    for case in cases:
        outcome = by_case.get(case.case_id)
        status = "no outcome" if outcome is None else outcome.status
        with st.expander(f"{case.case_id} — {status}"):
            st.markdown("**Article text**")
            st.text(canonical_document(case))
            st.markdown("**Model findings**")
            if outcome is None:
                st.text("No recorded outcome for this case.")
            elif outcome.status == "ok":
                _render_findings(outcome.findings or [])
            else:
                st.text("No parsed findings: this is an invalid output.")
            st.markdown("**Gold findings**")
            _render_gold_findings(case)
            if outcome is not None:
                st.markdown("**Parsing, matching, and error status**")
                _render_outcome_metric(outcome)
                st.markdown("**Timing and recorded cost**")
                st.markdown(
                    f"- **Latency:** {_latency(outcome.latency_ms)}\n"
                    f"- **Recorded cost:** "
                    f"{_outcome_cost(outcome)}\n"
                    f"- **Request IDs:** "
                    f"{', '.join(outcome.request_ids) or 'none'}"
                )


def _render_findings(findings: Sequence[object]) -> None:
    if not findings:
        st.text("(no findings; a valid empty list)")
        return
    for index, finding in enumerate(findings, start=1):
        code = getattr(finding, "code", "unknown")
        evidence = getattr(finding, "evidence", "")
        explanation = getattr(finding, "explanation", "")
        st.markdown(f"**Finding {index} — {code}**")
        st.text(f"Evidence: {evidence}\nExplanation: {explanation}")


def _render_gold_findings(case: CaseRecord) -> None:
    if not case.gold_findings:
        st.text("(none; clean case)")
        return
    for gold in case.gold_findings:
        st.markdown(f"**{gold.issue_id} — {gold.code}**")
        st.text(f"Evidence: {gold.evidence}\nExplanation: {gold.explanation}")


def _render_outcome_metric(outcome: RecordedOutcome) -> None:
    metric = outcome.metric
    if not isinstance(metric, Mapping):
        st.warning(
            "Per-case metric details are not archived; validity is shown "
            "only through the outcome status."
        )
        if outcome.status != "ok":
            st.text(f"Error: {outcome.error or 'unknown error'}")
        return
    st.markdown(
        f"- **Valid output:** {metric.get('valid', 'N/A')}\n"
        f"- **Error kind:** {metric.get('error_kind') or 'none'}\n"
        f"- **Exact match:** {metric.get('exact_match', 'N/A')}\n"
        f"- **Reward:** {_number(metric.get('reward'))}\n"
        f"- **TP / FP / FN:** {metric.get('tp', 'N/A')} / "
        f"{metric.get('fp', 'N/A')} / {metric.get('fn', 'N/A')}\n"
        f"- **F1:** {_percent(metric.get('f1'))}\n"
        f"- **Predictions / gold:** {metric.get('prediction_count', 'N/A')} / "
        f"{metric.get('gold_count', 'N/A')}\n"
        f"- **Matched gold issues:** "
        f"{', '.join(metric.get('matched_issue_ids') or []) or 'none'}"
    )
    if outcome.error:
        st.text(f"Outcome error: {outcome.error}")
    if metric.get("error"):
        st.text(f"Metric error: {metric['error']}")
    false_positives = metric.get("false_positives")
    if isinstance(false_positives, list) and false_positives:
        st.markdown("**False positives**")
        for item in false_positives:
            if not isinstance(item, Mapping):
                continue
            st.text(
                f"{item.get('code', 'unknown')}: "
                f"{item.get('evidence', '')}\n"
                f"Reason: {item.get('reason', 'unrecorded')}"
            )
    missed = metric.get("missed_gold")
    if isinstance(missed, list) and missed:
        st.markdown("**Missed gold issues**")
        for item in missed:
            if not isinstance(item, Mapping):
                continue
            st.text(
                f"{item.get('issue_id', 'unknown')} — "
                f"{item.get('code', 'unknown')}: "
                f"{item.get('evidence', '')}"
            )


def _inspector_view_from_replay(archive: ReplayArchive) -> InspectorView:
    baseline_id = archive.benchmark.links.baseline_program_id
    optimized_id = archive.benchmark.links.optimized_program_id
    baseline = (
        archive.programs.get(baseline_id)
        if baseline_id is not None
        else None
    )
    optimized = (
        archive.programs.get(optimized_id)
        if optimized_id is not None
        else None
    )
    warnings = list(archive.optimization.warnings)
    if archive.candidate_stats is None:
        warnings.append(
            "Candidate statistics are not archived; only the recorded "
            "selected program can be inspected."
        )
    assert archive.benchmark.benchmark is not None
    return InspectorView(
        baseline=baseline,
        optimized=optimized,
        cases=archive.cases,
        outcomes=archive.outcomes,
        task_requests=archive.requests,
        optimization_requests=archive.optimization_requests,
        candidate_stats=archive.candidate_stats,
        rules_text=archive.rules_text,
        split_label="held-out test (recorded benchmark snapshot)",
        release_label=(
            "released"
            if archive.benchmark.benchmark.released
            else "not released"
        ),
        measured=archive.is_measured,
        historical=True,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _inspector_view_from_live(context: LiveContext) -> InspectorView:
    development = context.development
    assert development is not None
    baseline_id = development.manifest.links.baseline_program_id
    baseline: ProgramSnapshot | None = None
    warnings = list(development.warnings)
    if baseline_id is not None:
        baseline = development.programs.get(baseline_id)
        if baseline is None:
            warnings.append(
                "Baseline program artifact is not part of this run."
            )
    linked_entries = [
        entry
        for entry in context.optimization_entries
        if entry.manifest is not None
        and entry.manifest.links.baseline_development_run_id
        == development.manifest.run_id
    ]
    optimized: ProgramSnapshot | None = None
    optimization_requests: Mapping[str, RecordedRequest] = {}
    candidate_stats: Mapping[str, object] | None = None
    if not linked_entries:
        warnings.append(
            "No recorded optimization run is linked to this development "
            "run; only the baseline program is available."
        )
    else:
        labels = [_run_entry_label(entry) for entry in linked_entries]
        selected_label = st.selectbox(
            "Recorded optimization run",
            labels,
            index=0,
            key="live_optimization_run",
        )
        try:
            opt_entry = linked_entries[labels.index(selected_label)]
        except ValueError:
            opt_entry = linked_entries[0]
        if opt_entry.manifest is None:
            warnings.append(
                "Selected optimization run manifest is unavailable."
            )
        else:
            try:
                optimization = open_optimization_run(
                    opt_entry.run_dir,
                    runs_root=RUNS_ROOT,
                    programs_root=PROGRAMS_ROOT,
                )
            except ArtifactError as exc:
                warnings.append(
                    f"Optimization run cannot be read: {exc}"
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(
                    f"Optimization run cannot be read: "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                warnings.extend(optimization.warnings)
                optimization_requests = optimization.requests
                candidate_stats = optimization.candidate_stats
                optimized_id = (
                    optimization.manifest.links.optimized_program_id
                )
                if optimized_id is not None:
                    optimized = optimization.programs.get(optimized_id)
    return InspectorView(
        baseline=baseline,
        optimized=optimized,
        cases=development.cases,
        outcomes=development.outcomes,
        task_requests=development.requests,
        optimization_requests=optimization_requests,
        candidate_stats=candidate_stats,
        rules_text=development.rules_text,
        split_label=(
            "validation (adaptive development split) — not held-out"
        ),
        release_label="not a benchmark archive",
        measured=False,
        historical=False,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _render_inspector_screen(view: InspectorView) -> None:
    st.subheader("Prompt Inspector")
    if view.historical:
        st.caption(
            "Replay Experiment: instructions, schema, and sent requests from "
            "the archive. Diffs are computed only from archived text; no "
            "prompt is regenerated and labeled as sent."
        )
    else:
        st.caption(
            "Live workspace: server-local recorded instructions and "
            "requests, read-only. No paid action is available."
        )
    st.markdown(
        f"- **Articles inspected:** {view.split_label}\n"
        f"- **Release status:** {view.release_label}\n"
        f"- **Measured experiment:** {'yes' if view.measured else 'no'}"
    )
    if view.historical and not view.measured:
        st.warning(
            "Held-out test content comes from a non-measured archive. It is "
            "shown for inspection only and is not a released result."
        )
    for warning in view.warnings:
        st.warning(warning)
    baseline = view.baseline
    optimized = view.optimized
    if baseline is None:
        st.error(
            "The baseline program artifact is unavailable; the inspector "
            "cannot show a trustworthy instruction or schema snapshot."
        )
        return
    selected = optimized or baseline
    st.markdown("### Program identity")
    st.markdown(
        f"- **Baseline program:** `{baseline.manifest.program_id}`\n"
        f"- **Selected program:** `{selected.manifest.program_id}`\n"
        f"- **Selected parent:** "
        f"`{selected.manifest.parent_program_id or 'none'}`\n"
        f"- **Origin / baseline:** {selected.manifest.origin} / "
        f"{selected.manifest.baseline}\n"
        f"- **Rules / prompt version:** "
        f"{selected.manifest.rules_version} / "
        f"{selected.manifest.prompt_version}\n"
        f"- **Adapter:** {selected.manifest.adapter} "
        f"v{selected.manifest.adapter_version}\n"
        f"- **Demos:** {len(selected.manifest.demos)} (zero)"
    )
    if not view.historical:
        st.caption(
            "Demos are zero by the frozen program contract; this stage does "
            "not store demonstrations."
        )
    st.markdown("### Instructions and diff")
    st.markdown("**Baseline instructions**")
    st.code(baseline.manifest.instructions, language="text")
    candidate_records, candidate_error = _candidate_records(
        view.candidate_stats
    )
    if candidate_error:
        st.warning(candidate_error)
    inspect_selected = selected.manifest.instructions
    if candidate_records:
        options = ["Selected program (archived)"]
        for record in candidate_records:
            score = record.get("val_aggregate_score")
            marker = " — recorded selection" if record.get("selected") else ""
            options.append(
                f"Candidate {record['candidate_index']} — validation score "
                f"{_number(score)}{marker}"
            )
        choice = st.selectbox(
            "Instructions to inspect",
            options,
            index=0,
            key="inspector_candidate",
        )
        if choice != options[0]:
            chosen_index = options.index(choice) - 1
            inspect_selected = str(
                candidate_records[chosen_index]["instructions"]
            )
    st.markdown("**Selected instructions**")
    st.code(inspect_selected, language="text")
    if inspect_selected == baseline.manifest.instructions:
        st.info(
            "No instruction change selected. The recorded selection is the "
            "unchanged baseline."
        )
    else:
        diff = _unified_diff(
            baseline.manifest.instructions,
            inspect_selected,
            fromfile="baseline",
            tofile="selected",
        )
        st.markdown("**Unified diff**")
        st.code(diff, language="diff")
    st.markdown(
        f"**Demos:** {len(selected.manifest.demos)} (zero) in the selected "
        "program artifact. GEPA changes instructions; this MVP stores no "
        "demonstrations, and candidate records contain instructions only."
    )
    st.markdown("### Full signature and schema")
    st.markdown("**Input/output signature schema**")
    st.code(
        json.dumps(
            baseline.manifest.signature_schema,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        language="json",
    )
    st.markdown("**Field descriptions**")
    st.code(
        json.dumps(
            baseline.manifest.signature_fields,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        language="json",
    )
    if not _same_schema(
        baseline.manifest, selected.manifest
    ):
        st.warning(
            "The selected program schema differs from the baseline schema. "
            "This must not occur in a valid experiment."
        )
    st.markdown("### Captured task requests")
    if not view.cases:
        st.info("No archived cases are available for request inspection.")
        return
    case_labels = [
        f"{case.case_id} — {case.title}" for case in view.cases
    ]
    case_label = st.selectbox(
        "Article for request inspection",
        case_labels,
        index=0,
        key="inspector_case",
    )
    case = view.cases[case_labels.index(case_label)]
    outcome_by_phase: dict[str, RecordedOutcome] = {}
    for outcome in view.outcomes:
        if outcome.case_id == case.case_id:
            outcome_by_phase[outcome.phase] = outcome
    for phase in sorted(outcome_by_phase):
        outcome = outcome_by_phase[phase]
        st.markdown(f"**{phase.capitalize()} arm — case {case.case_id}**")
        if not outcome.request_ids:
            st.info(
                "No sent request trace is archived for this case and arm. "
                "Nothing is regenerated."
            )
        for request_id in outcome.request_ids:
            request = view.task_requests.get(request_id)
            if request is None:
                st.warning(
                    f"Referenced request `{request_id}` is not archived."
                )
                continue
            _render_sent_request(request)
    _render_optimization_requests(view.optimization_requests)
    _render_preview(case, inspect_selected, view.rules_text)


def _render_sent_request(request: RecordedRequest) -> None:
    st.markdown("**SENT REQUEST — captured at the LM boundary**")
    try:
        trace = format_sent_trace(request)
    except ArtifactError as exc:
        st.error(f"Recorded request is not displayable: {exc}")
        return
    st.markdown(
        f"- **Source:** sent\n"
        f"- **Phase / status:** {trace.phase} / {trace.status}\n"
        f"- **Model / provider:** {trace.model} / "
        f"{trace.provider or 'not recorded'}\n"
        f"- **Generation ID:** {trace.generation_id or 'not recorded'}\n"
        f"- **Cost status:** {trace.cost_status or 'not recorded'}\n"
        f"- **Estimated cost:** "
        f"{trace.estimated_cost_usd or 'not recorded'} USD\n"
        f"- **Reported cost:** "
        f"{trace.reported_cost_usd or 'not recorded'} USD\n"
        f"- **Finish reason:** {trace.finish_reason or 'not recorded'}"
    )
    st.caption(
        "This is the client-side DSPy/LiteLLM request captured before "
        "dispatch; provider internals may still transform it."
    )
    for message in trace.messages:
        role = str(message.get("role", "unknown"))
        st.markdown(f"**{role}**")
        st.text(_message_content(message.get("content")))
    st.markdown("**Request parameters**")
    st.code(
        json.dumps(
            trace.request_kwargs,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        language="json",
    )
    st.markdown("**Usage**")
    st.code(
        json.dumps(
            trace.usage,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        language="json",
    )
    if trace.response_text is not None:
        st.markdown("**Raw recorded response text**")
        st.code(trace.response_text, language="text")


def _render_optimization_requests(
    requests: Mapping[str, RecordedRequest],
) -> None:
    if not requests:
        st.info(
            "No optimization request trace is archived. Task and reflection "
            "messages are unavailable; nothing is regenerated."
        )
        return
    ordered = sorted(requests.values(), key=lambda item: item.sequence)
    labels = [
        f"#{request.sequence} — {request.phase} — {request.status}"
        for request in ordered
    ]
    reflection_positions = [
        index
        for index, request in enumerate(ordered)
        if request.phase == "reflection"
    ]
    default_index = (
        reflection_positions[0] if reflection_positions else 0
    )
    st.markdown("### Recorded optimization requests")
    st.caption(
        "Historical task/reflection requests. The archive does not always "
        "associate a reflection request with one candidate."
    )
    label = st.selectbox(
        "Recorded optimization request",
        labels,
        index=default_index,
        key="inspector_optimization_request",
    )
    _render_sent_request(ordered[labels.index(label)])


def _render_preview(
    case: CaseRecord,
    instructions: str,
    rules_text: str,
) -> None:
    st.markdown("### Preview (not sent)")
    inputs = case_inputs(case, rules_text)
    product_facts = inputs["product_facts"]
    sources = inputs["sources"]
    messages = [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": (
                f"Document:\n{inputs['document_text']}\n\n"
                "Product facts:\n"
                + json.dumps(
                    product_facts.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n\nSources:\n"
                + json.dumps(
                    [
                        source.model_dump(mode="json")
                        for source in sources
                    ],
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            ),
        },
    ]
    preview = format_preview_trace(
        messages,
        {"purpose": "inspector input preview", "sent": False},
    )
    st.warning(
        "PREVIEW ONLY — this input sketch was not sent and has no provider "
        "usage, generation ID, or cost. It is not the captured original "
        "adapter-formatted request."
    )
    st.caption(preview.notice)
    for message in preview.messages:
        st.markdown(f"**{message.get('role', 'unknown')}**")
        st.text(_message_content(message.get("content")))
    st.text(f"Demos: 0 (zero). Document length: {len(inputs['document_text'])} characters.")


def _candidate_records(
    stats: Mapping[str, object] | None,
) -> tuple[list[dict[str, object]], str | None]:
    if not isinstance(stats, Mapping):
        return [], "No candidate statistics are archived."
    detailed = stats.get("detailed_results")
    if not isinstance(detailed, Mapping):
        return [], "Candidate statistics do not contain detailed_results."
    candidates = detailed.get("candidates")
    if not isinstance(candidates, list):
        return [], "Candidate statistics do not contain a candidate list."
    scores = detailed.get("val_aggregate_scores")
    best_idx = detailed.get("best_idx")
    records: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            continue
        instruction = candidate.get("predict")
        if not isinstance(instruction, str) or not instruction.strip():
            nested = candidate.get("instructions")
            if isinstance(nested, Mapping):
                instruction = nested.get("predict")
        if not isinstance(instruction, str) or not instruction.strip():
            continue
        score = (
            scores[index]
            if isinstance(scores, list) and index < len(scores)
            else None
        )
        records.append(
            {
                "candidate_index": index,
                "instructions": instruction,
                "val_aggregate_score": score,
                "selected": index == best_idx,
            }
        )
    if not records:
        return [], "Candidate statistics contain no inspectable instructions."
    return records, None


def _same_schema(
    left: ProgramManifest, right: ProgramManifest
) -> bool:
    return (
        left.signature_schema == right.signature_schema
        and left.signature_fields == right.signature_fields
    )


def _unified_diff(
    before: str,
    after: str,
    *,
    fromfile: str,
    tofile: str,
) -> str:
    return "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=fromfile,
            tofile=tofile,
            lineterm="",
        )
    )


def _render_historical_costs(summary: Mapping[str, object]) -> None:
    costs = summary.get("costs")
    if not isinstance(costs, Mapping):
        st.warning(
            "Historical cost is not archived for this run. It is not shown "
            "as zero."
        )
        return
    st.markdown(
        f"- **Optimization cost:** "
        f"{_cost_value(costs.get('optimization_cost_usd'), costs.get('optimization_cost_status'))}\n"
        f"- **Baseline development cost:** "
        f"{_cost_value(costs.get('baseline_development_cost_usd'), costs.get('baseline_development_cost_status'))}\n"
        f"- **Benchmark inference cost:** "
        f"{_cost_value(costs.get('benchmark_inference_known_cost_usd'), costs.get('benchmark_inference_cost_status'))}\n"
        f"- **One-time known cost:** "
        f"{_cost_value(costs.get('one_time_known_cost_usd'), costs.get('one_time_cost_status'))}"
    )
    st.caption(
        "Original measurements retain their recorded actual/estimated/"
        "pending labels. Replay does not reprice or settle them."
    )


def _cost_value(value: object, status: object) -> str:
    if value is None:
        return f"unavailable ({status or 'unknown status'})"
    return f"{value} USD ({status or 'unknown status'})"


def _cost_status_from_payload(payload: Mapping[str, object]) -> str:
    if "actual_cost_usd" in payload or "estimated_cost_usd" in payload:
        unresolved = _decimal_or_none(payload.get("unresolved_cost_usd"))
        if payload.get("pending_cost_calls") or (
            unresolved is not None and unresolved > 0
        ):
            return "pending"
        actual = _decimal_or_none(payload.get("actual_cost_usd"))
        estimated = _decimal_or_none(payload.get("estimated_cost_usd"))
        committed = _decimal_or_none(payload.get("committed_cost_usd"))
        if actual is not None and actual > 0:
            return "actual"
        if estimated is not None and estimated > 0:
            return "estimated"
        if committed is not None and committed > 0:
            return "committed"
        return "none"
    return str(payload.get("cost_status", "unavailable"))


def _decimal_or_none(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        decimal = Decimal(str(value))
    except Exception:  # noqa: BLE001 - display fallback
        return None
    return decimal if decimal.is_finite() else None


def _outcome_cost(outcome: RecordedOutcome) -> str:
    if outcome.cost_usd is None:
        return f"unavailable ({outcome.cost_status})"
    return f"{outcome.cost_usd} USD ({outcome.cost_status})"


def _percent(value: object) -> str:
    if value is None or isinstance(value, bool):
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.1f}%"
    return "N/A"


def _number(value: object) -> str:
    if value is None or isinstance(value, bool):
        return "N/A"
    if isinstance(value, (int, float)):
        number = float(value)
        text = f"{number:.4f}".rstrip("0").rstrip(".")
        if number != 0.0 and text in ("0", "-0"):
            return f"{number:.6g}"
        return text
    if isinstance(value, Decimal):
        return format(value, "f")
    return str(value)


def _int_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _latency(value: float | None) -> str:
    if value is None:
        return "not recorded"
    return f"{value:.1f} ms"


def _message_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping):
                text = part.get("text")
                parts.append(str(text) if text is not None else str(part))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
