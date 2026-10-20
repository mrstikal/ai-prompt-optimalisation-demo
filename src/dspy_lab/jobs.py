"""Stable job specifications, gatekeeping, and worker dispatch for the UI.

Streamlit re-executes ``app.py`` on every rerun. The cached ``JobManager``
keeps a runner created by the first execution, so the specification classes it
compares must live in a normally imported module. Defining them in ``app.py``
would create a new class object per script run and break ``isinstance``
dispatch for benchmark and Playground jobs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .artifacts import (
    RUNS_ROOT,
    append_jsonl,
    atomic_write_json,
    read_run_manifest,
)
from .config import PROJECT_ROOT, Settings
from .llm import BudgetLedger, BudgetSnapshot
from .schemas import CaseRecord, ProductFacts, SourceRecord

DATA_DIR = PROJECT_ROOT / "data" / "v1"
MODE_LIVE = "Live Experiment"


class JobLedgerHandle(Protocol):
    ledger: BudgetLedger


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


def run_paid_job(spec: object, handle: JobLedgerHandle) -> object:
    """Dispatch one confirmed Live worker. Imports stay off the Replay path."""
    if isinstance(spec, BenchmarkJobSpec):
        return _run_benchmark_job(spec, handle)
    if isinstance(spec, PlaygroundJobSpec):
        return _run_playground_job(spec, handle)
    if isinstance(spec, OptimizationJobSpec):
        return _run_optimization_job(spec, handle)
    raise TypeError(f"unsupported paid job spec: {type(spec).__name__}")


def _run_optimization_job(
    spec: OptimizationJobSpec, handle: JobLedgerHandle
) -> object:
    """Default optimization worker body."""
    from .experiments import run_optimization

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
    spec: BenchmarkJobSpec, handle: JobLedgerHandle
) -> object:
    """Load the sealed split only inside the authorized benchmark worker."""
    from .experiments import load_benchmark_cases, run_benchmark
    from .program import load_editorial_rules

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
    spec: PlaygroundJobSpec, handle: JobLedgerHandle
) -> object:
    """Run both frozen artifacts without any gold-based scoring."""
    from .experiments import run_playground

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


__all__ = [
    "BenchmarkJobSpec",
    "JobLedgerHandle",
    "OptimizationJobSpec",
    "PlaygroundJobSpec",
    "run_paid_job",
]


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


def _stable_process_instance_id(pid: int) -> str:
    """Return a process identity that is stable across Streamlit reruns.

    The worker's active-job lock must remain valid when the script is
    re-executed in the same process. PID plus the kernel start token survives
    reruns and changes when the process is actually restarted.
    """
    start_token = _process_start_token(pid)
    return f"{pid}:{start_token}" if start_token else str(pid)


PROCESS_INSTANCE_ID = _stable_process_instance_id(os.getpid())


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
    if payload.get("pid") == os.getpid():
        recorded_instance = payload.get("process_instance_id")
        if recorded_instance not in (None, PROCESS_INSTANCE_ID):
            # A transient /proc read may have fallen back to the bare PID for
            # this rerun while the lock retains pid:start-token. Treat a lock
            # from our own PID as ours; the start-token check below still
            # rejects a genuinely restarted process.
            same_process_fallback = (
                (
                    PROCESS_INSTANCE_ID == str(os.getpid())
                    and isinstance(recorded_instance, str)
                    and recorded_instance.startswith(
                        f"{os.getpid()}:"
                    )
                )
                or recorded_instance == str(os.getpid())
            )
            if not same_process_fallback:
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
        self._runner = runner or run_paid_job
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


__all__ += [
    "ACTIVE_JOB_LOCK_FILENAME",
    "JOB_ACTIVE_STATUSES",
    "JOB_FOLDERS",
    "JOB_KIND_BENCHMARK",
    "JOB_KIND_OPTIMIZATION",
    "JOB_KIND_PLAYGROUND",
    "JOB_REGISTRY_DIRNAME",
    "JOB_RUN_ID_PATTERN",
    "JOB_SCHEMA_VERSION",
    "JOB_STATUS_CANCELLING",
    "JOB_STATUS_COMPLETED",
    "JOB_STATUS_FAILED",
    "JOB_STATUS_INTERRUPTED",
    "JOB_STATUS_RUNNING",
    "MODE_LIVE",
    "PROCESS_INSTANCE_ID",
    "JobBusyError",
    "JobGateError",
    "JobHandle",
    "JobManager",
    "JobStaleError",
    "_new_job_run_id",
    "_pid_alive",
    "_process_start_token",
    "_redact_secret",
    "_stable_process_instance_id",
    "_utc_now",
    "read_active_job_lock",
]
