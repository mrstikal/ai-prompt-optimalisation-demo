"""Fake S13 worker runners for lifecycle tests; no provider access."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

CALLS = 0
MODE = "block"
LAST_LEDGER = None
BENCHMARK_CALLS = 0
PLAYGROUND_CALLS = 0
LAST_BENCHMARK_SPEC = None
LAST_PLAYGROUND_SPEC = None
_LOCK = threading.Lock()


def reset() -> None:
    global CALLS, MODE, LAST_LEDGER
    global BENCHMARK_CALLS, PLAYGROUND_CALLS
    global LAST_BENCHMARK_SPEC, LAST_PLAYGROUND_SPEC
    with _LOCK:
        CALLS = 0
        MODE = "block"
        LAST_LEDGER = None
        BENCHMARK_CALLS = 0
        PLAYGROUND_CALLS = 0
        LAST_BENCHMARK_SPEC = None
        LAST_PLAYGROUND_SPEC = None


@dataclass(frozen=True, slots=True)
class FakeManifest:
    run_id: str
    status: str
    origin: str = "live"


@dataclass(frozen=True, slots=True)
class FakeResult:
    manifest: FakeManifest
    selected_program: object | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _append(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
        handle.flush()


def make_runner():
    def runner(spec, handle):
        global CALLS, LAST_LEDGER
        with _LOCK:
            CALLS += 1
            LAST_LEDGER = handle.ledger
        events = (
            spec.runs_root / "optimization" / spec.run_id / "events.jsonl"
        )
        _append(
            events,
            {
                "at": "2026-10-19T12:00:00Z",
                "event": "optimization_started",
                "run_id": spec.run_id,
                "train_cases": len(spec.train_cases),
                "validation_cases": len(spec.validation_cases),
            },
        )
        if MODE == "fail":
            raise RuntimeError("fixture worker failure")
        if MODE == "complete":
            _append(
                events,
                {
                    "at": "2026-10-19T12:00:01Z",
                    "event": "optimization_finished",
                    "status": "completed",
                },
            )
            return FakeResult(
                FakeManifest(spec.run_id, "completed"),
                SimpleNamespace(program_id="program-fixture-selected"),
            )
        while not handle.ledger.cancelled:
            time.sleep(0.01)
        _append(
            events,
            {
                "at": "2026-10-19T12:00:02Z",
                "event": "optimization_finished",
                "status": "stopped",
                "stop_reason": "cancelled",
            },
        )
        return FakeResult(
            FakeManifest(spec.run_id, "stopped"),
            SimpleNamespace(program_id="program-fixture-selected"),
            ("fixture cancellation",),
        )

    return runner


@dataclass(frozen=True, slots=True)
class FakeArm:
    arm: str
    program_id: str
    status: str = "ok"
    findings: tuple[object, ...] | None = field(default_factory=tuple)
    error: str | None = None
    error_kind: str | None = None
    request_ids: tuple[str, ...] = field(default_factory=tuple)
    requests: tuple[object, ...] = field(default_factory=tuple)
    cost_usd: object | None = None
    cost_status: str = "estimated"
    latency_ms: float | None = 120.0
    input_tokens: int | None = 10
    output_tokens: int | None = 5


def make_playground_runner():
    def runner(spec, handle):
        global PLAYGROUND_CALLS, LAST_PLAYGROUND_SPEC
        with _LOCK:
            PLAYGROUND_CALLS += 1
            LAST_PLAYGROUND_SPEC = spec
        events = (
            spec.runs_root / "playground" / spec.run_id / "events.jsonl"
        )
        _append(
            events,
            {
                "at": "2026-10-19T12:00:00Z",
                "event": "playground_started",
                "run_id": spec.run_id,
            },
        )
        if MODE == "fail":
            raise RuntimeError("fixture playground failure")
        _append(
            events,
            {
                "at": "2026-10-19T12:00:01Z",
                "event": "playground_finished",
                "status": "completed",
            },
        )
        return SimpleNamespace(
            status="completed",
            optimization_run_id=spec.optimization_run_id,
            baseline=FakeArm(
                arm="baseline", program_id="program-fixture-baseline"
            ),
            optimized=FakeArm(
                arm="optimized", program_id="program-fixture-optimized"
            ),
            warnings=(),
            retained_content=spec.retain_content,
        )

    return runner


def make_benchmark_runner():
    def runner(spec, handle):
        global BENCHMARK_CALLS, LAST_BENCHMARK_SPEC
        with _LOCK:
            BENCHMARK_CALLS += 1
            LAST_BENCHMARK_SPEC = spec
        events = (
            spec.runs_root / "benchmarks" / spec.run_id / "events.jsonl"
        )
        _append(
            events,
            {
                "at": "2026-10-19T12:00:00Z",
                "event": "benchmark_started",
                "run_id": spec.run_id,
            },
        )
        if MODE == "fail":
            raise RuntimeError("fixture benchmark failure")
        _append(
            events,
            {
                "at": "2026-10-19T12:00:01Z",
                "event": "benchmark_finished",
                "status": "completed",
            },
        )
        return SimpleNamespace(
            status="completed",
            manifest=SimpleNamespace(
                run_id=spec.run_id,
                status="completed",
                origin="live",
                benchmark=SimpleNamespace(released=spec.release_results),
            ),
            summary={
                "verdict": "tie",
                "verdict_reason": "fixture benchmark",
            },
            warnings=(),
        )

    return runner
