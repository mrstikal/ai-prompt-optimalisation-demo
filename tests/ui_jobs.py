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
_LOCK = threading.Lock()


def reset() -> None:
    global CALLS, MODE, LAST_LEDGER
    with _LOCK:
        CALLS = 0
        MODE = "block"
        LAST_LEDGER = None


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
