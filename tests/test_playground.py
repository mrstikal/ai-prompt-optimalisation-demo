from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import dspy.clients.lm as dspy_lm
import pytest

from dspy_lab.artifacts import RUN_REQUESTS_FILENAME
from dspy_lab.config import Settings
from dspy_lab.experiments import (
    ExperimentConfigurationError,
    ExperimentIncompleteError,
    ExperimentNotAuthorized,
    run_playground,
)
from dspy_lab.llm import BudgetLedger

from ui_fixture import build_ui_archive

SECRET = "sk-or-v1-s14-playground-sentinel"
TITLE = "Zlatý slitk s novým povrchem"
ARTICLE = (
    "Producent představil zlatý slitk s ochrannou vrstvou. Podle "
    "materiálu je cena garantovaná a výnos jistý."
)


class Reply:
    def __init__(self, content: str, cost: float) -> None:
        self.content = content
        self.cost = cost


def response_for(reply: Reply, request: dict):
    from types import SimpleNamespace

    return SimpleNamespace(
        id="gen-s14",
        model=request.get("model"),
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason="stop",
                message=SimpleNamespace(
                    content=reply.content,
                    tool_calls=None,
                    reasoning_content=None,
                ),
                logprobs=None,
            )
        ],
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        _hidden_params={"response_cost": reply.cost},
    )


def install_transport(monkeypatch, handler):
    calls: list[dict] = []

    def fake_completion(*, request, num_retries=0):
        calls.append(request)
        content, cost = handler(request, len(calls))
        return response_for(Reply(content, cost), request)

    monkeypatch.setattr(dspy_lm, "litellm_completion", fake_completion)
    return calls


def make_settings(**overrides) -> Settings:
    values = {
        "mode": "live",
        "openrouter_api_key": SECRET,
        "max_playground_calls": 2,
        "max_playground_cost_usd": Decimal("0.15"),
        "request_timeout_seconds": 5.0,
    }
    values.update(overrides)
    return Settings(**values)


def fixture_root(tmp_path: Path, *, measured: bool = True) -> Path:
    root = tmp_path / "fixture"
    build_ui_archive(root, changed=True, measured=measured)
    return root


def run_fixture_playground(
    tmp_path: Path,
    monkeypatch,
    *,
    handler=None,
    measured: bool = True,
    retain: bool = False,
    run_id: str = "play-s14",
    confirm: bool = True,
    ledger: BudgetLedger | None = None,
):
    root = fixture_root(tmp_path, measured=measured)
    calls = install_transport(
        monkeypatch,
        handler
        or (lambda _request, _index: (json.dumps({"findings": []}), 0.00001)),
    )
    settings = make_settings()
    result = run_playground(
        settings,
        optimization_run_id="opt-ui",
        title=TITLE,
        article_text=ARTICLE,
        retain_content=retain,
        confirm_budget=confirm,
        run_id=run_id,
        runs_root=root / "runs",
        programs_root=root / "programs",
        ledger=ledger,
    )
    return root, calls, result


def test_playground_runs_both_arms_once_and_is_unscored(
    tmp_path, monkeypatch
) -> None:
    root, calls, result = run_fixture_playground(tmp_path, monkeypatch)
    assert len(calls) == 2
    assert result.status == "completed"
    assert result.baseline.status == "ok"
    assert result.optimized.status == "ok"
    assert result.baseline.findings == ()
    assert result.optimized.findings == ()
    assert result.baseline.request_ids
    assert result.optimized.request_ids
    assert result.ledger.admitted_calls == 2
    assert result.ledger.max_calls == 2
    assert result.ledger.max_cost_usd == Decimal("0.15")
    assert result.retained_content is False
    assert result.baseline.program_id != result.optimized.program_id
    assert all(
        call["model"].endswith(make_settings().task_model)
        for call in calls
    )
    assert {call["temperature"] for call in calls} == {0.0}
    first_messages = calls[0]["messages"]
    second_messages = calls[1]["messages"]
    assert first_messages[0]["content"] != second_messages[0]["content"]
    for messages in (first_messages, second_messages):
        assert any(ARTICLE in str(message.get("content")) for message in messages)
    summary = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["unscored"] is True
    assert summary["gold_used"] is False
    assert "reward" not in summary
    assert "micro_f1" not in summary
    assert "f1" not in summary
    assert not (result.run_dir / RUN_REQUESTS_FILENAME).exists()
    config_text = (result.run_dir / "config.json").read_text(encoding="utf-8")
    summary_text = (result.run_dir / "summary.json").read_text(
        encoding="utf-8"
    )
    assert TITLE not in config_text
    assert ARTICLE not in config_text
    assert TITLE.split()[0] not in summary_text
    assert "garantovaná" not in summary_text
    assert result.run_dir.parent.parent == root / "runs"


def test_playground_requires_confirmation_and_live_artifacts(
    tmp_path, monkeypatch
) -> None:
    root = fixture_root(tmp_path)
    calls = install_transport(
        monkeypatch,
        lambda _request, _index: (json.dumps({"findings": []}), 0.00001),
    )
    settings = make_settings()
    with pytest.raises(ExperimentNotAuthorized):
        run_playground(
            settings,
            optimization_run_id="opt-ui",
            title=TITLE,
            article_text=ARTICLE,
            confirm_budget=False,
            run_id="play-no-confirm",
            runs_root=root / "runs",
            programs_root=root / "programs",
        )
    assert calls == []
    mock_root = fixture_root(tmp_path / "mock", measured=False)
    with pytest.raises(ExperimentIncompleteError, match="Live"):
        run_playground(
            settings,
            optimization_run_id="opt-ui",
            title=TITLE,
            article_text=ARTICLE,
            confirm_budget=True,
            run_id="play-mock",
            runs_root=mock_root / "runs",
            programs_root=mock_root / "programs",
        )
    assert calls == []


def test_playground_rejects_mismatched_ledger(tmp_path, monkeypatch) -> None:
    root = fixture_root(tmp_path)
    install_transport(
        monkeypatch,
        lambda _request, _index: (json.dumps({"findings": []}), 0.00001),
    )
    settings = make_settings()
    mismatched = BudgetLedger(max_calls=3, max_cost_usd="0.15")
    with pytest.raises(ExperimentConfigurationError, match="ledger limits"):
        run_playground(
            settings,
            optimization_run_id="opt-ui",
            title=TITLE,
            article_text=ARTICLE,
            confirm_budget=True,
            run_id="play-mismatch",
            runs_root=root / "runs",
            programs_root=root / "programs",
            ledger=mismatched,
        )


def test_playground_retains_content_only_when_explicitly_enabled(
    tmp_path, monkeypatch
) -> None:
    finding = {
        "code": "UNWARRANTED_CERTAINTY",
        "evidence": "výnos jistý",
        "explanation": "Fixture guarantee.",
    }

    def handler(_request, _index):
        return json.dumps({"findings": [finding]}), 0.00001

    root, calls, result = run_fixture_playground(
        tmp_path,
        monkeypatch,
        handler=handler,
        retain=True,
    )
    assert len(calls) == 2
    assert result.retained_content is True
    assert result.baseline.findings is not None
    assert result.baseline.findings[0].evidence == "výnos jistý"
    requests_path = result.run_dir / RUN_REQUESTS_FILENAME
    assert requests_path.is_file()
    requests = requests_path.read_text(encoding="utf-8")
    assert TITLE in requests
    assert ARTICLE in requests
    summary = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["retain_content"] is True
    assert summary["arms"]["baseline"]["findings"][0]["evidence"] == "výnos jistý"


def test_playground_parse_failure_is_preserved_and_not_retried(
    tmp_path, monkeypatch
) -> None:
    _root, calls, result = run_fixture_playground(
        tmp_path,
        monkeypatch,
        handler=lambda _request, _index: ("not json", 0.00002),
    )
    assert len(calls) == 2
    assert result.status == "completed"
    assert result.baseline.status == "error"
    assert result.optimized.status == "error"
    assert "adapterparseerror" in (result.baseline.error or "").lower()
    summary = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["arms"]["baseline"]["status"] == "error"
    assert summary["arms"]["baseline"]["error_kind"] == "parse_error"


def test_playground_cancelled_ledger_dispatches_nothing(
    tmp_path, monkeypatch
) -> None:
    settings = make_settings()
    ledger = BudgetLedger.from_limits(settings.playground_budget())
    ledger.cancel("cancelled")
    _root, calls, result = run_fixture_playground(
        tmp_path, monkeypatch, ledger=ledger
    )
    assert calls == []
    assert result.status == "stopped"
    assert result.ledger.admitted_calls == 0
    assert result.baseline.status == "error"
    assert result.optimized.status == "error"
    assert result.baseline.request_ids == ()
