from __future__ import annotations

import socket
from decimal import Decimal
from types import SimpleNamespace

import pytest

from dspy_lab import config
from dspy_lab.config import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_OPTIMIZATION_CALLS,
    DEFAULT_MAX_OPTIMIZATION_COST_USD,
    DEFAULT_MAX_OPTIMIZATION_METRIC_CALLS,
    DEFAULT_REFLECTION_MODEL,
    DEFAULT_REFLECTION_PROVIDER,
    DEFAULT_TASK_MODEL,
    DEFAULT_TASK_PROVIDER,
    DEFAULT_TEMPERATURE,
    InvalidConfigurationError,
    MissingConfigurationError,
    UnsupportedPythonError,
    load_settings,
    to_litellm_model_id,
    validate_model_id,
)

BASE_LIVE_ENV = {"OPENROUTER_API_KEY": "sk-or-v1-test-key"}


def test_replay_is_the_default_and_needs_no_key() -> None:
    settings = load_settings(env={})
    assert settings.mode == "replay"
    assert settings.openrouter_api_key is None
    assert settings.task_model == DEFAULT_TASK_MODEL


def test_replay_ignores_environment_values() -> None:
    settings = load_settings(
        mode="replay",
        env={
            "OPENROUTER_API_KEY": "sk-or-v1-ignored",
            "TASK_MODEL": "not-a-valid-model",
            "TEMPERATURE": "not-a-number",
        },
    )
    assert settings.mode == "replay"
    assert settings.openrouter_api_key is None
    assert settings.task_model == DEFAULT_TASK_MODEL


def test_live_requires_an_api_key() -> None:
    with pytest.raises(MissingConfigurationError):
        load_settings(mode="live", env={})
    with pytest.raises(MissingConfigurationError):
        load_settings(mode="live", env={"OPENROUTER_API_KEY": "   "})


def test_live_rejects_an_api_key_with_whitespace() -> None:
    with pytest.raises(InvalidConfigurationError):
        load_settings(
            mode="live", env={"OPENROUTER_API_KEY": "sk-or-v1-test key"}
        )


def test_live_defaults_are_frozen_and_parse_without_network() -> None:
    settings = load_settings(mode="live", env=BASE_LIVE_ENV)
    assert settings.mode == "live"
    assert settings.openrouter_api_key == "sk-or-v1-test-key"
    assert settings.task_model == DEFAULT_TASK_MODEL
    assert settings.reflection_model == DEFAULT_REFLECTION_MODEL
    assert settings.task_provider == DEFAULT_TASK_PROVIDER
    assert settings.reflection_provider == DEFAULT_REFLECTION_PROVIDER
    assert settings.temperature == DEFAULT_TEMPERATURE
    assert settings.max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS
    assert settings.max_optimization_calls == DEFAULT_MAX_OPTIMIZATION_CALLS
    assert (
        settings.max_optimization_cost_usd
        == DEFAULT_MAX_OPTIMIZATION_COST_USD
    )
    assert (
        settings.max_optimization_metric_calls
        == DEFAULT_MAX_OPTIMIZATION_METRIC_CALLS
    )


def test_live_environment_overrides_are_parsed_exactly() -> None:
    settings = load_settings(
        mode="live",
        env={
            **BASE_LIVE_ENV,
            "TASK_MODEL": "openai/gpt-4.1-mini",
            "REFLECTION_MODEL": "anthropic/claude-sonnet-4.5",
            "TASK_PROVIDER": "openai",
            "REFLECTION_PROVIDER": "claude-on-aws",
            "TEMPERATURE": "0.2",
            "MAX_OUTPUT_TOKENS": "1024",
            "REFLECTION_TEMPERATURE": "0.9",
            "REFLECTION_MAX_OUTPUT_TOKENS": "2048",
            "MAX_OPTIMIZATION_CALLS": "50",
            "MAX_OPTIMIZATION_COST_USD": "1.25",
            "MAX_OPTIMIZATION_METRIC_CALLS": "40",
            "MAX_BASELINE_DEV_CALLS": "5",
            "MAX_BASELINE_DEV_COST_USD": "0.20",
            "MAX_BENCHMARK_CALLS": "20",
            "MAX_BENCHMARK_COST_USD": "0.30",
            "MAX_PLAYGROUND_CALLS": "2",
            "MAX_PLAYGROUND_COST_USD": "0.05",
            "REQUEST_TIMEOUT_SECONDS": "60",
        },
    )
    assert settings.task_model == "openai/gpt-4.1-mini"
    assert settings.reflection_model == "anthropic/claude-sonnet-4.5"
    assert settings.task_provider == "openai"
    assert settings.reflection_provider == "claude-on-aws"
    assert settings.temperature == 0.2
    assert settings.max_output_tokens == 1024
    assert settings.reflection_temperature == 0.9
    assert settings.reflection_max_output_tokens == 2048
    assert settings.max_optimization_calls == 50
    assert settings.max_optimization_cost_usd == Decimal("1.25")
    assert settings.max_optimization_metric_calls == 40
    assert settings.max_baseline_dev_calls == 5
    assert settings.max_baseline_dev_cost_usd == Decimal("0.20")
    assert settings.max_benchmark_calls == 20
    assert settings.max_benchmark_cost_usd == Decimal("0.30")
    assert settings.max_playground_calls == 2
    assert settings.max_playground_cost_usd == Decimal("0.05")
    assert settings.request_timeout_seconds == 60.0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TASK_MODEL", "openrouter/google/gemini-2.5-flash-lite"),
        ("TASK_MODEL", "google/gemini-2.5-flash-lite:free"),
        ("TASK_MODEL", "google/gemini-2.5-flash-lite:batch"),
        ("TASK_MODEL", "google/gemini-2.5-flash-lite:nitro"),
        ("TASK_MODEL", "google/gemini-2.5-flash-lite:floor"),
        ("TASK_MODEL", "google/gemini-2.5-flash-lite:latest"),
        ("TASK_MODEL", "google/gemini-latest"),
        ("TASK_MODEL", "google/latest"),
        ("TASK_MODEL", "not-a-model"),
        ("TASK_MODEL", "google/"),
        ("TASK_MODEL", ""),
        ("TASK_PROVIDER", ""),
        ("TASK_PROVIDER", "google vertex"),
        ("TEMPERATURE", "-0.1"),
        ("TEMPERATURE", "2.1"),
        ("TEMPERATURE", "nan"),
        ("TEMPERATURE", "inf"),
        ("TEMPERATURE", "abc"),
        ("MAX_OUTPUT_TOKENS", "0"),
        ("MAX_OUTPUT_TOKENS", "1.5"),
        ("MAX_OPTIMIZATION_CALLS", "0"),
        ("MAX_OPTIMIZATION_COST_USD", "-0.01"),
        ("MAX_OPTIMIZATION_COST_USD", "NaN"),
        ("MAX_OPTIMIZATION_COST_USD", "Infinity"),
        ("MAX_OPTIMIZATION_METRIC_CALLS", "0"),
        ("MAX_BASELINE_DEV_CALLS", "0"),
        ("MAX_BENCHMARK_CALLS", "0"),
        ("MAX_PLAYGROUND_CALLS", "0"),
        ("REQUEST_TIMEOUT_SECONDS", "0"),
    ],
)
def test_invalid_live_values_are_rejected(name: str, value: str) -> None:
    with pytest.raises(InvalidConfigurationError):
        load_settings(mode="live", env={**BASE_LIVE_ENV, name: value})


def test_metric_call_target_cannot_exceed_total_call_cap() -> None:
    with pytest.raises(InvalidConfigurationError):
        load_settings(
            mode="live",
            env={
                **BASE_LIVE_ENV,
                "MAX_OPTIMIZATION_CALLS": "10",
                "MAX_OPTIMIZATION_METRIC_CALLS": "11",
            },
        )


def test_invalid_mode_is_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        load_settings(mode="offline", env={})


def test_python_315_is_rejected_even_when_load_settings_is_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config, "sys", SimpleNamespace(version_info=(3, 15, 0))
    )
    with pytest.raises(UnsupportedPythonError):
        config.load_settings(mode="replay", env={})


def test_secret_is_not_exposed_in_repr_or_redacted_output() -> None:
    settings = load_settings(
        mode="live", env={"OPENROUTER_API_KEY": "sk-or-v1-super-secret"}
    )
    assert "sk-or-v1-super-secret" not in repr(settings)
    redacted = settings.redacted()
    assert redacted["openrouter_api_key"] == "***"
    assert "sk-or-v1-super-secret" not in str(redacted)


def test_litellm_prefix_is_added_exactly_once() -> None:
    assert (
        to_litellm_model_id("google/gemini-2.5-flash-lite")
        == "openrouter/google/gemini-2.5-flash-lite"
    )
    with pytest.raises(InvalidConfigurationError):
        to_litellm_model_id("openrouter/google/gemini-2.5-flash-lite")


def test_validate_model_id_returns_the_trimmed_bare_id() -> None:
    assert validate_model_id("  google/gemini-2.5-flash-lite  ") == (
        "google/gemini-2.5-flash-lite"
    )


def test_validate_model_id_allows_an_incidental_latest_substring() -> None:
    assert validate_model_id("author/slatest-v2") == "author/slatest-v2"


def test_configuration_loading_does_not_open_a_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NetworkBlocked(RuntimeError):
        pass

    class BlockedSocket:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise NetworkBlocked("network access is forbidden in config loading")

    def blocked(*args: object, **kwargs: object) -> None:
        raise NetworkBlocked("network access is forbidden in config loading")

    monkeypatch.setattr(socket, "socket", BlockedSocket)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    settings = load_settings(mode="live", env=BASE_LIVE_ENV)
    assert settings.mode == "live"
