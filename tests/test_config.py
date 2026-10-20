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
    DEFAULT_MAX_PREFLIGHT_COST_USD,
    DEFAULT_REFLECTION_MODEL,
    DEFAULT_REFLECTION_PROVIDER,
    DEFAULT_TASK_MODEL,
    DEFAULT_TASK_PROVIDER,
    DEFAULT_TEMPERATURE,
    InvalidConfigurationError,
    MissingConfigurationError,
    UnsupportedProviderRouteError,
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
            "REFLECTION_MODEL": "openai/gpt-6-sol",
            "TASK_PROVIDER": "openai",
            "REFLECTION_PROVIDER": "openai",
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
            "MAX_PREFLIGHT_COST_USD": "0.40",
            "REQUEST_TIMEOUT_SECONDS": "60",
        },
    )
    assert settings.task_model == "openai/gpt-4.1-mini"
    assert settings.reflection_model == "openai/gpt-6-sol"
    assert settings.task_provider == "openai"
    assert settings.reflection_provider == "openai"
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
    assert settings.max_preflight_cost_usd == Decimal("0.40")
    assert settings.request_timeout_seconds == 60.0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TASK_MODEL", "openrouter/deepseek/deepseek-v4.1-flash-1017"),
        ("TASK_MODEL", "deepseek/deepseek-v4.1-flash-1017:free"),
        ("TASK_MODEL", "deepseek/deepseek-v4.1-flash-1017:batch"),
        ("TASK_MODEL", "deepseek/deepseek-v4.1-flash-1017:nitro"),
        ("TASK_MODEL", "deepseek/deepseek-v4.1-flash-1017:floor"),
        ("TASK_MODEL", "deepseek/deepseek-v4.1-flash-1017:latest"),
        ("TASK_MODEL", "deepseek/deepseek-v4.1-flash-latest"),
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
        ("MAX_PREFLIGHT_COST_USD", "-0.01"),
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
        to_litellm_model_id("deepseek/deepseek-v4.1-flash-1017")
        == "openrouter/deepseek/deepseek-v4.1-flash-1017"
    )
    with pytest.raises(InvalidConfigurationError):
        to_litellm_model_id("openrouter/deepseek/deepseek-v4.1-flash-1017")


def test_validate_model_id_returns_the_trimmed_bare_id() -> None:
    assert validate_model_id("  deepseek/deepseek-v4.1-flash-1017  ") == (
        "deepseek/deepseek-v4.1-flash-1017"
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


def test_default_provider_routes_are_frozen() -> None:
    settings = load_settings(mode="live", env=BASE_LIVE_ENV)
    task = settings.task_route()
    reflection = settings.reflection_route()
    assert task.model == DEFAULT_TASK_MODEL
    assert task.provider == DEFAULT_TASK_PROVIDER
    assert task.expected_schema_mode == "native"
    assert task.context_limit == 1_048_576
    assert task.endpoint_max_output_tokens == 131_072
    assert task.input_usd_per_million == Decimal("0.098")
    assert task.output_usd_per_million == Decimal("0.294")
    assert reflection.model == DEFAULT_REFLECTION_MODEL
    assert reflection.provider == DEFAULT_REFLECTION_PROVIDER
    assert reflection.expected_schema_mode == "native"
    assert reflection.context_limit == 1_050_000
    assert reflection.endpoint_max_output_tokens == 128_000
    assert reflection.input_usd_per_million == Decimal("4.00")
    assert reflection.output_usd_per_million == Decimal("15.00")
    assert reflection.cache_write_usd_per_million == Decimal("5.00")
    assert task.supports_temperature is True
    assert reflection.supports_temperature is False


def test_cheap_reflection_route_is_predeclared() -> None:
    settings = load_settings(
        mode="live",
        env={
            **BASE_LIVE_ENV,
            "REFLECTION_MODEL": "deepseek/deepseek-v4.1-flash-1017",
            "REFLECTION_PROVIDER": "deepinfra/fp8",
        },
    )
    reflection = settings.reflection_route()
    assert reflection.expected_schema_mode == "json_object"
    assert reflection.context_limit == 1_048_576
    assert reflection.endpoint_max_output_tokens == 131_072
    assert reflection.input_usd_per_million == Decimal("0.098")
    assert reflection.output_usd_per_million == Decimal("0.294")
    assert reflection.supports_temperature is True


def test_same_cheap_model_can_serve_both_roles() -> None:
    settings = load_settings(
        mode="live",
        env={
            **BASE_LIVE_ENV,
            "TASK_MODEL": "deepseek/deepseek-v4.1-flash-1017",
            "TASK_PROVIDER": "deepinfra/fp8",
            "REFLECTION_MODEL": "deepseek/deepseek-v4.1-flash-1017",
            "REFLECTION_PROVIDER": "deepinfra/fp8",
        },
    )
    assert settings.task_route().expected_schema_mode == "native"
    assert settings.reflection_route().expected_schema_mode == "json_object"


def test_unknown_route_fails_closed() -> None:
    settings = load_settings(
        mode="live",
        env={
            **BASE_LIVE_ENV,
            "TASK_MODEL": "openai/gpt-4.1-mini",
            "TASK_PROVIDER": "openai",
        },
    )
    with pytest.raises(UnsupportedProviderRouteError):
        settings.task_route()


def test_route_role_mismatch_fails_closed() -> None:
    settings = load_settings(mode="live", env=BASE_LIVE_ENV)
    with pytest.raises(UnsupportedProviderRouteError):
        config.resolve_provider_route(
            settings.reflection_model,
            settings.reflection_provider,
            role="task",
        )


def test_preflight_budget_is_fixed_to_three_calls() -> None:
    settings = load_settings(mode="live", env=BASE_LIVE_ENV)
    budget = settings.preflight_budget()
    assert budget.phase == "preflight"
    assert budget.max_calls == 3
    assert budget.max_cost_usd == DEFAULT_MAX_PREFLIGHT_COST_USD
