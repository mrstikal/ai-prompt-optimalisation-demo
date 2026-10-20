"""Offline fixtures shared by the test suite."""

from __future__ import annotations

from decimal import Decimal

import pytest

from dspy_lab.config import load_settings


def _catalog_payload(profile, output_cap: int) -> dict[str, object]:
    assert profile.endpoint_max_output_tokens >= output_cap
    return {
        "data": {
            "endpoints": [
                {
                    "tag": profile.provider,
                    "model_id": profile.model,
                    "provider_name": "Offline fixture",
                    "context_length": profile.context_limit,
                    "max_completion_tokens": profile.endpoint_max_output_tokens,
                    "supported_parameters": [
                        "response_format",
                        "structured_outputs",
                        "max_tokens",
                        "reasoning",
                        *(
                            ["temperature"]
                            if profile.supports_temperature
                            else []
                        ),
                    ],
                    "pricing": {
                        "prompt": str(
                            profile.input_usd_per_million
                            / Decimal("1000000")
                        ),
                        "completion": str(
                            profile.output_usd_per_million
                            / Decimal("1000000")
                        ),
                        "input_cache_write": str(
                            profile.cache_write_usd_per_million
                            / Decimal("1000000")
                        ),
                    },
                }
            ]
        }
    }


@pytest.fixture(autouse=True)
def offline_route_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep paid phases offline while exercising their catalog gate."""
    import dspy_lab.experiments as experiments

    settings = load_settings(
        mode="live",
        env={"OPENROUTER_API_KEY": "sk-or-v1-catalog-fixture"},
    )
    routes = {
        settings.task_model: (
            settings.task_route(),
            settings.max_output_tokens,
        ),
        settings.reflection_model: (
            settings.reflection_route(),
            settings.reflection_max_output_tokens,
        ),
    }

    def fake_fetch(model: str) -> dict[str, object]:
        try:
            profile, output_cap = routes[model]
        except KeyError as exc:
            raise AssertionError(
                f"test catalog has no fixture for model {model!r}"
            ) from exc
        return _catalog_payload(profile, output_cap)

    monkeypatch.setattr(experiments, "fetch_endpoint_catalog", fake_fetch)
