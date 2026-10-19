"""Runtime configuration and frozen API contracts for the DSPy lab.

S1 loads and validates local settings only. Importing this module performs no
network I/O and constructs no LM. Live provider capability and pricing checks
are a later, separately authorized stage.
"""

from __future__ import annotations

import math
import os
import platform
import re
import sys
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal, Mapping

from . import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_PYTHON_PIN = "3.14.8"
SUPPORTED_PYTHON_MIN = (3, 14)
SUPPORTED_PYTHON_MAX_EXCLUSIVE = (3, 15)

Mode = Literal["live", "replay"]

EXPECTED_DSPY_VERSION = "3.4.1"
EXPECTED_GEPA_VERSION = "0.1.4"
EXPECTED_LITELLM_VERSION = "1.106.0"
EXPECTED_PYDANTIC_VERSION = "2.14.1"
EXPECTED_STREAMLIT_VERSION = "1.66.0"

DEFAULT_TASK_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_REFLECTION_MODEL = "deepseek/deepseek-chat-v3.1"
DEFAULT_TASK_PROVIDER = "google-vertex/eu"
DEFAULT_REFLECTION_PROVIDER = "deepinfra/fp4"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 2_048
DEFAULT_REFLECTION_TEMPERATURE = 1.0
DEFAULT_REFLECTION_MAX_OUTPUT_TOKENS = 4_096
DEFAULT_MAX_OPTIMIZATION_CALLS = 160
DEFAULT_MAX_OPTIMIZATION_COST_USD = Decimal("2.00")
DEFAULT_MAX_OPTIMIZATION_METRIC_CALLS = 120
DEFAULT_MAX_BASELINE_DEV_CALLS = 10
DEFAULT_MAX_BASELINE_DEV_COST_USD = Decimal("0.50")
DEFAULT_MAX_BENCHMARK_CALLS = 20
DEFAULT_MAX_BENCHMARK_COST_USD = Decimal("0.50")
DEFAULT_MAX_PLAYGROUND_CALLS = 2
DEFAULT_MAX_PLAYGROUND_COST_USD = Decimal("0.10")
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0

GEPA_SEED = 42
GEPA_REFLECTION_MINIBATCH_SIZE = 3
GEPA_CANDIDATE_SELECTION_STRATEGY = "pareto"
NUM_THREADS = 1
LOCAL_CACHE = False
NUM_RETRIES = 0

MODEL_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._:-]*$"
)
FORBIDDEN_MODEL_SUFFIXES = (":free", ":batch", ":nitro", ":floor")


class ConfigurationError(ValueError):
    """Base class for invalid or missing runtime configuration."""


class MissingConfigurationError(ConfigurationError):
    """Raised when Live mode is selected without a required setting."""


class InvalidConfigurationError(ConfigurationError):
    """Raised when a configured value violates the frozen contract."""


class UnsupportedPythonError(ConfigurationError):
    """Raised when the project runs on an unsupported Python interpreter."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated settings shared by the Live workflow and offline tests."""

    mode: Mode
    openrouter_api_key: str | None = field(default=None, repr=False)
    task_model: str = DEFAULT_TASK_MODEL
    reflection_model: str = DEFAULT_REFLECTION_MODEL
    task_provider: str = DEFAULT_TASK_PROVIDER
    reflection_provider: str = DEFAULT_REFLECTION_PROVIDER
    temperature: float = DEFAULT_TEMPERATURE
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    reflection_temperature: float = DEFAULT_REFLECTION_TEMPERATURE
    reflection_max_output_tokens: int = DEFAULT_REFLECTION_MAX_OUTPUT_TOKENS
    max_optimization_calls: int = DEFAULT_MAX_OPTIMIZATION_CALLS
    max_optimization_cost_usd: Decimal = DEFAULT_MAX_OPTIMIZATION_COST_USD
    max_optimization_metric_calls: int = DEFAULT_MAX_OPTIMIZATION_METRIC_CALLS
    max_baseline_dev_calls: int = DEFAULT_MAX_BASELINE_DEV_CALLS
    max_baseline_dev_cost_usd: Decimal = DEFAULT_MAX_BASELINE_DEV_COST_USD
    max_benchmark_calls: int = DEFAULT_MAX_BENCHMARK_CALLS
    max_benchmark_cost_usd: Decimal = DEFAULT_MAX_BENCHMARK_COST_USD
    max_playground_calls: int = DEFAULT_MAX_PLAYGROUND_CALLS
    max_playground_cost_usd: Decimal = DEFAULT_MAX_PLAYGROUND_COST_USD
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS

    def redacted(self) -> dict[str, object]:
        """Return a display-safe mapping that never exposes the API key."""
        return {
            "mode": self.mode,
            "openrouter_api_key": "***" if self.openrouter_api_key else None,
            "task_model": self.task_model,
            "reflection_model": self.reflection_model,
            "task_provider": self.task_provider,
            "reflection_provider": self.reflection_provider,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "reflection_temperature": self.reflection_temperature,
            "reflection_max_output_tokens": self.reflection_max_output_tokens,
            "max_optimization_calls": self.max_optimization_calls,
            "max_optimization_cost_usd": self.max_optimization_cost_usd,
            "max_optimization_metric_calls": self.max_optimization_metric_calls,
            "max_baseline_dev_calls": self.max_baseline_dev_calls,
            "max_baseline_dev_cost_usd": self.max_baseline_dev_cost_usd,
            "max_benchmark_calls": self.max_benchmark_calls,
            "max_benchmark_cost_usd": self.max_benchmark_cost_usd,
            "max_playground_calls": self.max_playground_calls,
            "max_playground_cost_usd": self.max_playground_cost_usd,
            "request_timeout_seconds": self.request_timeout_seconds,
        }


def ensure_supported_python(version_info: tuple[int, ...] | None = None) -> None:
    """Reject interpreters outside the project's supported Python range."""
    current = sys.version_info if version_info is None else version_info
    major = int(current[0])
    minor = int(current[1])
    if (major, minor) < SUPPORTED_PYTHON_MIN or (
        major,
        minor,
    ) >= SUPPORTED_PYTHON_MAX_EXCLUSIVE:
        raise UnsupportedPythonError(
            "Project Python must be "
            f">={SUPPORTED_PYTHON_MIN[0]}.{SUPPORTED_PYTHON_MIN[1]},"
            f"<{SUPPORTED_PYTHON_MAX_EXCLUSIVE[0]}."
            f"{SUPPORTED_PYTHON_MAX_EXCLUSIVE[1]} "
            f"({SUPPORTED_PYTHON_MIN[0]}.{SUPPORTED_PYTHON_MIN[1]}.x); "
            f"got {major}.{minor}. "
            "The host Python 3.15 installation is intentionally unsupported "
            "because DSPy 3.4.1 omits GEPA and LiteLLM on Python >=3.15."
        )


def validate_model_id(model: str, *, field_name: str = "model") -> str:
    """Validate a bare, reproducible OpenRouter model identifier."""
    if not isinstance(model, str) or not model.strip():
        raise InvalidConfigurationError(
            f"{field_name} must be a non-empty OpenRouter model ID"
        )
    candidate = model.strip()
    lowered = candidate.lower()
    if lowered.startswith("openrouter/"):
        raise InvalidConfigurationError(
            f"{field_name} must be a bare OpenRouter ID; the LiteLLM prefix "
            "is added exactly once by the LM factory"
        )
    if lowered.endswith(FORBIDDEN_MODEL_SUFFIXES):
        raise InvalidConfigurationError(
            f"{field_name} must not use :free, :batch, :nitro, or :floor aliases"
        )
    model_slug = lowered.rsplit("/", 1)[-1]
    if (
        model_slug == "latest"
        or model_slug.endswith(":latest")
        or model_slug.endswith("-latest")
    ):
        raise InvalidConfigurationError(
            f"{field_name} must not use a floating latest alias"
        )
    if MODEL_ID_PATTERN.fullmatch(candidate) is None:
        raise InvalidConfigurationError(
            f"{field_name} must look like 'author/model-slug', got {candidate!r}"
        )
    return candidate


def to_litellm_model_id(model: str) -> str:
    """Add the OpenRouter LiteLLM prefix exactly once."""
    return f"openrouter/{validate_model_id(model)}"


def load_settings(
    mode: Mode = "replay", env: Mapping[str, str] | None = None
) -> Settings:
    """Load validated settings for a mode without contacting any provider."""
    if mode not in ("live", "replay"):
        raise InvalidConfigurationError(f"Unsupported mode: {mode!r}")
    ensure_supported_python()
    if mode == "replay":
        return Settings(mode="replay")

    source = os.environ if env is None else env
    settings = Settings(
        mode="live",
        openrouter_api_key=_read_required(source, "OPENROUTER_API_KEY"),
        task_model=_read_model(source, "TASK_MODEL", DEFAULT_TASK_MODEL),
        reflection_model=_read_model(
            source, "REFLECTION_MODEL", DEFAULT_REFLECTION_MODEL
        ),
        task_provider=_read_provider(
            source, "TASK_PROVIDER", DEFAULT_TASK_PROVIDER
        ),
        reflection_provider=_read_provider(
            source, "REFLECTION_PROVIDER", DEFAULT_REFLECTION_PROVIDER
        ),
        temperature=_read_float(
            source,
            "TEMPERATURE",
            DEFAULT_TEMPERATURE,
            minimum=0.0,
            maximum=2.0,
        ),
        max_output_tokens=_read_int(
            source,
            "MAX_OUTPUT_TOKENS",
            DEFAULT_MAX_OUTPUT_TOKENS,
            minimum=1,
        ),
        reflection_temperature=_read_float(
            source,
            "REFLECTION_TEMPERATURE",
            DEFAULT_REFLECTION_TEMPERATURE,
            minimum=0.0,
            maximum=2.0,
        ),
        reflection_max_output_tokens=_read_int(
            source,
            "REFLECTION_MAX_OUTPUT_TOKENS",
            DEFAULT_REFLECTION_MAX_OUTPUT_TOKENS,
            minimum=1,
        ),
        max_optimization_calls=_read_int(
            source,
            "MAX_OPTIMIZATION_CALLS",
            DEFAULT_MAX_OPTIMIZATION_CALLS,
            minimum=1,
        ),
        max_optimization_cost_usd=_read_decimal(
            source,
            "MAX_OPTIMIZATION_COST_USD",
            DEFAULT_MAX_OPTIMIZATION_COST_USD,
        ),
        max_optimization_metric_calls=_read_int(
            source,
            "MAX_OPTIMIZATION_METRIC_CALLS",
            DEFAULT_MAX_OPTIMIZATION_METRIC_CALLS,
            minimum=1,
        ),
        max_baseline_dev_calls=_read_int(
            source,
            "MAX_BASELINE_DEV_CALLS",
            DEFAULT_MAX_BASELINE_DEV_CALLS,
            minimum=1,
        ),
        max_baseline_dev_cost_usd=_read_decimal(
            source,
            "MAX_BASELINE_DEV_COST_USD",
            DEFAULT_MAX_BASELINE_DEV_COST_USD,
        ),
        max_benchmark_calls=_read_int(
            source,
            "MAX_BENCHMARK_CALLS",
            DEFAULT_MAX_BENCHMARK_CALLS,
            minimum=1,
        ),
        max_benchmark_cost_usd=_read_decimal(
            source,
            "MAX_BENCHMARK_COST_USD",
            DEFAULT_MAX_BENCHMARK_COST_USD,
        ),
        max_playground_calls=_read_int(
            source,
            "MAX_PLAYGROUND_CALLS",
            DEFAULT_MAX_PLAYGROUND_CALLS,
            minimum=1,
        ),
        max_playground_cost_usd=_read_decimal(
            source,
            "MAX_PLAYGROUND_COST_USD",
            DEFAULT_MAX_PLAYGROUND_COST_USD,
        ),
        request_timeout_seconds=_read_float(
            source,
            "REQUEST_TIMEOUT_SECONDS",
            DEFAULT_REQUEST_TIMEOUT_SECONDS,
            minimum=1.0,
            maximum=3_600.0,
        ),
    )
    if settings.max_optimization_metric_calls > settings.max_optimization_calls:
        raise InvalidConfigurationError(
            "MAX_OPTIMIZATION_METRIC_CALLS must not exceed "
            "MAX_OPTIMIZATION_CALLS"
        )
    return settings


def collect_runtime_versions() -> dict[str, str]:
    """Return installed versions without importing or contacting providers."""
    versions = {
        "python": platform.python_version(),
        "application": __version__,
    }
    for package in (
        "dspy",
        "gepa",
        "litellm",
        "streamlit",
        "pydantic",
        "pytest",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def verify_runtime_contract() -> dict[str, str]:
    """Verify that the locked DSPy/GEPA/LiteLLM runtime is genuine."""
    versions = collect_runtime_versions()
    if versions["python"] != PROJECT_PYTHON_PIN:
        raise UnsupportedPythonError(
            f"Project Python must be {PROJECT_PYTHON_PIN}; "
            f"got {versions['python']}"
        )
    expected = {
        "dspy": EXPECTED_DSPY_VERSION,
        "gepa": EXPECTED_GEPA_VERSION,
        "litellm": EXPECTED_LITELLM_VERSION,
        "pydantic": EXPECTED_PYDANTIC_VERSION,
        "streamlit": EXPECTED_STREAMLIT_VERSION,
    }
    for package, expected_version in expected.items():
        if versions[package] != expected_version:
            raise ConfigurationError(
                f"Expected {package}=={expected_version}; "
                f"found {versions[package]}"
            )
    try:
        import dspy
        import gepa
        import litellm
    except ImportError as exc:
        raise ConfigurationError(
            f"Required runtime dependency is missing: {exc}"
        ) from exc
    if not isinstance(dspy.GEPA, type):
        raise ConfigurationError(
            "dspy.GEPA is not an available class; the optional GEPA "
            "dependency is missing or stubbed"
        )
    if not dspy.GEPA.__module__.startswith("dspy.teleprompt.gepa"):
        raise ConfigurationError(
            f"dspy.GEPA comes from an unexpected module: {dspy.GEPA.__module__}"
        )
    if not callable(getattr(dspy.GEPA, "compile", None)):
        raise ConfigurationError("dspy.GEPA does not expose compile()")
    if not callable(getattr(litellm, "completion", None)):
        raise ConfigurationError(
            "litellm does not expose completion(); the provider backend is "
            "not usable"
        )
    if not hasattr(gepa, "optimize"):
        raise ConfigurationError(
            "gepa does not expose optimize(); the optimizer package is not "
            "usable"
        )
    return versions


def _read_required(env: Mapping[str, str], name: str) -> str:
    raw = env.get(name)
    if not isinstance(raw, str) or not raw.strip():
        raise MissingConfigurationError(f"{name} is required in Live mode")
    value = raw.strip()
    if any(char.isspace() for char in value):
        raise InvalidConfigurationError(f"{name} must not contain whitespace")
    return value


def _read_optional(env: Mapping[str, str], name: str, default: str) -> str:
    if name not in env:
        return default
    raw = env[name]
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidConfigurationError(f"{name} is set but empty")
    return raw.strip()


def _read_provider(
    env: Mapping[str, str], name: str, default: str
) -> str:
    value = _read_optional(env, name, default)
    if any(char.isspace() for char in value):
        raise InvalidConfigurationError(f"{name} must not contain whitespace")
    return value


def _read_model(env: Mapping[str, str], name: str, default: str) -> str:
    value = _read_optional(env, name, default)
    return validate_model_id(value, field_name=name)


def _read_float(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = _read_optional(env, name, str(default))
    try:
        value = float(raw)
    except ValueError:
        raise InvalidConfigurationError(
            f"{name} must be a number, got {raw!r}"
        ) from None
    if not math.isfinite(value):
        raise InvalidConfigurationError(f"{name} must be finite")
    if value < minimum or value > maximum:
        raise InvalidConfigurationError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _read_int(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    raw = _read_optional(env, name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise InvalidConfigurationError(
            f"{name} must be an integer, got {raw!r}"
        ) from None
    if value < minimum:
        raise InvalidConfigurationError(f"{name} must be at least {minimum}")
    return value


def _read_decimal(
    env: Mapping[str, str], name: str, default: Decimal
) -> Decimal:
    raw = _read_optional(env, name, str(default))
    try:
        value = Decimal(raw)
    except InvalidOperation:
        raise InvalidConfigurationError(
            f"{name} must be a decimal number, got {raw!r}"
        ) from None
    if not value.is_finite():
        raise InvalidConfigurationError(f"{name} must be finite")
    if value < 0:
        raise InvalidConfigurationError(f"{name} must be non-negative")
    return value
