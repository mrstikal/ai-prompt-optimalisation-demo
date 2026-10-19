from __future__ import annotations

import os
import platform
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from dspy_lab.config import (
    PROJECT_PYTHON_PIN,
    PROJECT_ROOT,
    UnsupportedPythonError,
    collect_runtime_versions,
    ensure_supported_python,
    verify_runtime_contract,
)


def test_running_interpreter_is_the_pinned_project_python() -> None:
    assert platform.python_version() == PROJECT_PYTHON_PIN
    assert sys.version_info[:3] == (3, 14, 8)
    assert (PROJECT_ROOT / ".python-version").read_text().strip() == (
        PROJECT_PYTHON_PIN
    )


def test_pyproject_requires_the_supported_python_range() -> None:
    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert pyproject["project"]["requires-python"] == ">=3.14,<3.15"
    assert pyproject["project"]["dependencies"] == [
        "dspy==3.4.1",
        "gepa[dspy]==0.1.4",
        "litellm>=1.65.8",
        "pydantic>=2.11,<3",
        "streamlit==1.66.0",
    ]


def test_lockfile_is_frozen_and_pins_the_contract_versions() -> None:
    lock_path = PROJECT_ROOT / "uv.lock"
    assert lock_path.is_file()
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    packages = {entry["name"]: entry["version"] for entry in lock["package"]}
    assert lock["requires-python"] == "==3.14.*"
    assert packages["dspy"] == "3.4.1"
    assert packages["gepa"] == "0.1.4"
    assert packages["litellm"] == "1.106.0"
    assert packages["pydantic"] == "2.14.1"
    assert packages["streamlit"] == "1.66.0"


@pytest.mark.parametrize(
    "version_info",
    [(3, 15, 0), (3, 15, 1), (3, 16, 0), (3, 13, 9), (2, 7, 18)],
)
def test_unsupported_python_versions_are_rejected(
    version_info: tuple[int, ...],
) -> None:
    with pytest.raises(UnsupportedPythonError):
        ensure_supported_python(version_info)


@pytest.mark.parametrize("version_info", [(3, 14, 0), (3, 14, 8)])
def test_supported_python_versions_are_accepted(
    version_info: tuple[int, ...],
) -> None:
    ensure_supported_python(version_info)


def test_runtime_versions_are_installed() -> None:
    versions = collect_runtime_versions()
    assert versions["python"] == PROJECT_PYTHON_PIN
    assert versions["application"] == "0.1.0"
    assert versions["dspy"] == "3.4.1"
    assert versions["gepa"] == "0.1.4"
    assert versions["litellm"] == "1.106.0"
    assert versions["pydantic"] == "2.14.1"
    assert versions["streamlit"] == "1.66.0"


def test_gepa_and_litellm_are_genuine_non_stub_imports() -> None:
    versions = verify_runtime_contract()
    assert versions["dspy"] == "3.4.1"
    assert versions["gepa"] == "0.1.4"
    assert versions["litellm"] == "1.106.0"
    assert versions["pydantic"] == "2.14.1"

    import dspy
    import gepa
    import litellm

    assert isinstance(dspy.GEPA, type)
    assert dspy.GEPA.__module__.startswith("dspy.teleprompt.gepa")
    assert callable(dspy.GEPA.compile)
    assert callable(gepa.optimize)
    assert callable(litellm.completion)


def _run_with_network_blocked(imports: str) -> subprocess.CompletedProcess[str]:
    code = f"""
import socket

class NetworkBlocked(RuntimeError):
    pass

class BlockedSocket:
    def __init__(self, *args, **kwargs):
        raise NetworkBlocked("network access during import")

def blocked(*args, **kwargs):
    raise NetworkBlocked("network access during import")

socket.socket = BlockedSocket
socket.create_connection = blocked
socket.getaddrinfo = blocked
{imports}
"""
    env = os.environ.copy()
    source_path = str(PROJECT_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        source_path if not existing else os.pathsep.join([source_path, existing])
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_package_import_performs_no_network_io() -> None:
    result = _run_with_network_blocked("import dspy_lab\nimport dspy_lab.config")
    assert result.returncode == 0, result.stderr


def test_dependency_import_performs_no_network_io() -> None:
    result = _run_with_network_blocked(
        "import dspy_lab\nimport dspy\nimport gepa\nimport litellm\n"
        "import streamlit"
    )
    assert result.returncode == 0, result.stderr


def test_package_disables_litellm_remote_cost_map_fetch() -> None:
    code = """
import os
os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
import dspy_lab
assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
import litellm
"""
    result = _run_with_network_blocked(code)
    assert result.returncode == 0, result.stderr


def test_env_example_has_names_and_no_usable_secret() -> None:
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "OPENROUTER_API_KEY=\n" in text
    assert "sk-or-" not in text
    for name in (
        "TASK_MODEL",
        "REFLECTION_MODEL",
        "TASK_PROVIDER",
        "REFLECTION_PROVIDER",
        "MAX_OPTIMIZATION_CALLS",
        "MAX_OPTIMIZATION_COST_USD",
        "MAX_OPTIMIZATION_METRIC_CALLS",
    ):
        assert f"{name}=" in text


def test_gitignore_protects_secrets_and_local_run_artifacts() -> None:
    lines = {
        line.strip()
        for line in (PROJECT_ROOT / ".gitignore").read_text().splitlines()
    }
    assert ".env" in lines
    assert "!.env.example" in lines
    assert "/artifacts/" in lines
    assert "/runs/" in lines
    assert "/logs/" in lines


def test_readme_documents_setup_and_stage_status() -> None:
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Status: S1-S4 implemented" in text
    assert "uv python install 3.14.8" in text
    assert "uv sync --frozen" in text
    assert "3.14.8" in text
    assert "data/editorial_rules_v1.md" in text
    assert "data/v1/train.jsonl" in text
    assert "data/v1/validation.jsonl" in text
    assert "data/v1/test.jsonl" in text
    assert "data/v1/split_manifest.json" in text
    assert "data/v1/reviews.jsonl" in text
    assert "data/v1/coverage_report.json" in text
