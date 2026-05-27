"""Test package metadata for Pydantic AI Agent."""

import json
from pathlib import Path
import tomllib


def _repo_root() -> Path:
    """Return the repository root for metadata checks."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise AssertionError("Could not locate repository root")


_REPO_ROOT = _repo_root()
_EXPLICIT_RUNTIME_REQUIREMENTS = {
    "logfire==4.33.0",
    "pydantic-ai-slim==1.97.0",
    "anthropic>=0.97.0",
    "google-genai>=1.70.0",
    "tiktoken>=0.12.0",
    "fastmcp-slim[client,server]>=3.3.0",
    "markdownify>=1.2",
    "names-generator==0.2.0",
}


def test_runtime_requirements_are_explicit_for_home_assistant_installer() -> None:
    """Test runtime requirements do not rely on nested extras installation."""
    manifest = json.loads(
        (_REPO_ROOT / "custom_components/pydantic_ai_agent/manifest.json").read_text()
    )
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())

    manifest_requirements = set(manifest["requirements"])
    pyproject_dependencies = set(pyproject["project"]["dependencies"])

    assert _EXPLICIT_RUNTIME_REQUIREMENTS <= manifest_requirements
    assert _EXPLICIT_RUNTIME_REQUIREMENTS <= pyproject_dependencies
