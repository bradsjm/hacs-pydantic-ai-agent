"""Shared configuration for live provider integration tests."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from homeassistant.const import CONF_API_KEY, CONF_NAME

from custom_components.pydantic_ai_agent.const import (
    CONF_BASE_URL,
    CONF_PROVIDER_MODE,
    OUTPUT_MODE_NATIVE,
    OUTPUT_MODE_PROMPTED,
    OUTPUT_MODE_TOOL,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
)
from custom_components.pydantic_ai_agent.provider_validation import (
    ProviderValidationError,
)

REPO_ROOT = Path(__file__).parents[4]
ENV_FILE = REPO_ROOT / ".env"

REQUIRED_CONNECTION_ENV = ("OPENAI_API_KEY", "OPENAI_BASE_URL")
DEFAULT_MODEL_LIMIT = 5
TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
MODEL_LIST_TIMEOUT = 30.0
PROVIDER_INTEGRATION_TIMEOUT = 60.0

CONVERSATION_SENTINEL = "PAI_E2E_CONVERSATION_OK"
AI_TASK_SENTINEL = "PAI_E2E_AI_TASK_OK"
AI_TASK_STRUCTURED_SENTINEL = "PAI_E2E_AI_TASK_STRUCTURED_OK"
STREAM_SENTINEL = "PAI_E2E_STREAM_OK"
TOOL_SENTINEL = "PAI_E2E_TOOL_OK"
SKILL_SENTINEL = "PAI_E2E_SKILL_OK"

TEST_LLM_API_ID = "pydantic-ai-agent-integration-test"
PROVIDER_ID = "provider_integration_provider"
MODEL_PROFILE_ID = "provider_integration_model_profile"
MODEL_REF = f"{PROVIDER_ID}:{MODEL_PROFILE_ID}"
WORKSPACE_SKILL_ID = "pydantic_ai_agent_integration_skill"
UNSELECTED_WORKSPACE_SKILL_ID = "pydantic_ai_agent_integration_unselected_skill"

STRUCTURED_OUTPUT_SKIP_REASONS = {
    "invalid_model",
    "invalid_provider_config",
    "unsupported_output_mode",
}
STRUCTURED_OUTPUT_MODES = (
    OUTPUT_MODE_TOOL,
    OUTPUT_MODE_NATIVE,
    OUTPUT_MODE_PROMPTED,
)


@dataclass(frozen=True, kw_only=True)
class ProviderIntegrationConfig:
    """Live provider configuration loaded from the test environment."""

    api_key: str = field(repr=False)
    model: str
    base_url: str

    @property
    def provider_data(self) -> dict[str, str]:
        """Return config-entry provider data for OpenAI-compatible mode."""
        return {
            CONF_NAME: "Live OpenAI-compatible Provider",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: self.api_key,
            CONF_BASE_URL: self.base_url,
        }


@dataclass(frozen=True, kw_only=True)
class ModelParam:
    """One provider model parameter for integration tests."""

    model: str
    skip_reason: str | None = None


@dataclass(frozen=True, kw_only=True)
class StructuredOutputSupport:
    """Structured output modes supported by the configured provider model."""

    supported_modes: tuple[str, ...]
    failures: Mapping[str, ProviderValidationError]

    def skip_if_unsupported(self, output_mode: str) -> None:
        """Skip the current test if the output mode is unsupported."""
        if output_mode in self.supported_modes:
            return
        err = self.failures[output_mode]
        pytest.skip(
            f"Configured provider integration model does not support {output_mode} "
            f"structured output: {err.reason}: {err.message}"
        )


class Secret(str):
    """String value with a redacted repr for pytest failure output."""

    def __repr__(self) -> str:
        """Return a redacted representation."""
        return "'<redacted>'"
