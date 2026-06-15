"""Shared configuration for live provider integration tests."""

from dataclasses import dataclass, field
from pathlib import Path

from custom_components.pydantic_ai_agent.const import (
    CONF_BASE_URL,
    CONF_DEFAULT_MODEL_PROFILE_ID,
    CONF_ENABLED,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_PROVIDER_MODE,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
)
from homeassistant.const import CONF_API_KEY, CONF_NAME

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

@dataclass(frozen=True, kw_only=True)
class ProviderIntegrationConfig:
    """Live provider configuration loaded from the test environment."""

    api_key: str = field(repr=False)
    model: str
    base_url: str

    @property
    def provider_data(self) -> dict[str, object]:
        """Return config-entry provider data for OpenAI-compatible mode."""
        return {
            CONF_NAME: "Live OpenAI-compatible Provider",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: self.api_key,
            CONF_BASE_URL: self.base_url,
            CONF_DEFAULT_MODEL_PROFILE_ID: MODEL_PROFILE_ID,
            CONF_MODEL_PROFILES: {
                MODEL_PROFILE_ID: {
                    "id": MODEL_PROFILE_ID,
                    CONF_MODEL: self.model,
                    CONF_ENABLED: True,
                    "thinking_support": "supported",
                    "structured_output_support": "json_schema",
                    "supports_tools": True,
                    "openai_supports_strict_tool_definition": True,
                    "openai_supports_encrypted_reasoning_content": False,
                }
            },
        }


@dataclass(frozen=True, kw_only=True)
class ModelParam:
    """One provider model parameter for integration tests."""

    model: str
    skip_reason: str | None = None

class Secret(str):
    """String value with a redacted repr for pytest failure output."""

    def __repr__(self) -> str:
        """Return a redacted representation."""
        return "'<redacted>'"
