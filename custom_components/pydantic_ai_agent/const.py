"""Constants for Pydantic AI Agent."""

from logging import getLogger

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.helpers import llm

DOMAIN = "pydantic_ai_agent"
LOGGER = getLogger(__package__)

CONF_AGENT_NAME = "agent_name"
CONF_BASE_URL = "base_url"
CONF_CONFIGURE_ADVANCED_MODEL_SETTINGS = "configure_advanced_model_settings"
CONF_MODEL = "model"
CONF_MODEL_SETTINGS = "model_settings"
CONF_PROMPT = "prompt"
CONF_PROVIDER_MODE = "provider_mode"

PROVIDER_OPENAI = "openai"
PROVIDER_OPENAI_COMPATIBLE = "openai_compatible"
PROVIDER_MODES = (PROVIDER_OPENAI, PROVIDER_OPENAI_COMPATIBLE)

SUBENTRY_TYPE_CONVERSATION = "conversation"
SUBENTRY_TYPE_AI_TASK = "ai_task_data"

DEFAULT_AGENT_NAME = "Pydantic AI Agent"
DEFAULT_SERVICE_NAME = "Pydantic AI Agent"
DEFAULT_TIMEOUT = 10.0

DEFAULT_CONVERSATION_OPTIONS = {
    CONF_AGENT_NAME: DEFAULT_AGENT_NAME,
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
}

__all__ = ["CONF_LLM_HASS_API"]
