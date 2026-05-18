"""Constants for Pydantic AI Agent."""

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.helpers import llm

DOMAIN = "pydantic_ai_agent"

CONF_AGENT_NAME = "agent_name"
CONF_AI_TASK_NAME = "ai_task_name"
CONF_BASE_URL = "base_url"
CONF_CHAT_TEMPLATE_KWARGS = "chat_template_kwargs"
CONF_CHAT_TEMPLATE_KWARG_KEY = "key"
CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE = "value_template"
CONF_CONFIGURE_ADVANCED_MODEL_SETTINGS = "configure_advanced_model_settings"
CONF_FALLBACK_MODEL_SUBENTRY_IDS = "fallback_model_subentry_ids"
CONF_LOGFIRE_INCLUDE_CONTENT = "logfire_include_content"
CONF_LOGFIRE_TOKEN = "logfire_token"
CONF_ENABLE_SKILL_SCRIPT_EXECUTION = "enable_skill_script_execution"
CONF_MAX_ITERATIONS = "max_iterations"
CONF_MODEL = "model"
CONF_MODEL_SETTINGS = "model_settings"
CONF_MODEL_SUBENTRY_ID = "model_subentry_id"
CONF_MCP_ALLOWED_TOOLS = "mcp_allowed_tools"
CONF_MCP_HEADERS = "mcp_headers"
CONF_MCP_SERVER_IDS = "mcp_server_ids"
CONF_MCP_URL = "mcp_url"
CONF_OUTPUT_MODE = "output_mode"
CONF_PROMPT = "prompt"
CONF_PROVIDER_HEADERS = "provider_headers"
CONF_PROVIDER_MODE = "provider_mode"
CONF_SKILLS = "skills"
CONF_SKILLS_FOLDER = "skills_folder"
CONF_WEB_FETCH_ENABLED = "web_fetch_enabled"

PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS = "openai_compatible_completions"
PROVIDER_OPENAI_COMPATIBLE_RESPONSES = "openai_compatible_responses"
PROVIDER_MODES = (
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
)

SUBENTRY_TYPE_CONVERSATION = "conversation"
SUBENTRY_TYPE_AI_TASK = "ai_task_data"
SUBENTRY_TYPE_MCP_SERVER = "mcp_server"
SUBENTRY_TYPE_MODEL = "model"

OUTPUT_MODE_TOOL = "tool"
OUTPUT_MODE_NATIVE = "native"
OUTPUT_MODE_PROMPTED = "prompted"
STRUCTURED_OUTPUT_MODES = (
    OUTPUT_MODE_TOOL,
    OUTPUT_MODE_NATIVE,
    OUTPUT_MODE_PROMPTED,
)
DEFAULT_OUTPUT_MODE = OUTPUT_MODE_TOOL

DEFAULT_AGENT_NAME = "Conversation agent"
DEFAULT_AI_TASK_NAME = "AI task"
DEFAULT_SERVICE_NAME = "OpenAI-compatible"
DEFAULT_TIMEOUT = 10.0
DEFAULT_MCP_TIMEOUT = 10.0
DEFAULT_SKILLS_FOLDER = "/config/skills"


def default_conversation_options() -> dict[str, object]:
    """Return default conversation subentry options."""
    return {
        CONF_AGENT_NAME: DEFAULT_AGENT_NAME,
        CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    }


__all__ = ["CONF_LLM_HASS_API"]
