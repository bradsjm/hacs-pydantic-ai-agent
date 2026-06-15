"""Constants for Pydantic AI Agent."""

from homeassistant.const import (
    CONF_API_KEY as HA_CONF_API_KEY,
)
from homeassistant.const import (
    CONF_LLM_HASS_API as HA_CONF_LLM_HASS_API,
)
from homeassistant.const import (
    CONF_NAME as HA_CONF_NAME,
)

CONF_API_KEY = HA_CONF_API_KEY
CONF_LLM_HASS_API = HA_CONF_LLM_HASS_API
CONF_NAME = HA_CONF_NAME

DOMAIN = "pydantic_ai_agent"

CONF_AGENT_NAME = "agent_name"
CONF_AI_TASK_NAME = "ai_task_name"
CONF_BASE_URL = "base_url"
CONF_CHAT_TEMPLATE_KWARG_KEY = "key"
CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE = "value_template"
CONF_CUSTOM_MODEL_NAMES = "custom_model_names"
CONF_DEFAULT_MODEL_PROFILE_ID = "default_model_profile_id"
CONF_DESCRIPTION = "description"
CONF_DISCOVERED = "discovered"
CONF_DISCOVERED_MODELS = "discovered_models"
CONF_DISCOVERED_MODELS_AT = "discovered_models_at"
CONF_DISCOVERED_MODELS_CACHE_KEY = "discovered_models_cache_key"
CONF_ENABLED = "enabled"
CONF_FALLBACK_MODEL_REFS = "fallback_model_refs"
CONF_LOGFIRE_INCLUDE_CONTENT = "logfire_include_content"
CONF_LOGFIRE_TOKEN = "logfire_token"
CONF_MAX_ITERATIONS = "max_iterations"
CONF_MAX_TOKENS = "max_tokens"
CONF_MCP_ALLOWED_TOOLS = "mcp_allowed_tools"
CONF_MCP_CALL_CACHE_ENABLED = "mcp_call_cache_enabled"
CONF_MCP_CALL_CACHE_TTL = "mcp_call_cache_ttl"
CONF_MCP_DEFERRED_LOADING = "mcp_deferred_loading"
CONF_MCP_HEADERS = "mcp_headers"
CONF_MCP_SECRET_HEADER_KEYS = "mcp_secret_header_keys"
CONF_MCP_INCLUDE_RETURN_SCHEMA = "mcp_include_return_schema"
CONF_MCP_SERVER_IDS = "mcp_server_ids"
CONF_MCP_URL = "mcp_url"
CONF_KEY_VALUE_JSON_VALUE = "json_value"
CONF_KEY_VALUE_IS_SECRET = "is_secret"
CONF_KEY_VALUE_KEY = "key"
CONF_KEY_VALUE_VALUE = "value"
CONF_MODEL = "model"
CONF_MODEL_PRICING = "model_pricing"
CONF_MODEL_PROFILES = "model_profiles"
CONF_MODEL_SETTINGS = "model_settings"
CONF_OPENAI_SUPPORTS_ENCRYPTED_REASONING_CONTENT = (
    "openai_supports_encrypted_reasoning_content"
)
CONF_OPENAI_SUPPORTS_STRICT_TOOL_DEFINITION = (
    "openai_supports_strict_tool_definition"
)
CONF_OUTPUT_MODE = "output_mode"
CONF_PRIMARY_MODEL_REF = "primary_model_ref"
CONF_PROMPT = "prompt"
CONF_PROVIDER_EXTRA_BODY = "provider_extra_body"
CONF_TEMPLATED_EXTRA_BODY = "templated_extra_body"
CONF_PROVIDER_HEADERS = "provider_headers"
CONF_PROVIDER_SECRET_HEADER_KEYS = "provider_secret_header_keys"
CONF_PROVIDER_METADATA = "provider_metadata"
CONF_PROVIDER_MODE = "provider_mode"
CONF_PROVIDER_SUBENTRY_ID = "provider_subentry_id"
CONF_SKILL_CONTENT = "content"
CONF_SKILL_REFERENCES = "references"
CONF_SKILLS = "skills"
CONF_STREAMING_ENABLED = "streaming_enabled"
CONF_STRUCTURED_OUTPUT_SUPPORT = "structured_output_support"
CONF_SUPPORTS_TOOLS = "supports_tools"
CONF_THINKING = "thinking"
CONF_THINKING_SUPPORT = "thinking_support"
CONF_TIMEOUT = "timeout"
CONF_TOOL_RETRIES = "tool_retries"
CONF_TODO_LIST_ENTITY_ID = "todo_list_entity_id"
CONF_VIRTUAL_WORKSPACE_ENABLED = "virtual_workspace_enabled"
CONF_WEB_FETCH_ENABLED = "web_fetch_enabled"

PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS = "openai_compatible_completions"
PROVIDER_OPENAI_COMPATIBLE_RESPONSES = "openai_compatible_responses"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GOOGLE_GEMINI = "google_gemini"
PROVIDER_MODES = (
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE_GEMINI,
)

SUBENTRY_TYPE_AI_TASK = "ai_task_data"
SUBENTRY_TYPE_CONVERSATION = "conversation"
SUBENTRY_TYPE_MCP_SERVER = "mcp_server"
SUBENTRY_TYPE_PROVIDER = "provider"
SUBENTRY_TYPE_SKILL = "skill"

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
DEFAULT_SKILL_NAME = "Skill"
DEFAULT_WORKSPACE_NAME = "Pydantic AI Agent"
DEFAULT_MCP_TIMEOUT = 10.0
DEFAULT_MCP_CALL_CACHE_TTL = 300
DEFAULT_TIMEOUT = 10.0
DEFAULT_TOOL_RETRIES = 3


def default_conversation_options() -> dict[str, object]:
    """Return default conversation subentry options."""
    return {
        CONF_AGENT_NAME: DEFAULT_AGENT_NAME,
    }


__all__ = ["CONF_API_KEY", "CONF_LLM_HASS_API", "CONF_NAME"]
