"""Constants for the provider setup wizard."""

from datetime import timedelta

CATALOG_SOURCE_URL = "https://models.dev/api.json"
CATALOG_RETRY_PROVIDER_ID = "retry_catalog"
CUSTOM_PROVIDER_ID = "custom"
DATA_CATALOG_MANAGER = "provider_wizard_catalog_manager"
DEFAULT_MODEL_FILTER_THRESHOLD = 100
CONF_DRIVER = "driver"
CONF_FAMILY = "family"
CONF_INCLUDE_DEPRECATED = "include_deprecated"
CONF_INCLUDE_NON_TEXT_OUTPUT = "include_non_text_output"
CONF_INCLUDE_WITHOUT_STRUCTURED_OUTPUT = "include_without_structured_output"
CONF_INCLUDE_WITHOUT_TOOL_CALL = "include_without_tool_call"
CONF_PROVIDER_ID = "provider_id"
CONF_SELECTED_MODEL_IDS = "selected_model_ids"
MODEL_CATALOG_IDLE_TTL = timedelta(minutes=15)
MODEL_CATALOG_HARD_TTL = timedelta(hours=1)
MODEL_CATALOG_TIMEOUT = 10.0

MODE_LABELS = {
    "openai_compatible_completions": "Chat Completions",
    "openai_compatible_responses": "Responses",
    "anthropic": "Anthropic",
    "google_gemini": "Google Gemini",
}

OPENAI_COMPATIBLE_NPM = "@ai-sdk/openai-compatible"
OPENROUTER_NPM = "@openrouter/ai-sdk-provider"
