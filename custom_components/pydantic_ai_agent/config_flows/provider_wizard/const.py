"""Constants for the provider setup wizard."""

from datetime import timedelta

CATALOG_SOURCE_URL = "https://models.dev/api.json"
CATALOG_RETRY_PROVIDER_ID = "retry_catalog"
CUSTOM_PROVIDER_ID = "custom"
DATA_CATALOG_MANAGER = "provider_wizard_catalog_manager"
DEFAULT_MODEL_FILTER_THRESHOLD = 100
CONF_DRIVER = "driver"
CONF_FAMILY = "family"
CONF_CATALOG_PROVIDER_ID = "catalog_provider_id"
CONF_HIDE_DEPRECATED = "hide_deprecated"
CONF_HIDE_NON_TEXT_OUTPUT = "hide_non_text_output"
CONF_HIDE_WITHOUT_STRUCTURED_OUTPUT = "hide_without_structured_output"
CONF_HIDE_WITHOUT_TOOL_CALL = "hide_without_tool_call"
CONF_PROVIDER_ID = "provider_id"
CONF_SELECTED_MODEL_IDS = "selected_model_ids"
SECTION_ADVANCED_FILTERS = "advanced_filters"
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
