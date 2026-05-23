"""Constants for the provider setup wizard."""

from datetime import timedelta

CATALOG_SOURCE_URL = "https://models.dev/api.json"
CUSTOM_PROVIDER_ID = "custom"
DATA_CATALOG_MANAGER = "provider_wizard_catalog_manager"
DEFAULT_MODEL_FILTER_THRESHOLD = 100
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
