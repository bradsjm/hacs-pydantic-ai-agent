"""Constants for the provider setup wizard."""

CATALOG_SOURCE_URL = "https://models.dev/api.json"
CUSTOM_PROVIDER_ID = "custom"
DEFAULT_MODEL_FILTER_THRESHOLD = 100

MODE_LABELS = {
    "openai_compatible_completions": "Chat Completions",
    "openai_compatible_responses": "Responses",
    "anthropic": "Anthropic",
    "google_gemini": "Google Gemini",
}

OPENAI_COMPATIBLE_NPM = "@ai-sdk/openai-compatible"
OPENROUTER_NPM = "@openrouter/ai-sdk-provider"
