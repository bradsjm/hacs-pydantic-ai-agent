"""Test provider construction helpers."""

from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant

from custom_components.pydantic_ai_agent.const import CONF_BASE_URL, CONF_PROVIDER_HEADERS
from custom_components.pydantic_ai_agent.provider import (
    normalise_base_url,
    openai_compatible_completions_model_from_config,
    openai_compatible_client_from_config,
    openai_compatible_responses_model_from_config,
)
from custom_components.pydantic_ai_agent.pydantic_ai_openai_compatible import (
    OpenAICompatibleChatModel,
    OpenAICompatibleResponsesModel,
)


def test_normalise_base_url_strips_trailing_slash_and_preserves_empty() -> None:
    """Test base URL normalization only stores explicit configured URLs."""
    assert normalise_base_url(None) is None
    assert normalise_base_url("") is None
    assert normalise_base_url("https://api.example.com/v1/") == (
        "https://api.example.com/v1"
    )


def test_openai_compatible_client_from_config_uses_entry_data(
    hass: HomeAssistant,
) -> None:
    """Test client construction uses HA HTTP client and normalized config data."""
    client = openai_compatible_client_from_config(
        hass,
        {
            CONF_API_KEY: "sk-test",
            CONF_BASE_URL: "https://api.example.com/v1/",
            CONF_PROVIDER_HEADERS: {"X-Test": "enabled"},
        },
    )

    assert client.api_key == "sk-test"
    assert client.base_url == "https://api.example.com/v1"
    assert client.headers == {"X-Test": "enabled"}
    assert client.auth_headers == {
        "Authorization": "Bearer sk-test",
        "X-Test": "enabled",
    }


def test_openai_compatible_completions_model_from_config_uses_in_repo_provider(
    hass: HomeAssistant,
) -> None:
    """Test model construction uses the in-repo OpenAI-compatible adapter."""
    model = openai_compatible_completions_model_from_config(
        hass,
        {
            CONF_API_KEY: "sk-test",
            CONF_PROVIDER_HEADERS: {"X-Test": "enabled"},
        },
        "gpt-test",
    )

    assert isinstance(model, OpenAICompatibleChatModel)
    assert model.model_name == "gpt-test"
    assert model.client.base_url == "https://api.openai.com/v1"
    assert model.client.headers == {"X-Test": "enabled"}


def test_openai_compatible_responses_model_from_config_uses_in_repo_provider(
    hass: HomeAssistant,
) -> None:
    """Test Responses model construction uses the in-repo adapter."""
    model = openai_compatible_responses_model_from_config(
        hass,
        {
            CONF_API_KEY: "sk-test",
            CONF_PROVIDER_HEADERS: {"X-Test": "enabled"},
        },
        "gpt-test",
    )

    assert isinstance(model, OpenAICompatibleResponsesModel)
    assert model.model_name == "gpt-test"
    assert model.client.base_url == "https://api.openai.com/v1"
    assert model.client.headers == {"X-Test": "enabled"}
