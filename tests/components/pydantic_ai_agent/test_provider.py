"""Test provider construction helpers."""

import httpx
import pytest
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant

from custom_components.pydantic_ai_agent.const import (
    CONF_BASE_URL,
    CONF_PROVIDER_HEADERS,
)
from custom_components.pydantic_ai_agent.provider import (
    list_anthropic_model_names,
    list_google_gemini_model_names,
    normalise_base_url,
    openai_compatible_completions_model_from_config,
    openai_compatible_client_from_config,
    openai_compatible_responses_model_from_config,
)
from custom_components.pydantic_ai_agent.openai_compatible_adapter import (
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


async def test_list_anthropic_model_names_parses_paginated_response(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test Anthropic model discovery extracts IDs from all pages."""
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.params.get("after_id") == "claude-1":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "claude-2", "display_name": "Claude 2"},
                    ],
                    "has_more": False,
                },
            )
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "claude-1", "display_name": "Claude 1"},
                ],
                "last_id": "claude-1",
                "has_more": True,
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(
        "custom_components.pydantic_ai_agent.provider.get_async_client",
        lambda _hass: http_client,
    )
    try:
        model_names = await list_anthropic_model_names(
            hass,
            {
                CONF_API_KEY: "sk-ant-test",
                CONF_BASE_URL: "https://api.anthropic.com/v1",
                CONF_PROVIDER_HEADERS: {"X-Proxy": "on"},
            },
            timeout=10.0,
        )
    finally:
        await http_client.aclose()

    assert model_names == ["claude-1", "claude-2"]
    assert str(requests[0].url) == "https://api.anthropic.com/v1/models?limit=1000"
    assert requests[0].headers["x-api-key"] == "sk-ant-test"
    assert requests[0].headers["anthropic-version"] == "2023-06-01"
    assert requests[0].headers["x-proxy"] == "on"


async def test_list_google_gemini_model_names_filters_generate_content_models(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test Gemini model discovery keeps text generation models only."""
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "models/gemini-2.5-pro",
                        "baseModelId": "gemini-2.5-pro",
                        "displayName": "Gemini 2.5 Pro",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/gemini-embed",
                        "baseModelId": "gemini-embed",
                        "supportedGenerationMethods": ["embedContent"],
                    },
                    {
                        "name": "models/gemini-name-only",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                ]
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(
        "custom_components.pydantic_ai_agent.provider.get_async_client",
        lambda _hass: http_client,
    )
    try:
        model_names = await list_google_gemini_model_names(
            hass,
            {
                CONF_API_KEY: "google-test",
                CONF_BASE_URL: "https://generativelanguage.googleapis.com/v1beta",
                CONF_PROVIDER_HEADERS: {"X-Proxy": "on"},
            },
            timeout=10.0,
        )
    finally:
        await http_client.aclose()

    assert model_names == ["gemini-2.5-pro", "gemini-name-only"]
    assert str(requests[0].url) == (
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000"
    )
    assert requests[0].headers["x-goog-api-key"] == "google-test"
    assert requests[0].headers["x-proxy"] == "on"
