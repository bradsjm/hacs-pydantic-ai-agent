"""Pydantic AI provider helpers."""

from collections.abc import Mapping
from typing import Any

from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.helpers.httpx_client import get_async_client

from .const import CONF_BASE_URL, CONF_PROVIDER_HEADERS
from .openai_compatible_client import AsyncOpenAICompatible
from .pydantic_ai_openai_compatible import (
    OpenAICompatibleChatModel,
    OpenAICompatibleProvider,
    OpenAICompatibleResponsesModel,
)


def normalise_base_url(value: object) -> str | None:
    """Return a normalized provider base URL."""
    if not value:
        return None
    return str(value).rstrip("/")


def openai_compatible_completions_model(
    hass: HomeAssistant,
    *,
    api_key: str,
    base_url: str | None,
    headers: dict[str, str] | None = None,
    model_name: str,
) -> Any:
    """Build a Pydantic AI OpenAI-compatible Completions model."""
    provider = OpenAICompatibleProvider(
        api_key=api_key,
        base_url=normalise_base_url(base_url),
        headers=headers,
        name="openai-compatible-completions",
        # Reuse Home Assistant's shared async client for its SSL, proxy, and
        # connection-pooling configuration.
        http_client=get_async_client(hass),
    )
    return OpenAICompatibleChatModel(model_name, provider=provider)


def openai_compatible_responses_model(
    hass: HomeAssistant,
    *,
    api_key: str,
    base_url: str | None,
    headers: dict[str, str] | None = None,
    model_name: str,
) -> Any:
    """Build a Pydantic AI OpenAI-compatible Responses model."""
    provider = OpenAICompatibleProvider(
        api_key=api_key,
        base_url=normalise_base_url(base_url),
        headers=headers,
        name="openai-compatible-responses",
        # Reuse Home Assistant's shared async client for its SSL, proxy, and
        # connection-pooling configuration.
        http_client=get_async_client(hass),
    )
    return OpenAICompatibleResponsesModel(model_name, provider=provider)


def openai_compatible_client_from_config(
    hass: HomeAssistant, data: Mapping[str, Any]
) -> AsyncOpenAICompatible:
    """Build a lightweight OpenAI-compatible client from config entry data."""
    headers = data.get(CONF_PROVIDER_HEADERS)
    provider = OpenAICompatibleProvider(
        api_key=data[CONF_API_KEY],
        base_url=normalise_base_url(data.get(CONF_BASE_URL)),
        headers=dict(headers) if isinstance(headers, Mapping) else None,
        http_client=get_async_client(hass),
    )
    return provider.client


def openai_compatible_completions_model_from_config(
    hass: HomeAssistant, data: Mapping[str, Any], model_name: str
) -> Any:
    """Build a Pydantic AI Completions model from config entry data."""
    headers = data.get(CONF_PROVIDER_HEADERS)
    return openai_compatible_completions_model(
        hass,
        api_key=data[CONF_API_KEY],
        base_url=normalise_base_url(data.get(CONF_BASE_URL)),
        headers=dict(headers) if isinstance(headers, Mapping) else None,
        model_name=model_name,
    )


def openai_compatible_responses_model_from_config(
    hass: HomeAssistant, data: Mapping[str, Any], model_name: str
) -> Any:
    """Build a Pydantic AI Responses model from config entry data."""
    headers = data.get(CONF_PROVIDER_HEADERS)
    return openai_compatible_responses_model(
        hass,
        api_key=data[CONF_API_KEY],
        base_url=normalise_base_url(data.get(CONF_BASE_URL)),
        headers=dict(headers) if isinstance(headers, Mapping) else None,
        model_name=model_name,
    )
