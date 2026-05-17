"""Pydantic AI provider helpers."""

from collections.abc import Mapping
from typing import Any

from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.helpers.httpx_client import get_async_client

from .const import CONF_BASE_URL
from .pydantic_ai_openai_compatible import (
    OpenAICompatibleChatModel,
    OpenAICompatibleProvider,
)


def normalise_base_url(value: object) -> str | None:
    """Return a normalized provider base URL."""
    if not value:
        return None
    return str(value).rstrip("/")


def openai_compatible_chat_model(
    hass: HomeAssistant,
    *,
    api_key: str,
    base_url: str | None,
    model_name: str,
) -> Any:
    """Build a Pydantic AI OpenAI-compatible chat model."""
    provider = OpenAICompatibleProvider(
        api_key=api_key,
        base_url=normalise_base_url(base_url),
        # Reuse Home Assistant's shared async client for its SSL, proxy, and
        # connection-pooling configuration.
        http_client=get_async_client(hass),
    )
    return OpenAICompatibleChatModel(model_name, provider=provider)


def openai_compatible_chat_model_from_config(
    hass: HomeAssistant, data: Mapping[str, Any], model_name: str
) -> Any:
    """Build a Pydantic AI OpenAI-compatible chat model from config entry data."""
    return openai_compatible_chat_model(
        hass,
        api_key=data[CONF_API_KEY],
        base_url=normalise_base_url(data.get(CONF_BASE_URL)),
        model_name=model_name,
    )
