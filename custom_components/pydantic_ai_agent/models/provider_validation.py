"""Provider validation helpers for Pydantic AI Agent."""

from collections.abc import Mapping
from typing import Any

from homeassistant.core import HomeAssistant

from ..const import (
    CONF_PROVIDER_MODE,
    DEFAULT_TIMEOUT,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE_GEMINI,
)
from ._provider_validation_errors import (
    ProviderValidationError,
    format_api_error,
    map_http_error,
)
from .provider import (
    list_anthropic_model_names,
    list_google_gemini_model_names,
    openai_compatible_client_from_config,
)

__all__ = [
    "ProviderValidationError",
    "_format_api_error",
    "_map_http_error",
    "async_list_provider_model_names",
]

_format_api_error = format_api_error
_map_http_error = map_http_error


async def async_list_provider_model_names(
    hass: HomeAssistant, data: Mapping[str, Any]
) -> list[str]:
    """Return model names advertised by the configured provider."""
    provider_mode = data[CONF_PROVIDER_MODE]
    if provider_mode == PROVIDER_ANTHROPIC:
        return await list_anthropic_model_names(hass, data, timeout=DEFAULT_TIMEOUT)
    if provider_mode == PROVIDER_GOOGLE_GEMINI:
        return await list_google_gemini_model_names(hass, data, timeout=DEFAULT_TIMEOUT)
    client = openai_compatible_client_from_config(hass, data)
    return await client.models.list(timeout=DEFAULT_TIMEOUT)
