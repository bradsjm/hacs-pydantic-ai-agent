"""Diagnostics for Pydantic AI Agent."""

from collections.abc import Mapping
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant

from ._redaction import redact_data
from .const import (
    CONF_BASE_URL,
    CONF_LOGFIRE_TOKEN,
    CONF_MODEL,
    CONF_MODEL_SETTINGS,
    CONF_PROMPT,
)
from .const import CONF_MCP_HEADERS
from .logfire_support import (
    logfire_active_for_entry,
    logfire_enabled,
    logfire_include_content,
    logfire_token_conflict,
)

_SENSITIVE_KEYS = {
    CONF_API_KEY,
    CONF_LOGFIRE_TOKEN,
    CONF_PROMPT,
    "api_key",
    "authorization",
    "cookie",
    "extra_headers",
    CONF_MCP_HEADERS,
    "password",
    "secret",
    "token",
    "x-api-key",
}


def _redact(data: dict[str, Any]) -> dict[str, Any]:
    """Return diagnostics data with sensitive fields redacted."""
    return redact_data(data, _SENSITIVE_KEYS)


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    subentries = []
    for subentry in entry.subentries.values():
        model_settings = subentry.data.get(CONF_MODEL_SETTINGS)
        subentries.append(
            {
                "subentry_id": subentry.subentry_id,
                "subentry_type": subentry.subentry_type,
                "title": subentry.title,
                "data": dict(subentry.data),
                "model": subentry.data.get(CONF_MODEL),
                "ha_tools_enabled": bool(subentry.data.get(CONF_LLM_HASS_API)),
                "model_settings_keys": sorted(model_settings)
                if isinstance(model_settings, Mapping)
                else [],
            }
        )

    diagnostics = {
        "entry": {
            "entry_id": entry.entry_id,
            "title": entry.title,
            "state": entry.state.value,
            "data": dict(entry.data),
            "base_url_configured": bool(entry.data.get(CONF_BASE_URL)),
            "logfire_enabled": logfire_enabled(entry),
            "logfire_active": logfire_active_for_entry(entry),
            "logfire_include_content": logfire_include_content(entry),
            "logfire_token_conflict": logfire_token_conflict(entry),
        },
        "subentries": subentries,
        "runtime": {
            "loaded": hasattr(entry, "runtime_data"),
        },
    }
    return _redact(diagnostics)
