"""Diagnostics for Pydantic AI Agent."""

from collections.abc import Mapping
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers.redact import async_redact_data

from .const import (
    CONF_BASE_URL,
    CONF_LOGFIRE_TOKEN,
    CONF_MODEL,
    CONF_MODEL_SETTINGS,
    CONF_PROMPT,
)
from .const import CONF_MCP_HEADERS, CONF_MCP_URL
from .logfire_support import (
    logfire_active_for_entry,
    logfire_enabled,
    logfire_include_content,
    logfire_token_conflict,
)
from .mcp import redact_mcp_url_password

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


def _redaction_keys(data: object) -> set[object]:
    """Return keys that should be redacted, preserving original key casing."""
    keys: set[object] = set()
    if isinstance(data, Mapping):
        for key, value in data.items():
            key_text = str(key).lower()
            if key in _SENSITIVE_KEYS or key_text in _SENSITIVE_KEYS:
                keys.add(key)
            elif key_text.endswith("_token") or key_text.endswith("-token"):
                keys.add(key)
            keys.update(_redaction_keys(value))
    elif isinstance(data, list):
        for item in data:
            keys.update(_redaction_keys(item))
    return keys


def _redact(data: dict[str, Any]) -> dict[str, Any]:
    """Return diagnostics data with sensitive fields redacted."""
    redacted = async_redact_data(data, _redaction_keys(data))
    _redact_mcp_urls(redacted)
    return redacted


def _redact_mcp_urls(data: object) -> None:
    """Redact MCP URL passwords in-place without modifying query parameters."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key == CONF_MCP_URL and isinstance(value, str):
                data[key] = redact_mcp_url_password(value)
            else:
                _redact_mcp_urls(value)
    elif isinstance(data, list):
        for item in data:
            _redact_mcp_urls(item)


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
