"""Diagnostics for Pydantic AI Agent."""

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr

from ._redaction import redact_data
from .const import (
    CONF_CHAT_TEMPLATE_KWARGS,
    CONF_DEFAULT_MODEL_PROFILE_ID,
    CONF_LOGFIRE_TOKEN,
    CONF_MCP_HEADERS,
    CONF_MCP_URL,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_PROMPT,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    CONF_SKILL_CONTENT,
    CONF_SKILL_REFERENCES,
    DOMAIN,
    SUBENTRY_TYPE_PROVIDER,
    SUBENTRY_TYPE_SKILL,
)
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
    CONF_MCP_URL,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    "password",
    "secret",
    "token",
    "x-api-key",
}

_MODEL_PROFILE_SENSITIVE_KEYS = _SENSITIVE_KEYS | {
    CONF_CHAT_TEMPLATE_KWARGS,
    "extra_body",
}


def _redact(data: dict[str, Any]) -> dict[str, Any]:
    """Return diagnostics data with sensitive fields redacted."""
    return redact_data(data, _SENSITIVE_KEYS)


def _redact_subentry_data(subentry_type: str, data: dict[str, Any]) -> dict[str, Any]:
    """Return redacted subentry data."""
    if subentry_type == SUBENTRY_TYPE_SKILL:
        return redact_data(
            data,
            _SENSITIVE_KEYS | {CONF_SKILL_CONTENT, CONF_SKILL_REFERENCES},
        )
    if subentry_type != SUBENTRY_TYPE_PROVIDER:
        return redact_data(data, _SENSITIVE_KEYS)
    redacted = dict(data)
    raw_profiles = redacted.pop(CONF_MODEL_PROFILES, None)
    redacted = redact_data(redacted, _SENSITIVE_KEYS)
    if isinstance(raw_profiles, Mapping):
        redacted[CONF_MODEL_PROFILES] = {
            profile_id: redact_data(dict(profile), _MODEL_PROFILE_SENSITIVE_KEYS)
            for profile_id, profile in raw_profiles.items()
            if isinstance(profile_id, str) and isinstance(profile, Mapping)
        }
    return redacted


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    subentries = []
    for subentry in entry.subentries.values():
        model_settings = subentry.data.get(CONF_MODEL_SETTINGS)
        model_profiles = subentry.data.get(CONF_MODEL_PROFILES)
        subentries.append(
            {
                "subentry_id": subentry.subentry_id,
                "subentry_type": subentry.subentry_type,
                "title": subentry.title,
                "data": _redact_subentry_data(
                    subentry.subentry_type, dict(subentry.data)
                ),
                "model": subentry.data.get(CONF_MODEL),
                "default_model_profile_id": subentry.data.get(
                    CONF_DEFAULT_MODEL_PROFILE_ID
                ),
                "model_profile_count": len(model_profiles)
                if isinstance(model_profiles, Mapping)
                else 0,
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
            "logfire_enabled": logfire_enabled(hass, entry),
            "logfire_active": logfire_active_for_entry(hass, entry),
            "logfire_include_content": logfire_include_content(hass, entry),
            "logfire_token_conflict": logfire_token_conflict(hass, entry),
        },
        "subentries": subentries,
        "runtime": {
            "loaded": hasattr(entry, "runtime_data"),
            **_runtime_diagnostics(entry),
        },
    }
    return _redact(diagnostics)


async def async_get_device_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry, device: dr.DeviceEntry
) -> dict[str, Any]:
    """Return diagnostics for one subentry device."""
    subentry_id = _device_subentry_id(device)
    subentries = []
    if subentry_id is not None:
        subentry = entry.subentries.get(subentry_id)
        if subentry is not None:
            model_settings = subentry.data.get(CONF_MODEL_SETTINGS)
            model_profiles = subentry.data.get(CONF_MODEL_PROFILES)
            subentries.append(
                {
                    "subentry_id": subentry.subentry_id,
                    "subentry_type": subentry.subentry_type,
                    "title": subentry.title,
                    "data": _redact_subentry_data(
                        subentry.subentry_type, dict(subentry.data)
                    ),
                    "model": subentry.data.get(CONF_MODEL),
                    "default_model_profile_id": subentry.data.get(
                        CONF_DEFAULT_MODEL_PROFILE_ID
                    ),
                    "model_profile_count": len(model_profiles)
                    if isinstance(model_profiles, Mapping)
                    else 0,
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
        },
        "device": {"subentry_id": subentry_id},
        "subentries": subentries,
        "runtime": {
            "loaded": hasattr(entry, "runtime_data"),
            "metrics": _runtime_metrics(entry, subentry_id),
        },
    }
    return _redact(diagnostics)


def _runtime_diagnostics(entry: ConfigEntry) -> dict[str, Any]:
    """Return safe config-entry runtime diagnostics."""
    runtime_data = getattr(entry, "runtime_data", None)
    if runtime_data is None:
        return {
            "configured_mcp_server_count": 0,
            "cached_mcp_server_count": 0,
            "cached_mcp_tool_counts": {},
        }
    return {
        "configured_mcp_server_count": len(runtime_data.mcp_servers),
        "cached_mcp_server_count": len(runtime_data.mcp_tool_cache),
        "cached_mcp_tool_counts": {
            server_id: len(tools)
            for server_id, tools in runtime_data.mcp_tool_cache.items()
        },
    }


def _runtime_metrics(entry: ConfigEntry, subentry_id: str | None) -> dict[str, Any]:
    """Return safe runtime metrics for one subentry."""
    runtime_data = getattr(entry, "runtime_data", None)
    if runtime_data is None or subentry_id is None:
        return {}
    return asdict(runtime_data.metrics.record_for(subentry_id))


def _device_subentry_id(device: dr.DeviceEntry) -> str | None:
    """Return the integration subentry id represented by a device."""
    for domain, identifier in device.identifiers:
        if domain == DOMAIN:
            parts = identifier.split(":", 2)
            if len(parts) != 3:
                return None
            subentry_id = parts[2]
            return subentry_id
    return None
