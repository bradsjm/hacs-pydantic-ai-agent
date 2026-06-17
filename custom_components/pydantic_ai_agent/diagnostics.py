"""Diagnostics for Pydantic AI Agent."""

from collections.abc import Mapping
from typing import Any, cast

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr

from .const import (
    CONF_AGENT_NAME,
    CONF_AI_TASK_NAME,
    CONF_DEFAULT_MODEL_PROFILE_ID,
    CONF_FALLBACK_MODEL_REFS,
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_DEFERRED_LOADING,
    CONF_MCP_HEADERS,
    CONF_MCP_INCLUDE_RETURN_SCHEMA,
    CONF_MCP_SERVER_IDS,
    CONF_MCP_TOOL_MODE,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_PRIMARY_MODEL_REF,
    CONF_SKILL_CONTENT,
    CONF_SKILL_REFERENCES,
    CONF_SKILLS,
    CONF_TODO_LIST_ENTITY_ID,
    CONF_VIRTUAL_WORKSPACE_ENABLED,
    CONF_WEB_FETCH_ENABLED,
    DOMAIN,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_MCP_SERVER,
    SUBENTRY_TYPE_PROVIDER,
    SUBENTRY_TYPE_SKILL,
)
from .mcp.entry_helpers import effective_mcp_tool_mode
from .models.model_profiles import primary_model_profile
from .models.structured_output import resolved_structured_output_mode
from .observability.logfire_support import (
    logfire_active_for_entry,
    logfire_enabled,
    logfire_include_content,
    logfire_token_conflict,
)
from .observability.run_diagnostics import bound_diagnostics_data
from .runtime.redaction import redact_data


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    subentries = []
    for subentry in entry.subentries.values():
        subentries.append(_subentry_diagnostics(entry, subentry))

    diagnostics = {
        "entry": {
            "entry_id": entry.entry_id,
            "title": entry.title,
            "state": entry.state.value,
            "data": redact_data(dict(entry.data)),
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
    return cast(dict[str, Any], bound_diagnostics_data(diagnostics))


async def async_get_device_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry, device: dr.DeviceEntry
) -> dict[str, Any]:
    """Return diagnostics for one subentry device."""
    subentry_id = _device_subentry_id(device)
    subentries = []
    if subentry_id is not None:
        subentry = entry.subentries.get(subentry_id)
        if subentry is not None:
            subentries.append(_subentry_diagnostics(entry, subentry))
    diagnostics = {
        "entry": {
            "entry_id": entry.entry_id,
            "title": entry.title,
            "state": entry.state.value,
        },
        "device": {"subentry_id": subentry_id},
        "subentries": subentries,
        "runtime": {"loaded": hasattr(entry, "runtime_data")},
    }
    return cast(dict[str, Any], bound_diagnostics_data(diagnostics))


def _subentry_diagnostics(
    entry: ConfigEntry, subentry: ConfigSubentry
) -> dict[str, Any]:
    """Return redacted diagnostics for one config subentry."""
    model_settings = subentry.data.get(CONF_MODEL_SETTINGS)
    model_profiles = subentry.data.get(CONF_MODEL_PROFILES)
    return {
        "subentry_id": subentry.subentry_id,
        "subentry_type": subentry.subentry_type,
        "title": subentry.title,
        "data": redact_data(dict(subentry.data)),
        "configuration_summary": _configuration_summary(entry, subentry),
        "model": subentry.data.get(CONF_MODEL),
        "default_model_profile_id": subentry.data.get(CONF_DEFAULT_MODEL_PROFILE_ID),
        "model_profile_count": len(model_profiles)
        if isinstance(model_profiles, Mapping)
        else 0,
        "ha_tools_enabled": bool(subentry.data.get(CONF_LLM_HASS_API)),
        "model_settings_keys": sorted(model_settings)
        if isinstance(model_settings, Mapping)
        else [],
    }


def _runtime_diagnostics(entry: ConfigEntry) -> dict[str, Any]:
    """Return safe config-entry runtime diagnostics."""
    runtime_data = getattr(entry, "runtime_data", None)
    if runtime_data is None:
        return {}
    diagnostics: dict[str, Any] = {}
    if runtime_data.latest_run_diagnostics:
        diagnostics["latest_run_diagnostics"] = redact_data(
            runtime_data.latest_run_diagnostics
        )
    if runtime_data.latest_stream_traces:
        diagnostics["latest_stream_traces"] = redact_data(
            runtime_data.latest_stream_traces
        )
    diagnostics["mcp_server_count"] = len(getattr(runtime_data, "mcp_servers", {}))
    diagnostics["cached_mcp_server_count"] = len(
        getattr(runtime_data, "mcp_tool_cache", {})
    )
    return diagnostics


def _configuration_summary(
    entry: ConfigEntry, subentry: ConfigSubentry
) -> dict[str, Any]:
    """Return a compact, unredacted configuration summary for one subentry."""
    data = subentry.data
    summary: dict[str, Any] = {
        "subentry_type": subentry.subentry_type,
    }
    if subentry.subentry_type in {SUBENTRY_TYPE_CONVERSATION, SUBENTRY_TYPE_AI_TASK}:
        skill_ids = data.get(CONF_SKILLS)
        mcp_server_ids = data.get(CONF_MCP_SERVER_IDS)
        fallback_refs = data.get(CONF_FALLBACK_MODEL_REFS)
        summary.update(
            {
                "name": data.get(CONF_AGENT_NAME, data.get(CONF_AI_TASK_NAME)),
                CONF_PRIMARY_MODEL_REF: data.get(CONF_PRIMARY_MODEL_REF),
                "fallback_model_profile_count": len(fallback_refs)
                if isinstance(fallback_refs, list)
                else 0,
                "mcp_server_count": len(mcp_server_ids)
                if isinstance(mcp_server_ids, list)
                else 0,
                "skill_count": len(skill_ids) if isinstance(skill_ids, list) else 0,
                CONF_LLM_HASS_API: data.get(CONF_LLM_HASS_API),
                CONF_WEB_FETCH_ENABLED: bool(data.get(CONF_WEB_FETCH_ENABLED, False)),
                CONF_VIRTUAL_WORKSPACE_ENABLED: bool(
                    data.get(CONF_VIRTUAL_WORKSPACE_ENABLED, False)
                ),
            }
        )
        if subentry.subentry_type == SUBENTRY_TYPE_AI_TASK:
            if output_mode := _structured_output_mode_summary(entry, subentry):
                summary["structured_output_mode"] = output_mode
            summary["todo_workspace_enabled"] = bool(data.get(CONF_TODO_LIST_ENTITY_ID))
    elif subentry.subentry_type == SUBENTRY_TYPE_PROVIDER:
        model_profiles = data.get(CONF_MODEL_PROFILES)
        summary.update(
            {
                "default_model_profile_id": data.get(CONF_DEFAULT_MODEL_PROFILE_ID),
                "model_profile_count": len(model_profiles)
                if isinstance(model_profiles, Mapping)
                else 0,
            }
        )
    elif subentry.subentry_type == SUBENTRY_TYPE_SKILL:
        summary.update(
            {
                "has_skill_content": bool(data.get(CONF_SKILL_CONTENT)),
                "skill_reference_count": len(references)
                if isinstance((references := data.get(CONF_SKILL_REFERENCES)), list)
                else 0,
            }
        )
    elif subentry.subentry_type == SUBENTRY_TYPE_MCP_SERVER:
        summary.update(
            {
                "has_headers": bool(data.get(CONF_MCP_HEADERS)),
                CONF_MCP_TOOL_MODE: effective_mcp_tool_mode(data),
                "allowed_tool_count": len(data.get(CONF_MCP_ALLOWED_TOOLS, []))
                if isinstance(data.get(CONF_MCP_ALLOWED_TOOLS), list)
                else 0,
                CONF_MCP_INCLUDE_RETURN_SCHEMA: bool(
                    data.get(CONF_MCP_INCLUDE_RETURN_SCHEMA, True)
                ),
                CONF_MCP_DEFERRED_LOADING: bool(
                    data.get(CONF_MCP_DEFERRED_LOADING, False)
                ),
            }
        )
    return summary


def _structured_output_mode_summary(
    entry: ConfigEntry, subentry: ConfigSubentry
) -> str | None:
    """Return the computed AI task structured output mode when resolvable."""
    if subentry.subentry_type != SUBENTRY_TYPE_AI_TASK:
        return None
    try:
        return resolved_structured_output_mode(primary_model_profile(entry, subentry))
    except Exception:
        return None


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
