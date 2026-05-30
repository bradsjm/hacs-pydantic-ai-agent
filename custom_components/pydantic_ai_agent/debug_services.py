"""Read-only debug response services for Pydantic AI Agent."""

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
import voluptuous as vol

from .const import (
    CONF_AGENT_NAME,
    CONF_AI_TASK_NAME,
    CONF_DEFAULT_MODEL_PROFILE_ID,
    CONF_DESCRIPTION,
    CONF_DISCOVERED,
    CONF_ENABLED,
    CONF_FALLBACK_MODEL_REFS,
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_HEADERS,
    CONF_MCP_SERVER_IDS,
    CONF_MODEL,
    CONF_MODEL_PRICING,
    CONF_OUTPUT_MODE,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROVIDER_MODE,
    CONF_SKILL_CONTENT,
    CONF_SKILL_REFERENCES,
    CONF_SKILLS,
    CONF_TODO_LIST_ENTITY_ID,
    CONF_VIRTUAL_WORKSPACE_ENABLED,
    CONF_WEB_FETCH_ENABLED,
    DOMAIN,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_SKILL,
)
from .mcp import MCPValidationError, cached_mcp_tools, mcp_subentries
from .metrics import AgentRunMetrics
from .model_profiles import (
    model_profile_ref,
    provider_model_profiles,
    provider_subentries,
)

SERVICE_GET_WORKSPACE_STATUS = "get_workspace_status"
SERVICE_LIST_MODEL_PROFILES = "list_model_profiles"
SERVICE_GET_AGENT_METRICS = "get_agent_metrics"
SERVICE_GET_TOOL_SOURCE_STATUS = "get_tool_source_status"

ATTR_CONFIG_ENTRY_ID = "config_entry_id"
ATTR_PROVIDER_SUBENTRY_ID = "provider_subentry_id"
ATTR_SUBENTRY_ID = "subentry_id"
ATTR_INCLUDE_SUBENTRIES = "include_subentries"
ATTR_INCLUDE_RUNTIME = "include_runtime"
ATTR_ENABLED_ONLY = "enabled_only"
ATTR_INCLUDE_TOOL_NAMES = "include_tool_names"
ATTR_LIMIT = "limit"

_OPTIONAL_ENTRY_SCHEMA = {
    vol.Optional(ATTR_CONFIG_ENTRY_ID): str,
}

_WORKSPACE_STATUS_SCHEMA = vol.Schema(
    {
        **_OPTIONAL_ENTRY_SCHEMA,
        vol.Optional(ATTR_INCLUDE_SUBENTRIES, default=True): bool,
        vol.Optional(ATTR_INCLUDE_RUNTIME, default=True): bool,
    }
)
_LIST_MODEL_PROFILES_SCHEMA = vol.Schema(
    {
        **_OPTIONAL_ENTRY_SCHEMA,
        vol.Optional(ATTR_PROVIDER_SUBENTRY_ID): str,
        vol.Optional(ATTR_ENABLED_ONLY, default=False): bool,
    }
)
_AGENT_METRICS_SCHEMA = vol.Schema(
    {
        **_OPTIONAL_ENTRY_SCHEMA,
        vol.Optional(ATTR_SUBENTRY_ID): str,
    }
)
_TOOL_SOURCE_STATUS_SCHEMA = vol.Schema(
    {
        **_OPTIONAL_ENTRY_SCHEMA,
        vol.Optional(ATTR_SUBENTRY_ID): str,
        vol.Optional(ATTR_INCLUDE_TOOL_NAMES, default=True): bool,
        vol.Optional(ATTR_LIMIT, default=50): vol.All(vol.Coerce(int), vol.Range(min=1, max=200)),
    }
)


async def async_setup_services(hass: HomeAssistant) -> None:
    """Register read-only debug response services."""

    async def async_get_workspace_status(call: ServiceCall) -> dict[str, Any]:
        return _get_workspace_status(hass, call)

    async def async_list_model_profiles(call: ServiceCall) -> dict[str, Any]:
        return _list_model_profiles(hass, call)

    async def async_get_agent_metrics(call: ServiceCall) -> dict[str, Any]:
        return _get_agent_metrics(hass, call)

    async def async_get_tool_source_status(call: ServiceCall) -> dict[str, Any]:
        return _get_tool_source_status(hass, call)

    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_WORKSPACE_STATUS,
        async_get_workspace_status,
        schema=_WORKSPACE_STATUS_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_LIST_MODEL_PROFILES,
        async_list_model_profiles,
        schema=_LIST_MODEL_PROFILES_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_AGENT_METRICS,
        async_get_agent_metrics,
        schema=_AGENT_METRICS_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_TOOL_SOURCE_STATUS,
        async_get_tool_source_status,
        schema=_TOOL_SOURCE_STATUS_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )


def _get_workspace_status(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Return compact config/runtime status for matching workspaces."""
    entries = _entries_for_service(hass, call.data.get(ATTR_CONFIG_ENTRY_ID))
    return {
        "success": True,
        "count": len(entries),
        "entries": [
            _workspace_status(
                entry,
                include_subentries=call.data[ATTR_INCLUDE_SUBENTRIES],
                include_runtime=call.data[ATTR_INCLUDE_RUNTIME],
            )
            for entry in entries
        ],
    }


def _list_model_profiles(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Return configured provider-owned model profiles without probing providers."""
    entries = _entries_for_service(hass, call.data.get(ATTR_CONFIG_ENTRY_ID))
    profiles: list[dict[str, Any]] = []
    for entry in entries:
        for provider_subentry in provider_subentries(entry):
            if (
                provider_id := call.data.get(ATTR_PROVIDER_SUBENTRY_ID)
            ) is not None and provider_subentry.subentry_id != provider_id:
                continue
            profiles.extend(
                _model_profile_summaries(
                    entry,
                    provider_subentry,
                    enabled_only=call.data[ATTR_ENABLED_ONLY],
                )
            )
    return {"success": True, "count": len(profiles), "profiles": profiles}


def _get_agent_metrics(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Return metrics records currently stored in workspace runtime data."""
    entries = _entries_for_service(hass, call.data.get(ATTR_CONFIG_ENTRY_ID))
    subentry_id = call.data.get(ATTR_SUBENTRY_ID)
    return {
        "success": True,
        "count": len(entries),
        "entries": [
            _metrics_status(entry, subentry_id=subentry_id)
            for entry in entries
        ],
    }


def _get_tool_source_status(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Return MCP and native Skill tool-source status without refreshing tools."""
    entries = _entries_for_service(hass, call.data.get(ATTR_CONFIG_ENTRY_ID))
    subentry_id = call.data.get(ATTR_SUBENTRY_ID)
    return {
        "success": True,
        "count": len(entries),
        "entries": [
            _tool_source_status(
                entry,
                subentry_id=subentry_id,
                include_tool_names=call.data[ATTR_INCLUDE_TOOL_NAMES],
                limit=call.data[ATTR_LIMIT],
            )
            for entry in entries
        ],
    }


def _entries_for_service(
    hass: HomeAssistant, entry_id: str | None
) -> list[ConfigEntry]:
    """Return matching Pydantic AI Agent config entries."""
    if entry_id is not None:
        entry = hass.config_entries.async_get_entry(entry_id)
        return [entry] if entry is not None and entry.domain == DOMAIN else []
    return [entry for entry in hass.config_entries.async_entries(DOMAIN)]


def _workspace_status(
    entry: ConfigEntry, *, include_subentries: bool, include_runtime: bool
) -> dict[str, Any]:
    """Return one workspace status payload."""
    runtime_data = getattr(entry, "runtime_data", None)
    data: dict[str, Any] = {
        "config_entry_id": entry.entry_id,
        "title": entry.title,
        "state": _entry_state(entry),
        "loaded": runtime_data is not None,
        "subentry_counts": _subentry_counts(entry),
    }
    if include_subentries:
        data["subentries"] = _subentries_status(entry)
    if include_runtime:
        data["runtime"] = _runtime_status(runtime_data)
    return data


def _entry_state(entry: ConfigEntry) -> str:
    """Return a JSON-safe config entry state."""
    state = getattr(entry, "state", None)
    return str(getattr(state, "name", state)) if state is not None else "unknown"


def _subentry_counts(entry: ConfigEntry) -> dict[str, int]:
    """Return counts by subentry type."""
    counts: dict[str, int] = {}
    for subentry in entry.subentries.values():
        counts[subentry.subentry_type] = counts.get(subentry.subentry_type, 0) + 1
    return counts


def _subentries_status(entry: ConfigEntry) -> dict[str, list[dict[str, Any]]]:
    """Return compact subentry summaries grouped by type."""
    return {
        "providers": [_provider_status(entry, subentry) for subentry in provider_subentries(entry)],
        "conversations": [
            _agent_subentry_status(subentry)
            for subentry in entry.subentries.values()
            if subentry.subentry_type == SUBENTRY_TYPE_CONVERSATION
        ],
        "ai_tasks": [
            _agent_subentry_status(subentry)
            for subentry in entry.subentries.values()
            if subentry.subentry_type == SUBENTRY_TYPE_AI_TASK
        ],
        "mcp_servers": [_mcp_subentry_status(entry, subentry) for subentry in mcp_subentries(entry)],
        "skills": [
            _skill_subentry_status(subentry)
            for subentry in entry.subentries.values()
            if subentry.subentry_type == SUBENTRY_TYPE_SKILL
        ],
    }


def _provider_status(entry: ConfigEntry, subentry: ConfigSubentry) -> dict[str, Any]:
    """Return compact provider status without credentials."""
    profiles = provider_model_profiles(subentry)
    enabled_count = sum(
        1 for profile in profiles.values() if bool(profile.get(CONF_ENABLED, False))
    )
    runtime_data = getattr(entry, "runtime_data", None)
    runtime_providers = getattr(runtime_data, "providers", {}) if runtime_data else {}
    return {
        "subentry_id": subentry.subentry_id,
        "title": subentry.title,
        "provider_mode": subentry.data.get(CONF_PROVIDER_MODE),
        "has_api_key": bool(subentry.data.get(CONF_API_KEY)),
        "default_model_profile_id": subentry.data.get(CONF_DEFAULT_MODEL_PROFILE_ID),
        "profile_count": len(profiles),
        "enabled_profile_count": enabled_count,
        "runtime_loaded": subentry.subentry_id in runtime_providers,
    }


def _agent_subentry_status(subentry: ConfigSubentry) -> dict[str, Any]:
    """Return compact conversation or AI task subentry status."""
    data = subentry.data
    return {
        "subentry_id": subentry.subentry_id,
        "title": subentry.title,
        "type": subentry.subentry_type,
        "name": data.get(CONF_AGENT_NAME) or data.get(CONF_AI_TASK_NAME),
        "primary_model_ref": data.get(CONF_PRIMARY_MODEL_REF),
        "fallback_model_count": _list_count(data.get(CONF_FALLBACK_MODEL_REFS)),
        "mcp_server_count": _list_count(data.get(CONF_MCP_SERVER_IDS)),
        "skill_count": _list_count(data.get(CONF_SKILLS)),
        "ha_llm_api_count": _list_count(data.get(CONF_LLM_HASS_API)),
        "web_fetch_enabled": bool(data.get(CONF_WEB_FETCH_ENABLED, False)),
        "virtual_workspace_enabled": bool(data.get(CONF_VIRTUAL_WORKSPACE_ENABLED, False)),
        "todo_workspace_enabled": bool(data.get(CONF_TODO_LIST_ENTITY_ID)),
        "output_mode": data.get(CONF_OUTPUT_MODE),
    }


def _mcp_subentry_status(entry: ConfigEntry, subentry: ConfigSubentry) -> dict[str, Any]:
    """Return compact MCP subentry status without URL or headers."""
    tools = _cached_tools(entry, subentry.subentry_id)
    allowed_tools = subentry.data.get(CONF_MCP_ALLOWED_TOOLS)
    return {
        "subentry_id": subentry.subentry_id,
        "title": subentry.title,
        "has_headers": bool(subentry.data.get(CONF_MCP_HEADERS)),
        "allowed_tool_count": _list_count(allowed_tools),
        "cached": tools is not None,
        "cached_tool_count": len(tools or ()),
    }


def _skill_subentry_status(subentry: ConfigSubentry) -> dict[str, Any]:
    """Return compact native Skill status."""
    content = subentry.data.get(CONF_SKILL_CONTENT)
    references = subentry.data.get(CONF_SKILL_REFERENCES)
    return {
        "subentry_id": subentry.subentry_id,
        "title": subentry.title,
        "name": subentry.data.get(CONF_NAME),
        "description": subentry.data.get(CONF_DESCRIPTION) or "",
        "content_length": len(content) if isinstance(content, str) else 0,
        "reference_count": _list_count(references),
    }


def _runtime_status(runtime_data: Any) -> dict[str, Any] | None:
    """Return compact runtime status for a loaded workspace."""
    if runtime_data is None:
        return None
    metrics = getattr(runtime_data, "metrics", None)
    records = getattr(metrics, "_records", {}) if metrics is not None else {}
    latest_run_diagnostics = getattr(runtime_data, "latest_run_diagnostics", {})
    return {
        "workspace_name": getattr(runtime_data, "workspace_name", None),
        "provider_count": len(getattr(runtime_data, "providers", {})),
        "model_profile_count": len(getattr(runtime_data, "model_profiles", {})),
        "mcp_server_count": len(getattr(runtime_data, "mcp_servers", {})),
        "mcp_cached_server_count": len(getattr(runtime_data, "mcp_tool_cache", {})),
        "metrics_record_count": len(records),
        "latest_run_diagnostic_count": len(latest_run_diagnostics),
        "logfire_enabled": bool(getattr(runtime_data, "logfire_enabled", False)),
        "logfire_include_content": bool(getattr(runtime_data, "logfire_include_content", False)),
    }


def _model_profile_summaries(
    entry: ConfigEntry, provider_subentry: ConfigSubentry, *, enabled_only: bool
) -> list[dict[str, Any]]:
    """Return model profile summaries for one provider subentry."""
    runtime_data = getattr(entry, "runtime_data", None)
    runtime_profiles = getattr(runtime_data, "model_profiles", {}) if runtime_data else {}
    summaries: list[dict[str, Any]] = []
    default_profile_id = provider_subentry.data.get(CONF_DEFAULT_MODEL_PROFILE_ID)
    for profile_id, profile in provider_model_profiles(provider_subentry).items():
        enabled = bool(profile.get(CONF_ENABLED, False))
        if enabled_only and not enabled:
            continue
        ref = model_profile_ref(provider_subentry.subentry_id, profile_id)
        model = profile.get(CONF_MODEL)
        title = profile.get(CONF_NAME)
        summaries.append(
            {
                "config_entry_id": entry.entry_id,
                "provider_subentry_id": provider_subentry.subentry_id,
                "provider_title": provider_subentry.title,
                "provider_mode": provider_subentry.data.get(CONF_PROVIDER_MODE),
                "profile_id": profile_id,
                "ref": ref,
                "title": title if isinstance(title, str) and title else model,
                "model": model,
                "enabled": enabled,
                "default": profile_id == default_profile_id,
                "discovered": profile.get(CONF_DISCOVERED),
                "pricing_present": bool(profile.get(CONF_MODEL_PRICING)),
                "runtime_loaded": ref in runtime_profiles,
            }
        )
    return summaries


def _metrics_status(entry: ConfigEntry, *, subentry_id: str | None) -> dict[str, Any]:
    """Return metrics status for one workspace."""
    runtime_data = getattr(entry, "runtime_data", None)
    metrics = getattr(runtime_data, "metrics", None) if runtime_data is not None else None
    records: Mapping[str, Any] = getattr(metrics, "_records", {}) if metrics is not None else {}
    selected_ids = [subentry_id] if subentry_id is not None else sorted(records)
    return {
        "config_entry_id": entry.entry_id,
        "loaded": runtime_data is not None,
        "record_count": len(records),
        "records": [
            {
                "subentry_id": selected_id,
                "subentry_type": _subentry_type(entry, selected_id),
                "metrics": _jsonable_dataclass(records.get(selected_id)),
            }
            for selected_id in selected_ids
            if selected_id is not None
        ],
    }


def _tool_source_status(
    entry: ConfigEntry,
    *,
    subentry_id: str | None,
    include_tool_names: bool,
    limit: int,
) -> dict[str, Any]:
    """Return MCP and Skill tool-source status for one workspace."""
    mcp_servers = [
        _mcp_tool_source_status(
            entry,
            subentry,
            include_tool_names=include_tool_names,
            limit=limit,
        )
        for subentry in mcp_subentries(entry)
        if subentry_id is None or subentry.subentry_id == subentry_id
    ]
    skills = [
        _skill_subentry_status(subentry)
        for subentry in entry.subentries.values()
        if subentry.subentry_type == SUBENTRY_TYPE_SKILL
        and (subentry_id is None or subentry.subentry_id == subentry_id)
    ]
    return {
        "config_entry_id": entry.entry_id,
        "mcp_servers": mcp_servers,
        "skills": skills,
        "mcp_server_count": len(mcp_servers),
        "skill_count": len(skills),
    }


def _mcp_tool_source_status(
    entry: ConfigEntry,
    subentry: ConfigSubentry,
    *,
    include_tool_names: bool,
    limit: int,
) -> dict[str, Any]:
    """Return one MCP server cache status."""
    base = _mcp_subentry_status(entry, subentry)
    tools = _cached_tools(entry, subentry.subentry_id) or []
    if include_tool_names:
        base["tool_names"] = [
            tool["name"]
            for tool in tools[:limit]
            if isinstance(tool.get("name"), str)
        ]
        base["tool_name_limit"] = limit
    return base


def _cached_tools(entry: ConfigEntry, subentry_id: str) -> list[dict[str, Any]] | None:
    """Return cached MCP tools if the entry is loaded and cache is populated."""
    try:
        return cached_mcp_tools(entry, subentry_id)
    except MCPValidationError:
        return None


def _subentry_type(entry: ConfigEntry, subentry_id: str) -> str | None:
    """Return subentry type for one subentry id."""
    subentry = entry.subentries.get(subentry_id)
    return None if subentry is None else subentry.subentry_type


def _list_count(value: object) -> int:
    """Return length for persisted list-like config values."""
    return len(value) if isinstance(value, list) else 0


def _jsonable_dataclass(value: Any) -> dict[str, Any] | None:
    """Return a JSON-safe dict for simple dataclass records."""
    if isinstance(value, AgentRunMetrics):
        return asdict(value)
    return None
