"""Private response builders for debug services."""

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME

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
    CONF_PRIMARY_MODEL_REF,
    CONF_PROVIDER_MODE,
    CONF_SKILL_CONTENT,
    CONF_SKILL_REFERENCES,
    CONF_SKILLS,
    CONF_TODO_LIST_ENTITY_ID,
    CONF_VIRTUAL_WORKSPACE_ENABLED,
    CONF_WEB_FETCH_ENABLED,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_MCP_SERVER,
    SUBENTRY_TYPE_SKILL,
)
from .mcp import cached_mcp_tools
from .metrics import AgentRunMetrics
from .model_profiles import (
    model_profile_ref,
    primary_model_profile,
    provider_model_profiles,
    provider_subentries,
)
from .structured_output import resolved_structured_output_mode


def workspace_status(
    entry: ConfigEntry, *, include_subentries: bool, include_runtime: bool
) -> dict[str, Any]:
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


def model_profile_summaries(
    entry: ConfigEntry, provider_subentry: ConfigSubentry, *, enabled_only: bool
) -> list[dict[str, Any]]:
    runtime_data = getattr(entry, "runtime_data", None)
    runtime_profiles = (
        getattr(runtime_data, "model_profiles", {}) if runtime_data else {}
    )
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


def metrics_status(entry: ConfigEntry, *, subentry_id: str | None) -> dict[str, Any]:
    runtime_data = getattr(entry, "runtime_data", None)
    metrics = (
        getattr(runtime_data, "metrics", None) if runtime_data is not None else None
    )
    records: Mapping[str, Any] = (
        getattr(metrics, "_records", {}) if metrics is not None else {}
    )
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


def tool_source_status(
    entry: ConfigEntry,
    *,
    subentry_id: str | None,
    include_tool_names: bool,
    limit: int,
) -> dict[str, Any]:
    mcp_servers = [
        _mcp_tool_source_status(
            entry,
            subentry,
            include_tool_names=include_tool_names,
            limit=limit,
        )
        for subentry in entry.subentries.values()
        if subentry.subentry_type == SUBENTRY_TYPE_MCP_SERVER
        and (subentry_id is None or subentry.subentry_id == subentry_id)
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
        "mcp_server_count": len(mcp_servers),
        "skills": skills,
        "skill_count": len(skills),
    }


def _entry_state(entry: ConfigEntry) -> str:
    state = getattr(entry, "state", None)
    return str(getattr(state, "name", state)) if state is not None else "unknown"


def _subentry_counts(entry: ConfigEntry) -> dict[str, int]:
    counts: dict[str, int] = {}
    for subentry in entry.subentries.values():
        counts[subentry.subentry_type] = counts.get(subentry.subentry_type, 0) + 1
    return counts


def _subentries_status(entry: ConfigEntry) -> dict[str, list[dict[str, Any]]]:
    return {
        "providers": [
            _provider_status(entry, subentry) for subentry in provider_subentries(entry)
        ],
        "mcp_servers": [
            _mcp_server_status(entry, subentry)
            for subentry in entry.subentries.values()
            if subentry.subentry_type == SUBENTRY_TYPE_MCP_SERVER
        ],
        "conversations": [
            _agent_subentry_status(entry, subentry)
            for subentry in entry.subentries.values()
            if subentry.subentry_type == SUBENTRY_TYPE_CONVERSATION
        ],
        "ai_tasks": [
            _agent_subentry_status(entry, subentry)
            for subentry in entry.subentries.values()
            if subentry.subentry_type == SUBENTRY_TYPE_AI_TASK
        ],
        "skills": [
            _skill_subentry_status(subentry)
            for subentry in entry.subentries.values()
            if subentry.subentry_type == SUBENTRY_TYPE_SKILL
        ],
    }


def _provider_status(entry: ConfigEntry, subentry: ConfigSubentry) -> dict[str, Any]:
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


def _agent_subentry_status(
    entry: ConfigEntry, subentry: ConfigSubentry
) -> dict[str, Any]:
    data = subentry.data
    status = {
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
        "virtual_workspace_enabled": bool(
            data.get(CONF_VIRTUAL_WORKSPACE_ENABLED, False)
        ),
        "todo_workspace_enabled": bool(data.get(CONF_TODO_LIST_ENTITY_ID)),
    }
    if subentry.subentry_type == SUBENTRY_TYPE_AI_TASK:
        try:
            status["structured_output_mode"] = resolved_structured_output_mode(
                primary_model_profile(entry, subentry)
            )
        except Exception:
            status["structured_output_mode"] = None
    return status


def _mcp_server_status(entry: ConfigEntry, subentry: ConfigSubentry) -> dict[str, Any]:
    runtime_data = getattr(entry, "runtime_data", None)
    runtime_servers = getattr(runtime_data, "mcp_servers", {}) if runtime_data else {}
    cached_tools = (
        cached_mcp_tools(entry, subentry.subentry_id) if runtime_data else None
    )
    allowed_tools = subentry.data.get(CONF_MCP_ALLOWED_TOOLS)
    return {
        "subentry_id": subentry.subentry_id,
        "title": subentry.title,
        "has_headers": bool(subentry.data.get(CONF_MCP_HEADERS)),
        "allowed_tool_count": _list_count(allowed_tools),
        "cached_tool_count": len(cached_tools) if cached_tools is not None else 0,
        "runtime_loaded": subentry.subentry_id in runtime_servers,
    }


def _skill_subentry_status(subentry: ConfigSubentry) -> dict[str, Any]:
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


def _runtime_status(runtime_data: object) -> dict[str, Any] | None:
    if runtime_data is None:
        return None
    metrics = getattr(runtime_data, "metrics", None)
    records = getattr(metrics, "_records", {}) if metrics is not None else {}
    latest_run_diagnostics = getattr(runtime_data, "latest_run_diagnostics", {})
    return {
        "workspace_name": getattr(runtime_data, "workspace_name", None),
        "provider_count": len(getattr(runtime_data, "providers", {})),
        "mcp_server_count": len(getattr(runtime_data, "mcp_servers", {})),
        "cached_mcp_tool_server_count": len(
            getattr(runtime_data, "mcp_tool_cache", {})
        ),
        "model_profile_count": len(getattr(runtime_data, "model_profiles", {})),
        "metrics_record_count": len(records),
        "latest_run_diagnostic_count": len(latest_run_diagnostics),
        "logfire_enabled": bool(getattr(runtime_data, "logfire_enabled", False)),
        "logfire_include_content": bool(
            getattr(runtime_data, "logfire_include_content", False)
        ),
    }


def _mcp_tool_source_status(
    entry: ConfigEntry,
    subentry: ConfigSubentry,
    *,
    include_tool_names: bool,
    limit: int,
) -> dict[str, Any]:
    runtime_data = getattr(entry, "runtime_data", None)
    cached_tools = (
        cached_mcp_tools(entry, subentry.subentry_id)
        if runtime_data is not None
        else None
    )
    tool_names = []
    if include_tool_names and cached_tools is not None:
        tool_names = [
            str(tool.get("name")) for tool in cached_tools[:limit] if tool.get("name")
        ]
    return {
        "subentry_id": subentry.subentry_id,
        "title": subentry.title,
        "has_headers": bool(subentry.data.get(CONF_MCP_HEADERS)),
        "allowed_tool_count": _list_count(subentry.data.get(CONF_MCP_ALLOWED_TOOLS)),
        "cached_tool_count": len(cached_tools) if cached_tools is not None else 0,
        "tool_names": tool_names,
    }


def _subentry_type(entry: ConfigEntry, subentry_id: str) -> str | None:
    subentry = entry.subentries.get(subentry_id)
    return None if subentry is None else subentry.subentry_type


def _list_count(value: object) -> int:
    return len(value) if isinstance(value, list) else 0


def _jsonable_dataclass(value: object) -> dict[str, Any] | None:
    return asdict(value) if isinstance(value, AgentRunMetrics) else None
