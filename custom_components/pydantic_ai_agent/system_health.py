"""System health support for Pydantic AI Agent."""

from typing import Any

from homeassistant.components import system_health
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback

from .const import (
    CONF_MCP_SERVER_IDS,
    CONF_MODEL_PROFILES,
    CONF_PROVIDER_MODE,
    CONF_SKILLS,
    DOMAIN,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_MCP_SERVER,
    SUBENTRY_TYPE_PROVIDER,
    SUBENTRY_TYPE_SKILL,
)
from .observability.logfire_support import logfire_enabled


@callback
def async_register(
    hass: HomeAssistant, register: system_health.SystemHealthRegistration
) -> None:
    """Register system health callbacks."""
    register.async_register_info(system_health_info)


async def system_health_info(hass: HomeAssistant) -> dict[str, Any]:
    """Return non-secret aggregate integration health information."""
    entries = hass.config_entries.async_entries(DOMAIN)
    loaded_entries = [entry for entry in entries if hasattr(entry, "runtime_data")]
    return {
        "configured_entry_count": len(entries),
        "loaded_entry_count": len(loaded_entries),
        "provider_modes": _provider_mode_counts(entries),
        "provider_count": _subentry_count(entries, SUBENTRY_TYPE_PROVIDER),
        "model_profile_count": _model_profile_count(entries),
        "conversation_count": _subentry_count(entries, SUBENTRY_TYPE_CONVERSATION),
        "ai_task_count": _subentry_count(entries, SUBENTRY_TYPE_AI_TASK),
        "mcp_server_count": _subentry_count(entries, SUBENTRY_TYPE_MCP_SERVER),
        "skill_count": _subentry_count(entries, SUBENTRY_TYPE_SKILL),
        "selected_mcp_server_count": _selected_mcp_server_count(entries),
        "cached_mcp_tool_count": _cached_mcp_tool_count(loaded_entries),
        "selected_skill_count": _selected_skill_count(entries),
        "logfire_enabled_count": sum(
            1 for entry in entries if logfire_enabled(hass, entry)
        ),
    }


def _provider_mode_counts(entries: list[ConfigEntry]) -> dict[str, int]:
    """Return configured provider mode counts."""
    counts: dict[str, int] = {}
    for entry in entries:
        for subentry in entry.subentries.values():
            if subentry.subentry_type != SUBENTRY_TYPE_PROVIDER:
                continue
            mode = subentry.data.get(CONF_PROVIDER_MODE)
            if isinstance(mode, str):
                counts[mode] = counts.get(mode, 0) + 1
    return dict(sorted(counts.items()))


def _subentry_count(entries: list[ConfigEntry], subentry_type: str) -> int:
    """Return the number of configured subentries of a type."""
    return sum(
        1
        for entry in entries
        for subentry in entry.subentries.values()
        if subentry.subentry_type == subentry_type
    )


def _model_profile_count(entries: list[ConfigEntry]) -> int:
    """Return the number of configured provider-owned model profiles."""
    return sum(
        len(subentry.data.get(CONF_MODEL_PROFILES, {}))
        for entry in entries
        for subentry in entry.subentries.values()
        if subentry.subentry_type == SUBENTRY_TYPE_PROVIDER
        and isinstance(subentry.data.get(CONF_MODEL_PROFILES), dict)
    )


def _selected_skill_count(entries: list[ConfigEntry]) -> int:
    """Return total native Skill selections across agents and AI tasks."""
    return sum(
        len(skill_ids)
        for entry in entries
        for subentry in entry.subentries.values()
        if subentry.subentry_type in {SUBENTRY_TYPE_CONVERSATION, SUBENTRY_TYPE_AI_TASK}
        and isinstance((skill_ids := subentry.data.get(CONF_SKILLS)), list)
    )


def _selected_mcp_server_count(entries: list[ConfigEntry]) -> int:
    """Return total MCP server selections across agents and AI tasks."""
    return sum(
        len(server_ids)
        for entry in entries
        for subentry in entry.subentries.values()
        if subentry.subentry_type in {SUBENTRY_TYPE_CONVERSATION, SUBENTRY_TYPE_AI_TASK}
        and isinstance((server_ids := subentry.data.get(CONF_MCP_SERVER_IDS)), list)
    )


def _cached_mcp_tool_count(entries: list[ConfigEntry]) -> int:
    """Return total cached MCP tools across loaded workspaces."""
    return sum(
        sum(
            len(tools)
            for tools in getattr(entry.runtime_data, "mcp_tool_cache", {}).values()
        )
        for entry in entries
    )
