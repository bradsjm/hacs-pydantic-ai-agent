"""System health support for Pydantic AI Agent."""

from typing import Any

from homeassistant.components import system_health
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback

from .const import (
    CONF_MODEL_PROFILES,
    CONF_ENABLE_SKILL_SCRIPT_EXECUTION,
    CONF_PROVIDER_MODE,
    DOMAIN,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_MCP_SERVER,
    SUBENTRY_TYPE_PROVIDER,
)
from .logfire_support import logfire_enabled


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
        "cached_mcp_server_count": sum(
            len(entry.runtime_data.mcp_tool_cache) for entry in loaded_entries
        ),
        "cached_mcp_tool_count": sum(
            len(tools)
            for entry in loaded_entries
            for tools in entry.runtime_data.mcp_tool_cache.values()
        ),
        "logfire_enabled_count": sum(
            1 for entry in entries if logfire_enabled(hass, entry)
        ),
        "skill_script_execution_count": sum(
            1
            for entry in entries
            for subentry in entry.subentries.values()
            if subentry.subentry_type
            in {SUBENTRY_TYPE_CONVERSATION, SUBENTRY_TYPE_AI_TASK}
            and subentry.data.get(CONF_ENABLE_SKILL_SCRIPT_EXECUTION, False)
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
