"""Pydantic AI Agent integration."""

import logging
from dataclasses import replace
from typing import Any

import voluptuous as vol
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import ServiceValidationError

from ._migration import (
    _async_remove_removed_memory_store,
    _migrate_profile_templated_extra_body,
    _remove_ai_task_legacy_output_mode,
    _remove_removed_device_registry_entry,
    _remove_removed_entity_registry_entries,
    _remove_removed_llm_api_refs,
    _remove_stale_subentry_registry_entries,
)
from ._run_diagnostics_service import async_register_run_diagnostics_service
from ._setup_helpers import (
    _mcp_server_runtimes,
    _provider_runtimes,
    _resolved_model_profiles,
)
from ._types import (
    MCPServerRuntimeData as MCPServerRuntimeData,
)
from ._types import (
    ProviderRuntimeData as ProviderRuntimeData,
)
from ._types import (
    PydanticAIAgentConfigEntry,
)
from ._types import (
    WorkspaceRuntimeData as WorkspaceRuntimeData,
)
from .const import CONF_NAME, DOMAIN
from .debug_services import async_setup_services as async_setup_debug_services
from .logfire_support import (
    async_configure_logfire,
    async_release_logfire,
    logfire_enabled,
    logfire_include_content,
)
from .mcp import (
    MCPValidationError,
    async_refresh_mcp_tools,
    cached_mcp_tools,
    mcp_subentries,
)
from .metrics import (
    EVENT_MCP_TOOL_REFRESH_COMPLETED,
    EVENT_MCP_TOOL_REFRESH_FAILED,
    fire_integration_event,
)
from .repair_issues import (
    async_delete_entry_repair_issues,
    async_delete_logfire_token_conflict_issue,
    async_delete_model_validation_issues,
    async_delete_stale_provider_auth_issues,
)

_LOGGER = logging.getLogger(__name__)

SERVICE_LIST_MCP_TOOLS = "list_mcp_tools"
SERVICE_REFRESH_MCP_TOOLS = "refresh_mcp_tools"
ATTR_CONFIG_ENTRY_ID = "config_entry_id"
ATTR_MCP_SERVER_ID = "mcp_server_id"

_MCP_SERVICE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CONFIG_ENTRY_ID): str,
        vol.Optional(ATTR_MCP_SERVER_ID): str,
    }
)

PLATFORMS: tuple[Platform, ...] = (
    Platform.CONVERSATION,
    Platform.AI_TASK,
    Platform.SENSOR,
    Platform.BINARY_SENSOR,
)


async def async_setup(hass: HomeAssistant, config: dict[str, Any]) -> bool:
    """Set up integration-wide services."""

    async def async_list_mcp_tools(call: ServiceCall) -> dict[str, Any]:
        """Return cached MCP tools, discovering them if needed."""
        return await _async_mcp_tools_service(hass, call, refresh=False)

    async def async_refresh_mcp_tools_service(call: ServiceCall) -> dict[str, Any]:
        """Refresh and return MCP tools."""
        return await _async_mcp_tools_service(hass, call, refresh=True)

    hass.services.async_register(
        DOMAIN,
        SERVICE_LIST_MCP_TOOLS,
        async_list_mcp_tools,
        schema=_MCP_SERVICE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_REFRESH_MCP_TOOLS,
        async_refresh_mcp_tools_service,
        schema=_MCP_SERVICE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    async_register_run_diagnostics_service(hass)
    await async_setup_debug_services(hass)
    return True


async def async_setup_entry(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> bool:
    """Build workspace runtime data, then set up entity platforms."""
    provider_runtimes = _provider_runtimes(entry)
    mcp_runtime = _mcp_server_runtimes(entry)
    model_profiles = _resolved_model_profiles(entry, provider_runtimes)
    forwarded_platforms = False
    await async_configure_logfire(hass, entry)
    try:
        entry.runtime_data = WorkspaceRuntimeData(
            workspace_name=entry.data[CONF_NAME],
            providers=provider_runtimes,
            mcp_servers=mcp_runtime,
            model_profiles=model_profiles,
            logfire_enabled=logfire_enabled(hass, entry),
            logfire_include_content=logfire_include_content(hass, entry),
        )
        async_delete_model_validation_issues(hass, entry)
        async_delete_stale_provider_auth_issues(
            hass,
            entry,
            set(provider_runtimes),
        )
        _remove_stale_subentry_registry_entries(hass, entry)
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
        forwarded_platforms = True
        await async_configure_logfire(hass, entry)
        entry.runtime_data = replace(
            entry.runtime_data,
            logfire_enabled=logfire_enabled(hass, entry),
            logfire_include_content=logfire_include_content(hass, entry),
        )
    except BaseException:
        try:
            if forwarded_platforms:
                await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
        finally:
            await async_release_logfire(hass, entry)
        raise
    entry.async_on_unload(entry.add_update_listener(async_update_entry))
    return True


async def async_unload_entry(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> bool:
    """Unload a config entry."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded:
        await async_release_logfire(hass, entry)
        async_delete_logfire_token_conflict_issue(hass, entry)
    return unloaded


async def async_migrate_entry(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> bool:
    """Migrate workspace entries."""
    if entry.version != 2:
        _LOGGER.error(
            "Pydantic AI Agent config entry %s uses unsupported schema version %s.%s; "
            "delete and recreate the workspace entry.",
            entry.entry_id,
            entry.version,
            entry.minor_version,
        )
        return False

    if entry.minor_version == 0:
        _remove_removed_llm_api_refs(hass, entry)
        _remove_removed_entity_registry_entries(hass, entry)
        _remove_removed_device_registry_entry(hass, entry)
        hass.config_entries.async_update_entry(entry, minor_version=1)
    if entry.minor_version == 1:
        _migrate_profile_templated_extra_body(hass, entry)
        hass.config_entries.async_update_entry(entry, minor_version=2)
    if entry.minor_version == 2:
        _remove_ai_task_legacy_output_mode(hass, entry)
        hass.config_entries.async_update_entry(entry, minor_version=3)
    if entry.minor_version == 3:
        return True

    _LOGGER.error(
        "Pydantic AI Agent config entry %s uses unsupported schema version %s.%s; "
        "delete and recreate the workspace entry.",
        entry.entry_id,
        entry.version,
        entry.minor_version,
    )
    return False


async def async_remove_entry(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> None:
    """Clean up repair issues when a config entry is permanently removed."""
    await async_release_logfire(hass, entry)
    async_delete_entry_repair_issues(hass, entry)
    _remove_removed_entity_registry_entries(hass, entry)
    _remove_removed_device_registry_entry(hass, entry)
    _remove_stale_subentry_registry_entries(hass, entry)
    await _async_remove_removed_memory_store(hass, entry)


async def async_update_entry(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> None:
    """Reload the entry after config entry or subentry updates."""
    await hass.config_entries.async_reload(entry.entry_id)


def _mcp_error_response(err: MCPValidationError) -> dict[str, Any]:
    """Return a response-service error payload for an expected MCP failure."""
    return {
        "reason": err.reason,
        "message": err.message,
        "action": "Check the MCP server configuration and try again.",
        "status_code": err.status_code,
        "server_id": err.server_id,
        "tool_name": err.tool_name,
    }


def _config_entry_for_service(
    hass: HomeAssistant, entry_id: str
) -> PydanticAIAgentConfigEntry:
    """Return a config entry for an MCP response service."""
    entry = hass.config_entries.async_get_entry(entry_id)
    if entry is None or entry.domain != DOMAIN:
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="config_entry_not_found",
            translation_placeholders={"config_entry_id": entry_id},
        )
    return entry


def _mcp_service_subentry_ids(
    entry: PydanticAIAgentConfigEntry, requested_id: str | None
) -> list[str]:
    """Return target MCP subentry IDs for a response service call."""
    if requested_id is not None:
        return [requested_id]
    return [subentry.subentry_id for subentry in mcp_subentries(entry)]


async def _async_mcp_tools_service(
    hass: HomeAssistant,
    call: ServiceCall,
    *,
    refresh: bool,
) -> dict[str, Any]:
    """List or refresh MCP tools for one config entry."""
    errors: list[dict[str, Any]] = []
    tools_by_server: dict[str, list[dict[str, Any]]] = {}
    entry = _config_entry_for_service(hass, call.data[ATTR_CONFIG_ENTRY_ID])
    subentry_ids = _mcp_service_subentry_ids(entry, call.data.get(ATTR_MCP_SERVER_ID))
    if not subentry_ids:
        return {"success": True, "servers": {}, "tools": []}
    for subentry_id in subentry_ids:
        try:
            tools = None if refresh else cached_mcp_tools(entry, subentry_id)
            if tools is None:
                tools = await async_refresh_mcp_tools(hass, entry, subentry_id)
            tools_by_server[subentry_id] = tools
            if refresh:
                fire_integration_event(
                    hass,
                    EVENT_MCP_TOOL_REFRESH_COMPLETED,
                    {
                        "config_entry_id": entry.entry_id,
                        "mcp_server_id": subentry_id,
                        "tool_count": len(tools),
                    },
                )
        except MCPValidationError as err:
            _LOGGER.warning(
                "MCP tool discovery failed: reason=%s server_id=%s",
                err.reason,
                err.server_id,
            )
            errors.append(_mcp_error_response(err))
            if refresh:
                fire_integration_event(
                    hass,
                    EVENT_MCP_TOOL_REFRESH_FAILED,
                    {
                        "config_entry_id": entry.entry_id,
                        "mcp_server_id": subentry_id,
                        "reason": err.reason,
                    },
                )
    flat_tools = [tool for tools in tools_by_server.values() for tool in tools]
    return {
        "success": not errors,
        "servers": tools_by_server,
        "tools": flat_tools,
        "errors": errors,
    }
