"""Pydantic AI Agent integration."""

import logging
from dataclasses import replace
from typing import Any

from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from ._migration import (
    _async_remove_removed_memory_store,
    _remove_removed_device_registry_entry,
    _remove_removed_entity_registry_entries,
    _remove_removed_llm_api_refs,
    _remove_stale_subentry_registry_entries,
)
from ._model_validation import _async_validate_configured_models
from ._run_diagnostics_service import async_register_run_diagnostics_service
from ._setup_helpers import _provider_runtimes, _resolved_model_profiles
from ._types import (
    ProviderRuntimeData as ProviderRuntimeData,
)
from ._types import (
    PydanticAIAgentConfigEntry,
)
from ._types import (
    WorkspaceRuntimeData as WorkspaceRuntimeData,
)
from .const import CONF_NAME
from .debug_services import async_setup_services as async_setup_debug_services
from .logfire_support import (
    async_configure_logfire,
    async_release_logfire,
    logfire_enabled,
    logfire_include_content,
)
from .repairs import (
    async_delete_entry_repair_issues,
    async_delete_logfire_token_conflict_issue,
)

_LOGGER = logging.getLogger(__name__)

PLATFORMS: tuple[Platform, ...] = (
    Platform.CONVERSATION,
    Platform.AI_TASK,
    Platform.SENSOR,
    Platform.BINARY_SENSOR,
)


async def async_setup(hass: HomeAssistant, config: dict[str, Any]) -> bool:
    """Set up integration-wide services."""
    async_register_run_diagnostics_service(hass)
    await async_setup_debug_services(hass)
    return True


async def async_setup_entry(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> bool:
    """Build workspace runtime data, then set up entity platforms."""
    provider_runtimes = _provider_runtimes(entry)
    model_profiles = _resolved_model_profiles(entry, provider_runtimes)
    forwarded_platforms = False
    await async_configure_logfire(hass, entry)
    try:
        entry.runtime_data = WorkspaceRuntimeData(
            workspace_name=entry.data[CONF_NAME],
            providers=provider_runtimes,
            model_profiles=model_profiles,
            logfire_enabled=logfire_enabled(hass, entry),
            logfire_include_content=logfire_include_content(hass, entry),
        )
        model_validation_failures = await _async_validate_configured_models(hass, entry)
        entry.runtime_data = replace(
            entry.runtime_data,
            model_validation_failures=model_validation_failures,
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
    if entry.version == 2 and entry.minor_version == 0:
        _remove_removed_llm_api_refs(hass, entry)
        _remove_removed_entity_registry_entries(hass, entry)
        _remove_removed_device_registry_entry(hass, entry)
        hass.config_entries.async_update_entry(entry, minor_version=1)
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
