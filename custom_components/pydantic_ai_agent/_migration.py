"""Config entry migration and cleanup helpers."""

import logging
from typing import Any

from homeassistant.const import CONF_LLM_HASS_API, Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.storage import Store

from ._types import PydanticAIAgentConfigEntry
from .const import (
    DOMAIN,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_PROVIDER,
)

_LOGGER = logging.getLogger(__name__)

_REMOVED_IN_REPO_LLM_API_PREFIX = "pydantic_ai_agent_home_semantic_"
_REMOVED_IN_REPO_MEMORY_STORE_VERSION = 1
_REMOVED_IN_REPO_ENTITY_UNIQUE_ID_KEYS: tuple[tuple[str, str], ...] = (
    (Platform.BINARY_SENSOR, "semantic_index_ready"),
    (Platform.SENSOR, "semantic_index_generation"),
    (Platform.SENSOR, "semantic_document_count"),
    (Platform.SENSOR, "semantic_last_refresh_duration"),
)


def _remove_removed_llm_api_refs(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> None:
    """Remove persisted LLM API selections for the deleted in-repo API."""
    for subentry in entry.subentries.values():
        if subentry.subentry_type not in {
            SUBENTRY_TYPE_CONVERSATION,
            SUBENTRY_TYPE_AI_TASK,
        }:
            continue
        api_ids = subentry.data.get(CONF_LLM_HASS_API)
        if not isinstance(api_ids, list):
            continue
        cleaned_api_ids = [
            api_id
            for api_id in api_ids
            if not (
                isinstance(api_id, str)
                and api_id.startswith(_REMOVED_IN_REPO_LLM_API_PREFIX)
            )
        ]
        if cleaned_api_ids == api_ids:
            continue
        data = dict(subentry.data)
        if cleaned_api_ids:
            data[CONF_LLM_HASS_API] = cleaned_api_ids
        else:
            data.pop(CONF_LLM_HASS_API, None)
        hass.config_entries.async_update_subentry(entry, subentry, data=data)


def _remove_removed_entity_registry_entries(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> None:
    """Remove registry entries for diagnostic entities that no longer exist."""
    entity_registry = er.async_get(hass)
    for domain, key in _REMOVED_IN_REPO_ENTITY_UNIQUE_ID_KEYS:
        unique_id = f"{DOMAIN}_{entry.entry_id}_{key}"
        entity_id = entity_registry.async_get_entity_id(domain, DOMAIN, unique_id)
        if entity_id is not None:
            entity_registry.async_remove(entity_id)


def _remove_removed_device_registry_entry(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> None:
    """Remove obsolete workspace-level diagnostic device when it is empty."""
    device_registry = dr.async_get(hass)
    entity_registry = er.async_get(hass)
    device = device_registry.async_get_device(identifiers={(DOMAIN, entry.entry_id)})
    if device is None or er.async_entries_for_device(entity_registry, device.id, True):
        return
    device_registry.async_remove_device(device.id)


def _remove_stale_subentry_registry_entries(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> None:
    """Remove entities and empty devices for subentries no longer in the entry."""
    live_subentry_ids = set(entry.subentries)
    entity_registry = er.async_get(hass)
    for entity_entry in er.async_entries_for_config_entry(
        entity_registry, entry.entry_id
    ):
        subentry_id = entity_entry.config_subentry_id or _subentry_id_from_unique_id(
            entity_entry.unique_id, entry
        )
        if subentry_id is not None and subentry_id not in live_subentry_ids:
            entity_registry.async_remove(entity_entry.entity_id)

    device_registry = dr.async_get(hass)
    entity_registry = er.async_get(hass)
    for device in dr.async_entries_for_config_entry(device_registry, entry.entry_id):
        subentry_id = _subentry_id_from_device(device, entry)
        if subentry_id is None or subentry_id in live_subentry_ids:
            continue
        if er.async_entries_for_device(entity_registry, device.id, True):
            continue
        device_registry.async_remove_device(device.id)


def _subentry_id_from_unique_id(
    unique_id: str, entry: PydanticAIAgentConfigEntry
) -> str | None:
    """Return the subentry ID from an integration-owned entity unique ID."""
    prefix = f"{DOMAIN}_{entry.entry_id}_"
    if not unique_id.startswith(prefix):
        return None
    remainder = unique_id.removeprefix(prefix)
    for subentry_type in (
        SUBENTRY_TYPE_CONVERSATION,
        SUBENTRY_TYPE_AI_TASK,
        SUBENTRY_TYPE_PROVIDER,
    ):
        type_prefix = f"{subentry_type}_"
        if not remainder.startswith(type_prefix):
            continue
        subentry_and_key = remainder.removeprefix(type_prefix)
        for subentry_id in entry.subentries:
            if subentry_and_key == subentry_id or subentry_and_key.startswith(
                f"{subentry_id}_"
            ):
                return subentry_id
        return subentry_and_key.rsplit("_", 1)[0]
    return None


def _subentry_id_from_device(
    device: dr.DeviceEntry, entry: PydanticAIAgentConfigEntry
) -> str | None:
    """Return the subentry ID represented by an integration-owned device."""
    prefix = f"{entry.entry_id}:"
    for domain, identifier in device.identifiers:
        if domain != DOMAIN or not identifier.startswith(prefix):
            continue
        parts = identifier.split(":", 2)
        if len(parts) == 3:
            return parts[2]
    return None


async def _async_remove_removed_memory_store(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> None:
    """Remove obsolete per-entry memory from the deleted in-repo API."""
    store: Store[dict[str, Any]] = Store(
        hass,
        _REMOVED_IN_REPO_MEMORY_STORE_VERSION,
        f"{DOMAIN}.home_semantic.{entry.entry_id}",
    )
    await store.async_remove()
