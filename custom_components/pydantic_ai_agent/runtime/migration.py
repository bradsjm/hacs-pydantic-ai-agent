"""Config entry migration and cleanup helpers."""

import logging
from collections.abc import Mapping
from typing import Any

from homeassistant.const import CONF_LLM_HASS_API, Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.storage import Store

from ..const import (
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
    CONF_CONTEXT_MANAGEMENT_MODE,
    CONF_CONTEXT_SUMMARIZATION_MODEL_REF,
    CONF_CONTEXT_WINDOW_SOURCE,
    CONF_CONTEXT_WINDOW_TOKENS,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_TEMPLATED_EXTRA_BODY,
    CONTEXT_MANAGEMENT_CONTEXT_MANAGER,
    CONTEXT_MANAGEMENT_MODES,
    CONTEXT_MANAGEMENT_SLIDING_WINDOW,
    CONTEXT_WINDOW_SOURCE_DEFAULT,
    CONTEXT_WINDOW_SOURCES,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    DOMAIN,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_PROVIDER,
)
from .types import PydanticAIAgentConfigEntry

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


def _migrate_profile_templated_extra_body(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> None:
    """Migrate old profile chat_template_kwargs rows to templated extra body."""
    for subentry in entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_PROVIDER:
            continue
        profiles = subentry.data.get(CONF_MODEL_PROFILES)
        if not isinstance(profiles, Mapping):
            continue
        updated_profiles, changed = _migrated_profiles(profiles)
        if changed:
            data = dict(subentry.data)
            data[CONF_MODEL_PROFILES] = updated_profiles
            hass.config_entries.async_update_subentry(entry, subentry, data=data)


def _remove_ai_task_legacy_output_mode(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> None:
    """Remove deprecated stored AI task output-mode selections."""
    for subentry in entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_AI_TASK:
            continue
        if "output_mode" not in subentry.data:
            continue
        data = dict(subentry.data)
        data.pop("output_mode", None)
        hass.config_entries.async_update_subentry(entry, subentry, data=data)


def _migrate_context_management_defaults(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> None:
    """Backfill context-management settings added in schema 2.4."""
    for subentry in entry.subentries.values():
        if subentry.subentry_type == SUBENTRY_TYPE_PROVIDER:
            data, changed = _migrated_provider_context_windows(subentry.data)
            if changed:
                hass.config_entries.async_update_subentry(entry, subentry, data=data)
            continue
        if subentry.subentry_type not in {
            SUBENTRY_TYPE_CONVERSATION,
            SUBENTRY_TYPE_AI_TASK,
        }:
            continue
        default_mode = (
            CONTEXT_MANAGEMENT_SLIDING_WINDOW
            if subentry.subentry_type == SUBENTRY_TYPE_AI_TASK
            else CONTEXT_MANAGEMENT_CONTEXT_MANAGER
        )
        data = dict(subentry.data)
        changed = False
        if data.get(CONF_CONTEXT_MANAGEMENT_MODE) not in CONTEXT_MANAGEMENT_MODES:
            data[CONF_CONTEXT_MANAGEMENT_MODE] = default_mode
            changed = True
        if not data.get(CONF_CONTEXT_SUMMARIZATION_MODEL_REF):
            changed = changed or CONF_CONTEXT_SUMMARIZATION_MODEL_REF in data
            data.pop(CONF_CONTEXT_SUMMARIZATION_MODEL_REF, None)
        if changed:
            hass.config_entries.async_update_subentry(entry, subentry, data=data)


def _migrated_provider_context_windows(
    provider_data: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Return provider data with context-window fields on every profile."""
    profiles = provider_data.get(CONF_MODEL_PROFILES)
    if not isinstance(profiles, Mapping):
        return dict(provider_data), False
    updated_profiles: dict[str, object] = {}
    changed = False
    for profile_id, profile in profiles.items():
        if not isinstance(profile, Mapping):
            updated_profiles[str(profile_id)] = profile
            continue
        updated_profile = dict(profile)
        tokens = updated_profile.get(CONF_CONTEXT_WINDOW_TOKENS)
        if isinstance(tokens, bool) or not isinstance(tokens, int | float | str):
            updated_profile[CONF_CONTEXT_WINDOW_TOKENS] = DEFAULT_CONTEXT_WINDOW_TOKENS
            changed = True
        else:
            try:
                parsed_tokens = int(tokens)
            except ValueError:
                parsed_tokens = DEFAULT_CONTEXT_WINDOW_TOKENS
            if parsed_tokens <= 0:
                parsed_tokens = DEFAULT_CONTEXT_WINDOW_TOKENS
            if parsed_tokens != tokens:
                changed = True
            updated_profile[CONF_CONTEXT_WINDOW_TOKENS] = parsed_tokens
        source = updated_profile.get(CONF_CONTEXT_WINDOW_SOURCE)
        if not isinstance(source, str) or source not in CONTEXT_WINDOW_SOURCES:
            updated_profile[CONF_CONTEXT_WINDOW_SOURCE] = CONTEXT_WINDOW_SOURCE_DEFAULT
            changed = True
        updated_profiles[str(profile_id)] = updated_profile
    if not changed:
        return dict(provider_data), False
    data = dict(provider_data)
    data[CONF_MODEL_PROFILES] = updated_profiles
    return data, True


def _migrated_profiles(
    profiles: Mapping[str, object],
) -> tuple[dict[str, object], bool]:
    """Return migrated provider profiles and whether any profile changed."""
    updated_profiles = dict(profiles)
    changed = False
    for profile_id, profile in profiles.items():
        updated_profile = _migrated_profile(profile)
        if updated_profile is None:
            continue
        updated_profiles[profile_id] = updated_profile
        changed = True
    return updated_profiles, changed


def _migrated_profile(profile: object) -> dict[str, object] | None:
    """Return one migrated profile when it contains legacy chat-template rows."""
    if not isinstance(profile, Mapping):
        return None
    model_settings = profile.get(CONF_MODEL_SETTINGS)
    if not isinstance(model_settings, Mapping):
        return None
    legacy_rows = model_settings.get("chat_template_kwargs")
    if not isinstance(legacy_rows, list):
        return None
    updated_model_settings = dict(model_settings)
    updated_model_settings.pop("chat_template_kwargs", None)
    migrated_rows = _migrated_legacy_rows(legacy_rows)
    if migrated_rows and CONF_TEMPLATED_EXTRA_BODY not in updated_model_settings:
        updated_model_settings[CONF_TEMPLATED_EXTRA_BODY] = migrated_rows
    updated_profile = dict(profile)
    if updated_model_settings:
        updated_profile[CONF_MODEL_SETTINGS] = updated_model_settings
    else:
        updated_profile.pop(CONF_MODEL_SETTINGS, None)
    return updated_profile


def _migrated_legacy_rows(legacy_rows: list[object]) -> list[dict[str, str]]:
    """Return migrated templated extra-body rows from legacy chat-template rows."""
    migrated_rows: list[dict[str, str]] = []
    for row in legacy_rows:
        if not isinstance(row, Mapping):
            continue
        key = row.get(CONF_CHAT_TEMPLATE_KWARG_KEY)
        value_template = row.get(CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE)
        if isinstance(key, str) and isinstance(value_template, str):
            migrated_rows.append(
                {
                    CONF_CHAT_TEMPLATE_KWARG_KEY: f"chat_template_kwargs.{key.strip()}",
                    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: value_template,
                }
            )
    return migrated_rows
