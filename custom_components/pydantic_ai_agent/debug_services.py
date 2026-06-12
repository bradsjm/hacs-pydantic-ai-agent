"""Read-only debug response services for Pydantic AI Agent."""

from typing import Any

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import ServiceValidationError

from ._debug_service_responses import (
    metrics_status,
    model_profile_summaries,
    tool_source_status,
    workspace_status,
)
from .const import DOMAIN
from .model_profiles import provider_subentries

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

_OPTIONAL_ENTRY_SCHEMA = {vol.Optional(ATTR_CONFIG_ENTRY_ID): str}
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
        vol.Optional(ATTR_LIMIT, default=50): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=200)
        ),
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
    entries = _entries_for_service(hass, call.data.get(ATTR_CONFIG_ENTRY_ID))
    return {
        "success": True,
        "count": len(entries),
        "entries": [
            workspace_status(
                entry,
                include_subentries=call.data[ATTR_INCLUDE_SUBENTRIES],
                include_runtime=call.data[ATTR_INCLUDE_RUNTIME],
            )
            for entry in entries
        ],
    }


def _list_model_profiles(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    entries = _entries_for_service(hass, call.data.get(ATTR_CONFIG_ENTRY_ID))
    profiles: list[dict[str, Any]] = []
    for entry in entries:
        for provider_subentry in provider_subentries(entry):
            if (
                provider_id := call.data.get(ATTR_PROVIDER_SUBENTRY_ID)
            ) is not None and provider_subentry.subentry_id != provider_id:
                continue
            profiles.extend(
                model_profile_summaries(
                    entry,
                    provider_subentry,
                    enabled_only=call.data[ATTR_ENABLED_ONLY],
                )
            )
    return {"success": True, "count": len(profiles), "profiles": profiles}


def _get_agent_metrics(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    entries = _entries_for_service(hass, call.data.get(ATTR_CONFIG_ENTRY_ID))
    subentry_id = call.data.get(ATTR_SUBENTRY_ID)
    return {
        "success": True,
        "count": len(entries),
        "entries": [
            metrics_status(entry, subentry_id=subentry_id) for entry in entries
        ],
    }


def _get_tool_source_status(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    entries = _entries_for_service(hass, call.data.get(ATTR_CONFIG_ENTRY_ID))
    subentry_id = call.data.get(ATTR_SUBENTRY_ID)
    return {
        "success": True,
        "count": len(entries),
        "entries": [
            tool_source_status(
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
        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="config_entry_not_found",
                translation_placeholders={"config_entry_id": entry_id},
            )
        return [entry]
    return [entry for entry in hass.config_entries.async_entries(DOMAIN)]
