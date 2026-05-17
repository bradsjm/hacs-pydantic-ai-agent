"""Pydantic AI Agent integration."""

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, CONF_NAME, Platform
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import (
    ConfigEntryAuthFailed,
    ConfigEntryNotReady,
)
import voluptuous as vol

from .config_flow import ProviderValidationError, async_probe_model
from .const import (
    CONF_BASE_URL,
    CONF_ENABLE_SKILL_SCRIPT_EXECUTION,
    CONF_FALLBACK_MODEL_SUBENTRY_IDS,
    CONF_MAX_ITERATIONS,
    CONF_MODEL,
    CONF_MODEL_SETTINGS,
    CONF_MODEL_SUBENTRY_ID,
    CONF_OUTPUT_MODE,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_MODE,
    CONF_SKILLS_FOLDER,
    DEFAULT_SKILLS_FOLDER,
    DOMAIN,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_MODEL,
)
from .logfire_support import (
    async_configure_logfire,
    logfire_enabled,
    logfire_include_content,
)
from .mcp import (
    MCPValidationError,
    async_refresh_mcp_tools,
    cached_mcp_tools,
    mcp_subentries,
)
from .repairs import (
    async_delete_logfire_token_conflict_issue,
    async_create_model_validation_issue,
    async_delete_model_validation_issue,
    async_delete_stale_model_validation_issues,
    model_validation_issue_id,
)
from .structured_output import structured_output_mode

_LOGGER = logging.getLogger(__name__)

_AUTH_FAILURE_REASONS = {"invalid_auth"}
_RECONFIGURABLE_MODEL_FAILURE_REASONS = {
    "invalid_model",
    "invalid_provider_config",
    "model_does_not_support_streaming",
    "permission_denied",
}
_MODEL_VALIDATION_OUTPUT_MODE_KEY = "_pydantic_ai_agent_output_mode"

PLATFORMS: tuple[Platform, ...] = (Platform.CONVERSATION, Platform.AI_TASK)
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


@dataclass(frozen=True, kw_only=True)
class PydanticAIAgentRuntimeData:
    """Provider connection data shared by subentry-backed entities."""

    provider_mode: str
    name: str
    api_key: str
    base_url: str | None
    logfire_enabled: bool
    logfire_include_content: bool
    skills_folder: str
    enable_skill_script_execution: bool
    provider_headers: dict[str, str] = field(default_factory=dict)
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    mcp_tool_cache: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


type PydanticAIAgentConfigEntry = ConfigEntry[PydanticAIAgentRuntimeData]


async def async_setup(hass: HomeAssistant, config: dict[str, Any]) -> bool:
    """Set up integration-wide MCP discovery services."""

    async def async_list_mcp_tools(call: ServiceCall) -> dict[str, Any]:
        """Return cached MCP tools, discovering them if needed."""
        return await _async_mcp_tools_service(hass, call, refresh=False)

    async def async_refresh_mcp_tools(call: ServiceCall) -> dict[str, Any]:
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
        async_refresh_mcp_tools,
        schema=_MCP_SERVICE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    return True


async def async_setup_entry(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> bool:
    """Validate configured subentries, then set up entity platforms."""
    await _async_validate_configured_models(hass, entry)
    await async_configure_logfire(hass, entry)

    entry.runtime_data = PydanticAIAgentRuntimeData(
        provider_mode=entry.data[CONF_PROVIDER_MODE],
        name=entry.data[CONF_NAME],
        api_key=entry.data[CONF_API_KEY],
        base_url=entry.data.get(CONF_BASE_URL),
        provider_headers=dict(entry.data.get(CONF_PROVIDER_HEADERS, {})),
        logfire_enabled=logfire_enabled(hass, entry),
        logfire_include_content=logfire_include_content(hass, entry),
        skills_folder=entry.data.get(CONF_SKILLS_FOLDER, DEFAULT_SKILLS_FOLDER),
        enable_skill_script_execution=bool(
            entry.data.get(CONF_ENABLE_SKILL_SCRIPT_EXECUTION, False)
        ),
        mcp_servers=[
            {CONF_NAME: subentry.title, **dict(subentry.data)}
            for subentry in mcp_subentries(entry)
        ],
    )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_update_entry))
    return True


async def async_unload_entry(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> bool:
    """Unload a config entry."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded:
        async_delete_logfire_token_conflict_issue(hass, entry)
    return unloaded


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
        raise MCPValidationError(
            "config_entry_not_found",
            "Pydantic AI Agent config entry was not found.",
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
    try:
        entry = _config_entry_for_service(hass, call.data[ATTR_CONFIG_ENTRY_ID])
        subentry_ids = _mcp_service_subentry_ids(
            entry, call.data.get(ATTR_MCP_SERVER_ID)
        )
        if not subentry_ids:
            return {"success": True, "servers": {}, "tools": []}
        for subentry_id in subentry_ids:
            try:
                tools = None if refresh else cached_mcp_tools(entry, subentry_id)
                if tools is None:
                    tools = await async_refresh_mcp_tools(hass, entry, subentry_id)
                tools_by_server[subentry_id] = tools
            except MCPValidationError as err:
                _LOGGER.warning(
                    "MCP tool discovery failed: reason=%s server_id=%s",
                    err.reason,
                    err.server_id,
                )
                errors.append(_mcp_error_response(err))
    except MCPValidationError as err:
        errors.append(_mcp_error_response(err))
    flat_tools = [tool for tools in tools_by_server.values() for tool in tools]
    return {
        "success": not errors,
        "servers": tools_by_server,
        "tools": flat_tools,
        "errors": errors,
    }


def _normalise_model_settings(settings: Mapping[str, Any]) -> str:
    """Return a stable representation of model settings for de-duplication."""
    provider_settings = dict(settings)
    provider_settings.pop(CONF_MAX_ITERATIONS, None)
    return json.dumps(provider_settings, sort_keys=True, separators=(",", ":"))


def _configured_subentry_models(
    entry: PydanticAIAgentConfigEntry,
) -> list[tuple[str, str, dict[str, Any], str | None]]:
    """Return unique model probes needed before the entry can load."""
    model_profiles = {
        subentry.subentry_id: subentry
        for subentry in entry.subentries.values()
        if subentry.subentry_type == SUBENTRY_TYPE_MODEL
    }
    models: list[tuple[str, str, dict[str, Any], str | None]] = []
    seen: set[tuple[str, str, str | None]] = set()

    def add_model(profile_id: str, output_mode: str | None) -> None:
        profile = model_profiles.get(profile_id)
        if profile is None:
            return
        model = profile.data.get(CONF_MODEL)
        if not isinstance(model, str) or not model:
            return
        settings = profile.data.get(CONF_MODEL_SETTINGS)
        model_settings = dict(settings) if isinstance(settings, Mapping) else {}
        dedupe_key = (
            model,
            _normalise_model_settings(model_settings),
            output_mode,
        )
        # Several subentries can target the same model/settings pair, so probe
        # each unique runtime capability once during setup.
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        models.append((profile_id, model, model_settings, output_mode))

    for profile_id in model_profiles:
        add_model(profile_id, None)

    for subentry in entry.subentries.values():
        if subentry.subentry_type not in (
            SUBENTRY_TYPE_CONVERSATION,
            SUBENTRY_TYPE_AI_TASK,
        ):
            continue
        primary_id = subentry.data.get(CONF_MODEL_SUBENTRY_ID)
        if not isinstance(primary_id, str) or not primary_id:
            _LOGGER.warning(
                "Skipping legacy %s subentry without model profile: %s",
                subentry.subentry_type,
                subentry.subentry_id,
            )
            continue
        fallback_ids = subentry.data.get(CONF_FALLBACK_MODEL_SUBENTRY_IDS, [])
        if isinstance(fallback_ids, str) or not isinstance(fallback_ids, list):
            fallback_ids = []
        output_mode = (
            structured_output_mode(subentry.data.get(CONF_OUTPUT_MODE))
            if subentry.subentry_type == SUBENTRY_TYPE_AI_TASK
            else None
        )
        for profile_id in [primary_id, *fallback_ids]:
            if profile_id not in model_profiles:
                _LOGGER.warning(
                    "Skipping stale model profile reference %s for subentry %s",
                    profile_id,
                    subentry.subentry_id,
                )
                continue
            add_model(profile_id, output_mode)
    return models


def _repair_issue_model_settings(
    model_settings: Mapping[str, Any], output_mode: str | None
) -> dict[str, Any]:
    """Return settings material that separates chat and structured probes."""
    if output_mode is None:
        return dict(model_settings)
    # Repair issue ids include the output mode so probes for the same model do
    # not collide when different subentries require different capabilities.
    return {
        **model_settings,
        _MODEL_VALIDATION_OUTPUT_MODE_KEY: output_mode,
    }


async def _async_validate_configured_models(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> None:
    """Probe configured models and surface user-fixable failures as repairs."""
    current_issue_ids: set[str] = set()
    for (
        model_subentry_id,
        model,
        model_settings,
        output_mode,
    ) in _configured_subentry_models(entry):
        repair_settings = _repair_issue_model_settings(model_settings, output_mode)
        current_issue_ids.add(
            model_validation_issue_id(entry, model_subentry_id, repair_settings)
        )
        try:
            if output_mode is None:
                await async_probe_model(hass, entry.data, model, model_settings)
            else:
                await async_probe_model(
                    hass,
                    entry.data,
                    model,
                    model_settings,
                    structured_output_mode=output_mode,
                )
        except ProviderValidationError as err:
            _LOGGER.warning(
                'Provider validation failed during setup for model "%s": '
                "reason=%s status_code=%s",
                model,
                err.reason,
                err.status_code,
            )
            # Auth failures require reauth, model/configuration failures can be
            # repaired after load, and transient provider failures should retry.
            if err.reason in _AUTH_FAILURE_REASONS:
                raise ConfigEntryAuthFailed(err.message) from err
            if err.reason in _RECONFIGURABLE_MODEL_FAILURE_REASONS:
                async_create_model_validation_issue(
                    hass, entry, model_subentry_id, model, repair_settings, err
                )
                continue
            raise ConfigEntryNotReady(err.message) from err
        async_delete_model_validation_issue(
            hass, entry, model_subentry_id, model, repair_settings
        )
    async_delete_stale_model_validation_issues(hass, entry, current_issue_ids)
