"""Pydantic AI Agent integration."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_API_KEY, CONF_NAME, Platform
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
import voluptuous as vol

from .const import (
    CONF_BASE_URL,
    CONF_ENABLED,
    CONF_FALLBACK_MODEL_REFS,
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_HEADERS,
    CONF_MCP_URL,
    CONF_MODEL,
    CONF_MODEL_SETTINGS,
    CONF_OUTPUT_MODE,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_MODE,
    DOMAIN,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_PROVIDER,
)
from .provider_validation import ProviderValidationError, async_probe_model
from .logfire_support import (
    async_configure_logfire,
    async_release_logfire,
    logfire_enabled,
    logfire_include_content,
)
from .metrics import (
    EVENT_MCP_TOOL_REFRESH_COMPLETED,
    EVENT_MCP_TOOL_REFRESH_FAILED,
    MetricsStore,
    fire_integration_event,
)
from .model_settings import (
    normalise_applied_model_settings,
    validation_probe_model_settings,
)
from .mcp import (
    MCPValidationError,
    async_refresh_mcp_tools,
    cached_mcp_tools,
    mcp_subentries,
)
from .model_profiles import (
    enabled_model_profile_refs,
    parse_model_profile_ref,
    provider_model_profiles,
    provider_subentries,
    resolve_model_profile,
    ResolvedModelProfile,
)
from .repairs import (
    async_delete_logfire_token_conflict_issue,
    async_create_model_validation_issue,
    async_delete_entry_repair_issues,
    async_delete_model_validation_issue,
    async_delete_stale_model_validation_issues,
    model_validation_issue_id,
)
from .structured_output import structured_output_mode
from .home_semantic import HomeSemanticIndexManager
from .home_semantic.llm_api import HomeSemanticAPI

_LOGGER = logging.getLogger(__name__)

_MODEL_VALIDATION_OUTPUT_MODE_KEY = "_pydantic_ai_agent_output_mode"


@dataclass(frozen=True, kw_only=True)
class _ConfiguredModelProbe:
    """One setup-time model validation probe."""

    provider_subentry: ConfigSubentry
    issue_profile_id: str
    model: str
    model_settings: dict[str, Any]
    output_mode: str | None


PLATFORMS: tuple[Platform, ...] = (
    Platform.CONVERSATION,
    Platform.AI_TASK,
    Platform.SENSOR,
    Platform.BINARY_SENSOR,
)
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
class ProviderRuntimeData:
    """Provider runtime data owned by one workspace provider subentry."""

    provider_subentry_id: str
    name: str
    api_key: str
    provider_mode: str
    base_url: str | None
    provider_headers: dict[str, str] = field(default_factory=dict)
    provider_extra_body: dict[str, Any] = field(default_factory=dict)
    client: Any | None = None
    discovered_models: list[str] | None = None


@dataclass(frozen=True, kw_only=True)
class MCPServerRuntimeData:
    """MCP runtime data owned by one workspace MCP subentry."""

    subentry_id: str
    name: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    discovered_tools: list[dict[str, Any]] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    client: Any | None = None


@dataclass(frozen=True, kw_only=True)
class WorkspaceRuntimeData:
    """Workspace data shared by subentry-backed entities."""

    workspace_name: str
    providers: dict[str, ProviderRuntimeData] = field(default_factory=dict)
    mcp_servers: dict[str, MCPServerRuntimeData] = field(default_factory=dict)
    model_profiles: dict[str, ResolvedModelProfile] = field(default_factory=dict)
    mcp_tool_cache: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    metrics: MetricsStore = field(default_factory=MetricsStore)
    home_semantic: HomeSemanticIndexManager | None = None
    logfire_enabled: bool = False
    logfire_include_content: bool = False


type PydanticAIAgentConfigEntry = ConfigEntry[WorkspaceRuntimeData]


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
    """Build workspace runtime data, then set up entity platforms."""
    provider_runtimes = _provider_runtimes(entry)
    mcp_runtime = _mcp_server_runtimes(entry)
    model_profiles = _resolved_model_profiles(entry, provider_runtimes)
    home_semantic = HomeSemanticIndexManager(hass, entry)
    await async_configure_logfire(hass, entry)
    try:
        entry.runtime_data = WorkspaceRuntimeData(
            workspace_name=entry.data[CONF_NAME],
            providers=provider_runtimes,
            mcp_servers=mcp_runtime,
            model_profiles=model_profiles,
            home_semantic=home_semantic,
            logfire_enabled=logfire_enabled(hass, entry),
            logfire_include_content=logfire_include_content(hass, entry),
        )
        await _async_validate_configured_models(hass, entry)
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
        await async_configure_logfire(hass, entry)
        entry.runtime_data = replace(
            entry.runtime_data,
            logfire_enabled=logfire_enabled(hass, entry),
            logfire_include_content=logfire_include_content(hass, entry),
        )
    except BaseException:
        await async_release_logfire(hass, entry)
        raise
    entry.async_on_unload(llm.async_register_api(hass, HomeSemanticAPI(hass, entry)))
    home_semantic.async_start()
    entry.async_on_unload(home_semantic.async_stop)
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
    """Reject legacy config entries that predate the workspace schema."""
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
    except MCPValidationError as err:
        errors.append(_mcp_error_response(err))
    flat_tools = [tool for tools in tools_by_server.values() for tool in tools]
    return {
        "success": not errors,
        "servers": tools_by_server,
        "tools": flat_tools,
        "errors": errors,
    }


def _provider_runtimes(
    entry: PydanticAIAgentConfigEntry,
) -> dict[str, ProviderRuntimeData]:
    """Return runtime provider data for structurally valid provider subentries."""
    runtimes: dict[str, ProviderRuntimeData] = {}
    for subentry in provider_subentries(entry):
        api_key = subentry.data.get(CONF_API_KEY)
        provider_mode = subentry.data.get(CONF_PROVIDER_MODE)
        if not isinstance(api_key, str) or not api_key:
            _LOGGER.warning(
                "Skipping provider subentry %s without an API key",
                subentry.subentry_id,
            )
            continue
        if not isinstance(provider_mode, str) or not provider_mode:
            _LOGGER.warning(
                "Skipping provider subentry %s without a provider mode",
                subentry.subentry_id,
            )
            continue
        headers = subentry.data.get(CONF_PROVIDER_HEADERS)
        provider_extra_body = subentry.data.get(CONF_PROVIDER_EXTRA_BODY)
        runtimes[subentry.subentry_id] = ProviderRuntimeData(
            provider_subentry_id=subentry.subentry_id,
            name=subentry.title,
            api_key=api_key,
            provider_mode=provider_mode,
            base_url=subentry.data.get(CONF_BASE_URL),
            provider_headers=dict(headers) if isinstance(headers, Mapping) else {},
            provider_extra_body=dict(provider_extra_body)
            if isinstance(provider_extra_body, Mapping)
            else {},
        )
    return runtimes


def _mcp_server_runtimes(
    entry: PydanticAIAgentConfigEntry,
) -> dict[str, MCPServerRuntimeData]:
    """Return runtime MCP server data for configured MCP subentries."""
    runtimes: dict[str, MCPServerRuntimeData] = {}
    for subentry in mcp_subentries(entry):
        headers = subentry.data.get(CONF_MCP_HEADERS)
        allowed_tools = subentry.data.get(CONF_MCP_ALLOWED_TOOLS)
        runtimes[subentry.subentry_id] = MCPServerRuntimeData(
            subentry_id=subentry.subentry_id,
            name=subentry.title,
            url=str(subentry.data.get(CONF_MCP_URL, "")),
            headers=dict(headers) if isinstance(headers, Mapping) else {},
            allowed_tools=list(allowed_tools)
            if isinstance(allowed_tools, list)
            else [],
        )
    return runtimes


def _resolved_model_profiles(
    entry: PydanticAIAgentConfigEntry,
    provider_runtimes: Mapping[str, ProviderRuntimeData],
) -> dict[str, ResolvedModelProfile]:
    """Return enabled model profiles for providers that loaded successfully."""
    profiles: dict[str, ResolvedModelProfile] = {}
    for ref in enabled_model_profile_refs(entry):
        provider_subentry_id, _profile_id = parse_model_profile_ref(ref)
        if provider_subentry_id not in provider_runtimes:
            continue
        profiles[ref] = resolve_model_profile(entry, ref)
    return profiles


def _configured_subentry_models(
    entry: PydanticAIAgentConfigEntry,
) -> list[_ConfiguredModelProbe]:
    """Return unique model probes needed before the entry can load."""
    models: list[_ConfiguredModelProbe] = []
    seen: set[tuple[str, str, str, str | None]] = set()

    def add_model(
        provider_subentry: ConfigSubentry,
        profile_ref: str,
        subentry_data: Mapping[str, Any],
        output_mode: str | None,
    ) -> None:
        provider_subentry_id, profile_id = parse_model_profile_ref(profile_ref)
        if provider_subentry.subentry_id != provider_subentry_id:
            return
        profile = provider_model_profiles(provider_subentry).get(profile_id)
        if profile is None or not bool(profile.get(CONF_ENABLED, False)):
            return
        model = profile.get(CONF_MODEL)
        if not isinstance(model, str) or not model:
            return
        settings = profile.get(CONF_MODEL_SETTINGS)
        model_settings = validation_probe_model_settings(
            settings if isinstance(settings, Mapping) else {}, subentry_data
        )
        dedupe_key = (
            provider_subentry.subentry_id,
            model,
            normalise_applied_model_settings(model_settings),
            output_mode,
        )
        # Several subentries can target the same model/settings pair, so probe
        # each unique runtime capability once during setup.
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        models.append(
            _ConfiguredModelProbe(
                provider_subentry=provider_subentry,
                issue_profile_id=profile_ref,
                model=model,
                model_settings=model_settings,
                output_mode=output_mode,
            )
        )

    for subentry in entry.subentries.values():
        if subentry.subentry_type not in (
            SUBENTRY_TYPE_CONVERSATION,
            SUBENTRY_TYPE_AI_TASK,
        ):
            continue
        primary_ref = subentry.data.get(CONF_PRIMARY_MODEL_REF)
        if not isinstance(primary_ref, str) or not primary_ref:
            _LOGGER.warning(
                "Skipping legacy %s subentry without model profile: %s",
                subentry.subentry_type,
                subentry.subentry_id,
            )
            continue
        fallback_refs = subentry.data.get(CONF_FALLBACK_MODEL_REFS, [])
        if isinstance(fallback_refs, str) or not isinstance(fallback_refs, list):
            fallback_refs = []
        output_mode = (
            structured_output_mode(subentry.data.get(CONF_OUTPUT_MODE))
            if subentry.subentry_type == SUBENTRY_TYPE_AI_TASK
            else None
        )
        refs = [primary_ref, *[ref for ref in fallback_refs if isinstance(ref, str)]]
        for ref in refs:
            try:
                provider_subentry_id, _profile_id = parse_model_profile_ref(ref)
            except HomeAssistantError:
                _LOGGER.warning(
                    "Skipping malformed model profile reference %s for subentry %s",
                    ref,
                    subentry.subentry_id,
                )
                continue
            provider_subentry = entry.subentries.get(provider_subentry_id)
            if (
                provider_subentry is None
                or provider_subentry.subentry_type != SUBENTRY_TYPE_PROVIDER
            ):
                _LOGGER.warning(
                    "Skipping stale model profile reference %s for subentry %s",
                    ref,
                    subentry.subentry_id,
                )
                continue
            add_model(provider_subentry, ref, subentry.data, output_mode)
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
    """Probe configured models and surface provider/profile repairs."""
    current_issue_ids: set[str] = set()
    for probe in _configured_subentry_models(entry):
        repair_settings = _repair_issue_model_settings(
            probe.model_settings, probe.output_mode
        )
        current_issue_ids.add(
            model_validation_issue_id(entry, probe.issue_profile_id, repair_settings)
        )
        try:
            if probe.output_mode is None:
                await async_probe_model(
                    hass,
                    probe.provider_subentry.data,
                    probe.model,
                    probe.model_settings,
                )
            else:
                await async_probe_model(
                    hass,
                    probe.provider_subentry.data,
                    probe.model,
                    probe.model_settings,
                    structured_output_mode=probe.output_mode,
                )
        except ProviderValidationError as err:
            _LOGGER.warning(
                'Provider validation failed during setup for model "%s": '
                "reason=%s status_code=%s",
                probe.model,
                err.reason,
                err.status_code,
            )
            async_create_model_validation_issue(
                hass,
                entry,
                probe.issue_profile_id,
                probe.model,
                repair_settings,
                err,
            )
            continue
        async_delete_model_validation_issue(
            hass, entry, probe.issue_profile_id, probe.model, repair_settings
        )
    async_delete_stale_model_validation_issues(hass, entry, current_issue_ids)
