"""Pydantic AI Agent integration."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME, Platform
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.storage import Store
import voluptuous as vol

from .const import (
    CONF_BASE_URL,
    CONF_ENABLED,
    CONF_FALLBACK_MODEL_REFS,
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
from .metrics import MetricsStore
from .model_settings import (
    normalise_applied_model_settings,
    validation_probe_model_settings,
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
    async_create_provider_auth_issue,
    async_delete_entry_repair_issues,
    async_delete_model_validation_issue,
    async_delete_provider_auth_issue,
    async_delete_stale_model_validation_issues,
    async_delete_stale_provider_auth_issues,
    model_validation_issue_id,
    provider_validation_is_auth_failure,
)
from .structured_output import structured_output_mode
from .debug_services import async_setup_services as async_setup_debug_services

_LOGGER = logging.getLogger(__name__)

_MODEL_VALIDATION_OUTPUT_MODE_KEY = "_pydantic_ai_agent_output_mode"
_REMOVED_IN_REPO_LLM_API_PREFIX = "pydantic_ai_agent_home_semantic_"
_REMOVED_IN_REPO_MEMORY_STORE_VERSION = 1
_REMOVED_IN_REPO_ENTITY_UNIQUE_ID_KEYS: tuple[tuple[str, str], ...] = (
    (Platform.BINARY_SENSOR, "semantic_index_ready"),
    (Platform.SENSOR, "semantic_index_generation"),
    (Platform.SENSOR, "semantic_document_count"),
    (Platform.SENSOR, "semantic_last_refresh_duration"),
)


@dataclass(frozen=True, kw_only=True)
class _ConfiguredModelProbe:
    """One setup-time model validation probe."""

    provider_subentry: ConfigSubentry
    issue_profile_id: str
    failure_keys: tuple[str, ...]
    model: str
    model_settings: dict[str, Any]
    output_mode: str | None


PLATFORMS: tuple[Platform, ...] = (
    Platform.CONVERSATION,
    Platform.AI_TASK,
    Platform.SENSOR,
    Platform.BINARY_SENSOR,
)
SERVICE_GET_AGENT_RUN_DIAGNOSTICS = "get_agent_run_diagnostics"
ATTR_CONFIG_ENTRY_ID = "config_entry_id"
ATTR_SUBENTRY_ID = "subentry_id"
ATTR_SECTION = "section"
ATTR_OFFSET = "offset"
ATTR_LIMIT = "limit"

_RUN_DIAGNOSTIC_SECTIONS = (
    "summary",
    "usage",
    "tool_definitions",
    "tool_calls",
    "model_profile",
    "output",
    "errors",
    "timeline",
)

_RUN_DIAGNOSTICS_SERVICE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CONFIG_ENTRY_ID): str,
        vol.Required(ATTR_SUBENTRY_ID): str,
        vol.Optional(ATTR_SECTION): vol.In(_RUN_DIAGNOSTIC_SECTIONS),
        vol.Optional(ATTR_OFFSET, default=0): vol.All(
            vol.Coerce(int), vol.Range(min=0)
        ),
        vol.Optional(ATTR_LIMIT, default=25): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=100)
        ),
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
class WorkspaceRuntimeData:
    """Workspace data shared by subentry-backed entities."""

    workspace_name: str
    providers: dict[str, ProviderRuntimeData] = field(default_factory=dict)
    model_profiles: dict[str, ResolvedModelProfile] = field(default_factory=dict)
    metrics: MetricsStore = field(default_factory=MetricsStore)
    latest_stream_traces: dict[str, dict[str, Any]] = field(default_factory=dict)
    latest_run_diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_validation_failures: dict[str, str] = field(default_factory=dict)
    runtime_provider_auth_failures: dict[str, list[str]] = field(default_factory=dict)
    logfire_enabled: bool = False
    logfire_include_content: bool = False


type PydanticAIAgentConfigEntry = ConfigEntry[WorkspaceRuntimeData]


async def async_setup(hass: HomeAssistant, config: dict[str, Any]) -> bool:
    """Set up integration-wide services."""

    async def async_get_agent_run_diagnostics(call: ServiceCall) -> dict[str, Any]:
        """Return targeted latest-run diagnostics for one agent subentry."""
        return _agent_run_diagnostics_service(hass, call)

    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_AGENT_RUN_DIAGNOSTICS,
        async_get_agent_run_diagnostics,
        schema=_RUN_DIAGNOSTICS_SERVICE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    await async_setup_debug_services(hass)
    return True


async def async_setup_entry(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> bool:
    """Build workspace runtime data, then set up entity platforms."""
    provider_runtimes = _provider_runtimes(entry)
    model_profiles = _resolved_model_profiles(entry, provider_runtimes)
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
        await async_configure_logfire(hass, entry)
        entry.runtime_data = replace(
            entry.runtime_data,
            logfire_enabled=logfire_enabled(hass, entry),
            logfire_include_content=logfire_include_content(hass, entry),
        )
    except BaseException:
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


async def async_update_entry(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> None:
    """Reload the entry after config entry or subentry updates."""
    await hass.config_entries.async_reload(entry.entry_id)


def _config_entry_for_service(
    hass: HomeAssistant, entry_id: str
) -> PydanticAIAgentConfigEntry:
    """Return a config entry for a response service."""
    entry = hass.config_entries.async_get_entry(entry_id)
    if entry is None or entry.domain != DOMAIN:
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="config_entry_not_found",
            translation_placeholders={"config_entry_id": entry_id},
        )
    return entry


def _agent_run_diagnostics_service(
    hass: HomeAssistant,
    call: ServiceCall,
) -> dict[str, Any]:
    """Return compact latest-run diagnostics for one agent subentry."""
    entry = _config_entry_for_service(hass, call.data[ATTR_CONFIG_ENTRY_ID])
    subentry_id = call.data[ATTR_SUBENTRY_ID]
    if subentry_id not in entry.subentries:
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="subentry_not_found",
            translation_placeholders={
                "config_entry_id": entry.entry_id,
                "subentry_id": subentry_id,
            },
        )
    runtime_data = getattr(entry, "runtime_data", None)
    diagnostics = (
        None
        if runtime_data is None
        else runtime_data.latest_run_diagnostics.get(subentry_id)
    )
    base: dict[str, Any] = {
        "success": True,
        "config_entry_id": entry.entry_id,
        "subentry_id": subentry_id,
        "section": call.data.get(ATTR_SECTION),
        "errors": [],
    }
    if diagnostics is None:
        return {**base, "status": "no_run_yet", "run": None}
    return {
        **base,
        "status": diagnostics.get("status"),
        "run": _agent_run_diagnostics_section(
            diagnostics,
            section=call.data.get(ATTR_SECTION),
            offset=call.data[ATTR_OFFSET],
            limit=call.data[ATTR_LIMIT],
        ),
    }


def _agent_run_diagnostics_section(
    diagnostics: Mapping[str, Any], *, section: str | None, offset: int, limit: int
) -> dict[str, Any]:
    """Return one compact section from a bounded run diagnostics payload."""
    summary = _mapping(diagnostics.get("summary"))
    timeline = diagnostics.get("timeline")
    if section == "timeline":
        return {"timeline": _timeline_page(timeline, offset=offset, limit=limit)}
    if section == "summary":
        return {"summary": summary}
    if section == "usage":
        return {"usage": summary.get("usage")}
    if section == "model_profile":
        return {"model_profile": summary.get("model_profile")}
    if section == "output":
        return {"output": summary.get("output")}
    if section == "errors":
        return {"errors": _run_errors(summary)}
    if section == "tool_definitions":
        return {"tool_definitions": _tool_definitions(timeline)}
    if section == "tool_calls":
        return {"tool_calls": _tool_calls(timeline, offset=offset, limit=limit)}
    timeline_page = _timeline_page(timeline, offset=0, limit=0)
    return {
        "run_id": diagnostics.get("run_id"),
        "status": diagnostics.get("status"),
        "started_at": diagnostics.get("started_at"),
        "finished_at": diagnostics.get("finished_at"),
        "duration_ms": diagnostics.get("duration_ms"),
        "conversation_id": diagnostics.get("conversation_id"),
        "summary": {
            "model_profile": summary.get("model_profile"),
            "usage": summary.get("usage"),
            "duration": summary.get("duration"),
            "error": summary.get("error"),
            "failure": summary.get("failure"),
        },
        "timeline": {
            "total_count": timeline_page["total_count"],
            "available_count": timeline_page["available_count"],
            "omitted_middle_count": timeline_page.get("omitted_middle_count", 0),
        },
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return value when it is a mapping, otherwise an empty mapping."""
    return value if isinstance(value, Mapping) else {}


def _timeline_events(timeline: Any) -> tuple[list[dict[str, Any]], int, int]:
    """Return available timeline events, total count, and omitted middle count."""
    if isinstance(timeline, list):
        events = [event for event in timeline if isinstance(event, dict)]
        return events, len(events), 0
    if isinstance(timeline, Mapping):
        head = timeline.get("head", [])
        tail = timeline.get("tail", [])
        events = [event for event in [*head, *tail] if isinstance(event, dict)]
        return (
            events,
            int(timeline.get("total_count", len(events))),
            int(timeline.get("omitted_middle_count", 0)),
        )
    return [], 0, 0


def _timeline_page(timeline: Any, *, offset: int, limit: int) -> dict[str, Any]:
    """Return a compact page over the bounded available timeline."""
    events, total_count, omitted_middle_count = _timeline_events(timeline)
    page = events[offset : offset + limit] if limit else []
    return {
        "items": page,
        "offset": offset,
        "limit": limit,
        "total_count": total_count,
        "available_count": len(events),
        "has_more_available": offset + len(page) < len(events),
        "omitted_middle_count": omitted_middle_count,
    }


def _tool_definitions(timeline: Any) -> list[Any]:
    """Return model-visible HA tool definitions from the latest run."""
    events, _total_count, _omitted_middle_count = _timeline_events(timeline)
    for event in events:
        if event.get("event") != "messages_prepared":
            continue
        data = _mapping(event.get("data"))
        definitions = data.get("llm_tool_definitions")
        return definitions if isinstance(definitions, list) else []
    return []


def _tool_calls(timeline: Any, *, offset: int, limit: int) -> dict[str, Any]:
    """Return a page of HA LLM API tool call diagnostics."""
    events, _total_count, _omitted_middle_count = _timeline_events(timeline)
    calls = [event for event in events if event.get("phase") == "tool_call"]
    page = calls[offset : offset + limit]
    return {
        "items": page,
        "offset": offset,
        "limit": limit,
        "total_count": len(calls),
        "has_more": offset + len(page) < len(calls),
    }


def _run_errors(summary: Mapping[str, Any]) -> list[Any]:
    """Return compact run error/failure summary values."""
    errors = []
    if error := summary.get("error"):
        errors.append(error)
    if failure := summary.get("failure"):
        errors.append(failure)
    return errors


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
    seen: dict[tuple[str, str, str, str | None], int] = {}

    def add_model(
        provider_subentry: ConfigSubentry,
        subentry_id: str,
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
        failure_key = f"{subentry_id}:{profile_ref}"
        # Several subentries can target the same model/settings pair, so probe
        # each unique runtime capability once during setup.
        if dedupe_key in seen:
            index = seen[dedupe_key]
            probe = models[index]
            models[index] = replace(
                probe,
                failure_keys=(*probe.failure_keys, failure_key),
            )
            return
        seen[dedupe_key] = len(models)
        models.append(
            _ConfiguredModelProbe(
                provider_subentry=provider_subentry,
                issue_profile_id=profile_ref,
                failure_keys=(failure_key,),
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
            add_model(
                provider_subentry, subentry.subentry_id, ref, subentry.data, output_mode
            )
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
) -> dict[str, str]:
    """Probe configured models and surface provider/profile repairs."""
    current_issue_ids: set[str] = set()
    auth_failure_provider_ids: set[str] = set()
    validation_failures: dict[str, str] = {}
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
            validation_failures.update(
                {failure_key: err.reason for failure_key in probe.failure_keys}
            )
            if provider_validation_is_auth_failure(err):
                auth_failure_provider_ids.add(probe.provider_subentry.subentry_id)
                async_create_provider_auth_issue(
                    hass,
                    entry,
                    probe.provider_subentry.subentry_id,
                    probe.provider_subentry.title,
                    err,
                )
            continue
        async_delete_model_validation_issue(
            hass, entry, probe.issue_profile_id, probe.model, repair_settings
        )
    async_delete_stale_model_validation_issues(hass, entry, current_issue_ids)
    current_provider_ids = {
        subentry.subentry_id
        for subentry in entry.subentries.values()
        if subentry.subentry_type == SUBENTRY_TYPE_PROVIDER
    }
    for provider_id in current_provider_ids - auth_failure_provider_ids:
        async_delete_provider_auth_issue(hass, entry, provider_id)
    async_delete_stale_provider_auth_issues(
        hass,
        entry,
        current_provider_ids,
    )
    return validation_failures
