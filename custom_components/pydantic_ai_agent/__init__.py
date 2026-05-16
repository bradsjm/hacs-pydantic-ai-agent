"""Pydantic AI Agent integration."""

from collections.abc import Mapping
from dataclasses import dataclass
import json
import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, CONF_NAME, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import (
    ConfigEntryAuthFailed,
    ConfigEntryNotReady,
)

from .config_flow import ProviderValidationError, async_probe_model
from .const import (
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_MODEL_SETTINGS,
    CONF_PROVIDER_MODE,
    SUBENTRY_TYPE_CONVERSATION,
)
from .repairs import (
    async_create_model_validation_issue,
    async_delete_model_validation_issue,
    async_delete_stale_model_validation_issues,
    model_validation_issue_id,
)

_LOGGER = logging.getLogger(__name__)

_AUTH_FAILURE_REASONS = {"invalid_auth"}
_RECONFIGURABLE_MODEL_FAILURE_REASONS = {
    "invalid_model",
    "invalid_provider_config",
    "model_does_not_support_streaming",
    "permission_denied",
}

PLATFORMS: tuple[Platform, ...] = (Platform.CONVERSATION,)


@dataclass(frozen=True, kw_only=True)
class PydanticAIAgentRuntimeData:
    """Runtime data for one provider/service config entry."""

    provider_mode: str
    name: str
    api_key: str
    base_url: str | None


type PydanticAIAgentConfigEntry = ConfigEntry[PydanticAIAgentRuntimeData]


async def async_setup_entry(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> bool:
    """Set up Pydantic AI Agent from a config entry."""
    await _async_validate_configured_models(hass, entry)

    entry.runtime_data = PydanticAIAgentRuntimeData(
        provider_mode=entry.data[CONF_PROVIDER_MODE],
        name=entry.data[CONF_NAME],
        api_key=entry.data[CONF_API_KEY],
        base_url=entry.data.get(CONF_BASE_URL),
    )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_update_entry))
    return True


async def async_unload_entry(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_update_entry(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> None:
    """Reload the entry after config entry or subentry updates."""
    await hass.config_entries.async_reload(entry.entry_id)


def _normalise_model_settings(settings: Mapping[str, Any]) -> str:
    """Return a stable representation of model settings for de-duplication."""
    return json.dumps(settings, sort_keys=True, separators=(",", ":"))


def _configured_subentry_models(
    entry: PydanticAIAgentConfigEntry,
) -> list[tuple[str, dict[str, Any]]]:
    """Return configured subentry models and settings, deduped in storage order."""
    models: list[tuple[str, dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    for subentry in entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_CONVERSATION:
            continue
        if not (model := subentry.data.get(CONF_MODEL)):
            continue
        settings = subentry.data.get(CONF_MODEL_SETTINGS)
        model_settings = dict(settings) if isinstance(settings, Mapping) else {}
        dedupe_key = (model, _normalise_model_settings(model_settings))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        models.append((model, model_settings))
    return models


async def _async_validate_configured_models(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> None:
    """Validate configured models before marking an entry loaded."""
    current_issue_ids: set[str] = set()
    for model, model_settings in _configured_subentry_models(entry):
        current_issue_ids.add(model_validation_issue_id(entry, model, model_settings))
        try:
            await async_probe_model(hass, entry.data, model, model_settings)
        except ProviderValidationError as err:
            _LOGGER.warning(
                'Provider validation failed during setup for model "%s": '
                "reason=%s status_code=%s",
                model,
                err.reason,
                err.status_code,
            )
            if err.reason in _AUTH_FAILURE_REASONS:
                raise ConfigEntryAuthFailed(err.message) from err
            if err.reason in _RECONFIGURABLE_MODEL_FAILURE_REASONS:
                async_create_model_validation_issue(
                    hass, entry, model, model_settings, err
                )
                continue
            raise ConfigEntryNotReady(err.message) from err
        async_delete_model_validation_issue(hass, entry, model, model_settings)
    async_delete_stale_model_validation_issues(hass, entry, current_issue_ids)
