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
    CONF_OUTPUT_MODE,
    CONF_PROVIDER_MODE,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
)
from .repairs import (
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


@dataclass(frozen=True, kw_only=True)
class PydanticAIAgentRuntimeData:
    """Provider connection data shared by subentry-backed entities."""

    provider_mode: str
    name: str
    api_key: str
    base_url: str | None


type PydanticAIAgentConfigEntry = ConfigEntry[PydanticAIAgentRuntimeData]


async def async_setup_entry(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> bool:
    """Validate configured subentries, then set up entity platforms."""
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
) -> list[tuple[str, dict[str, Any], str | None]]:
    """Return unique model probes needed before the entry can load."""
    models: list[tuple[str, dict[str, Any], str | None]] = []
    seen: set[tuple[str, str, str | None]] = set()
    for subentry in entry.subentries.values():
        if subentry.subentry_type not in (
            SUBENTRY_TYPE_CONVERSATION,
            SUBENTRY_TYPE_AI_TASK,
        ):
            continue
        if not (model := subentry.data.get(CONF_MODEL)):
            continue
        if subentry.subentry_type == SUBENTRY_TYPE_CONVERSATION:
            settings = subentry.data.get(CONF_MODEL_SETTINGS)
            model_settings = dict(settings) if isinstance(settings, Mapping) else {}
            output_mode = None
        else:
            model_settings = {}
            output_mode = structured_output_mode(subentry.data.get(CONF_OUTPUT_MODE))
        dedupe_key = (
            model,
            _normalise_model_settings(model_settings),
            output_mode,
        )
        # Several subentries can target the same model/settings pair, so probe
        # each unique runtime capability once during setup.
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        models.append((model, model_settings, output_mode))
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
        model,
        model_settings,
        output_mode,
    ) in _configured_subentry_models(entry):
        repair_settings = _repair_issue_model_settings(
            model_settings, output_mode
        )
        current_issue_ids.add(model_validation_issue_id(entry, model, repair_settings))
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
                    hass, entry, model, repair_settings, err
                )
                continue
            raise ConfigEntryNotReady(err.message) from err
        async_delete_model_validation_issue(hass, entry, model, repair_settings)
    async_delete_stale_model_validation_issues(hass, entry, current_issue_ids)
