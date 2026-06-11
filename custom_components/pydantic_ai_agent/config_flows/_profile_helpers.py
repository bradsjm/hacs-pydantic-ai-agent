"""Model profile selection and management helpers for config flows."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any
from uuid import uuid4

import voluptuous as vol
from homeassistant.components.todo.const import DOMAIN as TODO_DOMAIN
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import section
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
)
from homeassistant.helpers.typing import VolDictType

from ..const import (
    CONF_DISCOVERED,
    CONF_ENABLED,
    CONF_FALLBACK_MODEL_REFS,
    CONF_MODEL,
    CONF_MODEL_PRICING,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_NAME,
    CONF_PRIMARY_MODEL_REF,
    CONF_TODO_LIST_ENTITY_ID,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
)
from ..model_profiles import (
    configured_model_profile_exists,
    model_profile_display_name,
    model_profile_ref,
    parse_model_profile_ref,
    provider_model_profiles,
    provider_subentries,
)
from ..provider_validation import ProviderValidationError
from ._constants import (
    _CONF_MODEL_PROFILE_ID,
    _MODEL_SETTING_TEMPERATURE,
    _SECTION_ADVANCED_MODEL_SETTINGS,
    _SECTION_MODEL_PRICING,
    _TODO_WORKSPACE_REQUIRED_FEATURES,
)
from ._settings_parsing import _model_settings_from_options
from .helpers import _sorted_select_options

_LOGGER = logging.getLogger(__name__)


def _provider_validation_placeholders(
    err: ProviderValidationError,
) -> dict[str, str]:
    """Return translation placeholders for provider validation errors."""
    placeholders = {"error_message": err.message}
    if err.status_code is not None:
        placeholders["status_code"] = str(err.status_code)
    return placeholders


def _log_provider_validation_failure(
    *, step: str, model_name: str, err: ProviderValidationError
) -> None:
    """Log provider validation failures without request details or credentials."""
    if err.status_code == 429:
        _LOGGER.warning(
            'Provider validation rate limited during %s for model "%s": '
            "reason=%s status_code=%s",
            step,
            model_name,
            err.reason,
            err.status_code,
        )
        return

    _LOGGER.warning(
        'Provider validation failed during %s for model "%s": reason=%s status_code=%s',
        step,
        model_name,
        err.reason,
        err.status_code,
    )


def _referenced_provider_profile_ids(
    entry: ConfigEntry, provider_subentry_id: str
) -> set[str]:
    """Return model profile IDs referenced by conversation or AI task subentries."""
    referenced: set[str] = set()
    for subentry in entry.subentries.values():
        if subentry.subentry_type not in {
            SUBENTRY_TYPE_CONVERSATION,
            SUBENTRY_TYPE_AI_TASK,
        }:
            continue
        for profile_ref in _selected_model_profile_refs(subentry.data):
            try:
                ref_provider_subentry_id, profile_id = parse_model_profile_ref(
                    profile_ref
                )
            except HomeAssistantError:
                continue
            if ref_provider_subentry_id == provider_subentry_id:
                referenced.add(profile_id)
    return referenced


def _normalise_provider_model_profiles(
    existing_profiles: Mapping[str, Any],
    model_names: list[str],
    discovered_model_names: Iterable[str],
    *,
    model_labels: Mapping[str, str] | None = None,
    keep_profile_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return provider-owned profile storage synced to provider model names."""
    model_labels = model_labels or {}
    discovered_set = set(discovered_model_names)
    model_set = set(model_names)
    keep_profile_ids = keep_profile_ids or set()
    existing_by_model: dict[str, tuple[str, dict[str, Any]]] = {}
    kept_profiles: dict[str, dict[str, Any]] = {}
    for profile_id, profile in existing_profiles.items():
        profile_info = _classify_existing_provider_profile(
            profile_id,
            profile,
            model_set=model_set,
            keep_profile_ids=keep_profile_ids,
        )
        if profile_info is None:
            continue
        model_name, existing_profile_id, profile, keep_profile = profile_info
        if keep_profile:
            kept_profiles[profile_id] = profile
            continue
        existing_by_model.setdefault(model_name, (existing_profile_id, profile))

    profiles: dict[str, dict[str, Any]] = dict(kept_profiles)
    for model_name in model_names:
        existing_profile = existing_by_model.get(model_name)
        if existing_profile is None:
            profile_id = uuid4().hex
            profile = _normalised_provider_profile(
                profile={},
                profile_id=profile_id,
                model_name=model_name,
                label=model_labels.get(model_name, model_name),
                discovered=model_name in discovered_set,
            )
        else:
            profile_id, profile = existing_profile
            profile = _normalised_provider_profile(
                profile=profile,
                profile_id=profile_id,
                model_name=model_name,
                label=model_labels.get(model_name, model_name),
                discovered=model_name in discovered_set,
            )
        profiles[profile_id] = profile
    return profiles


def _classify_existing_provider_profile(
    profile_id: object,
    profile: object,
    *,
    model_set: set[str],
    keep_profile_ids: set[str],
) -> tuple[str, str, dict[str, Any], bool] | None:
    """Return normalized existing profile data and whether it should be kept."""
    if not isinstance(profile_id, str) or not isinstance(profile, Mapping):
        return None
    model_name = profile.get(CONF_MODEL)
    if not isinstance(model_name, str) or not model_name.strip():
        return None
    normalized_profile = _normalised_provider_profile(
        profile=profile,
        profile_id=profile_id,
        model_name=model_name,
        label=model_name,
        discovered=bool(profile.get(CONF_DISCOVERED, False)),
    )
    keep_profile = model_name not in model_set and profile_id in keep_profile_ids
    return model_name, profile_id, normalized_profile, keep_profile


def _normalised_provider_profile(
    *,
    profile: Mapping[str, Any],
    profile_id: str,
    model_name: str,
    label: str,
    discovered: bool,
) -> dict[str, Any]:
    """Return one provider model profile in normalized stored form."""
    normalized_profile = dict(profile)
    normalized_profile["id"] = profile_id
    profile_name = str(normalized_profile.get(CONF_NAME) or "").strip()
    if not profile_name or profile_name == model_name:
        profile_name = label
    normalized_profile[CONF_NAME] = profile_name
    normalized_profile[CONF_MODEL] = model_name
    normalized_profile[CONF_ENABLED] = bool(normalized_profile.get(CONF_ENABLED, False))
    normalized_profile[CONF_DISCOVERED] = discovered
    model_settings = normalized_profile.get(CONF_MODEL_SETTINGS)
    if isinstance(model_settings, Mapping):
        normalized_profile[CONF_MODEL_SETTINGS] = _model_settings_from_options(
            normalized_profile
        )
    else:
        normalized_profile.pop(CONF_MODEL_SETTINGS, None)
    return normalized_profile


def _provider_model_profiles_for_discovery_mode(
    existing_profiles: Mapping[str, Any], *, keep_profile_ids: set[str]
) -> dict[str, dict[str, Any]]:
    """Return existing profiles that remain valid before discovery refresh."""
    profiles: dict[str, dict[str, Any]] = {}
    for profile_id, profile in existing_profiles.items():
        if not isinstance(profile_id, str) or not isinstance(profile, Mapping):
            continue
        if (
            not bool(profile.get(CONF_DISCOVERED, False))
            and profile_id not in keep_profile_ids
        ):
            continue
        model_name = profile.get(CONF_MODEL)
        if not isinstance(model_name, str) or not model_name.strip():
            continue
        profile = dict(profile)
        profile["id"] = profile_id
        profile[CONF_MODEL] = model_name
        profile[CONF_ENABLED] = bool(profile.get(CONF_ENABLED, False))
        model_settings = profile.get(CONF_MODEL_SETTINGS)
        if isinstance(model_settings, Mapping):
            profile[CONF_MODEL_SETTINGS] = _model_settings_from_options(profile)
        else:
            profile.pop(CONF_MODEL_SETTINGS, None)
        profiles[profile_id] = profile
    return profiles


def _provider_profile_options(
    data: Mapping[str, Any],
    model_ids: set[str] | None = None,
    *,
    enabled_only: bool = False,
) -> list[SelectOptionDict]:
    """Return provider model profiles as select options."""
    options: list[SelectOptionDict] = []
    profiles = data.get(CONF_MODEL_PROFILES)
    if not isinstance(profiles, Mapping):
        return []
    for profile_id, profile in profiles.items():
        if not isinstance(profile_id, str) or not isinstance(profile, Mapping):
            continue
        model_name = profile.get(CONF_MODEL)
        if not isinstance(model_name, str) or not model_name.strip():
            continue
        enabled = bool(profile.get(CONF_ENABLED, False))
        if enabled_only and not enabled:
            continue
        if model_ids is not None and model_name not in model_ids:
            continue
        label = model_profile_display_name(profile)
        if not enabled:
            label = f"{label} (disabled)"
        options.append(SelectOptionDict(label=label, value=profile_id))
    return _sorted_select_options(options)


def _provider_profile_dependents(entry: ConfigEntry, profile_ref: str) -> list[str]:
    """Return conversation and AI task titles that reference one profile."""
    dependents: list[str] = []
    for subentry in entry.subentries.values():
        if subentry.subentry_type not in {
            SUBENTRY_TYPE_CONVERSATION,
            SUBENTRY_TYPE_AI_TASK,
        }:
            continue
        refs = _selected_model_profile_refs(subentry.data)
        if profile_ref in refs:
            dependents.append(subentry.title)
    return dependents


def _provider_profile_selector_schema(
    data: Mapping[str, Any],
    model_ids: set[str] | None = None,
    *,
    enabled_only: bool = False,
) -> vol.Schema:
    """Return a selector schema for existing provider-owned profiles."""
    return vol.Schema(
        {
            vol.Required(_CONF_MODEL_PROFILE_ID): SelectSelector(
                SelectSelectorConfig(
                    options=_provider_profile_options(
                        data, model_ids, enabled_only=enabled_only
                    ),
                    mode=SelectSelectorMode.DROPDOWN,
                )
            )
        }
    )


def _model_profile_edit_schema(
    profile: Mapping[str, Any],
) -> vol.Schema:
    """Return the provider-owned model profile edit schema."""
    from ._schema_helpers import _model_pricing_schema, _model_settings_schema

    options: dict[str, Any] = {
        CONF_NAME: model_profile_display_name(profile),
        CONF_MODEL_PRICING: profile.get(CONF_MODEL_PRICING, {}),
        CONF_MODEL_SETTINGS: profile.get(CONF_MODEL_SETTINGS, {}),
    }
    schema: VolDictType = {
        vol.Required(CONF_NAME, default=options[CONF_NAME]): TextSelector(
            TextSelectorConfig()
        ),
    }
    if not bool(profile.get(CONF_DISCOVERED, False)):
        schema[vol.Required(CONF_MODEL, default=profile.get(CONF_MODEL, ""))] = (
            TextSelector(TextSelectorConfig())
        )
    schema[
        vol.Optional(
            _MODEL_SETTING_TEMPERATURE,
            description={
                "suggested_value": options[CONF_MODEL_SETTINGS].get(
                    _MODEL_SETTING_TEMPERATURE
                )
                if isinstance(options[CONF_MODEL_SETTINGS], Mapping)
                else None
            },
        )
    ] = NumberSelector(NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=0.1))
    schema[vol.Optional(_SECTION_ADVANCED_MODEL_SETTINGS, default={})] = section(
        _model_settings_schema(options), {"collapsed": True}
    )
    schema[vol.Optional(_SECTION_MODEL_PRICING, default={})] = section(
        _model_pricing_schema(options), {"collapsed": True}
    )
    return vol.Schema(schema)


def _model_profile_select_options(entry: ConfigEntry | None) -> list[SelectOptionDict]:
    """Return enabled workspace model profiles as select options."""
    if entry is None:
        return []
    options: list[SelectOptionDict] = []
    for provider_subentry in provider_subentries(entry):
        for profile_id, profile in provider_model_profiles(provider_subentry).items():
            if not bool(profile.get(CONF_ENABLED, False)):
                continue
            model_name = profile.get(CONF_MODEL)
            if not isinstance(model_name, str) or not model_name.strip():
                continue
            label = model_profile_display_name(profile)
            options.append(
                SelectOptionDict(
                    label=f"{provider_subentry.title} / {label}",
                    value=model_profile_ref(provider_subentry.subentry_id, profile_id),
                )
            )
    return _sorted_select_options(options)


def _normalise_fallback_model_refs(
    raw_refs: object,
) -> list[str]:
    """Return canonical workspace-local fallback refs, preserving order."""
    if isinstance(raw_refs, str) or not isinstance(raw_refs, list):
        return []
    refs: list[str] = []
    for raw_ref in raw_refs:
        if not isinstance(raw_ref, str) or not raw_ref:
            continue
        try:
            provider_subentry_id, profile_id = parse_model_profile_ref(raw_ref)
        except HomeAssistantError:
            continue
        refs.append(model_profile_ref(provider_subentry_id, profile_id))
    return refs


def _fallback_model_profile_select_options(
    hass: HomeAssistant, entry: ConfigEntry | None, selected_refs: object = None
) -> list[SelectOptionDict]:
    """Return workspace-local fallback profile options."""
    del hass
    options = _model_profile_select_options(entry)
    configured_refs = {str(option["value"]) for option in options if "value" in option}
    for ref in _normalise_fallback_model_refs(selected_refs):
        if ref not in configured_refs:
            options.append(SelectOptionDict(label=f"Unavailable / {ref}", value=ref))
    return options


def _selected_model_profile_refs(data: Mapping[str, Any]) -> list[str]:
    """Return selected primary plus ordered fallback profile refs."""
    primary_ref = data.get(CONF_PRIMARY_MODEL_REF)
    if not isinstance(primary_ref, str) or not primary_ref:
        return []
    fallback_refs = data.get(CONF_FALLBACK_MODEL_REFS, [])
    if isinstance(fallback_refs, str) or not isinstance(fallback_refs, list):
        fallback_refs = []
    return [primary_ref, *[item for item in fallback_refs if isinstance(item, str)]]


def _selected_model_profile_error(
    hass: HomeAssistant, entry: ConfigEntry, data: Mapping[str, Any]
) -> str | None:
    """Return a form error for missing or invalid model profile selections."""
    del hass
    primary_ref = data.get(CONF_PRIMARY_MODEL_REF)
    if not isinstance(primary_ref, str) or not primary_ref:
        return "model_profile_required"
    if not configured_model_profile_exists(entry, primary_ref):
        return "model_profile_not_found"
    fallback_refs = _normalise_fallback_model_refs(
        data.get(CONF_FALLBACK_MODEL_REFS, [])
    )
    if primary_ref in fallback_refs:
        return "primary_model_in_fallbacks"
    if len(fallback_refs) != len(set(fallback_refs)):
        return "duplicate_fallback_model"
    for profile_ref in fallback_refs:
        if not configured_model_profile_exists(entry, profile_ref):
            return "model_profile_not_found"
    return None


def _selected_todo_workspace_error(
    hass: HomeAssistant, data: Mapping[str, Any]
) -> str | None:
    """Return a form error for an invalid todo workspace entity."""
    entity_id = data.get(CONF_TODO_LIST_ENTITY_ID)
    if not entity_id:
        return None
    if not isinstance(entity_id, str) or not entity_id.startswith(f"{TODO_DOMAIN}."):
        return "todo_list_not_found"
    state = hass.states.get(entity_id)
    if state is None:
        return "todo_list_not_found"
    supported_features = state.attributes.get("supported_features", 0)
    if not isinstance(supported_features, int):
        return "todo_list_unsupported"
    if (
        supported_features & _TODO_WORKSPACE_REQUIRED_FEATURES
        != _TODO_WORKSPACE_REQUIRED_FEATURES
    ):
        return "todo_list_unsupported"
    return None
