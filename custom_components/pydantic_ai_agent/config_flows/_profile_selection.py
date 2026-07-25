"""Model profile selection helpers for config flows."""

from collections.abc import Mapping
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.selector import SelectOptionDict

from ..const import (
    CONF_ENABLED,
    CONF_FALLBACK_MODEL_REFS,
    CONF_MODEL,
    CONF_PRIMARY_MODEL_REF,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
)
from ..models.model_profiles import (
    configured_model_profile_exists,
    model_profile_display_name,
    model_profile_ref,
    parse_model_profile_ref,
    provider_model_profiles,
    provider_subentries,
)
from .helpers import _sorted_select_options


def _referenced_provider_profile_ids(entry: ConfigEntry, provider_subentry_id: str) -> set[str]:
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
                ref_provider_subentry_id, profile_id = parse_model_profile_ref(profile_ref)
            except HomeAssistantError:
                continue
            if ref_provider_subentry_id == provider_subentry_id:
                referenced.add(profile_id)
    return referenced


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


def _normalise_fallback_model_refs(raw_refs: object) -> list[str]:
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


def _deduplicate_fallback_model_refs(raw_refs: object) -> list[str]:
    """Return canonical fallback refs for selector defaults, preserving order."""
    return list(dict.fromkeys(_normalise_fallback_model_refs(raw_refs)))


def _fallback_model_profile_select_options(
    hass: HomeAssistant,
    entry: ConfigEntry | None,
    selected_refs: object = None,
    primary_ref: str | None = None,
) -> list[SelectOptionDict]:
    """Return workspace-local fallback profile options."""
    del hass
    options = [option for option in _model_profile_select_options(entry) if option.get("value") != primary_ref]
    configured_refs = {str(option["value"]) for option in options if "value" in option}
    for ref in _normalise_fallback_model_refs(selected_refs):
        if ref != primary_ref and ref not in configured_refs:
            options.append(SelectOptionDict(label=f"Unavailable / {ref}", value=ref))
            configured_refs.add(ref)
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


def _selected_model_profile_error(hass: HomeAssistant, entry: ConfigEntry, data: Mapping[str, Any]) -> str | None:
    """Return a form error for missing or invalid model profile selections."""
    del hass
    primary_ref = data.get(CONF_PRIMARY_MODEL_REF)
    if not isinstance(primary_ref, str) or not primary_ref:
        return "model_profile_required"
    if not configured_model_profile_exists(entry, primary_ref):
        return "model_profile_not_found"
    fallback_refs = _normalise_fallback_model_refs(data.get(CONF_FALLBACK_MODEL_REFS, []))
    if primary_ref in fallback_refs:
        return "primary_model_in_fallbacks"
    if len(fallback_refs) != len(set(fallback_refs)):
        return "duplicate_fallback_model"
    for profile_ref in fallback_refs:
        if not configured_model_profile_exists(entry, profile_ref):
            return "model_profile_not_found"
    return None
