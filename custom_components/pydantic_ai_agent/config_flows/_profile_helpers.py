"""Model profile selection and management helpers for config flows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.data_entry_flow import section
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.selector import (
    BooleanSelector,
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
    CONF_MODEL,
    CONF_MODEL_PRICING,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_PROVIDER_MODE,
    CONF_STRUCTURED_OUTPUT_SUPPORT,
    CONF_SUPPORTS_TOOLS,
    CONF_THINKING_SUPPORT,
)
from ..models.model_profiles import (
    enabled_model_profile_refs,
    model_profile_display_name,
    parse_model_profile_ref,
    provider_model_profiles,
    provider_profile_thinking_support,
)
from ..models.openai_compatible_profile import (
    default_openai_compatible_profile_data,
    is_openai_compatible_provider_mode,
)
from ._constants import (
    _CONF_MODEL_PROFILE_ID,
    _MODEL_SETTING_PARALLEL_TOOL_CALLS,
    _MODEL_SETTING_TEMPERATURE,
    _SECTION_ADVANCED_MODEL_SETTINGS,
    _SECTION_MODEL_PRICING,
    _SECTION_OPENAI_COMPATIBLE_CAPABILITIES,
)
from ._profile_validation_logging import (  # noqa: F401
    _log_provider_validation_failure,
    _provider_validation_placeholders,
    _selected_todo_workspace_error,
)
from ._settings_parsing import _model_settings_from_options
from .helpers import _section_schema_key, _sorted_select_options
from .provider_wizard.types import CatalogModelOption


@dataclass(frozen=True, kw_only=True)
class RunSettingsVisibility:
    """Resolved run-setting visibility for one config-flow form."""

    supports_thinking: bool = False
    can_disable_thinking: bool = False


def _run_settings_visibility(
    entry: ConfigEntry | None,
    selected_profile_refs: Iterable[object] = (),
) -> RunSettingsVisibility:
    """Return run-setting visibility from effective selected profile capabilities."""
    if entry is None:
        return RunSettingsVisibility()
    refs = [ref for ref in selected_profile_refs if isinstance(ref, str) and ref]
    selected_refs = bool(refs)
    if not refs:
        refs = enabled_model_profile_refs(entry)
    if not refs:
        return RunSettingsVisibility()
    supports_thinking = False
    can_disable_thinking = False
    for raw_ref in refs:
        try:
            provider_subentry_id, profile_id = parse_model_profile_ref(raw_ref)
        except HomeAssistantError:
            continue
        provider_subentry = entry.subentries.get(provider_subentry_id)
        if provider_subentry is None:
            continue
        profile = provider_model_profiles(provider_subentry).get(profile_id)
        provider_mode = provider_subentry.data.get(CONF_PROVIDER_MODE)
        if not isinstance(profile, Mapping) or not isinstance(provider_mode, str):
            continue
        try:
            supported, can_disable = provider_profile_thinking_support(
                provider_mode, profile
            )
        except KeyError, ValueError:
            continue
        supports_thinking = supports_thinking or supported
        can_disable_thinking = can_disable_thinking or can_disable
    if not selected_refs and not supports_thinking:
        return RunSettingsVisibility()
    return RunSettingsVisibility(
        supports_thinking=supports_thinking,
        can_disable_thinking=can_disable_thinking,
    )


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
    provider_mode: str,
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
    advanced_model_settings_schema = _model_settings_schema(options)
    schema[
        _section_schema_key(
            _SECTION_ADVANCED_MODEL_SETTINGS, advanced_model_settings_schema.schema
        )
    ] = section(advanced_model_settings_schema, {"collapsed": True})
    if is_openai_compatible_provider_mode(provider_mode):
        capability_defaults = default_openai_compatible_profile_data() | {
            key: profile[key]
            for key in default_openai_compatible_profile_data()
            if key in profile
        }
        capability_schema = vol.Schema(
            {
                vol.Required(
                    CONF_THINKING_SUPPORT,
                    default=capability_defaults[CONF_THINKING_SUPPORT],
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(label="None", value="none"),
                            SelectOptionDict(label="Supported", value="supported"),
                            SelectOptionDict(label="Always", value="always"),
                        ],
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Required(
                    CONF_STRUCTURED_OUTPUT_SUPPORT,
                    default=capability_defaults[CONF_STRUCTURED_OUTPUT_SUPPORT],
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(label="None", value="none"),
                            SelectOptionDict(label="JSON Object", value="json_object"),
                            SelectOptionDict(label="JSON Schema", value="json_schema"),
                        ],
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Required(
                    CONF_SUPPORTS_TOOLS,
                    default=capability_defaults[CONF_SUPPORTS_TOOLS],
                ): BooleanSelector(),
                vol.Required(
                    _MODEL_SETTING_PARALLEL_TOOL_CALLS,
                    default=(
                        bool(
                            options[CONF_MODEL_SETTINGS].get(
                                _MODEL_SETTING_PARALLEL_TOOL_CALLS, False
                            )
                        )
                        if isinstance(options[CONF_MODEL_SETTINGS], Mapping)
                        else False
                    ),
                ): BooleanSelector(),
            }
        )
        schema[
            _section_schema_key(
                _SECTION_OPENAI_COMPATIBLE_CAPABILITIES,
                capability_schema.schema,
            )
        ] = section(capability_schema, {"collapsed": True})
    model_pricing_schema = _model_pricing_schema(options)
    schema[_section_schema_key(_SECTION_MODEL_PRICING, model_pricing_schema.schema)] = (
        section(model_pricing_schema, {"collapsed": True})
    )
    return vol.Schema(schema)


def model_profile_description_placeholders(
    profile: Mapping[str, Any], model: CatalogModelOption | None
) -> dict[str, str]:
    """Return description placeholders for the edit model profile form."""
    return {"catalog_details": _model_profile_catalog_details(profile, model)}


def _model_profile_catalog_details(
    profile: Mapping[str, Any], model: CatalogModelOption | None
) -> str:
    """Return human-readable catalog details for one model profile."""
    del profile
    if model is None:
        return "Catalog details unavailable for this model."
    details: list[str] = []
    if model.family:
        details.append(f"Family: {model.family}")
    capabilities = _catalog_capabilities(model)
    if capabilities:
        details.append(f"Capabilities: {', '.join(capabilities)}")
    inputs = " ".join(_input_modality_labels(model))
    if inputs:
        details.append(f"Input: {inputs}")
    limits = _catalog_limits(model)
    if limits:
        details.append(f"Limits: {', '.join(limits)}")
    if model.status:
        details.append(f"Status: {model.status}")
    pricing = _catalog_pricing(model)
    if pricing:
        details.append(f"Pricing: {', '.join(pricing)}")
    if details:
        return "\n".join(details)
    return "Catalog details unavailable for this model."


def _catalog_capabilities(model: CatalogModelOption) -> list[str]:
    """Return human-readable catalog capability labels."""
    capabilities: list[str] = []
    if model.reasoning:
        capabilities.append("reasoning")
    if model.supports_tools:
        capabilities.append("tools")
    if model.structured_output:
        capabilities.append("structured output")
    return capabilities


def _catalog_limits(model: CatalogModelOption) -> list[str]:
    """Return human-readable catalog limit labels."""
    limits: list[str] = []
    if model.context_limit:
        limits.append(f"{model.context_limit:,} ctx")
    if model.output_limit:
        limits.append(f"{model.output_limit:,} output")
    return limits


def _catalog_pricing(model: CatalogModelOption) -> list[str]:
    """Return human-readable catalog pricing labels."""
    pricing: list[str] = []
    if model.input_price is not None:
        pricing.append(f"input ${model.input_price:g}/1M")
    if model.output_price is not None:
        pricing.append(f"output ${model.output_price:g}/1M")
    if model.cache_read_price is not None:
        pricing.append(f"cache read ${model.cache_read_price:g}/1M")
    return pricing


def _input_modality_labels(model: CatalogModelOption) -> tuple[str, ...]:
    """Return glyph and name labels for one model's input modalities."""
    labels = {
        "text": "⊤ text",  # noqa: RUF001
        "image": "▣ image",
        "video": "▶ video",
        "audio": "♪ audio",
        "pdf": "▤ pdf",
    }
    return tuple(
        labels[modality] for modality in model.input_modalities if modality in labels
    )
