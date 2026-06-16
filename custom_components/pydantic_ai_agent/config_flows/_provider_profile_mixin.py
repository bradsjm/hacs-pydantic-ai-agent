"""Model profile editing mixin for provider config flows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry, ConfigSubentry, SubentryFlowResult
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from ..const import (
    CONF_MODEL,
    CONF_MODEL_PRICING,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_NAME,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_METADATA,
    CONF_PROVIDER_MODE,
    CONF_TEMPLATED_EXTRA_BODY,
)
from ..models.openai_compatible_profile import (
    PersistedOpenAICompatibleProfile,
    is_openai_compatible_provider_mode,
)
from ..models.templated_extra_body import merge_extra_body, render_templated_extra_body
from ._constants import (
    _ADVANCED_MODEL_SETTING_KEYS,
    _CONF_MODEL_PROFILE_ID,
    _MAIN_MODEL_SETTING_KEYS,
    _MODEL_PRICING_CACHE_READ,
    _MODEL_PRICING_INPUT,
    _MODEL_PRICING_OUTPUT,
    _MODEL_SETTING_PARALLEL_TOOL_CALLS,
    _SECTION_ADVANCED_MODEL_SETTINGS,
    _SECTION_MODEL_PRICING,
    _SECTION_OPENAI_COMPATIBLE_CAPABILITIES,
)
from ._profile_helpers import (
    _model_profile_edit_schema,
    _provider_profile_options,
    _provider_profile_selector_schema,
    model_profile_description_placeholders,
)
from ._settings_parsing import (
    _merge_model_pricing,
    _merge_model_settings,
    _model_pricing_from_options,
    _model_profile_data_from_user_input,
    _model_settings_from_options,
    _parse_model_pricing,
    _parse_model_settings,
    _store_model_pricing,
    _store_model_settings,
)
from .helpers import _flatten_section_data
from .provider_wizard.catalog_cache import catalog_manager
from .provider_wizard.const import CONF_CATALOG_PROVIDER_ID
from .provider_wizard.models_dev import CatalogLoadError
from .provider_wizard.types import CatalogModelOption

_CONF_MODEL_PROFILE_ID = _CONF_MODEL_PROFILE_ID


class ProviderProfileMixin:
    """Mixin for provider model profile editing steps.

    Must be used with a ConfigSubentryFlow subclass.
    """

    # ---- parent class delegations (satisfied by ConfigSubentryFlow at runtime) ----
    # These forward to super() to satisfy pyright while preserving real behavior.
    hass: HomeAssistant

    def _get_entry(self) -> ConfigEntry:
        return super()._get_entry()  # type: ignore[misc]

    def _get_reconfigure_subentry(self) -> ConfigSubentry:
        return super()._get_reconfigure_subentry()  # type: ignore[misc]

    def async_show_form(self, *args: object, **kwargs: object) -> SubentryFlowResult:
        return super().async_show_form(*args, **kwargs)  # type: ignore[misc]

    def async_abort(self, *args: object, **kwargs: object) -> SubentryFlowResult:
        return super().async_abort(*args, **kwargs)  # type: ignore[misc]

    def async_update_and_abort(
        self, *args: object, **kwargs: object
    ) -> SubentryFlowResult:
        return super().async_update_and_abort(*args, **kwargs)  # type: ignore[misc]

    # ---- mixin attributes ----
    _profile_flow_data: dict[str, Any]
    _selected_profile_id: str | None
    _pending_profile_data: dict[str, Any]
    _pending_model_settings: dict[str, Any]
    _pending_model_pricing: dict[str, float]
    _pending_profile_error: tuple[str, dict[str, str]] | None
    _profile_refresh_error: str | None
    _profile_models: tuple[CatalogModelOption, ...]

    async def async_step_customize_model_profile(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Choose a provider-owned profile to edit."""
        del user_input
        self._profile_flow_data = dict(self._get_reconfigure_subentry().data)
        self._profile_refresh_error = None
        self._selected_profile_id = None
        self._profile_models = await self._async_profile_catalog_models(
            self._profile_flow_data
        )
        return await self.async_step_pick_model_profile()

    async def async_step_pick_model_profile(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Pick one existing provider-owned model profile."""
        data = self._current_profile_flow_data()
        if user_input is None:
            if not _provider_profile_options(data, enabled_only=True):
                return self.async_show_form(
                    step_id="pick_model_profile",
                    data_schema=vol.Schema({}),
                    errors={
                        "base": getattr(self, "_profile_refresh_error", None)
                        or "model_list_unavailable"
                    },
                )
            errors = {}
            if profile_refresh_error := getattr(self, "_profile_refresh_error", None):
                errors["base"] = profile_refresh_error
            return self.async_show_form(
                step_id="pick_model_profile",
                data_schema=_provider_profile_selector_schema(data, enabled_only=True),
                errors=errors,
            )
        self._selected_profile_id = str(user_input[_CONF_MODEL_PROFILE_ID])
        return await self.async_step_edit_model_profile()

    async def async_step_edit_model_profile(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Edit one provider-owned model profile."""
        profile_id = self._selected_profile_id
        if profile_id is None:
            return await self.async_step_pick_model_profile()
        profiles = self._current_profile_flow_data().get(CONF_MODEL_PROFILES, {})
        profile = profiles.get(profile_id) if isinstance(profiles, Mapping) else None
        if profile is None:
            return self.async_abort(reason="model_profile_not_found")
        if user_input is not None:
            flat_user_input = _flatten_section_data(
                user_input,
                (
                    _SECTION_ADVANCED_MODEL_SETTINGS,
                    _SECTION_MODEL_PRICING,
                    _SECTION_OPENAI_COMPATIBLE_CAPABILITIES,
                ),
            )
            parsed_settings, errors, cleared = _parse_model_settings(
                self.hass,
                flat_user_input,
                _MAIN_MODEL_SETTING_KEYS
                | _ADVANCED_MODEL_SETTING_KEYS
                | {_MODEL_SETTING_PARALLEL_TOOL_CALLS},
            )
            pricing_field_keys = {
                _MODEL_PRICING_INPUT,
                _MODEL_PRICING_OUTPUT,
                _MODEL_PRICING_CACHE_READ,
            }
            parsed_pricing, pricing_errors, pricing_cleared = _parse_model_pricing(
                flat_user_input, pricing_field_keys
            )
            pricing_submitted = any(
                key in flat_user_input for key in pricing_field_keys
            )
            errors.update(pricing_errors)
            data = _model_profile_data_from_user_input(flat_user_input)
            existing_settings = _model_settings_from_options(
                {CONF_MODEL_SETTINGS: profile.get(CONF_MODEL_SETTINGS, {})}
            )
            model_settings = _merge_model_settings(
                existing_settings, parsed_settings, cleared
            )
            try:
                merge_extra_body(
                    self._current_profile_flow_data().get(CONF_PROVIDER_EXTRA_BODY),
                    render_templated_extra_body(
                        self.hass, model_settings.get(CONF_TEMPLATED_EXTRA_BODY)
                    ),
                )
            except HomeAssistantError:
                errors[CONF_TEMPLATED_EXTRA_BODY] = "templated_extra_body_path_conflict"
            existing_pricing = _model_pricing_from_options(
                {CONF_MODEL_PRICING: profile.get(CONF_MODEL_PRICING, {})}
            )
            model_pricing = _merge_model_pricing(
                existing_pricing, parsed_pricing, pricing_cleared
            )
            if errors:
                return self.async_show_form(
                    step_id="edit_model_profile",
                    data_schema=_model_profile_edit_schema(
                        profile
                        | data
                        | {
                            CONF_MODEL_SETTINGS: model_settings,
                            CONF_MODEL_PRICING: model_pricing,
                        },
                        str(
                            self._current_profile_flow_data().get(
                                CONF_PROVIDER_MODE, ""
                            )
                        ),
                    ),
                    errors=errors,
                    description_placeholders=self._model_profile_description_placeholders(
                        profile,
                    ),
                )
            pending_profile = dict(profile) | data
            if is_openai_compatible_provider_mode(
                str(self._current_profile_flow_data().get(CONF_PROVIDER_MODE, ""))
            ):
                try:
                    PersistedOpenAICompatibleProfile.from_mapping(pending_profile)
                except KeyError, ValueError:
                    return self.async_show_form(
                        step_id="edit_model_profile",
                        data_schema=_model_profile_edit_schema(
                            pending_profile
                            | {
                                CONF_MODEL_SETTINGS: model_settings,
                                CONF_MODEL_PRICING: model_pricing,
                            },
                            str(
                                self._current_profile_flow_data().get(
                                    CONF_PROVIDER_MODE, ""
                                )
                            ),
                        ),
                        errors={"base": "openai_compatible_profile_incomplete"},
                        description_placeholders=self._model_profile_description_placeholders(
                            pending_profile,
                        ),
                    )
            self._pending_profile_data = pending_profile
            self._pending_model_settings = dict(model_settings)
            self._pending_model_pricing = dict(model_pricing)
            self._pending_profile_error = None
            _store_model_settings(
                self._pending_profile_data, self._pending_model_settings
            )
            if CONF_MODEL_PRICING in profile or pricing_submitted:
                _store_model_pricing(
                    self._pending_profile_data, self._pending_model_pricing
                )
            return await self.async_step_model_profile_finish()
        return self.async_show_form(
            step_id="edit_model_profile",
            data_schema=_model_profile_edit_schema(
                profile,
                str(self._current_profile_flow_data().get(CONF_PROVIDER_MODE, "")),
            ),
            description_placeholders=self._model_profile_description_placeholders(
                profile,
            ),
        )

    async def async_step_model_profile_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Persist a provider-owned model profile edit or replay validation errors."""
        del user_input
        provider_subentry = self._get_reconfigure_subentry()
        profile_id = self._selected_profile_id
        if profile_id is None:
            return await self.async_step_pick_model_profile()
        if self._pending_profile_error is not None:
            reason, placeholders = self._pending_profile_error
            return self.async_show_form(
                step_id="edit_model_profile",
                data_schema=_model_profile_edit_schema(
                    self._pending_profile_data,
                    str(self._current_profile_flow_data().get(CONF_PROVIDER_MODE, "")),
                ),
                errors={"base": reason},
                description_placeholders=placeholders,
            )
        data = self._current_profile_flow_data()
        profiles = dict(data.get(CONF_MODEL_PROFILES, {}))
        profile = dict(self._pending_profile_data)
        profile["id"] = profile_id
        profiles[profile_id] = profile
        data[CONF_MODEL_PROFILES] = profiles
        return self.async_update_and_abort(
            self._get_entry(), provider_subentry, title=data[CONF_NAME], data=data
        )

    def _current_profile_flow_data(self) -> dict[str, Any]:
        """Return transient provider data for the active profile edit flow."""
        if profile_flow_data := getattr(self, "_profile_flow_data", None):
            return profile_flow_data
        return dict(self._get_reconfigure_subentry().data)

    def _model_profile_description_placeholders(
        self, profile: Mapping[str, Any]
    ) -> dict[str, str]:
        """Return edit-model-profile description placeholders."""
        model_name = profile.get(CONF_MODEL)
        model = None
        if isinstance(model_name, str):
            model = next(
                (
                    candidate
                    for candidate in getattr(self, "_profile_models", ())
                    if candidate.id == model_name
                ),
                None,
            )
        return model_profile_description_placeholders(profile, model)

    async def _async_profile_catalog_models(
        self, data: Mapping[str, Any]
    ) -> tuple[CatalogModelOption, ...]:
        """Return catalog models for profile-edit descriptions when available."""
        metadata = data.get(CONF_PROVIDER_METADATA)
        catalog_provider_id = (
            metadata.get(CONF_CATALOG_PROVIDER_ID)
            if isinstance(metadata, Mapping)
            else None
        )
        if not isinstance(catalog_provider_id, str):
            return ()
        try:
            catalog = await catalog_manager(self.hass).async_get_catalog()
        except CatalogLoadError:
            return ()
        provider = catalog.providers.get(catalog_provider_id)
        if provider is None:
            return ()
        return catalog.models_for_provider(provider.id)
