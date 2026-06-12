"""Config subentry flow handlers for Pydantic AI Agent."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from homeassistant.config_entries import (
    SOURCE_USER,
    ConfigEntryState,
    ConfigSubentry,
    ConfigSubentryFlow,
    SubentryFlowResult,
)

from ..const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_CUSTOM_MODEL_NAMES,
    CONF_DISCOVERED_MODELS,
    CONF_DISCOVERED_MODELS_AT,
    CONF_DISCOVERED_MODELS_CACHE_KEY,
    CONF_MODEL_PROFILES,
    CONF_NAME,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_METADATA,
    CONF_PROVIDER_MODE,
)
from ..generated_titles import DEFAULT_SERVICE_TITLE_SUFFIX, generated_default_title
from ..provider_validation import ProviderValidationError
from ._key_value_rows import _format_key_value_json_rows
from ._profile_helpers import (
    _provider_validation_placeholders,
)
from ._provider_data import (
    _cached_provider_model_names,
    _clear_provider_model_cache,
    _format_custom_model_names,
    _format_http_headers,
    _normalise_provider_data,
    _provider_custom_model_names,
    _provider_data_matches,
    _provider_schema,
    _validate_provider_data,
)
from ._provider_flow_helpers import _catalog_provider_metadata_still_valid
from ._provider_model_mixin import ProviderModelManagementMixin
from ._provider_profile_mixin import ProviderProfileMixin
from ._provider_wizard_mixin import ProviderWizardMixin
from .helpers import _flatten_section_data
from .provider_wizard.const import CONF_SELECTED_MODEL_IDS
from .provider_wizard.filters import ModelFilterOptions
from .provider_wizard.flow import selected_models_by_id
from .provider_wizard.schemas import (
    SECTION_ADVANCED_MODELS,
    model_selection_schema,
)
from .provider_wizard.types import (
    CatalogModelOption,
    CatalogProviderOption,
    CompactCatalog,
)

_LOGGER = logging.getLogger(__name__)


class ProviderSubentryFlowHandler(
    ProviderWizardMixin,
    ProviderModelManagementMixin,
    ProviderProfileMixin,
    ConfigSubentryFlow,
):
    """Flow for managing workspace-owned provider subentries."""

    _options: dict[str, Any]
    _pending_data: dict[str, Any]
    _pending_error: tuple[str, str, dict[str, str]] | None
    _pending_storage_data: dict[str, Any]
    _pending_step_id: str
    _profile_flow_data: dict[str, Any]
    _profile_filter_provider: CatalogProviderOption | None
    _profile_filters: ModelFilterOptions
    _profile_models: tuple[CatalogModelOption, ...]
    _profile_refresh_error: str | None
    _selected_profile_id: str | None
    _pending_profile_data: dict[str, Any]
    _pending_profile_error: tuple[str, dict[str, str]] | None
    _wizard_catalog: CompactCatalog | None
    _wizard_catalog_error: str | None
    _wizard_connection_data: dict[str, Any]
    _wizard_connection_options: dict[str, Any]
    _wizard_driver: str | None
    _wizard_filters: ModelFilterOptions
    _wizard_models: tuple[CatalogModelOption, ...]
    _wizard_provider: CatalogProviderOption | None
    _wizard_selected_models: tuple[CatalogModelOption, ...]

    @property
    def _is_new(self) -> bool:
        """Return if this flow creates a new subentry."""
        return self.source == SOURCE_USER

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add a provider subentry."""
        self._options = {
            CONF_NAME: generated_default_title(
                DEFAULT_SERVICE_TITLE_SUFFIX,
                (subentry.title for subentry in self._provider_subentries()),
            )
        }
        self._pending_error = None
        self._pending_storage_data = {}
        self._pending_step_id = "init"
        self._selected_profile_id = None
        self._profile_flow_data = {}
        self._profile_filter_provider = None
        self._profile_filters = ModelFilterOptions()
        self._profile_models = ()
        self._profile_refresh_error = None
        self._wizard_catalog = None
        self._wizard_catalog_error = None
        self._wizard_connection_data = {}
        self._wizard_connection_options = {}
        self._wizard_driver = None
        self._wizard_filters = ModelFilterOptions()
        self._wizard_models = ()
        self._wizard_provider = None
        self._wizard_selected_models = ()
        return await self.async_step_pick_provider(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure a provider subentry."""
        self._options = self._provider_form_options(self._get_reconfigure_subentry())
        self._pending_error = None
        self._pending_storage_data = {}
        self._pending_step_id = "edit_connection"
        self._selected_profile_id = None
        self._profile_flow_data = {}
        self._profile_refresh_error = None
        return await self.async_step_reconfigure_menu()

    def _provider_form_options(self, subentry: ConfigSubentry) -> dict[str, Any]:
        """Return provider data expanded with form-only model-selection fields."""
        options = dict(subentry.data)
        options[CONF_CUSTOM_MODEL_NAMES] = _format_custom_model_names(options)
        options[CONF_PROVIDER_HEADERS] = _format_http_headers(
            options.get(CONF_PROVIDER_HEADERS)
        )
        options[CONF_PROVIDER_EXTRA_BODY] = _format_key_value_json_rows(
            options.get(CONF_PROVIDER_EXTRA_BODY)
        )
        return options

    def _provider_subentries(self) -> list[ConfigSubentry]:
        """Return provider subentries from the current entry."""
        from ..model_profiles import provider_subentries

        return provider_subentries(self._get_entry())

    def _provider_already_configured(self, data: Mapping[str, Any]) -> bool:
        """Return if another provider subentry already uses this connection."""
        current_subentry_id = (
            None if self._is_new else self._get_reconfigure_subentry().subentry_id
        )
        for provider_subentry in self._provider_subentries():
            if provider_subentry.subentry_id == current_subentry_id:
                continue
            if _provider_data_matches(provider_subentry.data, data):
                return True
        return False

    async def _async_show_provider_form(
        self,
        step_id: str,
        *,
        options: Mapping[str, Any],
        errors: dict[str, str] | None = None,
        description_placeholders: dict[str, str] | None = None,
    ) -> SubentryFlowResult:
        """Show a provider form."""
        return self.async_show_form(
            step_id=step_id,
            data_schema=_provider_schema(options),
            errors=dict(errors or {}),
            description_placeholders=description_placeholders,
        )

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Create a provider subentry."""
        return await self._async_provider_form_step("init", user_input)

    async def async_step_edit_connection(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Edit provider connection settings."""
        return await self._async_provider_form_step("edit_connection", user_input)

    async def _async_provider_form_step(
        self, step_id: str, user_input: dict[str, Any] | None
    ) -> SubentryFlowResult:
        """Handle the provider create/edit form."""
        entry = self._get_entry()
        if entry.state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if user_input is not None:
            try:
                data = _normalise_provider_data(user_input)
                _validate_provider_data(self.hass, data)
            except ProviderValidationError as err:
                return await self._async_show_provider_form(
                    step_id,
                    options=user_input,
                    errors={"base": err.reason},
                    description_placeholders=_provider_validation_placeholders(err),
                )
            if self._provider_already_configured(data):
                return self.async_abort(reason="already_configured")
            self._options = dict(data)
            self._pending_data = dict(data)
            self._pending_storage_data = {}
            self._pending_step_id = step_id
            self._pending_error = await self._async_validate_provider_form(data)
            if self._pending_error is not None:
                field, reason, placeholders = self._pending_error
                return await self._async_show_provider_form(
                    step_id,
                    options=self._pending_data,
                    errors={field: reason},
                    description_placeholders=placeholders,
                )
            return self._finish_provider_form()

        if not self._options and not self._is_new:
            self._options = self._provider_form_options(
                self._get_reconfigure_subentry()
            )
        return await self._async_show_provider_form(step_id, options=self._options)

    async def _async_validate_provider_form(
        self, data: dict[str, Any]
    ) -> tuple[str, str, dict[str, str]] | None:
        """Validate one provider form submission."""
        existing_data: Mapping[str, Any] = {}
        existing_profiles: Mapping[str, Any] = {}
        if not self._is_new:
            existing_data = self._get_reconfigure_subentry().data
            existing_profiles = existing_data.get(CONF_MODEL_PROFILES, {})
        storage_data: dict[str, Any] = {
            CONF_NAME: self._pending_data[CONF_NAME],
            CONF_PROVIDER_MODE: self._pending_data[CONF_PROVIDER_MODE],
            CONF_API_KEY: self._pending_data[CONF_API_KEY],
            CONF_MODEL_PROFILES: dict(existing_profiles)
            if isinstance(existing_profiles, Mapping)
            else {},
        }
        custom_model_names = _provider_custom_model_names(existing_data)
        if not self._is_new and custom_model_names:
            storage_data[CONF_CUSTOM_MODEL_NAMES] = custom_model_names
        if base_url := self._pending_data.get(CONF_BASE_URL):
            storage_data[CONF_BASE_URL] = base_url
        if provider_headers := self._pending_data.get(CONF_PROVIDER_HEADERS):
            storage_data[CONF_PROVIDER_HEADERS] = provider_headers
        if provider_extra_body := self._pending_data.get(CONF_PROVIDER_EXTRA_BODY):
            storage_data[CONF_PROVIDER_EXTRA_BODY] = dict(provider_extra_body)
        if isinstance(
            provider_metadata := existing_data.get(CONF_PROVIDER_METADATA), Mapping
        ) and _catalog_provider_metadata_still_valid(existing_data, storage_data):
            storage_data[CONF_PROVIDER_METADATA] = dict(provider_metadata)
        for key in (
            CONF_DISCOVERED_MODELS,
            CONF_DISCOVERED_MODELS_AT,
            CONF_DISCOVERED_MODELS_CACHE_KEY,
        ):
            if key in existing_data:
                storage_data[key] = existing_data[key]
        if _cached_provider_model_names(storage_data) is None:
            _clear_provider_model_cache(storage_data)
        self._pending_storage_data = storage_data
        return None

    def _finish_provider_form(self) -> SubentryFlowResult:
        """Create or update the provider subentry after validation."""
        data = self._pending_storage_data
        if self._is_new:
            return self.async_create_entry(title=data[CONF_NAME], data=data)
        return self.async_update_and_abort(
            self._get_entry(),
            self._get_reconfigure_subentry(),
            title=data[CONF_NAME],
            data=data,
        )

    async def async_step_reconfigure_menu(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Show the shallow provider-management menu."""
        del user_input
        return self.async_show_menu(
            step_id="reconfigure_menu",
            menu_options=[
                "edit_connection",
                "manage_models",
                "customize_model_profile",
            ],
        )

    async def async_step_manage_models(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage which provider-owned model profiles are available."""
        if not getattr(self, "_profile_flow_data", None):
            result = await self._async_prepare_manage_models_flow()
            if result is not None:
                return result
        if user_input is None:
            models = self._managed_models_for_selection()
            return self.async_show_form(
                step_id="manage_models",
                data_schema=model_selection_schema(
                    models,
                    self._enabled_model_ids_for_options(models),
                    custom_model_names=_format_custom_model_names(
                        self._current_profile_flow_data()
                    ),
                ),
                errors={"base": "no_models_available"} if not models else None,
            )
        flat_user_input = _flatten_section_data(user_input, (SECTION_ADVANCED_MODELS,))
        data = self._current_profile_flow_data()
        previous_custom_model_ids = set(_provider_custom_model_names(data))
        custom_model_names = _provider_custom_model_names(flat_user_input)
        if custom_model_names:
            data[CONF_CUSTOM_MODEL_NAMES] = custom_model_names
        else:
            data.pop(CONF_CUSTOM_MODEL_NAMES, None)
        models = self._managed_models_for_selection()
        raw_selected_model_ids = user_input.get(CONF_SELECTED_MODEL_IDS)
        if isinstance(raw_selected_model_ids, str) or not isinstance(
            raw_selected_model_ids, list
        ):
            return self.async_show_form(
                step_id="manage_models",
                data_schema=model_selection_schema(
                    models,
                    self._enabled_model_ids_for_options(models),
                    custom_model_names=_format_custom_model_names(data),
                ),
                errors={CONF_SELECTED_MODEL_IDS: "model_required"},
            )
        selected_model_ids = list(raw_selected_model_ids)
        for model_name in custom_model_names:
            if (
                model_name not in previous_custom_model_ids
                and model_name not in selected_model_ids
            ):
                selected_model_ids.append(model_name)
        selected_models = selected_models_by_id(models, selected_model_ids)
        error = self._sync_selected_model_profiles(
            selected_models, models, set(custom_model_names)
        )
        if error is not None:
            return self.async_show_form(
                step_id="manage_models",
                data_schema=model_selection_schema(
                    models,
                    self._enabled_model_ids_for_options(models),
                    custom_model_names=_format_custom_model_names(data),
                ),
                errors={"base": error},
            )
        return self.async_update_and_abort(
            self._get_entry(),
            self._get_reconfigure_subentry(),
            title=data[CONF_NAME],
            data=data,
        )
