"""Config subentry flow handlers for Pydantic AI Agent."""

# ruff: noqa: F403, F405

from __future__ import annotations

from .common import (
    Any,
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_CUSTOM_MODEL_NAMES,
    CONF_DISCOVERED,
    CONF_DISCOVERED_MODELS,
    CONF_DISCOVERED_MODELS_AT,
    CONF_DISCOVERED_MODELS_CACHE_KEY,
    CONF_ENABLED,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_NAME,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_METADATA,
    CONF_PROVIDER_MODE,
    ConfigEntryState,
    ConfigSubentry,
    ConfigSubentryFlow,
    Mapping,
    ProviderValidationError,
    SOURCE_USER,
    SubentryFlowResult,
    _ADVANCED_MODEL_SETTING_KEYS,
    _CONF_MODEL_PROFILE_ID,
    _LOGGER,
    _MAIN_MODEL_SETTING_KEYS,
    _SECTION_ADVANCED_MODEL_SETTINGS,
    _cached_provider_model_names,
    _clear_provider_model_cache,
    _flatten_section_data,
    _format_custom_model_names,
    _format_key_value_json_setting,
    _merge_model_settings,
    _normalise_base_url,
    _model_profile_data_from_user_input,
    _model_profile_edit_schema,
    _model_settings_from_options,
    _normalise_provider_data,
    _normalise_provider_model_profiles,
    _parse_model_settings,
    _provider_custom_model_names,
    _provider_data_matches,
    _provider_model_profiles_for_discovery_mode,
    _provider_profile_options,
    _provider_profile_selector_schema,
    _provider_schema,
    _provider_validation_placeholders,
    _referenced_provider_profile_ids,
    _store_model_settings,
    _store_provider_model_cache,
    _validate_provider_data,
    async_list_provider_model_names,
    provider_model_profiles,
    provider_subentries,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE_GEMINI,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
    vol,
)
from .provider_wizard.catalog_cache import catalog_manager
from .provider_wizard.const import (
    CATALOG_RETRY_PROVIDER_ID,
    CONF_CATALOG_PROVIDER_ID,
    CONF_DRIVER,
    CONF_PROVIDER_ID,
    CONF_SELECTED_MODEL_IDS,
    CUSTOM_PROVIDER_ID,
)
from .provider_wizard.filters import ModelFilterOptions, filtered_models
from .provider_wizard.flow import (
    build_model_profiles,
    build_provider_data,
    selected_models_by_id,
)
from .provider_wizard.models_dev import CatalogLoadError
from .provider_wizard.schemas import (
    connection_schema,
    driver_selection_schema,
    filters_from_user_input,
    model_filter_schema,
    model_selection_schema,
    needs_model_filter_step,
    provider_selection_schema,
)
from .provider_wizard.types import (
    CatalogModelOption,
    CatalogProviderOption,
    CompactCatalog,
)
from ..generated_titles import DEFAULT_SERVICE_TITLE_SUFFIX, generated_default_title

_DISCOVERED_PROVIDER_ID = "__provider_discovery__"


def _catalog_provider_metadata_still_valid(
    existing_data: Mapping[str, Any], storage_data: Mapping[str, Any]
) -> bool:
    """Return if stored catalog metadata still matches the edited endpoint."""
    return _effective_catalog_base_url(existing_data) == _effective_catalog_base_url(
        storage_data
    )


def _effective_catalog_base_url(data: Mapping[str, Any]) -> str | None:
    """Return the effective endpoint used by model discovery/catalog metadata."""
    base_url = _normalise_base_url(data)
    provider_mode = data.get(CONF_PROVIDER_MODE)
    if provider_mode == PROVIDER_ANTHROPIC:
        if base_url and base_url.endswith("/v1"):
            return base_url.removesuffix("/v1")
        return base_url or "https://api.anthropic.com"
    if provider_mode == PROVIDER_GOOGLE_GEMINI:
        if base_url and base_url.endswith(("/v1beta", "/v1")):
            return base_url.rsplit("/", 1)[0]
        return base_url or "https://generativelanguage.googleapis.com"
    if provider_mode in {
        PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
    }:
        return base_url or "https://api.openai.com/v1"
    return base_url


class ProviderSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing workspace-owned provider subentries."""

    _options: dict[str, Any]
    _pending_data: dict[str, Any]
    _pending_error: tuple[str, str, dict[str, str]] | None
    _pending_storage_data: dict[str, Any]
    _pending_step_id: str
    _selected_profile_id: str | None
    _pending_model_settings: dict[str, Any]
    _pending_profile_data: dict[str, Any]
    _pending_profile_error: tuple[str, dict[str, str]] | None
    _profile_flow_data: dict[str, Any]
    _profile_filter_provider: CatalogProviderOption | None
    _profile_filters: ModelFilterOptions
    _profile_models: tuple[CatalogModelOption, ...]
    _profile_refresh_error: str | None
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
                (subentry.title for subentry in provider_subentries(self._get_entry())),
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
        options[CONF_PROVIDER_EXTRA_BODY] = _format_key_value_json_setting(
            options.get(CONF_PROVIDER_EXTRA_BODY)
        )
        return options

    def _provider_already_configured(self, data: Mapping[str, Any]) -> bool:
        """Return if another provider subentry already uses this connection."""
        current_subentry_id = (
            None if self._is_new else self._get_reconfigure_subentry().subentry_id
        )
        for provider_subentry in provider_subentries(self._get_entry()):
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

    async def _async_load_model_catalog(self) -> SubentryFlowResult:
        """Load the provider model catalog before provider selection."""
        entry = self._get_entry()
        if entry.state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")
        manager = catalog_manager(self.hass)
        if catalog := manager.cached_catalog():
            self._wizard_catalog = catalog
            self._wizard_catalog_error = None
            return await self.async_step_pick_provider()
        task = manager.load_task()
        return self.async_show_progress(
            step_id="load_model_catalog_progress",
            progress_action="load_model_catalog",
            progress_task=task,
        )

    async def async_step_load_model_catalog_progress(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Finish provider catalog loading progress."""
        del user_input
        task = self.async_get_progress_task()
        if task is not None and not task.done():
            return self.async_show_progress(
                step_id="load_model_catalog_progress",
                progress_action="load_model_catalog",
                progress_task=task,
            )
        self._wizard_catalog = None
        self._wizard_catalog_error = None
        if task is not None:
            try:
                self._wizard_catalog = task.result()
            except CatalogLoadError:
                self._wizard_catalog_error = "model_catalog_unavailable"
            except Exception:
                _LOGGER.exception("Unexpected exception loading provider model catalog")
                self._wizard_catalog_error = "model_catalog_unavailable"
        return self.async_show_progress_done(next_step_id="load_model_catalog_finish")

    async def async_step_load_model_catalog_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Continue after provider catalog loading."""
        del user_input
        if self._wizard_catalog_error is not None or self._wizard_catalog is None:
            return await self.async_step_pick_provider()
        return await self.async_step_pick_provider()

    def _pick_provider_errors(self, field_error: str | None = None) -> dict[str, str]:
        """Return provider picker errors including catalog load failures."""
        errors: dict[str, str] = {}
        if self._wizard_catalog_error is not None:
            errors["base"] = self._wizard_catalog_error
        if field_error is not None:
            errors[CONF_PROVIDER_ID] = field_error
        return errors

    async def async_step_pick_provider(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Choose a catalog provider by name."""
        catalog = self._wizard_catalog
        has_catalog_error = self._wizard_catalog_error is not None
        if catalog is None:
            if not has_catalog_error:
                return await self._async_load_model_catalog()
            catalog = CompactCatalog(providers={}, models_by_provider={})
        if user_input is None:
            return self.async_show_form(
                step_id="pick_provider",
                data_schema=provider_selection_schema(
                    catalog, include_retry=has_catalog_error
                ),
                errors=self._pick_provider_errors(),
            )
        provider_id = user_input.get(CONF_PROVIDER_ID)
        if provider_id == CATALOG_RETRY_PROVIDER_ID and has_catalog_error:
            self._wizard_catalog_error = None
            return await self._async_load_model_catalog()
        if not isinstance(provider_id, str):
            return self.async_show_form(
                step_id="pick_provider",
                data_schema=provider_selection_schema(
                    catalog, include_retry=has_catalog_error
                ),
                errors=self._pick_provider_errors("invalid_provider_config"),
            )
        if provider_id == CUSTOM_PROVIDER_ID:
            return await self.async_step_init()
        provider = catalog.providers.get(provider_id)
        if provider is None:
            return self.async_show_form(
                step_id="pick_provider",
                data_schema=provider_selection_schema(
                    catalog, include_retry=has_catalog_error
                ),
                errors=self._pick_provider_errors("invalid_provider_config"),
            )
        self._wizard_provider = provider
        self._wizard_models = catalog.models_for_provider(provider.id)
        self._wizard_filters = ModelFilterOptions()
        if not provider.supported_drivers:
            return self.async_show_form(
                step_id="pick_provider",
                data_schema=provider_selection_schema(catalog),
                errors={CONF_PROVIDER_ID: "invalid_provider_config"},
            )
        if len(provider.supported_drivers) > 1:
            return await self.async_step_pick_driver()
        self._wizard_driver = next(iter(provider.supported_drivers))
        return await self.async_step_wizard_connection()

    async def async_step_pick_driver(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Choose an API mode when the provider supports more than one."""
        provider = self._wizard_provider
        if provider is None:
            return await self.async_step_pick_provider()
        if user_input is None:
            return self.async_show_form(
                step_id="pick_driver", data_schema=driver_selection_schema(provider)
            )
        driver = user_input.get(CONF_DRIVER)
        if driver not in provider.supported_drivers:
            return self.async_show_form(
                step_id="pick_driver",
                data_schema=driver_selection_schema(provider),
                errors={CONF_DRIVER: "invalid_provider_config"},
            )
        self._wizard_driver = str(driver)
        return await self.async_step_wizard_connection()

    async def async_step_wizard_connection(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Collect guided provider connection details."""
        provider = self._wizard_provider
        driver = self._wizard_driver
        if provider is None or driver is None:
            return await self.async_step_pick_provider()
        if user_input is None:
            return self.async_show_form(
                step_id="wizard_connection",
                data_schema=connection_schema(
                    provider, driver, self._wizard_connection_options
                ),
            )
        data_input = dict(user_input)
        data_input[CONF_PROVIDER_MODE] = driver
        try:
            data = _normalise_provider_data(data_input)
            _validate_provider_data(self.hass, data)
        except ProviderValidationError as err:
            self._wizard_connection_options = data_input
            return self.async_show_form(
                step_id="wizard_connection",
                data_schema=connection_schema(provider, driver, data_input),
                errors={"base": err.reason},
                description_placeholders=_provider_validation_placeholders(err),
            )
        if self._provider_already_configured(data):
            return self.async_abort(reason="already_configured")
        self._wizard_connection_data = data
        self._wizard_connection_options = dict(data_input)
        return await self._async_next_model_step()

    async def _async_next_model_step(self) -> SubentryFlowResult:
        """Advance to filter, model selection, or finish for guided setup."""
        default_models = filtered_models(self._wizard_models, self._wizard_filters)
        if not default_models:
            return await self.async_step_model_filters()
        if len(default_models) < len(self._wizard_models):
            return await self.async_step_model_filters()
        if needs_model_filter_step(self._wizard_models):
            return await self.async_step_model_filters()
        if len(default_models) == 1:
            self._wizard_selected_models = default_models
            return self._finish_guided_provider()
        return await self.async_step_pick_models()

    async def async_step_model_filters(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Filter a large or empty default catalog model list."""
        provider = self._wizard_provider
        if provider is None:
            return await self.async_step_pick_provider()
        errors: dict[str, str] = {}
        if user_input is not None:
            self._wizard_filters = filters_from_user_input(user_input)
            if not filtered_models(self._wizard_models, self._wizard_filters):
                errors["base"] = "no_models_available"
            else:
                return await self.async_step_pick_models()
        return self.async_show_form(
            step_id="model_filters",
            data_schema=model_filter_schema(provider, self._wizard_filters),
            errors=errors,
        )

    async def async_step_pick_models(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Select catalog models to enable."""
        models = filtered_models(self._wizard_models, self._wizard_filters)
        if user_input is None:
            return self.async_show_form(
                step_id="pick_models", data_schema=model_selection_schema(models)
            )
        selected_models = selected_models_by_id(
            models, user_input.get(CONF_SELECTED_MODEL_IDS)
        )
        if not selected_models:
            return self.async_show_form(
                step_id="pick_models",
                data_schema=model_selection_schema(models),
                errors={CONF_SELECTED_MODEL_IDS: "model_required"},
            )
        self._wizard_selected_models = selected_models
        return self._finish_guided_provider()

    def _finish_guided_provider(self) -> SubentryFlowResult:
        """Create a provider subentry from guided wizard selections."""
        provider = self._wizard_provider
        if provider is None:
            return self.async_abort(reason="invalid_provider_config")
        headers = self._wizard_connection_data.get(CONF_PROVIDER_HEADERS)
        extra_body = self._wizard_connection_data.get(CONF_PROVIDER_EXTRA_BODY)
        data = build_provider_data(
            provider,
            provider_mode=str(self._wizard_connection_data[CONF_PROVIDER_MODE]),
            api_key=str(self._wizard_connection_data[CONF_API_KEY]),
            selected_models=self._wizard_selected_models,
            provider_name=str(self._wizard_connection_data[CONF_NAME]),
            base_url=self._wizard_connection_data.get(CONF_BASE_URL),
            provider_headers=dict(headers) if isinstance(headers, Mapping) else None,
            provider_extra_body=dict(extra_body) if isinstance(extra_body, Mapping) else None,
        )
        return self.async_create_entry(title=str(data[CONF_NAME]), data=data)

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
            self._pending_error = None
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
        existing_profiles: Mapping[str, Any] = {}
        existing_data: Mapping[str, Any] = {}
        if not self._is_new:
            existing_data = self._get_reconfigure_subentry().data
            existing_profiles = existing_data.get(CONF_MODEL_PROFILES, {})
        custom_model_names = _provider_custom_model_names(self._pending_data)
        if custom_model_names:
            keep_profile_ids = (
                _referenced_provider_profile_ids(
                    self._get_entry(), self._get_reconfigure_subentry().subentry_id
                )
                if not self._is_new
                else set()
            )
            model_profiles = _normalise_provider_model_profiles(
                existing_profiles,
                custom_model_names,
                [],
                keep_profile_ids=keep_profile_ids,
            )
        elif isinstance(existing_profiles, Mapping):
            model_profiles = (
                _provider_model_profiles_for_discovery_mode(
                    existing_profiles,
                    keep_profile_ids=_referenced_provider_profile_ids(
                        self._get_entry(), self._get_reconfigure_subentry().subentry_id
                    ),
                )
                if not self._is_new
                else {}
            )
        else:
            model_profiles = {}
        storage_data: dict[str, Any] = {
            CONF_NAME: self._pending_data[CONF_NAME],
            CONF_PROVIDER_MODE: self._pending_data[CONF_PROVIDER_MODE],
            CONF_API_KEY: self._pending_data[CONF_API_KEY],
            CONF_MODEL_PROFILES: model_profiles,
        }
        if custom_model_names:
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
        if not custom_model_names:
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

    async def async_step_customize_model_profile(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Choose a provider-owned profile to edit."""
        del user_input
        self._profile_flow_data = dict(self._get_reconfigure_subentry().data)
        self._profile_refresh_error = None
        self._selected_profile_id = None
        return await self.async_step_pick_model_profile()

    async def async_step_manage_models(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage which provider-owned model profiles are available."""
        if not getattr(self, "_profile_flow_data", None):
            result = await self._async_prepare_manage_models_flow()
            if result is not None:
                return result
        models = self._managed_models_for_selection()
        if user_input is None:
            if not models:
                return self.async_show_form(
                    step_id="manage_models",
                    data_schema=vol.Schema({}),
                    errors={"base": "no_models_available"},
                )
            return self.async_show_form(
                step_id="manage_models",
                data_schema=model_selection_schema(
                    models, self._enabled_model_ids_for_options(models)
                ),
            )
        raw_selected_model_ids = user_input.get(CONF_SELECTED_MODEL_IDS)
        if isinstance(raw_selected_model_ids, str) or not isinstance(
            raw_selected_model_ids, list
        ):
            return self.async_show_form(
                step_id="manage_models",
                data_schema=model_selection_schema(
                    models, self._enabled_model_ids_for_options(models)
                ),
                errors={CONF_SELECTED_MODEL_IDS: "model_required"},
            )
        selected_models = selected_models_by_id(models, raw_selected_model_ids)
        error = self._sync_selected_model_profiles(selected_models, models)
        if error is not None:
            return self.async_show_form(
                step_id="manage_models",
                data_schema=model_selection_schema(
                    models, self._enabled_model_ids_for_options(models)
                ),
                errors={"base": error},
            )
        data = self._current_profile_flow_data()
        return self.async_update_and_abort(
            self._get_entry(),
            self._get_reconfigure_subentry(),
            title=data[CONF_NAME],
            data=data,
        )

    async def _async_prepare_manage_models_flow(self) -> SubentryFlowResult | None:
        """Prepare catalog-backed model availability management."""
        data = dict(self._get_reconfigure_subentry().data)
        self._profile_flow_data = data
        self._profile_filters = ModelFilterOptions()
        custom_model_names = _provider_custom_model_names(data)
        if custom_model_names:
            self._profile_filter_provider = None
            self._profile_models = tuple(
                CatalogModelOption(
                    id=model_name,
                    name=model_name,
                    provider_id=CUSTOM_PROVIDER_ID,
                    family=None,
                    tool_call=True,
                    structured_output=None,
                    reasoning=False,
                    attachment=False,
                    text_output=True,
                    context_limit=0,
                    output_limit=0,
                    status=None,
                )
                for model_name in custom_model_names
            )
            return None
        if CONF_PROVIDER_METADATA not in data:
            if discovered_models := await self._async_discovered_models_for_manage(data):
                self._profile_filter_provider = None
                self._profile_models = discovered_models
                return None
        provider, models = await self._async_catalog_models_for_provider_data(data)
        if provider is None or not models:
            if discovered_models := await self._async_discovered_models_for_manage(data):
                self._profile_filter_provider = None
                self._profile_models = discovered_models
                return None
            return self.async_show_form(
                step_id="manage_models",
                data_schema=vol.Schema({}),
                errors={"base": "model_catalog_unavailable"},
            )
        self._profile_filter_provider = provider
        self._profile_models = models
        if self._profile_needs_model_filter_step():
            return await self.async_step_manage_model_filters()
        return None

    async def _async_discovered_models_for_manage(
        self, data: Mapping[str, Any]
    ) -> tuple[CatalogModelOption, ...]:
        """Return provider-discovered model options for availability management."""
        model_names = _cached_provider_model_names(data)
        if model_names is None:
            try:
                model_names = await async_list_provider_model_names(self.hass, data)
            except Exception:
                _LOGGER.warning("Unable to list provider models for model management")
                return ()
            _store_provider_model_cache(self._profile_flow_data, model_names)
        return tuple(
            CatalogModelOption(
                id=model_name,
                name=model_name,
                provider_id=_DISCOVERED_PROVIDER_ID,
                family=None,
                tool_call=True,
                structured_output=None,
                reasoning=False,
                attachment=False,
                text_output=True,
                context_limit=0,
                output_limit=0,
                status=None,
            )
            for model_name in model_names
        )

    async def _async_catalog_models_for_provider_data(
        self, data: Mapping[str, Any]
    ) -> tuple[CatalogProviderOption | None, tuple[CatalogModelOption, ...]]:
        """Return catalog provider models matching stored provider metadata."""
        try:
            catalog = await catalog_manager(self.hass).async_get_catalog()
        except CatalogLoadError:
            return None, ()
        except Exception:
            _LOGGER.warning("Unable to load model catalog for model management")
            return None, ()
        metadata = data.get(CONF_PROVIDER_METADATA)
        catalog_provider_id = (
            metadata.get(CONF_CATALOG_PROVIDER_ID)
            if isinstance(metadata, Mapping)
            else None
        )
        provider = (
            catalog.providers.get(catalog_provider_id)
            if isinstance(catalog_provider_id, str)
            else None
        )
        profiles = provider_model_profiles(self._get_reconfigure_subentry())
        profile_model_ids = {
            str(profile[CONF_MODEL])
            for profile in profiles.values()
            if isinstance(profile.get(CONF_MODEL), str)
        }
        if provider is None:
            provider_matches = [
                (
                    len(
                        profile_model_ids
                        & {model.id for model in catalog.models_for_provider(candidate.id)}
                    ),
                    candidate,
                )
                for candidate in catalog.providers.values()
            ]
            best_score = max((score for score, _ in provider_matches), default=0)
            matches = [
                candidate
                for score, candidate in provider_matches
                if score == best_score and score > 0
            ]
            if len(matches) != 1:
                return None, ()
            provider = matches[0]
        return provider, catalog.models_for_provider(provider.id)

    def _enabled_model_ids_for_options(
        self, models: tuple[CatalogModelOption, ...]
    ) -> tuple[str, ...]:
        """Return enabled model IDs that are visible in a selection form."""
        visible_ids = {model.id for model in models}
        profiles = self._current_profile_flow_data().get(CONF_MODEL_PROFILES, {})
        if not isinstance(profiles, Mapping):
            return ()
        return tuple(
            model_name
            for profile in profiles.values()
            if isinstance(profile, Mapping)
            and bool(profile.get(CONF_ENABLED, False))
            and isinstance(model_name := profile.get(CONF_MODEL), str)
            and model_name in visible_ids
        )

    def _managed_models_for_selection(self) -> tuple[CatalogModelOption, ...]:
        """Return filtered models plus enabled models hidden by those filters."""
        models_by_id = {model.id: model for model in self._profile_models}
        managed_models = {
            model.id: model
            for model in filtered_models(self._profile_models, self._profile_filters)
        }
        profiles = self._current_profile_flow_data().get(CONF_MODEL_PROFILES, {})
        if isinstance(profiles, Mapping):
            for profile in profiles.values():
                if not isinstance(profile, Mapping) or not bool(
                    profile.get(CONF_ENABLED, False)
                ):
                    continue
                model_name = profile.get(CONF_MODEL)
                if not isinstance(model_name, str) or not model_name.strip():
                    continue
                if model_name in models_by_id:
                    managed_models[model_name] = models_by_id[model_name]
                    continue
                profile_name = profile.get(CONF_NAME)
                managed_models[model_name] = CatalogModelOption(
                    id=model_name,
                    name=str(profile_name or model_name),
                    provider_id=CUSTOM_PROVIDER_ID,
                    family=None,
                    tool_call=True,
                    structured_output=None,
                    reasoning=False,
                    attachment=False,
                    text_output=True,
                    context_limit=0,
                    output_limit=0,
                    status=None,
                )
        return tuple(managed_models.values())

    def _sync_selected_model_profiles(
        self,
        selected_models: tuple[CatalogModelOption, ...],
        managed_models: tuple[CatalogModelOption, ...],
    ) -> str | None:
        """Sync enabled model profiles for one availability selection."""
        data = self._current_profile_flow_data()
        profiles = data.get(CONF_MODEL_PROFILES, {})
        if not isinstance(profiles, Mapping):
            profiles = {}
        selected_model_ids = {model.id for model in selected_models}
        managed_model_ids = {model.id for model in managed_models}
        referenced_profile_ids = _referenced_provider_profile_ids(
            self._get_entry(), self._get_reconfigure_subentry().subentry_id
        )
        synced_profiles: dict[str, dict[str, Any]] = {}
        selected_existing_model_ids: set[str] = set()
        for profile_id, profile in profiles.items():
            if not isinstance(profile_id, str) or not isinstance(profile, Mapping):
                continue
            model_name = profile.get(CONF_MODEL)
            if not isinstance(model_name, str) or not model_name.strip():
                continue
            profile_data = dict(profile)
            profile_data["id"] = profile_id
            if model_name in selected_model_ids:
                profile_data[CONF_ENABLED] = True
                selected_existing_model_ids.add(model_name)
            elif model_name in managed_model_ids:
                if profile_id in referenced_profile_ids:
                    return "model_profile_in_use"
                profile_data[CONF_ENABLED] = False
            synced_profiles[profile_id] = profile_data
        missing_selected_models = tuple(
            model for model in selected_models if model.id not in selected_existing_model_ids
        )
        missing_profiles = build_model_profiles(missing_selected_models)
        custom_model_ids = {
            model.id for model in missing_selected_models if model.provider_id == CUSTOM_PROVIDER_ID
        }
        for profile in missing_profiles.values():
            if profile[CONF_MODEL] in custom_model_ids:
                profile[CONF_DISCOVERED] = False
        synced_profiles.update(missing_profiles)
        data[CONF_MODEL_PROFILES] = synced_profiles
        return None

    def _profile_needs_model_filter_step(self) -> bool:
        """Return if profile reconfiguration should show model filters first."""
        if self._profile_filter_provider is None or not self._profile_models:
            return False
        default_models = filtered_models(self._profile_models, self._profile_filters)
        return len(default_models) < len(self._profile_models) or needs_model_filter_step(
            self._profile_models
        )

    async def async_step_manage_model_filters(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Filter provider-owned models before managing availability."""
        provider = self._profile_filter_provider
        if provider is None:
            return await self.async_step_manage_models()
        errors: dict[str, str] = {}
        if user_input is not None:
            self._profile_filters = filters_from_user_input(user_input)
            if not self._managed_models_for_selection():
                errors["base"] = "no_models_available"
            else:
                return await self.async_step_manage_models()
        return self.async_show_form(
            step_id="manage_model_filters",
            data_schema=model_filter_schema(provider, self._profile_filters),
            errors=errors,
        )

    def _current_profile_flow_data(self) -> dict[str, Any]:
        """Return transient provider data for the active profile edit flow."""
        if profile_flow_data := getattr(self, "_profile_flow_data", None):
            return profile_flow_data
        return dict(self._get_reconfigure_subentry().data)

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
                user_input, (_SECTION_ADVANCED_MODEL_SETTINGS,)
            )
            parsed_settings, errors, cleared = _parse_model_settings(
                self.hass,
                flat_user_input,
                _MAIN_MODEL_SETTING_KEYS | _ADVANCED_MODEL_SETTING_KEYS,
            )
            data = _model_profile_data_from_user_input(flat_user_input)
            existing_settings = _model_settings_from_options(
                {CONF_MODEL_SETTINGS: profile.get(CONF_MODEL_SETTINGS, {})}
            )
            model_settings = _merge_model_settings(
                existing_settings, parsed_settings, cleared
            )
            if errors:
                return self.async_show_form(
                    step_id="edit_model_profile",
                    data_schema=_model_profile_edit_schema(
                        profile | data | {CONF_MODEL_SETTINGS: model_settings}
                    ),
                    errors=errors,
                )
            self._pending_profile_data = dict(profile) | data
            self._pending_model_settings = dict(model_settings)
            self._pending_profile_error = None
            _store_model_settings(
                self._pending_profile_data, self._pending_model_settings
            )
            return await self.async_step_model_profile_finish()
        return self.async_show_form(
            step_id="edit_model_profile",
            data_schema=_model_profile_edit_schema(profile),
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
                data_schema=_model_profile_edit_schema(self._pending_profile_data),
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
