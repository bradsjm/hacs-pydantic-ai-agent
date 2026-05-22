"""Config subentry flow handlers for Pydantic AI Agent."""

# ruff: noqa: F403, F405

from __future__ import annotations

from .common import (
    Any,
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_CUSTOM_MODEL_NAMES,
    CONF_DISCOVERED_MODELS,
    CONF_DISCOVERED_MODELS_AT,
    CONF_DISCOVERED_MODELS_CACHE_KEY,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_NAME,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
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
    _model_profile_data_from_user_input,
    _model_profile_edit_schema,
    _model_settings_from_options,
    _normalise_provider_data,
    _normalise_provider_model_profiles,
    _parse_model_settings,
    _provider_custom_model_names,
    _provider_data_matches,
    _provider_model_cache_key,
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
    vol,
)

class ProviderSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing workspace-owned provider subentries."""

    _model_names: list[str] | None
    _model_names_cache_key: str | None
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
    _profile_refresh_error: str | None

    @property
    def _is_new(self) -> bool:
        """Return if this flow creates a new subentry."""
        return self.source == SOURCE_USER

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add a provider subentry."""
        self._model_names = None
        self._model_names_cache_key = None
        self._options = {}
        self._pending_error = None
        self._pending_storage_data = {}
        self._pending_step_id = "init"
        self._selected_profile_id = None
        self._profile_flow_data = {}
        self._profile_refresh_error = None
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure a provider subentry."""
        self._options = self._provider_form_options(self._get_reconfigure_subentry())
        self._model_names = None
        self._model_names_cache_key = None
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

    async def _async_model_names(self, data: Mapping[str, Any]) -> list[str] | None:
        """Return discovered provider model names for this flow."""
        if _provider_custom_model_names(data):
            return None
        cache_key = _provider_model_cache_key(data)
        if self._model_names is not None and self._model_names_cache_key == cache_key:
            return self._model_names
        if cached_names := _cached_provider_model_names(data):
            self._model_names = cached_names
            self._model_names_cache_key = cache_key
            return cached_names
        try:
            self._model_names = await async_list_provider_model_names(self.hass, data)
            self._model_names_cache_key = cache_key
        except Exception:
            _LOGGER.warning("Unable to list provider models for provider form")
            return None
        return self._model_names

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

    async def async_step_reconfigure_menu(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Show the shallow provider-management menu."""
        del user_input
        return self.async_show_menu(
            step_id="reconfigure_menu",
            menu_options=[
                "edit_connection",
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
        (
            self._profile_flow_data,
            self._profile_refresh_error,
        ) = await self._async_prepare_profile_flow_data()
        self._selected_profile_id = None
        return await self.async_step_pick_model_profile()

    async def _async_prepare_profile_flow_data(
        self,
    ) -> tuple[dict[str, Any], str | None]:
        """Return provider data with refreshed model profiles for profile editing."""
        provider_subentry = self._get_reconfigure_subentry()
        data = dict(provider_subentry.data)
        existing_profiles = data.get(CONF_MODEL_PROFILES, {})
        custom_model_names = _provider_custom_model_names(data)
        if custom_model_names:
            _clear_provider_model_cache(data)
            data[CONF_MODEL_PROFILES] = _normalise_provider_model_profiles(
                existing_profiles,
                custom_model_names,
                [],
                keep_profile_ids=_referenced_provider_profile_ids(
                    self._get_entry(), provider_subentry.subentry_id
                ),
            )
            return data, None

        discovered_model_names = await self._async_model_names(data)
        if not discovered_model_names:
            return data, None if provider_model_profiles(
                provider_subentry
            ) else "model_list_unavailable"

        _store_provider_model_cache(data, discovered_model_names)
        data[CONF_MODEL_PROFILES] = _normalise_provider_model_profiles(
            existing_profiles,
            discovered_model_names,
            discovered_model_names,
            keep_profile_ids=_referenced_provider_profile_ids(
                self._get_entry(), provider_subentry.subentry_id
            ),
        )
        return data, None

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
            if not _provider_profile_options(data):
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
                data_schema=_provider_profile_selector_schema(data),
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
