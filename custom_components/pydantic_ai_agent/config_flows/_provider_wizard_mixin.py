"""Guided provider setup wizard mixin for provider config flows."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigEntryState,
    SubentryFlowResult,
)
from homeassistant.core import HomeAssistant

from ..const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_NAME,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_MODE,
    CONF_PROVIDER_SECRET_HEADER_KEYS,
)
from ._profile_helpers import (
    _provider_validation_placeholders,
)
from ._provider_data import (
    _normalise_provider_data,
    _validate_provider_data,
)
from .provider_wizard.catalog_cache import catalog_manager
from .provider_wizard.const import (
    CATALOG_RETRY_PROVIDER_ID,
    CONF_DRIVER,
    CONF_PROVIDER_ID,
    CONF_SELECTED_MODEL_IDS,
    CUSTOM_PROVIDER_ID,
)
from .provider_wizard.filters import ModelFilterOptions, filtered_models
from .provider_wizard.flow import (
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

_LOGGER = logging.getLogger(__name__)


class ProviderWizardMixin:
    """Mixin for guided provider setup wizard steps.

    Must be used with a ConfigSubentryFlow subclass.
    """

    # ---- parent class delegations (satisfied by ConfigSubentryFlow at runtime) ----
    # These forward to super() to satisfy pyright while preserving real behavior.
    hass: HomeAssistant

    def _get_entry(self) -> ConfigEntry:
        return super()._get_entry()  # type: ignore[misc]

    def async_show_form(self, *args: object, **kwargs: object) -> SubentryFlowResult:
        return super().async_show_form(*args, **kwargs)  # type: ignore[misc]

    def async_abort(self, *args: object, **kwargs: object) -> SubentryFlowResult:
        return super().async_abort(*args, **kwargs)  # type: ignore[misc]

    def async_create_entry(self, *args: object, **kwargs: object) -> SubentryFlowResult:
        return super().async_create_entry(*args, **kwargs)  # type: ignore[misc]

    def async_show_progress(
        self, *args: object, **kwargs: object
    ) -> SubentryFlowResult:
        return super().async_show_progress(*args, **kwargs)  # type: ignore[misc]

    def async_show_progress_done(
        self, *args: object, **kwargs: object
    ) -> SubentryFlowResult:
        return super().async_show_progress_done(*args, **kwargs)  # type: ignore[misc]

    def async_get_progress_task(
        self,
    ) -> asyncio.Task[Any] | None:
        return super().async_get_progress_task()  # type: ignore[misc]

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult: ...
    def _provider_already_configured(self, data: Mapping[str, Any]) -> bool: ...

    # ---- mixin attributes ----
    _wizard_catalog: CompactCatalog | None
    _wizard_catalog_error: str | None
    _wizard_connection_data: dict[str, Any]
    _wizard_connection_options: dict[str, Any]
    _wizard_driver: str | None
    _wizard_filters: ModelFilterOptions
    _wizard_models: tuple[CatalogModelOption, ...]
    _wizard_provider: CatalogProviderOption | None
    _wizard_selected_models: tuple[CatalogModelOption, ...]

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
        from ..models.provider_validation import ProviderValidationError

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
            data = _normalise_provider_data(data_input, self._wizard_connection_options)
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
            provider_secret_header_keys=self._wizard_connection_data.get(
                CONF_PROVIDER_SECRET_HEADER_KEYS
            ),
            provider_extra_body=dict(extra_body)
            if isinstance(extra_body, Mapping)
            else None,
        )
        return self.async_create_entry(title=str(data[CONF_NAME]), data=data)
