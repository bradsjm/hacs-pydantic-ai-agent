"""Model availability management mixin for provider config flows."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any, Literal

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry, ConfigSubentry, SubentryFlowResult
from homeassistant.core import HomeAssistant

from ..const import (
    CONF_DISCOVERED,
    CONF_ENABLED,
    CONF_MODEL,
    CONF_MODEL_PRICING,
    CONF_MODEL_PROFILES,
    CONF_OPENAI_SUPPORTS_ENCRYPTED_REASONING_CONTENT,
    CONF_OPENAI_SUPPORTS_STRICT_TOOL_DEFINITION,
    CONF_PROVIDER_METADATA,
    CONF_STRUCTURED_OUTPUT_SUPPORT,
    CONF_SUPPORTS_TOOLS,
    CONF_THINKING_SUPPORT,
)
from ..model_profiles import provider_model_profiles
from ._profile_helpers import (
    _referenced_provider_profile_ids,
)
from ._provider_data import (
    _cached_provider_model_names,
    _provider_custom_model_names,
    _store_provider_model_cache,
)
from ._provider_flow_helpers import (
    _catalog_model_pricing,
    _custom_model_options,
    _discovered_model_options,
)
from .provider_wizard.const import (
    CONF_CATALOG_PROVIDER_ID,
    CUSTOM_PROVIDER_ID,
)
from .provider_wizard.filters import ModelFilterOptions, filtered_models
from .provider_wizard.flow import (
    build_model_profiles,
)
from .provider_wizard.models_dev import CatalogLoadError
from .provider_wizard.schemas import (
    filters_from_user_input,
    model_filter_schema,
    needs_model_filter_step,
)
from .provider_wizard.types import (
    CatalogModelOption,
    CatalogProviderOption,
)

_LOGGER = logging.getLogger(__name__)
_DISCOVERED_PROVIDER_ID = "__provider_discovery__"


class ProviderModelManagementMixin:
    """Mixin for provider model availability management steps.

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

    def async_show_progress(
        self, *args: object, **kwargs: object
    ) -> SubentryFlowResult:
        return super().async_show_progress(*args, **kwargs)  # type: ignore[misc]

    def async_show_progress_done(
        self, *args: object, **kwargs: object
    ) -> SubentryFlowResult:
        return super().async_show_progress_done(*args, **kwargs)  # type: ignore[misc]

    def async_get_progress_task(self) -> asyncio.Task[Any] | None:
        return super().async_get_progress_task()  # type: ignore[misc]

    async def async_step_manage_models(
        self, user_input: dict[str, object] | None = None
    ) -> SubentryFlowResult: ...

    # ---- mixin attributes ----
    _profile_flow_data: dict[str, Any]
    _profile_filter_provider: CatalogProviderOption | None
    _profile_filters: ModelFilterOptions
    _profile_models: tuple[CatalogModelOption, ...]
    _profile_refresh_error: str | None
    _manage_models_prepared: bool
    _manage_models_prepare_result: SubentryFlowResult | None

    def _start_manage_models_prepare(self) -> SubentryFlowResult:
        """Start progress for provider model-management preparation."""
        return self.async_show_progress(
            step_id="manage_models_prepare",
            progress_action="discover_models",
            progress_task=self.hass.async_create_task(
                self._async_prepare_manage_models_flow()
            ),
        )

    async def async_step_manage_models_prepare(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Finish provider model-management preparation progress."""
        del user_input
        task = self.async_get_progress_task()
        if task is not None and not task.done():
            return self.async_show_progress(
                step_id="manage_models_prepare",
                progress_action="discover_models",
                progress_task=task,
            )
        self._manage_models_prepare_result = None
        if task is not None:
            self._manage_models_prepare_result = await task
        return self.async_show_progress_done(
            next_step_id="manage_models_prepare_finish"
        )

    async def async_step_manage_models_prepare_finish(
        self, user_input: dict[str, object] | None = None
    ) -> SubentryFlowResult:
        """Continue after provider model preparation progress."""
        del user_input
        self._manage_models_prepared = True
        if result := self._manage_models_prepare_result:
            self._manage_models_prepare_result = None
            return result
        return await self.async_step_manage_models()

    async def _async_prepare_manage_models_entry(self) -> SubentryFlowResult | None:
        """Prepare manage-models state before rendering or saving."""
        if result := self._manage_models_prepare_result:
            self._manage_models_prepare_result = None
            return result
        if not self._manage_models_prepared:
            return self._start_manage_models_prepare()
        if not getattr(self, "_profile_flow_data", None):
            result = await self._async_prepare_manage_models_flow()
            if result is not None:
                return result
        self._manage_models_prepared = True
        return None

    async def _async_prepare_manage_models_flow(self) -> SubentryFlowResult | None:
        """Prepare catalog-backed model availability management."""
        data = dict(self._get_reconfigure_subentry().data)
        self._profile_flow_data = data
        self._profile_filters = ModelFilterOptions()
        if CONF_PROVIDER_METADATA not in data and (
            discovered_models := await self._async_discovered_models_for_manage(data)
        ):
            self._profile_filter_provider = None
            self._profile_models = discovered_models
            return None
        provider, models = await self._async_catalog_models_for_provider_data(data)
        if provider is None or not models:
            if discovered_models := await self._async_discovered_models_for_manage(
                data
            ):
                self._profile_filter_provider = None
                self._profile_models = discovered_models
                return None
            if _provider_custom_model_names(data):
                self._profile_filter_provider = None
                self._profile_models = ()
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
            from ..provider_validation import async_list_provider_model_names

            try:
                model_names = await async_list_provider_model_names(self.hass, data)
            except Exception:
                _LOGGER.warning("Unable to list provider models for model management")
                return ()
            _store_provider_model_cache(self._profile_flow_data, model_names)
        return _discovered_model_options(model_names)

    async def _async_catalog_models_for_provider_data(
        self, data: Mapping[str, Any]
    ) -> tuple[CatalogProviderOption | None, tuple[CatalogModelOption, ...]]:
        """Return catalog provider models matching stored provider metadata."""
        from .provider_wizard.catalog_cache import catalog_manager

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
                        & {
                            model.id
                            for model in catalog.models_for_provider(candidate.id)
                        }
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
        from ..model_profiles import model_profile_display_name

        models_by_id = {model.id: model for model in self._profile_models}
        managed_models = {
            model.id: model
            for model in filtered_models(self._profile_models, self._profile_filters)
        }
        for model in _custom_model_options(
            _provider_custom_model_names(self._current_profile_flow_data())
        ):
            managed_models.setdefault(model.id, model)
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
                managed_models[model_name] = CatalogModelOption(
                    id=model_name,
                    name=model_profile_display_name(profile) or model_name,
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
                    thinking_support=str(
                        profile.get(CONF_THINKING_SUPPORT, "none")
                    ),
                    structured_output_support=str(
                        profile.get(CONF_STRUCTURED_OUTPUT_SUPPORT, "none")
                    ),
                    supports_tools=bool(profile.get(CONF_SUPPORTS_TOOLS, True)),
                    openai_supports_strict_tool_definition=bool(
                        profile.get(
                            CONF_OPENAI_SUPPORTS_STRICT_TOOL_DEFINITION,
                            True,
                        )
                    ),
                    openai_supports_encrypted_reasoning_content=bool(
                        profile.get(
                            CONF_OPENAI_SUPPORTS_ENCRYPTED_REASONING_CONTENT,
                            False,
                        )
                    ),
                )
        return tuple(managed_models.values())

    def _sync_selected_model_profiles(
        self,
        selected_models: tuple[CatalogModelOption, ...],
        managed_models: tuple[CatalogModelOption, ...],
        custom_model_ids: set[str],
    ) -> str | None:
        """Sync enabled model profiles for one availability selection."""
        data = self._current_profile_flow_data()
        profiles = data.get(CONF_MODEL_PROFILES, {})
        if not isinstance(profiles, Mapping):
            profiles = {}
        selected_model_ids = {model.id for model in selected_models}
        managed_model_ids = {model.id for model in managed_models}
        managed_models_by_id = {model.id: model for model in managed_models}
        referenced_profile_ids = _referenced_provider_profile_ids(
            self._get_entry(), self._get_reconfigure_subentry().subentry_id
        )
        synced_profiles: dict[str, dict[str, Any]] = {}
        selected_existing_model_ids: set[str] = set()
        for profile_id, profile in profiles.items():
            sync_result = self._sync_existing_model_profile(
                profile_id,
                profile,
                selected_model_ids=selected_model_ids,
                managed_model_ids=managed_model_ids,
                managed_models_by_id=managed_models_by_id,
                referenced_profile_ids=referenced_profile_ids,
                custom_model_ids=custom_model_ids,
            )
            if sync_result == "model_profile_in_use":
                return "model_profile_in_use"
            if sync_result is None:
                continue
            profile_data, model_name, selected_existing = sync_result
            if selected_existing:
                selected_existing_model_ids.add(model_name)
            synced_profiles[profile_id] = profile_data
        missing_selected_models = tuple(
            model
            for model in selected_models
            if model.id not in selected_existing_model_ids
        )
        missing_profiles = build_model_profiles(missing_selected_models)
        custom_model_ids_set = {
            model.id
            for model in missing_selected_models
            if model.provider_id == CUSTOM_PROVIDER_ID
        }
        for profile in missing_profiles.values():
            if profile[CONF_MODEL] in custom_model_ids_set:
                profile[CONF_DISCOVERED] = False
        synced_profiles.update(missing_profiles)
        data[CONF_MODEL_PROFILES] = synced_profiles
        return None

    def _sync_existing_model_profile(
        self,
        profile_id: object,
        profile: object,
        *,
        selected_model_ids: set[str],
        managed_model_ids: set[str],
        managed_models_by_id: dict[str, CatalogModelOption],
        referenced_profile_ids: set[str],
        custom_model_ids: set[str],
    ) -> tuple[dict[str, Any], str, bool] | Literal["model_profile_in_use"] | None:
        """Return synced existing profile data or an in-use error marker."""
        if not isinstance(profile_id, str) or not isinstance(profile, Mapping):
            return None
        model_name = profile.get(CONF_MODEL)
        if not isinstance(model_name, str) or not model_name.strip():
            return None
        profile_data = dict(profile)
        profile_data["id"] = profile_id
        if CONF_MODEL_PRICING not in profile_data:
            model_pricing = _catalog_model_pricing(managed_models_by_id.get(model_name))
            if model_pricing:
                profile_data[CONF_MODEL_PRICING] = model_pricing
        if model_name in selected_model_ids:
            profile_data[CONF_ENABLED] = True
            return profile_data, model_name, True
        if model_name in managed_model_ids:
            if profile_id in referenced_profile_ids:
                return "model_profile_in_use"
            profile_data[CONF_ENABLED] = False
            return profile_data, model_name, False
        if bool(profile.get(CONF_DISCOVERED, False)) or model_name in custom_model_ids:
            return profile_data, model_name, False
        if profile_id in referenced_profile_ids:
            return "model_profile_in_use"
        return None

    def _profile_needs_model_filter_step(self) -> bool:
        """Return if profile reconfiguration should show model filters first."""
        if self._profile_filter_provider is None or not self._profile_models:
            return False
        default_models = filtered_models(self._profile_models, self._profile_filters)
        return len(default_models) < len(
            self._profile_models
        ) or needs_model_filter_step(self._profile_models)

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
