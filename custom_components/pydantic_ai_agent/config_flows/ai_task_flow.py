"""Config subentry flow handlers for Pydantic AI Agent."""

from __future__ import annotations

from ..generated_titles import DEFAULT_AI_TASK_TITLE_SUFFIX, generated_default_title
from ..model_settings import validation_probe_model_settings
from .common import (
    _LOGGER,
    _SECTION_EXTERNAL_TOOLS,
    _SECTION_FALLBACK_MODELS,
    _SECTION_HASS_CONTROL,
    _SECTION_RUN_SETTINGS,
    _SECTION_SKILLS,
    CONF_AI_TASK_NAME,
    CONF_FALLBACK_MODEL_REFS,
    CONF_MODEL,
    CONF_MODEL_SETTINGS,
    CONF_OUTPUT_MODE,
    CONF_PRIMARY_MODEL_REF,
    CONF_TODO_LIST_ENTITY_ID,
    SOURCE_USER,
    SUBENTRY_TYPE_PROVIDER,
    Any,
    ConfigEntryState,
    ConfigSubentryFlow,
    Mapping,
    ProviderValidationError,
    RunSettingsValidationError,
    SubentryFlowResult,
    _agent_form_suggested_values,
    _ai_task_data_from_user_input,
    _ai_task_data_schema,
    _log_provider_validation_failure,
    _model_profile_select_options,
    _normalise_fallback_model_refs,
    _provider_validation_placeholders,
    _selected_model_profile_error,
    _selected_todo_workspace_error,
    async_probe_model,
    parse_model_profile_ref,
    provider_model_profiles,
)
from .helpers import _flatten_section_data
from .mcp_helpers import _selected_mcp_server_error
from .skill_helpers import _selected_skill_error


class AITaskDataSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing AI task data subentries."""

    _options: dict[str, Any]
    _pending_ai_task_data: dict[str, Any]
    _pending_ai_task_error: tuple[str, str, dict[str, str]] | None

    @property
    def _is_new(self) -> bool:
        """Return if this flow creates a new subentry."""
        return self.source == SOURCE_USER

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add an AI task data subentry."""
        self._options = {
            CONF_AI_TASK_NAME: generated_default_title(
                DEFAULT_AI_TASK_TITLE_SUFFIX,
                (subentry.title for subentry in self._get_entry().subentries.values()),
            )
        }
        self._pending_ai_task_data = {}
        self._pending_ai_task_error = None
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure an AI task data subentry."""
        subentry = self._get_reconfigure_subentry()
        self._options = subentry.data.copy()
        self._options.setdefault(CONF_AI_TASK_NAME, subentry.title)
        self._pending_ai_task_data = {}
        self._pending_ai_task_error = None
        return await self.async_step_init(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage AI task model options."""
        entry = self._get_entry()
        if entry.state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")
        if not _model_profile_select_options(entry):
            return self.async_abort(reason="no_models_configured")
        if user_input is not None:
            flat_user_input = _flatten_section_data(
                user_input,
                (
                    _SECTION_EXTERNAL_TOOLS,
                    _SECTION_FALLBACK_MODELS,
                    _SECTION_HASS_CONTROL,
                    _SECTION_RUN_SETTINGS,
                    _SECTION_SKILLS,
                ),
            )
            try:
                data = _ai_task_data_from_user_input(
                    flat_user_input,
                    self._options,
                )
            except RunSettingsValidationError as err:
                return self.async_show_form(
                    step_id="init",
                    data_schema=self.add_suggested_values_to_schema(
                        _ai_task_data_schema(
                            self.hass, self._options | flat_user_input, entry
                        ),
                        _agent_form_suggested_values(
                            self._options | flat_user_input, self.hass
                        ),
                    ),
                    errors=err.errors,
                )
            if model_error := _selected_model_profile_error(self.hass, entry, data):
                return self.async_show_form(
                    step_id="init",
                    data_schema=self.add_suggested_values_to_schema(
                        _ai_task_data_schema(self.hass, self._options | data, entry),
                        _agent_form_suggested_values(self._options | data, self.hass),
                    ),
                    errors={CONF_PRIMARY_MODEL_REF: model_error},
                )
            if todo_error := _selected_todo_workspace_error(self.hass, data):
                return self.async_show_form(
                    step_id="init",
                    data_schema=self.add_suggested_values_to_schema(
                        _ai_task_data_schema(self.hass, self._options | data, entry),
                        _agent_form_suggested_values(self._options | data, self.hass),
                    ),
                    errors={CONF_TODO_LIST_ENTITY_ID: todo_error},
                )
            if skill_error := _selected_skill_error(entry, data):
                return self.async_show_form(
                    step_id="init",
                    data_schema=self.add_suggested_values_to_schema(
                        _ai_task_data_schema(self.hass, self._options | data, entry),
                        _agent_form_suggested_values(self._options | data, self.hass),
                    ),
                    errors={"base": skill_error},
                )
            if mcp_error := _selected_mcp_server_error(entry, data):
                return self.async_show_form(
                    step_id="init",
                    data_schema=self.add_suggested_values_to_schema(
                        _ai_task_data_schema(self.hass, self._options | data, entry),
                        _agent_form_suggested_values(self._options | data, self.hass),
                    ),
                    errors={"base": mcp_error},
                )
            return await self._async_finish_ai_task_options(data)

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                _ai_task_data_schema(self.hass, self._options, entry),
                _agent_form_suggested_values(self._options, self.hass),
            ),
        )

    async def _async_finish_ai_task_options(
        self,
        data: dict[str, Any],
    ) -> SubentryFlowResult:
        """Probe the selected AI task model, then create or update the subentry."""
        entry = self._get_entry()
        if skill_error := _selected_skill_error(entry, data):
            return self.async_show_form(
                step_id="init",
                data_schema=_ai_task_data_schema(
                    self.hass, self._options | data, entry
                ),
                errors={"base": skill_error},
            )
        if mcp_error := _selected_mcp_server_error(entry, data):
            return self.async_show_form(
                step_id="init",
                data_schema=_ai_task_data_schema(
                    self.hass, self._options | data, entry
                ),
                errors={"base": mcp_error},
            )
        self._pending_ai_task_data = dict(data)
        self._pending_ai_task_error = None
        task = self.hass.async_create_task(self._async_probe_ai_task_options(data))
        return self.async_show_progress(
            step_id="ai_task_progress",
            progress_action="probe_model",
            progress_task=task,
        )

    async def _async_probe_ai_task_options(
        self, data: dict[str, Any]
    ) -> tuple[str, str, dict[str, str]] | None:
        """Return an AI task model validation error, if any."""
        entry = self._get_entry()
        current_model = ""
        try:
            profile_refs = [
                data[CONF_PRIMARY_MODEL_REF],
                *_normalise_fallback_model_refs(data.get(CONF_FALLBACK_MODEL_REFS, [])),
            ]
            for profile_ref in profile_refs:
                provider_subentry_id, profile_id = parse_model_profile_ref(profile_ref)
                provider_subentry = entry.subentries.get(provider_subentry_id)
                if (
                    provider_subentry is None
                    or provider_subentry.subentry_type != SUBENTRY_TYPE_PROVIDER
                ):
                    return "base", "model_profile_not_found", {}
                profile = provider_model_profiles(provider_subentry).get(profile_id)
                if not isinstance(profile, Mapping):
                    return "base", "model_profile_not_found", {}
                settings = profile.get(CONF_MODEL_SETTINGS)
                model_settings = validation_probe_model_settings(
                    settings if isinstance(settings, Mapping) else None, data
                )
                current_model = str(profile[CONF_MODEL])
                await async_probe_model(
                    self.hass,
                    provider_subentry.data,
                    current_model,
                    model_settings,
                    structured_output_mode=data[CONF_OUTPUT_MODE],
                )
        except ProviderValidationError as err:
            _log_provider_validation_failure(
                step="AI task subentry", model_name=current_model, err=err
            )
            return "base", err.reason, _provider_validation_placeholders(err)
        except Exception:
            _LOGGER.exception("Unexpected exception validating AI task model")
            return "base", "unknown", {}
        return None

    async def async_step_ai_task_progress(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Finish AI task validation progress."""
        task = self.async_get_progress_task()
        if task is not None and not task.done():
            return self.async_show_progress(
                step_id="ai_task_progress",
                progress_action="probe_model",
                progress_task=task,
            )
        self._pending_ai_task_error = None if task is None else task.result()
        return self.async_show_progress_done(next_step_id="ai_task_finish")

    async def async_step_ai_task_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Create/update the AI task or show the validation error."""
        data = self._pending_ai_task_data
        if self._pending_ai_task_error is not None:
            field, reason, placeholders = self._pending_ai_task_error
            return self.async_show_form(
                step_id="init",
                data_schema=self.add_suggested_values_to_schema(
                    _ai_task_data_schema(
                        self.hass, self._options | data, self._get_entry()
                    ),
                    _agent_form_suggested_values(self._options | data, self.hass),
                ),
                errors={field: reason},
                description_placeholders=placeholders,
            )
        if self._is_new:
            return self.async_create_entry(title=data[CONF_AI_TASK_NAME], data=data)
        return self.async_update_and_abort(
            self._get_entry(),
            self._get_reconfigure_subentry(),
            title=data[CONF_AI_TASK_NAME],
            data=data,
        )
