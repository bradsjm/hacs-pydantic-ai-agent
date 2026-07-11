"""Config subentry flow handlers for Pydantic AI Agent."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from homeassistant.config_entries import (
    SOURCE_USER,
    ConfigEntryState,
    ConfigSubentryFlow,
    SubentryFlowResult,
)

from .common import (
    _SECTION_EXTERNAL_TOOLS,
    _SECTION_FALLBACK_MODELS,
    _SECTION_HASS_CONTROL,
    _SECTION_RUN_SETTINGS,
    _SECTION_SKILLS,
    CONF_AI_TASK_NAME,
    CONF_PRIMARY_MODEL_REF,
    CONF_TODO_LIST_ENTITY_ID,
    RunSettingsValidationError,
    _agent_form_suggested_values,
    _ai_task_data_from_user_input,
    _ai_task_data_schema,
    _model_profile_select_options,
    _selected_model_profile_error,
    _selected_todo_workspace_error,
)
from .generated_titles import DEFAULT_AI_TASK_TITLE_SUFFIX, generated_default_title
from .helpers import _flatten_section_data
from .mcp_helpers import _selected_mcp_server_error
from .skill_helpers import _selected_skill_error


class AITaskDataSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing AI task data subentries."""

    _options: dict[str, Any]

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
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure an AI task data subentry."""
        subentry = self._get_reconfigure_subentry()
        self._options = subentry.data.copy()
        self._options.setdefault(CONF_AI_TASK_NAME, subentry.title)
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
                    entry,
                )
            except RunSettingsValidationError as err:
                return self._async_show_init_form(flat_user_input, err.errors)
            if model_error := _selected_model_profile_error(self.hass, entry, data):
                return self._async_show_init_form(
                    data, {CONF_PRIMARY_MODEL_REF: model_error}
                )
            if todo_error := _selected_todo_workspace_error(self.hass, data):
                return self._async_show_init_form(
                    data, {CONF_TODO_LIST_ENTITY_ID: todo_error}
                )
            if skill_error := _selected_skill_error(entry, data):
                return self._async_show_init_form(data, {"base": skill_error})
            if mcp_error := _selected_mcp_server_error(entry, data):
                return self._async_show_init_form(data, {"base": mcp_error})
            return self._async_finish_ai_task_options(data)

        return self._async_show_init_form({})

    def _async_show_init_form(
        self, data: Mapping[str, Any], errors: dict[str, str] | None = None
    ) -> SubentryFlowResult:
        """Render the AI task options form with suggested values."""
        options = self._options | dict(data)
        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                _ai_task_data_schema(self.hass, options, self._get_entry()),
                _agent_form_suggested_values(options, self.hass),
            ),
            errors=errors,
        )

    def _async_finish_ai_task_options(
        self,
        data: dict[str, Any],
    ) -> SubentryFlowResult:
        """Create or update the AI task subentry."""
        if self._is_new:
            return self.async_create_entry(title=data[CONF_AI_TASK_NAME], data=data)
        return self.async_update_and_abort(
            self._get_entry(),
            self._get_reconfigure_subentry(),
            title=data[CONF_AI_TASK_NAME],
            data=data,
        )
