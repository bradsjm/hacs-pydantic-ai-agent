"""Config subentry flow handlers for Pydantic AI Agent."""

# ruff: noqa: F403, F405

from __future__ import annotations

from .common import (
    Any,
    CONF_AGENT_NAME,
    CONF_MCP_SERVER_IDS,
    CONF_PRIMARY_MODEL_REF,
    CONF_SKILLS,
    CONF_SKILLS_FOLDER,
    ConfigEntryState,
    ConfigSubentryFlow,
    DEFAULT_SKILLS_FOLDER,
    ProviderValidationError,
    SOURCE_USER,
    SubentryFlowResult,
    _SECTION_EXTERNAL_TOOLS,
    _SECTION_SKILLS,
    _conversation_data_from_user_input,
    _conversation_schema,
    _flatten_section_data,
    _model_profile_select_options,
    _normalise_skills_folder,
    _provider_validation_placeholders,
    _selected_mcp_server_error,
    _selected_model_profile_error,
    _skill_source,
    _validate_skills_folder,
    async_available_skills,
    default_conversation_options,
)

class ConversationSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing conversation subentries."""

    _options: dict[str, Any]
    _pending_conversation_data: dict[str, Any]

    @property
    def _is_new(self) -> bool:
        """Return if this flow creates a new subentry."""
        return self.source == SOURCE_USER

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add a conversation subentry."""
        self._options = default_conversation_options()
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure a conversation subentry."""
        self._options = self._get_reconfigure_subentry().data.copy()
        return await self.async_step_init(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage conversation options."""
        entry = self._get_entry()
        if entry.state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")
        if not _model_profile_select_options(entry):
            return self.async_abort(reason="no_models_configured")
        available_skills = await async_available_skills(self.hass, self._options)

        if user_input is not None:
            flat_user_input = _flatten_section_data(
                user_input, (_SECTION_EXTERNAL_TOOLS, _SECTION_SKILLS)
            )
            try:
                _validate_skills_folder(
                    self.hass,
                    flat_user_input.get(CONF_SKILLS_FOLDER, DEFAULT_SKILLS_FOLDER),
                )
            except ProviderValidationError as err:
                return self.async_show_form(
                    step_id="init",
                    data_schema=_conversation_schema(
                        self.hass,
                        self._options | flat_user_input,
                        entry,
                        available_skills,
                    ),
                    errors={"base": err.reason},
                    description_placeholders=_provider_validation_placeholders(err),
                )
            if _skill_source(flat_user_input) != _skill_source(self._options):
                refreshed_options = dict(flat_user_input)
                refreshed_options[CONF_SKILLS_FOLDER] = _normalise_skills_folder(
                    refreshed_options.get(CONF_SKILLS_FOLDER)
                )
                refreshed_options.pop(CONF_SKILLS, None)
                self._options = refreshed_options
                refreshed_skills = await async_available_skills(
                    self.hass, refreshed_options
                )
                return self.async_show_form(
                    step_id="init",
                    data_schema=_conversation_schema(
                        self.hass,
                        refreshed_options,
                        entry,
                        refreshed_skills,
                    ),
                    errors={"base": "skills_refreshed"},
                )
            available_skills = await async_available_skills(self.hass, flat_user_input)
            data = _conversation_data_from_user_input(
                flat_user_input,
                self._options,
                available_skills=available_skills,
            )
            if model_error := _selected_model_profile_error(self.hass, entry, data):
                return self.async_show_form(
                    step_id="init",
                    data_schema=_conversation_schema(
                        self.hass,
                        self._options | data,
                        entry,
                        available_skills,
                    ),
                    errors={CONF_PRIMARY_MODEL_REF: model_error},
                )
            if mcp_error := _selected_mcp_server_error(entry, data):
                return self.async_show_form(
                    step_id="init",
                    data_schema=_conversation_schema(
                        self.hass,
                        self._options | data,
                        entry,
                        available_skills,
                    ),
                    errors={CONF_MCP_SERVER_IDS: mcp_error},
                )
            return self._async_finish_conversation_options(data)

        return self.async_show_form(
            step_id="init",
            data_schema=_conversation_schema(
                self.hass, self._options, entry, available_skills
            ),
        )

    def _async_finish_conversation_options(
        self,
        data: dict[str, Any],
    ) -> SubentryFlowResult:
        """Create or update the conversation subentry."""
        entry = self._get_entry()
        if self._is_new:
            return self.async_create_entry(
                title=data[CONF_AGENT_NAME],
                data=data,
            )
        return self.async_update_and_abort(
            entry,
            self._get_reconfigure_subentry(),
            title=data[CONF_AGENT_NAME],
            data=data,
        )
