"""Config subentry flow handlers for Pydantic AI Agent."""

# ruff: noqa: F403, F405

from __future__ import annotations

from .common import (
    Any,
    CONF_AGENT_NAME,
    CONF_MCP_SERVER_IDS,
    CONF_PRIMARY_MODEL_REF,
    ConfigEntryState,
    ConfigSubentryFlow,
    SOURCE_USER,
    SubentryFlowResult,
    _SECTION_EXTERNAL_TOOLS,
    _SECTION_FALLBACK_MODELS,
    _SECTION_HASS_CONTROL,
    _SECTION_SKILLS,
    _agent_form_suggested_values,
    _conversation_data_from_user_input,
    _conversation_schema,
    _flatten_section_data,
    _model_profile_select_options,
    _selected_mcp_server_error,
    _selected_model_profile_error,
    _selected_skill_error,
    default_conversation_options,
)
from ..generated_titles import DEFAULT_AGENT_TITLE_SUFFIX, generated_default_title


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
        self._options[CONF_AGENT_NAME] = generated_default_title(
            DEFAULT_AGENT_TITLE_SUFFIX,
            (subentry.title for subentry in self._get_entry().subentries.values()),
        )
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
        if user_input is not None:
            flat_user_input = _flatten_section_data(
                user_input,
                (
                    _SECTION_EXTERNAL_TOOLS,
                    _SECTION_FALLBACK_MODELS,
                    _SECTION_HASS_CONTROL,
                    _SECTION_SKILLS,
                ),
            )
            data = _conversation_data_from_user_input(
                flat_user_input,
                self._options,
            )
            if model_error := _selected_model_profile_error(self.hass, entry, data):
                return self.async_show_form(
                    step_id="init",
                    data_schema=self.add_suggested_values_to_schema(
                        _conversation_schema(
                            self.hass,
                            self._options | data,
                            entry,
                        ),
                        _agent_form_suggested_values(self._options | data, self.hass),
                    ),
                    errors={CONF_PRIMARY_MODEL_REF: model_error},
                )
            if mcp_error := _selected_mcp_server_error(entry, data):
                return self.async_show_form(
                    step_id="init",
                    data_schema=self.add_suggested_values_to_schema(
                        _conversation_schema(
                            self.hass,
                            self._options | data,
                            entry,
                        ),
                        _agent_form_suggested_values(self._options | data, self.hass),
                    ),
                    errors={CONF_MCP_SERVER_IDS: mcp_error},
                )
            if skill_error := _selected_skill_error(entry, data):
                return self.async_show_form(
                    step_id="init",
                    data_schema=self.add_suggested_values_to_schema(
                        _conversation_schema(
                            self.hass,
                            self._options | data,
                            entry,
                        ),
                        _agent_form_suggested_values(self._options | data, self.hass),
                    ),
                    errors={"base": skill_error},
                )
            return self._async_finish_conversation_options(data)

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                _conversation_schema(self.hass, self._options, entry),
                _agent_form_suggested_values(self._options, self.hass),
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
