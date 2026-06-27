"""Workspace config flow for Pydantic AI Agent."""

from __future__ import annotations

from .ai_task_flow import AITaskDataSubentryFlowHandler
from .common import (
    CONF_NAME,
    DOMAIN,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_MCP_SERVER,
    SUBENTRY_TYPE_PROVIDER,
    SUBENTRY_TYPE_SKILL,
    Any,
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    callback,
)
from .conversation_flow import ConversationSubentryFlowHandler
from .generated_titles import (
    DEFAULT_WORKSPACE_TITLE_SUFFIX,
    generated_default_title,
)
from .mcp_server_flow import MCPServerSubentryFlowHandler
from .provider_flow import ProviderSubentryFlowHandler
from .skill_flow import SkillSubentryFlowHandler
from .workspace_helpers import (
    _base_schema,
    _normalise_workspace_data,
    _provider_form_suggested_values,
)


class PydanticAIAgentConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Pydantic AI Agent."""

    VERSION = 2
    MINOR_VERSION = 4
    _default_workspace_name: str | None = None

    def _async_update_workspace_and_abort(
        self, entry: ConfigEntry, data: dict[str, Any]
    ) -> ConfigFlowResult:
        """Update a workspace entry using the active reload mechanism."""
        if entry.update_listeners:
            return self.async_update_and_abort(entry, data=data)
        return self.async_update_reload_and_abort(entry, data=data)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Create a new workspace config entry."""
        errors: dict[str, str] = {}

        if user_input is not None:
            data = _normalise_workspace_data(user_input)
            return self.async_create_entry(title=data[CONF_NAME], data=data)

        return self.async_show_form(
            step_id="user",
            data_schema=self.add_suggested_values_to_schema(
                _base_schema(
                    user_input or {CONF_NAME: self._new_workspace_default_name()}
                ),
                _provider_form_suggested_values(user_input),
            ),
            errors=errors,
        )

    def _new_workspace_default_name(self) -> str:
        """Return the generated default name for this workspace flow."""
        if self._default_workspace_name is None:
            self._default_workspace_name = generated_default_title(
                DEFAULT_WORKSPACE_TITLE_SUFFIX,
                (entry.title for entry in self._async_current_entries()),
            )
        return self._default_workspace_name

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Reconfigure workspace metadata and Logfire settings."""
        entry = self._get_reconfigure_entry()
        if user_input is None:
            entry_data = dict(entry.data)
            return self.async_show_form(
                step_id="reconfigure",
                data_schema=self.add_suggested_values_to_schema(
                    _base_schema(entry_data),
                    _provider_form_suggested_values(entry_data),
                ),
            )

        data = _normalise_workspace_data(user_input)
        return self._async_update_workspace_and_abort(entry, data)

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, _config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {
            SUBENTRY_TYPE_PROVIDER: ProviderSubentryFlowHandler,
            SUBENTRY_TYPE_CONVERSATION: ConversationSubentryFlowHandler,
            SUBENTRY_TYPE_AI_TASK: AITaskDataSubentryFlowHandler,
            SUBENTRY_TYPE_MCP_SERVER: MCPServerSubentryFlowHandler,
            SUBENTRY_TYPE_SKILL: SkillSubentryFlowHandler,
        }
