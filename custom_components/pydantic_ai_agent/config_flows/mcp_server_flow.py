"""Config subentry flow handlers for MCP servers."""

from homeassistant.helpers.selector import SelectOptionDict

from ..const import CONF_MCP_ALLOWED_TOOLS, CONF_MCP_HEADERS, CONF_MCP_URL
from ..mcp import (
    MCPValidationError,
    async_discover_mcp_tools_from_config,
    async_validate_mcp_url,
    parse_allowed_tools,
)
from .common import (
    CONF_NAME,
    SOURCE_USER,
    Any,
    ConfigEntryState,
    ConfigSubentryFlow,
    SubentryFlowResult,
    vol,
)
from .helpers import _flatten_section_data
from .mcp_helpers import (
    _SECTION_ADVANCED_MCP,
    _mcp_server_data_from_user_input,
    _mcp_server_schema,
    _mcp_tool_options,
    _mcp_tools_schema,
    _mcp_url_already_configured,
    _mcp_validation_placeholders,
)


class MCPServerSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing remote MCP server subentries."""

    _options: dict[str, Any]
    _pending_data: dict[str, Any]
    _pending_form_data: dict[str, Any]
    _pending_mcp_error: tuple[str, str, dict[str, str]] | None
    _tool_options: list[SelectOptionDict]

    @property
    def _is_new(self) -> bool:
        """Return if this flow creates a new subentry."""
        return self.source == SOURCE_USER

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add an MCP server subentry."""
        self._options = {}
        self._pending_data = {}
        self._pending_form_data = {}
        self._pending_mcp_error = None
        self._tool_options = []
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure an MCP server subentry."""
        self._options = self._get_reconfigure_subentry().data.copy()
        self._pending_data = {}
        self._pending_form_data = {}
        self._pending_mcp_error = None
        self._tool_options = []
        return await self.async_step_init(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage remote MCP server options."""
        entry = self._get_entry()
        if entry.state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if user_input is None:
            return self.async_show_form(
                step_id="init",
                data_schema=_mcp_server_schema(self._options),
            )

        flat_user_input = _flatten_section_data(user_input, (_SECTION_ADVANCED_MCP,))
        errors: dict[str, str] = {}
        description_placeholders: dict[str, str] = {}
        try:
            data = _mcp_server_data_from_user_input(flat_user_input)
        except MCPValidationError as err:
            errors[CONF_MCP_URL] = err.reason
            description_placeholders = _mcp_validation_placeholders(err)
            data = dict(flat_user_input)
        except vol.Invalid as err:
            reason = str(err) or "invalid_mcp_headers"
            if reason == "invalid_mcp_tools":
                errors[CONF_MCP_ALLOWED_TOOLS] = reason
            else:
                errors[CONF_MCP_HEADERS] = "invalid_mcp_headers"
            data = dict(flat_user_input)
        else:
            form_data = (
                self._options
                | data
                | {CONF_MCP_HEADERS: flat_user_input.get(CONF_MCP_HEADERS, "")}
            )
            current_subentry_id = None
            if not self._is_new:
                current_subentry_id = self._get_reconfigure_subentry().subentry_id
            self._pending_form_data = form_data
            self._pending_mcp_error = None
            task = self.hass.async_create_task(
                self._async_validate_mcp_server(data, current_subentry_id)
            )
            return self.async_show_progress(
                step_id="mcp_validation_progress",
                progress_action="validate_mcp",
                progress_task=task,
            )

        return self.async_show_form(
            step_id="init",
            data_schema=_mcp_server_schema(self._options | data),
            errors=errors,
            description_placeholders=description_placeholders,
        )

    async def _async_validate_mcp_server(
        self, data: dict[str, Any], current_subentry_id: str | None
    ) -> tuple[
        dict[str, Any] | None,
        list[SelectOptionDict],
        tuple[str, str, dict[str, str]] | None,
    ]:
        """Validate MCP connectivity and return discovered tool options."""
        try:
            data = dict(data)
            data[CONF_MCP_URL] = await async_validate_mcp_url(
                self.hass, data[CONF_MCP_URL]
            )
            tools = await async_discover_mcp_tools_from_config(
                self.hass,
                data,
                server_id=current_subentry_id or data[CONF_NAME],
                apply_allowlist=False,
            )
            existing_allowed_tools = parse_allowed_tools(
                self._options.get(CONF_MCP_ALLOWED_TOOLS)
            )
            tool_options = _mcp_tool_options(tools, existing_allowed_tools)
            if not tool_options:
                raise MCPValidationError(
                    "no_mcp_tools",
                    "The MCP server did not expose any tools.",
                )
        except MCPValidationError as err:
            target = CONF_MCP_URL if err.reason == "invalid_mcp_url" else "base"
            return None, [], (target, err.reason, _mcp_validation_placeholders(err))
        return data, tool_options, None

    async def async_step_mcp_validation_progress(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Finish MCP validation progress."""
        task = self.async_get_progress_task()
        if task is not None and not task.done():
            return self.async_show_progress(
                step_id="mcp_validation_progress",
                progress_action="validate_mcp",
                progress_task=task,
            )
        data, tool_options, error = (None, [], None) if task is None else task.result()
        self._pending_mcp_error = error
        if data is not None:
            self._pending_data = data
            self._tool_options = tool_options
        return self.async_show_progress_done(next_step_id="mcp_validation_finish")

    async def async_step_mcp_validation_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Advance to tool selection or show MCP validation errors."""
        if self._pending_mcp_error is not None:
            target, reason, placeholders = self._pending_mcp_error
            return self.async_show_form(
                step_id="init",
                data_schema=_mcp_server_schema(self._pending_form_data),
                errors={target: reason},
                description_placeholders=placeholders,
            )
        current_subentry_id = None
        if not self._is_new:
            current_subentry_id = self._get_reconfigure_subentry().subentry_id
        if _mcp_url_already_configured(
            self._get_entry(), self._pending_data[CONF_MCP_URL], current_subentry_id
        ):
            return self.async_abort(reason="already_configured")
        return await self.async_step_tools()

    async def async_step_tools(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Select the discovered MCP tools to allow."""
        if not self._pending_data or not self._tool_options:
            return await self.async_step_init()

        tool_names = [option["value"] for option in self._tool_options]
        default_tool_names = (
            tool_names
            if self._is_new
            else [
                tool_name
                for tool_name in parse_allowed_tools(
                    self._options.get(CONF_MCP_ALLOWED_TOOLS)
                )
                if tool_name in tool_names
            ]
        )

        if user_input is None:
            return self.async_show_form(
                step_id="tools",
                data_schema=_mcp_tools_schema(self._tool_options, default_tool_names),
            )

        allowed_tools = [
            tool_name
            for tool_name in parse_allowed_tools(user_input.get(CONF_MCP_ALLOWED_TOOLS))
            if tool_name in tool_names
        ]
        if not allowed_tools:
            return self.async_show_form(
                step_id="tools",
                data_schema=_mcp_tools_schema(self._tool_options, default_tool_names),
                errors={CONF_MCP_ALLOWED_TOOLS: "mcp_tools_not_allowlisted"},
            )

        data = {**self._pending_data, CONF_MCP_ALLOWED_TOOLS: allowed_tools}
        if self._is_new:
            return self.async_create_entry(title=data[CONF_NAME], data=data)
        return self.async_update_and_abort(
            self._get_entry(),
            self._get_reconfigure_subentry(),
            title=data[CONF_NAME],
            data=data,
        )
