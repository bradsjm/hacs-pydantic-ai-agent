"""Config subentry flow handlers for MCP servers."""

import asyncio
import logging

from homeassistant.helpers.selector import SelectOptionDict

from ..const import (
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_CALL_CACHE_ENABLED,
    CONF_MCP_CALL_CACHE_TTL,
    CONF_MCP_DEFERRED_LOADING,
    CONF_MCP_HEADERS,
    CONF_MCP_INCLUDE_RETURN_SCHEMA,
    CONF_MCP_SECRET_HEADER_KEYS,
    CONF_MCP_TIMEOUT,
    CONF_MCP_TOOL_MODE,
    CONF_MCP_URL,
    DEFAULT_MCP_TIMEOUT,
    MCP_TOOL_MODE_SPECIFIED,
)
from ..mcp import (
    MCPValidationError,
    async_discover_mcp_tools_from_config,
    async_validate_mcp_url,
    parse_allowed_tools,
)
from ..mcp.entry_helpers import stored_mcp_tool_configuration
from .common import (
    CONF_NAME,
    SOURCE_USER,
    Any,
    ConfigEntryState,
    ConfigSubentryFlow,
    SubentryFlowResult,
    vol,
)
from .helpers import (
    _flatten_section_data_with_presence,
    _merge_submitted_optional_fields,
)
from .mcp_helpers import (
    _SECTION_ADVANCED_MCP,
    _mcp_server_data_from_user_input,
    _mcp_server_form_options,
    _mcp_server_schema,
    _mcp_tool_mode,
    _mcp_tool_options,
    _mcp_tools_schema,
    _mcp_url_already_configured,
    _mcp_validation_placeholders,
)

_LOGGER = logging.getLogger(__name__)


def _validation_failure_log_details(
    err: MCPValidationError,
) -> tuple[str, tuple[object, ...]]:
    """Return extra secret-safe details for MCP validation warning logs."""
    cause = err.__cause__
    if cause is None:
        return "reason=%s", (err.reason,)
    if isinstance(cause, ImportError):
        return (
            "reason=%s cause=%s message=%s",
            (err.reason, type(cause).__name__, str(cause)),
        )
    return "reason=%s cause=%s", (err.reason, type(cause).__name__)


class MCPServerSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing remote MCP server subentries."""

    _options: dict[str, Any]
    _pending_mcp_form_options: dict[str, Any]
    _pending_mcp_progress_action: str
    _pending_mcp_server_step_id: str
    _pending_mcp_server_submitted_keys: set[str]
    _pending_mcp_server_user_input: dict[str, Any]
    _pending_mcp_tool_options: list[SelectOptionDict]
    _pending_manage_tools_data: dict[str, Any]
    _pending_manage_tools_tool_options: list[SelectOptionDict]
    _pending_mcp_validation_error: tuple[str, str, dict[str, str]] | None
    _pending_mcp_validated_data: dict[str, Any] | None
    _manage_tools_prepared: bool

    @property
    def _is_new(self) -> bool:
        """Return if this flow creates a new subentry."""
        return self.source == SOURCE_USER

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        """Add an MCP server subentry."""
        self._options = {}
        self._manage_tools_prepared = False
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        """Reconfigure an MCP server subentry."""
        self._options = _mcp_server_form_options(self._get_reconfigure_subentry().data)
        self._manage_tools_prepared = False
        return await self.async_step_reconfigure_menu(user_input)

    async def async_step_reconfigure_menu(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        """Show the shallow MCP server management menu."""
        del user_input
        return self.async_show_menu(
            step_id="reconfigure_menu",
            menu_options=["edit_server", "manage_tools"],
        )

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        """Manage remote MCP server options."""
        return await self._async_server_form_step("init", user_input)

    async def async_step_edit_server(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        """Edit remote MCP server options."""
        return await self._async_server_form_step("edit_server", user_input)

    async def _async_server_form_step(self, step_id: str, user_input: dict[str, Any] | None) -> SubentryFlowResult:
        """Handle the MCP server create/edit form."""
        entry = self._get_entry()
        if entry.state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if user_input is None:
            return self.async_show_form(
                step_id=step_id,
                data_schema=_mcp_server_schema(self._options),
            )

        flat_user_input, submitted_keys = _flatten_section_data_with_presence(user_input, (_SECTION_ADVANCED_MCP,))
        form_options = _mcp_server_form_options(flat_user_input)
        errors: dict[str, str] = {}
        description_placeholders: dict[str, str] = {}
        try:
            previous_data = None if self._is_new else self._get_reconfigure_subentry().data
            data = _mcp_server_data_from_user_input(flat_user_input, previous_data)
        except MCPValidationError as err:
            errors[CONF_MCP_URL] = err.reason
            description_placeholders = _mcp_validation_placeholders(err)
        except vol.Invalid as err:
            reason = str(err) or "invalid_mcp_headers"
            if reason == "invalid_mcp_tools":
                errors[CONF_MCP_ALLOWED_TOOLS] = reason
            elif reason == "invalid_mcp_call_cache_ttl":
                errors[CONF_MCP_CALL_CACHE_TTL] = reason
            elif reason == "invalid_mcp_timeout":
                errors[CONF_MCP_TIMEOUT] = reason
            else:
                errors[CONF_MCP_HEADERS] = "invalid_mcp_headers"
        else:
            self._pending_mcp_form_options = self._storage_data_with_preserved_allowlist(data, submitted_keys)
            self._pending_mcp_progress_action = "validate_mcp_server"
            self._pending_mcp_server_step_id = step_id
            self._pending_mcp_server_submitted_keys = submitted_keys
            self._pending_mcp_server_user_input = flat_user_input
            return self.async_show_progress(
                step_id="validate_mcp_server_progress",
                progress_action=self._pending_mcp_progress_action,
                progress_task=self.hass.async_create_task(
                    self._async_validate_mcp_server(data, self._current_subentry_id())
                ),
            )

        return self.async_show_form(
            step_id=step_id,
            data_schema=_mcp_server_schema(form_options),
            errors=errors,
            description_placeholders=description_placeholders,
        )

    def _current_subentry_id(self) -> str | None:
        """Return the current reconfigure subentry id when present."""
        if self._is_new:
            return None
        return self._get_reconfigure_subentry().subentry_id

    def _show_server_form_error(
        self,
        step_id: str,
        form_data: dict[str, Any],
        error: tuple[str, str, dict[str, str]],
    ) -> SubentryFlowResult:
        """Return the MCP server form with one validation error."""
        target, reason, placeholders = error
        return self.async_show_form(
            step_id=step_id,
            data_schema=_mcp_server_schema(form_data),
            errors={target: reason},
            description_placeholders=placeholders,
        )

    def _storage_data_with_preserved_allowlist(
        self,
        validated_data: dict[str, Any],
        submitted_keys: set[str],
    ) -> dict[str, Any]:
        """Return validated storage data preserving values not edited in this step."""
        storage_data = dict(validated_data)
        if self._is_new:
            return storage_data

        existing_data = self._get_reconfigure_subentry().data
        if CONF_MCP_TOOL_MODE in existing_data:
            storage_data.setdefault(CONF_MCP_TOOL_MODE, existing_data[CONF_MCP_TOOL_MODE])
        if CONF_MCP_ALLOWED_TOOLS in existing_data:
            storage_data.setdefault(
                CONF_MCP_ALLOWED_TOOLS,
                existing_data.get(CONF_MCP_ALLOWED_TOOLS),
            )

        _merge_submitted_optional_fields(
            storage_data,
            existing_data=existing_data,
            validated_data=validated_data,
            submitted_keys=submitted_keys,
            keys=(
                CONF_MCP_HEADERS,
                CONF_MCP_SECRET_HEADER_KEYS,
                CONF_MCP_CALL_CACHE_ENABLED,
                CONF_MCP_CALL_CACHE_TTL,
                CONF_MCP_TIMEOUT,
                CONF_MCP_INCLUDE_RETURN_SCHEMA,
                CONF_MCP_DEFERRED_LOADING,
            ),
            derived_sources={CONF_MCP_SECRET_HEADER_KEYS: CONF_MCP_HEADERS},
        )
        return storage_data

    def _async_finish_server_form(
        self,
        validated_data: dict[str, Any],
        submitted_keys: set[str],
    ) -> SubentryFlowResult:
        """Create or update the MCP server after validation."""
        storage_data = self._storage_data_with_preserved_allowlist(validated_data, submitted_keys)
        if self._is_new:
            return self.async_create_entry(
                title=storage_data[CONF_NAME],
                data=storage_data,
            )
        return self.async_update_and_abort(
            self._get_entry(),
            self._get_reconfigure_subentry(),
            title=storage_data[CONF_NAME],
            data=storage_data,
        )

    async def async_step_validate_mcp_server_progress(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Finish MCP server validation progress."""
        del user_input
        task = self.async_get_progress_task()
        if task is not None and not task.done():
            return self.async_show_progress(
                step_id="validate_mcp_server_progress",
                progress_action=self._pending_mcp_progress_action,
                progress_task=task,
            )
        self._pending_mcp_validated_data = None
        self._pending_mcp_tool_options = []
        self._pending_mcp_validation_error = None
        if task is not None:
            (
                self._pending_mcp_validated_data,
                self._pending_mcp_tool_options,
                self._pending_mcp_validation_error,
            ) = await task
        return self.async_show_progress_done(next_step_id="validate_mcp_server_finish")

    async def async_step_validate_mcp_server_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Resume MCP server flow after validation progress."""
        del user_input
        step_id = self._pending_mcp_server_step_id
        form_options = self._pending_mcp_form_options
        validated_data = self._pending_mcp_validated_data
        tool_options = self._pending_mcp_tool_options
        error = self._pending_mcp_validation_error
        if error is not None:
            return self._show_server_form_error(step_id, form_options, error)

        if step_id == "manage_tools":
            assert validated_data is not None
            self._pending_manage_tools_data = validated_data
            self._pending_manage_tools_tool_options = tool_options
            self._manage_tools_prepared = True
            return await self.async_step_manage_tools()

        assert validated_data is not None
        pending_submitted_keys = self._pending_mcp_server_submitted_keys
        if _mcp_url_already_configured(
            self._get_entry(),
            validated_data[CONF_MCP_URL],
            self._current_subentry_id(),
        ):
            return self.async_abort(reason="already_configured")

        return self._async_finish_server_form(validated_data, pending_submitted_keys)

    async def _async_validate_mcp_server(
        self, data: dict[str, Any], current_subentry_id: str | None
    ) -> tuple[
        dict[str, Any] | None,
        list[SelectOptionDict],
        tuple[str, str, dict[str, str]] | None,
    ]:
        """Validate MCP connectivity and return discovered tool options."""
        server_id = current_subentry_id or str(data[CONF_NAME])
        _LOGGER.debug("Validating MCP server %s", server_id)
        try:
            async with asyncio.timeout(data.get(CONF_MCP_TIMEOUT, DEFAULT_MCP_TIMEOUT) * 3):
                data = dict(data)
                data[CONF_MCP_URL] = await async_validate_mcp_url(self.hass, data[CONF_MCP_URL])
                tools = await async_discover_mcp_tools_from_config(
                    self.hass,
                    data,
                    server_id=server_id,
                    apply_allowlist=False,
                )
                existing_allowed_tools = parse_allowed_tools(self._options.get(CONF_MCP_ALLOWED_TOOLS))
                if not tools:
                    raise MCPValidationError(
                        "no_mcp_tools",
                        "The MCP server did not expose any tools.",
                    )
                tool_options = _mcp_tool_options(tools, existing_allowed_tools)
        except TimeoutError:
            validation_error = MCPValidationError(
                "timeout",
                "Timed out validating the MCP server.",
                server_id=server_id,
            )
            _LOGGER.warning("Timed out validating MCP server %s", server_id)
            return (
                None,
                [],
                (
                    "base",
                    validation_error.reason,
                    _mcp_validation_placeholders(validation_error),
                ),
            )
        except MCPValidationError as err:
            target = CONF_MCP_URL if err.reason == "invalid_mcp_url" else "base"
            log_message, log_args = _validation_failure_log_details(err)
            log_detail = log_message % log_args if log_args else log_message
            _LOGGER.warning(
                "MCP server validation failed for %s: %s",
                server_id,
                log_detail,
            )
            return None, [], (target, err.reason, _mcp_validation_placeholders(err))
        except Exception:
            _LOGGER.exception("Unexpected exception validating MCP server")
            return None, [], ("base", "unknown", {})
        _LOGGER.debug(
            "Validated MCP server %s with %s available tools",
            server_id,
            len(tool_options),
        )
        return data, tool_options, None

    async def async_step_manage_tools(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        """Select the discovered MCP tools to allow."""
        entry = self._get_entry()
        if entry.state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")
        if self._is_new:
            return await self.async_step_init()

        subentry = self._get_reconfigure_subentry()
        if user_input is None and not self._manage_tools_prepared:
            current_data = dict(subentry.data)
            self._pending_mcp_form_options = current_data
            self._pending_mcp_progress_action = "discover_mcp_tools"
            self._pending_mcp_server_step_id = "manage_tools"
            self._pending_mcp_server_user_input = {}
            return self.async_show_progress(
                step_id="validate_mcp_server_progress",
                progress_action=self._pending_mcp_progress_action,
                progress_task=self.hass.async_create_task(
                    self._async_validate_mcp_server(current_data, subentry.subentry_id)
                ),
            )

        validated_data = self._pending_manage_tools_data
        tool_options = self._pending_manage_tools_tool_options
        tool_names = [option["value"] for option in tool_options]
        default_tool_mode = _mcp_tool_mode(subentry.data)
        if default_tool_mode == MCP_TOOL_MODE_SPECIFIED:
            default_tool_names = [
                tool_name
                for tool_name in parse_allowed_tools(subentry.data.get(CONF_MCP_ALLOWED_TOOLS))
                if tool_name in tool_names
            ]
        else:
            default_tool_names = []

        if user_input is None:
            return self.async_show_form(
                step_id="manage_tools",
                data_schema=_mcp_tools_schema(tool_options, default_tool_mode, default_tool_names),
            )

        tool_mode = str(user_input.get(CONF_MCP_TOOL_MODE, default_tool_mode))
        allowed_tools = [
            tool_name
            for tool_name in parse_allowed_tools(user_input.get(CONF_MCP_ALLOWED_TOOLS))
            if tool_name in tool_names
        ]
        if tool_mode == MCP_TOOL_MODE_SPECIFIED and not allowed_tools:
            return self.async_show_form(
                step_id="manage_tools",
                data_schema=_mcp_tools_schema(tool_options, tool_mode, allowed_tools),
                errors={CONF_MCP_ALLOWED_TOOLS: "mcp_tools_not_allowlisted"},
            )

        self._manage_tools_prepared = False
        data = {**validated_data}
        data.pop(CONF_MCP_ALLOWED_TOOLS, None)
        data.update(stored_mcp_tool_configuration(tool_mode, allowed_tools))
        return self.async_update_and_abort(
            self._get_entry(),
            subentry,
            title=data[CONF_NAME],
            data=data,
        )
