"""MCP runtime toolset helpers."""

# ruff: noqa: ANN401

import logging
from collections.abc import Sequence
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.util import slugify
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import AbstractToolset

from .._redaction import redact_data
from ..const import (
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_DEFERRED_LOADING,
    CONF_MCP_HEADERS,
    CONF_MCP_INCLUDE_RETURN_SCHEMA,
    CONF_MCP_URL,
    DEFAULT_MCP_TIMEOUT,
)
from .client import _mcp_client
from .entry_helpers import mcp_config_from_subentry, mcp_subentries
from .errors import MCPValidationError
from .validation import async_validate_mcp_url_details

_LOGGER = logging.getLogger(__name__)


async def _async_runtime_mcp_toolset_for_subentry(
    hass: HomeAssistant,
    subentry: Any,
) -> AbstractToolset[Any]:
    """Return one runtime MCP toolset for a configured server subentry."""
    try:
        config = mcp_config_from_subentry(subentry)
        validated_url = await async_validate_mcp_url_details(hass, config[CONF_MCP_URL])
        config[CONF_MCP_URL] = validated_url.url
    except MCPValidationError as err:
        _LOGGER.warning(
            "Invalid selected MCP server %s for runtime: %s",
            subentry.subentry_id,
            err.message,
        )
        raise

    has_allowlist = CONF_MCP_ALLOWED_TOOLS in subentry.data
    if has_allowlist and not config[CONF_MCP_ALLOWED_TOOLS]:
        raise MCPValidationError(
            "mcp_tools_not_allowlisted",
            "Select at least one allowed MCP tool before enabling this server for "
            "runtime use.",
            server_id=subentry.subentry_id,
        )
    allowed_tools = set(config[CONF_MCP_ALLOWED_TOOLS])
    server_id = subentry.subentry_id

    async def process_tool_call(
        _ctx: Any,
        call_tool: Any,
        tool_name: str,
        tool_args: dict[str, Any],
        *,
        server_id: str = server_id,
        allowed_tools: set[str] = allowed_tools,
        has_allowlist: bool = has_allowlist,
    ) -> Any:
        if has_allowlist and tool_name not in allowed_tools:
            raise MCPValidationError(
                "mcp_tool_not_allowed",
                "MCP tool is not allowlisted for this server.",
                server_id=server_id,
                tool_name=tool_name,
            )
        _LOGGER.info("Calling MCP tool %s on server %s", tool_name, server_id)
        _LOGGER.debug("MCP tool call arguments: %s", redact_data(tool_args))
        result = await call_tool(tool_name, tool_args)
        _LOGGER.debug("MCP tool result: %s", redact_data({"result": result}))
        return result

    toolset: AbstractToolset[Any] = MCPToolset(
        _mcp_client(
            validated_url,
            config[CONF_MCP_HEADERS],
            DEFAULT_MCP_TIMEOUT,
        ),
        id=server_id,
        tool_error_behavior="error",
        process_tool_call=process_tool_call,
        include_return_schema=config[CONF_MCP_INCLUDE_RETURN_SCHEMA],
    )
    if has_allowlist:
        toolset = toolset.filtered(
            lambda _ctx, tool_def, allowed_tools=allowed_tools: (
                tool_def.name in allowed_tools
            )
        )
    toolset = toolset.prefixed(f"mcp_{slugify(server_id)}")
    if config[CONF_MCP_DEFERRED_LOADING]:
        toolset = toolset.defer_loading()
    return toolset


async def async_runtime_mcp_toolsets(
    hass: HomeAssistant,
    entry: ConfigEntry,
    selected_server_ids: Sequence[str] | None,
) -> list[AbstractToolset[Any]]:
    """Return agent MCP toolsets for explicitly allowlisted servers."""
    toolsets: list[AbstractToolset[Any]] = []
    selected_servers = set(selected_server_ids or [])
    if not selected_servers:
        return toolsets
    configured_server_ids: set[str] = set()
    for subentry in mcp_subentries(entry):
        if subentry.subentry_id not in selected_servers:
            continue
        configured_server_ids.add(subentry.subentry_id)
        toolsets.append(await _async_runtime_mcp_toolset_for_subentry(hass, subentry))
    missing_server_ids = selected_servers - configured_server_ids
    if missing_server_ids:
        missing_server_id = sorted(missing_server_ids)[0]
        raise MCPValidationError(
            "mcp_server_not_found",
            "Selected MCP server subentry was not found.",
            server_id=missing_server_id,
        )
    return toolsets
