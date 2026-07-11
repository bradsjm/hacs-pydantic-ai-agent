"""MCP discovery and cache helpers."""

# ruff: noqa: ANN401

import asyncio
from collections.abc import Mapping
from hashlib import sha256
import json
import logging
from typing import Any, cast

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
import httpx
from mcp.shared.exceptions import McpError
from pydantic_ai.mcp import MCPToolset

from ..const import (
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_HEADERS,
    CONF_MCP_TIMEOUT,
    CONF_MCP_TOOL_MODE,
    CONF_MCP_URL,
    MCP_TOOL_MODE_ALL,
)
from ..runtime.redaction import redact_data
from .client import _mcp_client
from .entry_helpers import (
    _mcp_config_from_data,
    get_mcp_subentry,
    mcp_config_from_subentry,
)
from .errors import MCPValidationError, is_mcp_timeout_error
from .validation import _jsonable, async_validate_mcp_url_details, schema_hash

_LOGGER = logging.getLogger(__name__)


def mcp_catalog_cache(entry: ConfigEntry) -> dict[str, Any]:
    """Return the entry-scoped MCP discovery cache."""
    runtime_data = getattr(entry, "runtime_data", None)
    if runtime_data is None:
        raise MCPValidationError(
            "config_entry_not_loaded",
            "Pydantic AI Agent config entry is not loaded.",
        )
    return cast(dict[str, Any], runtime_data.mcp_tool_cache)


def _cache_key(entry: ConfigEntry, subentry_id: str) -> str:
    """Return the cache key for one entry/subentry pair."""
    subentry = get_mcp_subentry(entry, subentry_id)
    fingerprint = sha256(
        json.dumps(
            _jsonable(subentry.data),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:16]
    return f"{entry.entry_id}:{subentry_id}:{fingerprint}"


async def async_discover_mcp_tools(
    hass: HomeAssistant,
    subentry: Any,
) -> list[dict[str, Any]]:
    """Discover tools exposed by one remote MCP server subentry."""
    config = mcp_config_from_subentry(subentry)
    return await async_discover_mcp_tools_from_config(
        hass,
        config,
        server_id=subentry.subentry_id,
    )


async def async_discover_mcp_tools_from_config(
    hass: HomeAssistant,
    data: Mapping[str, Any],
    *,
    server_id: str | None = None,
    apply_allowlist: bool = True,
    request_timeout: float | None = None,
) -> list[dict[str, Any]]:
    """Discover tools exposed by one remote MCP server configuration."""
    config = _mcp_config_from_data(data, server_id=server_id)
    timeout = request_timeout if request_timeout is not None else config[CONF_MCP_TIMEOUT]
    validated_url = await async_validate_mcp_url_details(hass, config[CONF_MCP_URL])
    config[CONF_MCP_URL] = validated_url.url
    allowed_tools: set[str] | None = None
    if apply_allowlist and config.get(CONF_MCP_TOOL_MODE) != MCP_TOOL_MODE_ALL:
        allowed_tools = set(config[CONF_MCP_ALLOWED_TOOLS])
    server_id = server_id or str(config[CONF_NAME])
    _LOGGER.info(
        "Discovering MCP tools for server %s (%s)",
        server_id,
        config[CONF_NAME],
    )
    tools = await _async_list_mcp_tools(
        validated_url,
        config[CONF_MCP_HEADERS],
        timeout,
        server_id,
    )
    discovered = [
        tool_data
        for tool in tools
        if (
            tool_data := _normalize_mcp_tool(
                tool,
                server_id=server_id,
                server_name=config[CONF_NAME],
                allowed_tools=allowed_tools,
            )
        )
    ]
    _LOGGER.info(
        "Discovered %s %s MCP tools for server %s",
        len(discovered),
        "allowed" if apply_allowlist else "available",
        server_id,
    )
    _LOGGER.debug("MCP server config used for discovery: %s", redact_data(config))
    return discovered


async def _async_list_mcp_tools(
    validated_url: Any,
    headers: dict[str, str],
    request_timeout: float,
    server_id: str,
) -> Any:
    """List MCP tools with a bounded transport lifecycle and typed errors."""
    try:
        toolset = MCPToolset(
            _mcp_client(validated_url, headers, request_timeout),
            id=server_id,
            tool_error_behavior="error",
        )
        # FastMCP uses the configured timeout for both session initialization and
        # the subsequent tools/list request, so the watchdog here must cover both
        # phases without allowing the config-flow progress task to hang forever.
        async with asyncio.timeout(request_timeout * 2):
            return await toolset.list_tools()
    except TimeoutError as err:
        raise MCPValidationError(
            "timeout",
            "Timed out connecting to the MCP server.",
            server_id=server_id,
        ) from err
    except McpError as err:
        if is_mcp_timeout_error(err):
            raise MCPValidationError(
                "timeout",
                "Timed out connecting to the MCP server.",
                server_id=server_id,
            ) from err
        raise MCPValidationError(
            "cannot_connect",
            "Could not connect to the MCP server.",
            server_id=server_id,
        ) from err
    except Exception as err:
        status_code = getattr(err, "status_code", None)
        if status_code is None and isinstance(err, httpx.HTTPStatusError):
            status_code = err.response.status_code
        if isinstance(status_code, int) and status_code in {401, 403}:
            raise MCPValidationError(
                "invalid_auth",
                "The MCP server rejected the configured HTTP headers.",
                status_code=status_code,
                server_id=server_id,
            ) from err
        raise MCPValidationError(
            "cannot_connect",
            "Could not connect to the MCP server.",
            status_code=status_code if isinstance(status_code, int) else None,
            server_id=server_id,
        ) from err


def _normalize_mcp_tool(
    tool: Any,
    *,
    server_id: str,
    server_name: str,
    allowed_tools: set[str] | None,
) -> dict[str, Any] | None:
    """Normalize one raw MCP tool, returning None when it is filtered out."""
    tool_data = _jsonable(tool)
    name = str(tool_data.get("name", ""))
    if not name or (allowed_tools is not None and name not in allowed_tools):
        return None
    input_schema = tool_data.get("inputSchema") or tool_data.get("input_schema") or {}
    if not isinstance(input_schema, Mapping):
        input_schema = {}
    return {
        "server_id": server_id,
        "server_name": server_name,
        "name": name,
        "description": tool_data.get("description") or "",
        "input_schema": _jsonable(input_schema),
        "schema_hash": schema_hash(input_schema),
    }


async def async_refresh_mcp_tools(
    hass: HomeAssistant,
    entry: ConfigEntry,
    subentry_id: str,
) -> list[dict[str, Any]]:
    """Refresh and cache tools for one MCP server subentry."""
    subentry = get_mcp_subentry(entry, subentry_id)
    tools = await async_discover_mcp_tools(hass, subentry)
    mcp_catalog_cache(entry)[_cache_key(entry, subentry_id)] = tools
    return tools


def cached_mcp_tools(
    entry: ConfigEntry,
    subentry_id: str,
) -> list[dict[str, Any]] | None:
    """Return cached MCP tools for one server if available."""
    get_mcp_subentry(entry, subentry_id)
    tools = mcp_catalog_cache(entry).get(_cache_key(entry, subentry_id))
    if tools is None:
        return None
    return list(tools)
