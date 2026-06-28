"""Home Assistant entry helpers for MCP."""

# ruff: noqa: ANN401

from collections.abc import Mapping
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME

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
    DEFAULT_MCP_CALL_CACHE_TTL,
    DEFAULT_MCP_TIMEOUT,
    MCP_TOOL_MODE_ALL,
    MCP_TOOL_MODE_DISABLED,
    MCP_TOOL_MODE_SPECIFIED,
    SUBENTRY_TYPE_MCP_SERVER,
)
from .errors import MCPValidationError
from .validation import normalise_mcp_url, parse_allowed_tools, parse_mcp_headers


def mcp_subentries(entry: ConfigEntry) -> list[Any]:
    """Return configured MCP server subentries."""
    return [
        subentry
        for subentry in entry.subentries.values()
        if subentry.subentry_type == SUBENTRY_TYPE_MCP_SERVER
    ]


def get_mcp_subentry(entry: ConfigEntry, subentry_id: str) -> Any:
    """Return a configured MCP server subentry by ID or raise a validation error."""
    subentry = entry.subentries.get(subentry_id)
    if subentry is None or subentry.subentry_type != SUBENTRY_TYPE_MCP_SERVER:
        raise MCPValidationError(
            "mcp_server_not_found",
            "MCP server subentry was not found.",
            server_id=subentry_id,
        )
    return subentry


def mcp_config_from_subentry(subentry: Any) -> dict[str, Any]:
    """Return normalized MCP server configuration from a subentry."""
    data = dict(subentry.data)
    tool_mode = effective_mcp_tool_mode(data)
    return {
        CONF_NAME: subentry.title,
        CONF_MCP_URL: normalise_mcp_url(data.get(CONF_MCP_URL)),
        CONF_MCP_HEADERS: parse_mcp_headers(data.get(CONF_MCP_HEADERS)),
        CONF_MCP_SECRET_HEADER_KEYS: data.get(CONF_MCP_SECRET_HEADER_KEYS, []),
        CONF_MCP_CALL_CACHE_ENABLED: bool(data.get(CONF_MCP_CALL_CACHE_ENABLED, False)),
        CONF_MCP_CALL_CACHE_TTL: int(
            data.get(CONF_MCP_CALL_CACHE_TTL, DEFAULT_MCP_CALL_CACHE_TTL)
        ),
        CONF_MCP_INCLUDE_RETURN_SCHEMA: bool(
            data.get(CONF_MCP_INCLUDE_RETURN_SCHEMA, True)
        ),
        CONF_MCP_TIMEOUT: float(data.get(CONF_MCP_TIMEOUT, DEFAULT_MCP_TIMEOUT)),
        CONF_MCP_DEFERRED_LOADING: bool(data.get(CONF_MCP_DEFERRED_LOADING, False)),
        CONF_MCP_TOOL_MODE: tool_mode,
        CONF_MCP_ALLOWED_TOOLS: parse_allowed_tools(data.get(CONF_MCP_ALLOWED_TOOLS)),
    }


def _mcp_config_from_data(
    data: Mapping[str, Any], *, server_id: str | None = None
) -> dict[str, Any]:
    """Return normalized MCP server configuration from raw data."""
    tool_mode = effective_mcp_tool_mode(data)
    return {
        CONF_NAME: data.get(CONF_NAME, server_id or "MCP server"),
        CONF_MCP_URL: normalise_mcp_url(data.get(CONF_MCP_URL)),
        CONF_MCP_HEADERS: parse_mcp_headers(data.get(CONF_MCP_HEADERS)),
        CONF_MCP_SECRET_HEADER_KEYS: data.get(CONF_MCP_SECRET_HEADER_KEYS, []),
        CONF_MCP_CALL_CACHE_ENABLED: bool(data.get(CONF_MCP_CALL_CACHE_ENABLED, False)),
        CONF_MCP_CALL_CACHE_TTL: int(
            data.get(CONF_MCP_CALL_CACHE_TTL, DEFAULT_MCP_CALL_CACHE_TTL)
        ),
        CONF_MCP_INCLUDE_RETURN_SCHEMA: bool(
            data.get(CONF_MCP_INCLUDE_RETURN_SCHEMA, True)
        ),
        CONF_MCP_TIMEOUT: float(data.get(CONF_MCP_TIMEOUT, DEFAULT_MCP_TIMEOUT)),
        CONF_MCP_DEFERRED_LOADING: bool(data.get(CONF_MCP_DEFERRED_LOADING, False)),
        CONF_MCP_TOOL_MODE: tool_mode,
        CONF_MCP_ALLOWED_TOOLS: parse_allowed_tools(data.get(CONF_MCP_ALLOWED_TOOLS)),
    }


def effective_mcp_tool_mode(data: Mapping[str, Any]) -> str:
    """Return effective MCP tool mode from stored data, including legacy state."""
    stored_mode = data.get(CONF_MCP_TOOL_MODE)
    if stored_mode in {
        MCP_TOOL_MODE_ALL,
        MCP_TOOL_MODE_SPECIFIED,
        MCP_TOOL_MODE_DISABLED,
    }:
        return str(stored_mode)

    if CONF_MCP_ALLOWED_TOOLS not in data:
        return MCP_TOOL_MODE_ALL
    if parse_allowed_tools(data.get(CONF_MCP_ALLOWED_TOOLS)):
        return MCP_TOOL_MODE_SPECIFIED
    return MCP_TOOL_MODE_DISABLED


def stored_mcp_tool_configuration(
    mode: str, allowed_tools: list[str]
) -> dict[str, Any]:
    """Return stored MCP tool configuration fields for one mode."""
    if mode == MCP_TOOL_MODE_ALL:
        return {CONF_MCP_TOOL_MODE: MCP_TOOL_MODE_ALL}
    if mode == MCP_TOOL_MODE_DISABLED:
        return {
            CONF_MCP_TOOL_MODE: MCP_TOOL_MODE_DISABLED,
            CONF_MCP_ALLOWED_TOOLS: [],
        }
    if mode == MCP_TOOL_MODE_SPECIFIED:
        if not allowed_tools:
            raise ValueError("specified mode requires at least one tool")
        return {
            CONF_MCP_TOOL_MODE: MCP_TOOL_MODE_SPECIFIED,
            CONF_MCP_ALLOWED_TOOLS: allowed_tools,
        }
    raise ValueError(f"unsupported MCP tool mode: {mode}")
