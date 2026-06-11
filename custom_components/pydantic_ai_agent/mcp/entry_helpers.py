"""Home Assistant entry helpers for MCP."""

# ruff: noqa: ANN401

from collections.abc import Mapping
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME

from ..const import (
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_DEFERRED_LOADING,
    CONF_MCP_HEADERS,
    CONF_MCP_INCLUDE_RETURN_SCHEMA,
    CONF_MCP_URL,
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
    return {
        CONF_NAME: subentry.title,
        CONF_MCP_URL: normalise_mcp_url(data.get(CONF_MCP_URL)),
        CONF_MCP_HEADERS: parse_mcp_headers(data.get(CONF_MCP_HEADERS)),
        CONF_MCP_INCLUDE_RETURN_SCHEMA: bool(
            data.get(CONF_MCP_INCLUDE_RETURN_SCHEMA, True)
        ),
        CONF_MCP_DEFERRED_LOADING: bool(data.get(CONF_MCP_DEFERRED_LOADING, False)),
        CONF_MCP_ALLOWED_TOOLS: parse_allowed_tools(data.get(CONF_MCP_ALLOWED_TOOLS)),
    }


def _mcp_config_from_data(
    data: Mapping[str, Any], *, server_id: str | None = None
) -> dict[str, Any]:
    """Return normalized MCP server configuration from raw data."""
    return {
        CONF_NAME: data.get(CONF_NAME, server_id or "MCP server"),
        CONF_MCP_URL: normalise_mcp_url(data.get(CONF_MCP_URL)),
        CONF_MCP_HEADERS: parse_mcp_headers(data.get(CONF_MCP_HEADERS)),
        CONF_MCP_INCLUDE_RETURN_SCHEMA: bool(
            data.get(CONF_MCP_INCLUDE_RETURN_SCHEMA, True)
        ),
        CONF_MCP_DEFERRED_LOADING: bool(data.get(CONF_MCP_DEFERRED_LOADING, False)),
        CONF_MCP_ALLOWED_TOOLS: parse_allowed_tools(data.get(CONF_MCP_ALLOWED_TOOLS)),
    }
