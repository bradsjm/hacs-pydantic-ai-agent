"""MCP config-flow helpers."""

from collections.abc import Iterable, Mapping
import logging
from typing import Any
from urllib.parse import parse_qsl, urlparse

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.data_entry_flow import section
from homeassistant.helpers.selector import (
    BooleanSelector,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TextSelector,
    TextSelectorConfig,
)

from ..const import (
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_DEFERRED_LOADING,
    CONF_MCP_HEADERS,
    CONF_MCP_INCLUDE_RETURN_SCHEMA,
    CONF_MCP_SERVER_IDS,
    CONF_MCP_URL,
    SUBENTRY_TYPE_MCP_SERVER,
)
from ..mcp import MCPValidationError, normalise_mcp_url, parse_allowed_tools, parse_mcp_headers
from .helpers import _flatten_section_data

_LOGGER = logging.getLogger(__name__)

_MCP_TOOL_DESCRIPTION_LABEL_MAX_LENGTH = 80
_SECTION_ADVANCED_MCP = "advanced_mcp"


def _mcp_validation_placeholders(err: MCPValidationError) -> dict[str, str]:
    """Return translation placeholders for MCP validation errors."""
    placeholders = {"error_message": err.message}
    if err.status_code is not None:
        placeholders["status_code"] = str(err.status_code)
    return placeholders


def _selected_mcp_server_error(
    entry: ConfigEntry, data: Mapping[str, Any]
) -> str | None:
    """Return a form error for selected MCP servers that cannot run."""
    for server_id in data.get(CONF_MCP_SERVER_IDS, []):
        subentry = entry.subentries.get(server_id)
        if subentry is None or subentry.subentry_type != SUBENTRY_TYPE_MCP_SERVER:
            return "mcp_server_not_found"
        if not parse_allowed_tools(subentry.data.get(CONF_MCP_ALLOWED_TOOLS)):
            return "mcp_tools_not_allowlisted"
    return None


def _mcp_server_schema(options: Mapping[str, Any] | None = None) -> vol.Schema:
    """Return the remote MCP server subentry schema."""
    options = _flatten_section_data(options or {}, (_SECTION_ADVANCED_MCP,))
    return vol.Schema(
        {
            vol.Required(CONF_NAME, default=options.get(CONF_NAME, "")): TextSelector(
                TextSelectorConfig()
            ),
            vol.Required(
                CONF_MCP_URL,
                default=options.get(CONF_MCP_URL, ""),
            ): TextSelector(TextSelectorConfig()),
            vol.Optional(_SECTION_ADVANCED_MCP, default={}): section(
                vol.Schema(
                    {
                        vol.Optional(
                            CONF_MCP_HEADERS,
                            default=_format_mcp_headers(options.get(CONF_MCP_HEADERS)),
                        ): TextSelector(TextSelectorConfig(multiline=True)),
                        vol.Optional(
                            CONF_MCP_INCLUDE_RETURN_SCHEMA,
                            default=options.get(CONF_MCP_INCLUDE_RETURN_SCHEMA, True),
                        ): BooleanSelector(),
                        vol.Optional(
                            CONF_MCP_DEFERRED_LOADING,
                            default=options.get(CONF_MCP_DEFERRED_LOADING, False),
                        ): BooleanSelector(),
                    }
                ),
                {"collapsed": True},
            ),
        }
    )


def _format_mcp_headers(headers: object) -> str:
    """Return headers as one HTTP header per line for the config form."""
    if headers is None:
        return ""
    if isinstance(headers, str):
        return headers
    if not isinstance(headers, Mapping):
        return ""
    return "\n".join(f"{name}: {headers[name]}" for name in sorted(headers))


def _truncate_mcp_tool_description(description: str) -> str:
    """Return a compact single-line MCP tool description for selector labels."""
    description = " ".join(description.split())
    if len(description) <= _MCP_TOOL_DESCRIPTION_LABEL_MAX_LENGTH:
        return description
    return f"{description[: _MCP_TOOL_DESCRIPTION_LABEL_MAX_LENGTH - 3].rstrip()}..."


def _mcp_tool_options(
    tools: Iterable[Mapping[str, Any]],
    extra_tool_names: Iterable[str] = (),
) -> list[SelectOptionDict]:
    """Return sorted MCP tool selector options from discovered metadata."""
    options_by_name: dict[str, SelectOptionDict] = {}
    for tool in tools:
        name = str(tool.get("name", "")).strip()
        if not name or name in options_by_name:
            continue
        description = _truncate_mcp_tool_description(
            str(tool.get("description", "")).strip()
        )
        label = f"{name} ({description})" if description else name
        options_by_name[name] = SelectOptionDict(label=label, value=name)
    for name in extra_tool_names:
        if name and name not in options_by_name:
            options_by_name[name] = SelectOptionDict(label=name, value=name)
    return [options_by_name[name] for name in sorted(options_by_name)]


def _mcp_tools_schema(
    tool_options: list[SelectOptionDict], default_tool_names: list[str]
) -> vol.Schema:
    """Return the MCP discovered tools selection schema."""
    return vol.Schema(
        {
            vol.Required(
                CONF_MCP_ALLOWED_TOOLS,
                default=default_tool_names,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=tool_options,
                    multiple=True,
                )
            )
        }
    )


def _mcp_server_data_from_user_input(user_input: Mapping[str, Any]) -> dict[str, Any]:
    """Return normalized remote MCP server subentry data."""
    user_input = _flatten_section_data(user_input, (_SECTION_ADVANCED_MCP,))
    data: dict[str, Any] = {
        CONF_NAME: str(user_input[CONF_NAME]).strip(),
        CONF_MCP_URL: normalise_mcp_url(user_input[CONF_MCP_URL]),
        CONF_MCP_INCLUDE_RETURN_SCHEMA: bool(
            user_input.get(CONF_MCP_INCLUDE_RETURN_SCHEMA, True)
        ),
        CONF_MCP_DEFERRED_LOADING: bool(
            user_input.get(CONF_MCP_DEFERRED_LOADING, False)
        ),
    }
    headers = parse_mcp_headers(user_input.get(CONF_MCP_HEADERS))
    if headers:
        data[CONF_MCP_HEADERS] = headers
    allowed_tools = parse_allowed_tools(user_input.get(CONF_MCP_ALLOWED_TOOLS))
    if allowed_tools:
        data[CONF_MCP_ALLOWED_TOOLS] = allowed_tools
    return data


def _mcp_url_already_configured(
    entry: ConfigEntry,
    url: str,
    current_subentry_id: str | None = None,
) -> bool:
    """Return if another MCP server subentry already uses this URL."""
    url_identity = _mcp_url_identity(url)
    for subentry in entry.subentries.values():
        if subentry.subentry_id == current_subentry_id:
            continue
        if subentry.subentry_type != SUBENTRY_TYPE_MCP_SERVER:
            continue
        try:
            existing_identity = _mcp_url_identity(subentry.data.get(CONF_MCP_URL))
        except MCPValidationError:
            _LOGGER.warning(
                "Ignoring invalid stored MCP URL while checking duplicates for subentry %s",
                subentry.subentry_id,
            )
            continue
        if existing_identity == url_identity:
            return True
    return False


def _mcp_url_identity(
    url: object,
) -> tuple[str, str, int, str, tuple[tuple[str, str], ...]]:
    """Return a canonical identity for duplicate MCP URL checks."""
    parsed = urlparse(normalise_mcp_url(url))
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return (
        parsed.scheme,
        (parsed.hostname or "").lower().rstrip("."),
        port,
        parsed.path or "/",
        tuple(sorted(parse_qsl(parsed.query, keep_blank_values=True))),
    )
