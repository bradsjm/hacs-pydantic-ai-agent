"""MCP config-flow helpers."""

import logging
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import parse_qsl, urlparse

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.data_entry_flow import section
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
)
from homeassistant.helpers.typing import VolDictType

from ..const import (
    CONF_KEY_VALUE_VALUE,
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_CALL_CACHE_ENABLED,
    CONF_MCP_CALL_CACHE_TTL,
    CONF_MCP_DEFERRED_LOADING,
    CONF_MCP_HEADERS,
    CONF_MCP_INCLUDE_RETURN_SCHEMA,
    CONF_MCP_SECRET_HEADER_KEYS,
    CONF_MCP_SERVER_IDS,
    CONF_MCP_URL,
    DEFAULT_MCP_CALL_CACHE_TTL,
    SUBENTRY_TYPE_MCP_SERVER,
)
from ..mcp import (
    MCPValidationError,
    normalise_mcp_url,
    parse_allowed_tools,
    parse_mcp_headers,
)
from ..runtime.header_metadata import (
    format_header_rows,
    normalize_secret_header_keys,
    parse_header_rows,
)
from .helpers import (
    _flatten_section_data,
    _key_value_rows_selector,
    _sorted_select_options,
)

_LOGGER = logging.getLogger(__name__)

_SECTION_ADVANCED_MCP = "advanced_mcp"
_MAX_MCP_CALL_CACHE_TTL = 86_400


def _normalise_selected_mcp_server_ids(raw_server_ids: object) -> list[str]:
    """Return selected MCP server subentry IDs in storage order without duplicates."""
    if isinstance(raw_server_ids, str):
        raw_values: Iterable[object] = (raw_server_ids,)
    elif isinstance(raw_server_ids, Iterable):
        raw_values = raw_server_ids
    else:
        return []
    server_ids: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        if not isinstance(raw_value, str):
            continue
        server_id = raw_value.strip()
        if not server_id or server_id in seen:
            continue
        seen.add(server_id)
        server_ids.append(server_id)
    return server_ids


def _mcp_validation_placeholders(err: MCPValidationError) -> dict[str, str]:
    """Return translation placeholders for MCP validation errors."""
    placeholders = {"error_message": err.message}
    if err.status_code is not None:
        placeholders["status_code"] = str(err.status_code)
    return placeholders


def _mcp_server_select_options(
    entry: ConfigEntry | None,
    selected_server_ids: object = None,
) -> list[SelectOptionDict]:
    """Return workspace MCP server subentries as select options."""
    if entry is None:
        return []
    options = _sorted_select_options(
        [
            SelectOptionDict(label=subentry.title, value=subentry.subentry_id)
            for subentry in entry.subentries.values()
            if subentry.subentry_type == SUBENTRY_TYPE_MCP_SERVER
        ]
    )
    configured_ids = {str(option["value"]) for option in options if "value" in option}
    for server_id in _normalise_selected_mcp_server_ids(selected_server_ids):
        if server_id not in configured_ids:
            options.append(
                SelectOptionDict(label=f"Unavailable / {server_id}", value=server_id)
            )
    return options


def _append_mcp_server_schema_fields(
    schema: VolDictType,
    options: Mapping[str, Any],
    entry: ConfigEntry | None,
) -> None:
    """Append per-agent MCP server selection controls to a subentry form."""
    server_options = _mcp_server_select_options(entry, options.get(CONF_MCP_SERVER_IDS))
    if not server_options:
        return
    schema[
        vol.Optional(
            CONF_MCP_SERVER_IDS,
            default=_normalise_selected_mcp_server_ids(
                options.get(CONF_MCP_SERVER_IDS)
            ),
        )
    ] = SelectSelector(
        SelectSelectorConfig(
            options=server_options,
            mode=SelectSelectorMode.DROPDOWN,
            multiple=True,
        )
    )


def _selected_mcp_server_error(
    entry: ConfigEntry, data: Mapping[str, Any]
) -> str | None:
    """Return a form error for selected MCP servers that cannot run."""
    for server_id in _normalise_selected_mcp_server_ids(data.get(CONF_MCP_SERVER_IDS)):
        subentry = entry.subentries.get(server_id)
        if subentry is None or subentry.subentry_type != SUBENTRY_TYPE_MCP_SERVER:
            return "mcp_server_not_found"
        if CONF_MCP_ALLOWED_TOOLS in subentry.data and not parse_allowed_tools(
            subentry.data.get(CONF_MCP_ALLOWED_TOOLS)
        ):
            return "mcp_tools_not_allowlisted"
    return None


def _normalise_mcp_server_selection(data: dict[str, Any]) -> None:
    """Store only current MCP server subentry IDs on agents and tasks."""
    server_ids = _normalise_selected_mcp_server_ids(data.get(CONF_MCP_SERVER_IDS))
    if server_ids:
        data[CONF_MCP_SERVER_IDS] = server_ids
    else:
        data.pop(CONF_MCP_SERVER_IDS, None)


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
                            default=_format_mcp_headers(
                                options.get(CONF_MCP_HEADERS),
                                options.get(CONF_MCP_SECRET_HEADER_KEYS),
                            ),
                        ): _key_value_rows_selector(
                            CONF_KEY_VALUE_VALUE,
                            {"text": None},
                            key_label="header name",
                            value_label="header value",
                            include_secret_toggle=True,
                            secret_default=False,
                            translation_key=CONF_MCP_HEADERS,
                        ),
                        vol.Optional(
                            CONF_MCP_INCLUDE_RETURN_SCHEMA,
                            default=options.get(CONF_MCP_INCLUDE_RETURN_SCHEMA, True),
                        ): BooleanSelector(),
                        vol.Optional(
                            CONF_MCP_CALL_CACHE_ENABLED,
                            default=options.get(CONF_MCP_CALL_CACHE_ENABLED, False),
                        ): BooleanSelector(),
                        vol.Optional(
                            CONF_MCP_CALL_CACHE_TTL,
                            default=options.get(
                                CONF_MCP_CALL_CACHE_TTL,
                                DEFAULT_MCP_CALL_CACHE_TTL,
                            ),
                        ): NumberSelector(
                            NumberSelectorConfig(
                                mode=NumberSelectorMode.BOX,
                                min=1,
                                max=_MAX_MCP_CALL_CACHE_TTL,
                                step=1,
                            )
                        ),
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


def _format_mcp_headers(
    headers: object, secret_header_keys: object = ()
) -> list[dict[str, str | bool]]:
    """Return headers in selector-compatible row shape for the config form."""
    return format_header_rows(headers, secret_header_keys)


def _mcp_server_form_options(
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return MCP server data expanded into form-friendly values."""
    form_options = _flatten_section_data(options or {}, (_SECTION_ADVANCED_MCP,))
    if CONF_MCP_HEADERS in form_options:
        form_options[CONF_MCP_HEADERS] = _format_mcp_headers(
            form_options.get(CONF_MCP_HEADERS),
            form_options.get(CONF_MCP_SECRET_HEADER_KEYS),
        )
    return form_options


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
        options_by_name[name] = SelectOptionDict(label=name, value=name)
    for name in extra_tool_names:
        if name and name not in options_by_name:
            options_by_name[name] = SelectOptionDict(label=name, value=name)
    return [options_by_name[name] for name in sorted(options_by_name)]


def _mcp_tools_schema(
    tool_options: list[SelectOptionDict],
    default_tool_names: list[str],
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
                    mode=SelectSelectorMode.DROPDOWN,
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
        CONF_MCP_CALL_CACHE_ENABLED: bool(
            user_input.get(CONF_MCP_CALL_CACHE_ENABLED, False)
        ),
        CONF_MCP_CALL_CACHE_TTL: _parse_mcp_call_cache_ttl(
            user_input.get(CONF_MCP_CALL_CACHE_TTL, DEFAULT_MCP_CALL_CACHE_TTL)
        ),
        CONF_MCP_INCLUDE_RETURN_SCHEMA: bool(
            user_input.get(CONF_MCP_INCLUDE_RETURN_SCHEMA, True)
        ),
        CONF_MCP_DEFERRED_LOADING: bool(
            user_input.get(CONF_MCP_DEFERRED_LOADING, False)
        ),
    }
    headers, secret_header_keys = _parse_mcp_headers_with_secrets(
        user_input.get(CONF_MCP_HEADERS)
    )
    if headers:
        data[CONF_MCP_HEADERS] = headers
        data[CONF_MCP_SECRET_HEADER_KEYS] = secret_header_keys
    allowed_tools = parse_allowed_tools(user_input.get(CONF_MCP_ALLOWED_TOOLS))
    if allowed_tools:
        data[CONF_MCP_ALLOWED_TOOLS] = allowed_tools
    return data


def _parse_mcp_headers_with_secrets(value: object) -> tuple[dict[str, str], list[str]]:
    """Parse optional HTTP headers and secret metadata from selector rows."""
    try:
        headers, secret_header_keys = parse_header_rows(value)
    except ValueError as err:
        raise vol.Invalid(str(err)) from err
    validated_headers = parse_mcp_headers(headers)
    return validated_headers, normalize_secret_header_keys(
        validated_headers, secret_header_keys
    )


def _parse_mcp_call_cache_ttl(value: object) -> int:
    """Return a validated MCP call cache TTL in seconds."""
    if isinstance(value, bool):
        raise vol.Invalid("invalid_mcp_call_cache_ttl")
    if not isinstance(value, str | int | float):
        raise vol.Invalid("invalid_mcp_call_cache_ttl")
    try:
        ttl = int(value)
    except (TypeError, ValueError) as err:
        raise vol.Invalid("invalid_mcp_call_cache_ttl") from err
    if ttl < 1 or ttl > _MAX_MCP_CALL_CACHE_TTL:
        raise vol.Invalid("invalid_mcp_call_cache_ttl")
    return ttl


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
                "Ignoring invalid stored MCP URL while checking duplicates "
                "for subentry %s",
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
