"""MCP tool discovery helpers for Pydantic AI Agent."""

# ruff: noqa: ANN401

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from types import TracebackType
from typing import Any
from urllib.parse import urlparse

import httpx
import voluptuous as vol
from fastmcp.client import Client as FastMCPClient
from fastmcp.client.transports import StreamableHttpTransport
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.httpx_client import (
    DEFAULT_LIMITS,
    SERVER_SOFTWARE,
    USER_AGENT,
    HassHttpXAsyncClient,
)
from homeassistant.util import slugify
from homeassistant.util.ssl import SSL_ALPN_HTTP11, client_context
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import AbstractToolset

from ._redaction import redact_data
from .const import (
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_DEFERRED_LOADING,
    CONF_MCP_HEADERS,
    CONF_MCP_INCLUDE_RETURN_SCHEMA,
    CONF_MCP_URL,
    DEFAULT_MCP_TIMEOUT,
    SUBENTRY_TYPE_MCP_SERVER,
)

_LOGGER = logging.getLogger(__name__)

_HTTP_HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


class _MCPHttpXClient(HassHttpXAsyncClient):
    """HA-configured HTTPX client that FastMCP owns for one session."""

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        """Close the per-session client when FastMCP exits its context."""
        await self.aclose()


@dataclass(slots=True)
class MCPValidationError(Exception):
    """MCP validation or discovery failed with a stable reason."""

    reason: str
    message: str
    status_code: int | None = None
    server_id: str | None = None
    tool_name: str | None = None


@dataclass(frozen=True, slots=True)
class ValidatedMCPURL:
    """MCP URL plus its exact origin."""

    url: str
    scheme: str
    hostname: str
    port: int


def _jsonable(value: Any) -> Any:
    """Return a JSON-compatible representation of MCP metadata."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    return value


def schema_hash(schema: Mapping[str, Any]) -> str:
    """Return a stable short hash for an MCP tool input schema."""
    payload = json.dumps(_jsonable(schema), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()[:16]


def normalise_mcp_url(url: object) -> str:
    """Validate and normalize a remote MCP URL."""
    if not isinstance(url, str) or not url.strip():
        raise MCPValidationError("invalid_mcp_url", "Enter an MCP server URL.")
    try:
        normalized = cv.url(url.strip())
    except vol.Invalid as err:
        raise MCPValidationError(
            "invalid_mcp_url",
            "Enter an HTTP or HTTPS Streamable HTTP MCP server URL.",
        ) from err
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise MCPValidationError(
            "invalid_mcp_url",
            "Enter an HTTP or HTTPS Streamable HTTP MCP server URL.",
        )
    try:
        hostname = parsed.hostname
        _port = parsed.port
    except ValueError as err:
        raise MCPValidationError(
            "invalid_mcp_url",
            "Enter an MCP server URL with a valid host and port.",
        ) from err
    if not hostname:
        raise MCPValidationError(
            "invalid_mcp_url",
            "Enter an MCP server URL with a host.",
        )
    if parsed.fragment:
        raise MCPValidationError(
            "invalid_mcp_url",
            "Enter an MCP server URL without a fragment.",
        )
    if parsed.username or parsed.password:
        raise MCPValidationError(
            "invalid_mcp_url",
            "Do not include credentials in MCP server URLs.",
        )
    return normalized


def _default_port(scheme: str, port: int | None) -> int:
    """Return the URL port or the scheme default."""
    if port is not None:
        return port
    if scheme == "http":
        return 80
    return 443


async def async_validate_mcp_url(_hass: HomeAssistant, url: str) -> str:
    """Validate an MCP URL."""
    return validate_mcp_url_details(url).url


async def async_validate_mcp_url_details(
    _hass: HomeAssistant, url: str
) -> ValidatedMCPURL:
    """Validate an MCP URL and return its exact origin."""
    return validate_mcp_url_details(url)


def validate_mcp_url_details(url: str) -> ValidatedMCPURL:
    """Validate an MCP URL and return its exact origin."""
    normalized = normalise_mcp_url(url)
    parsed = urlparse(normalized)
    hostname = parsed.hostname
    if hostname is None:
        raise MCPValidationError(
            "invalid_mcp_url", "Enter an MCP server URL with a host."
        )
    port = _default_port(parsed.scheme, parsed.port)
    return ValidatedMCPURL(normalized, parsed.scheme, hostname, port)


def parse_mcp_headers(value: object) -> dict[str, str]:
    """Parse optional one-header-per-line HTTP headers."""
    if value is None:
        return {}
    if isinstance(value, str):
        headers: dict[str, str] = {}
        for line in value.splitlines():
            line = line.strip()
            if not line:
                continue
            name, separator, header_value = line.partition(":")
            name = name.strip()
            if not separator or not _HTTP_HEADER_NAME_PATTERN.fullmatch(name):
                raise vol.Invalid("invalid_mcp_headers")
            headers[name] = header_value.strip()
        return headers
    if not isinstance(value, Mapping):
        raise vol.Invalid("invalid_mcp_headers")
    headers = dict(value)
    if not all(
        isinstance(key, str)
        and _HTTP_HEADER_NAME_PATTERN.fullmatch(key)
        and isinstance(item, str)
        for key, item in headers.items()
    ):
        raise vol.Invalid("invalid_mcp_headers")
    return headers


def parse_allowed_tools(value: object) -> list[str]:
    """Parse a comma/newline separated allowlist of MCP tool names."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        parts = value.replace("\n", ",").split(",")
        return sorted({part.strip() for part in parts if part.strip()})
    if isinstance(value, Sequence):
        tools = [str(item).strip() for item in value if str(item).strip()]
        return sorted(set(tools))
    raise vol.Invalid("invalid_mcp_tools")


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


def _origin_guard_hook(validated_url: ValidatedMCPURL) -> Any:
    """Return an HTTPX hook that rejects requests outside the MCP origin."""

    async def guard_origin(request: httpx.Request) -> None:
        request_port = _default_port(request.url.scheme, request.url.port)
        if (
            request.url.scheme != validated_url.scheme
            or request.url.host != validated_url.hostname
            or request_port != validated_url.port
        ):
            raise httpx.ConnectError(
                "MCP redirects must stay on the validated origin.",
                request=request,
            )

    return guard_origin


def _mcp_http_client_factory(validated_url: ValidatedMCPURL) -> Any:
    """Return an HTTP client factory for FastMCP Streamable HTTP connections."""

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
        follow_redirects: bool = False,
    ) -> httpx.AsyncClient:
        return _MCPHttpXClient(
            headers={USER_AGENT: SERVER_SOFTWARE, **(headers or {})},
            timeout=timeout,
            auth=auth,
            follow_redirects=follow_redirects,
            trust_env=False,
            limits=DEFAULT_LIMITS,
            verify=client_context(alpn_protocols=SSL_ALPN_HTTP11),
            event_hooks={"request": [_origin_guard_hook(validated_url)]},
        )

    return factory


def _mcp_client(
    validated_url: ValidatedMCPURL,
    headers: dict[str, str],
    timeout: float,
) -> FastMCPClient[Any]:
    """Return a FastMCP client pinned to Streamable HTTP transport."""
    transport = StreamableHttpTransport(
        validated_url.url,
        headers=headers,
        httpx_client_factory=_mcp_http_client_factory(validated_url),
    )
    return FastMCPClient(transport=transport, init_timeout=timeout, timeout=timeout)


def mcp_catalog_cache(entry: ConfigEntry) -> dict[str, Any]:
    """Return the entry-scoped MCP discovery cache."""
    runtime_data = getattr(entry, "runtime_data", None)
    if runtime_data is None:
        raise MCPValidationError(
            "config_entry_not_loaded",
            "Pydantic AI Agent config entry is not loaded.",
        )
    return runtime_data.mcp_tool_cache


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
    *,
    timeout: float = DEFAULT_MCP_TIMEOUT,
) -> list[dict[str, Any]]:
    """Discover tools exposed by one remote MCP server subentry."""
    config = mcp_config_from_subentry(subentry)
    return await async_discover_mcp_tools_from_config(
        hass,
        config,
        server_id=subentry.subentry_id,
        timeout=timeout,
    )


async def async_discover_mcp_tools_from_config(
    hass: HomeAssistant,
    data: Mapping[str, Any],
    *,
    server_id: str | None = None,
    apply_allowlist: bool = True,
    timeout: float = DEFAULT_MCP_TIMEOUT,
) -> list[dict[str, Any]]:
    """Discover tools exposed by one remote MCP server configuration."""
    config = _mcp_config_from_data(data, server_id=server_id)
    validated_url = await async_validate_mcp_url_details(hass, config[CONF_MCP_URL])
    config[CONF_MCP_URL] = validated_url.url
    allowed_tools = set(config[CONF_MCP_ALLOWED_TOOLS]) if apply_allowlist else set()
    server_id = server_id or str(config[CONF_NAME])
    _LOGGER.info(
        "Discovering MCP tools for server %s (%s)",
        server_id,
        config[CONF_NAME],
    )
    try:
        toolset = MCPToolset(
            _mcp_client(validated_url, config[CONF_MCP_HEADERS], timeout),
            id=server_id,
            tool_error_behavior="error",
        )
        async with toolset:
            tools = await toolset.list_tools()
    except TimeoutError as err:
        raise MCPValidationError(
            "timeout",
            "Timed out connecting to the MCP server.",
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
    discovered: list[dict[str, Any]] = []
    for tool in tools:
        tool_data = _jsonable(tool)
        name = str(tool_data.get("name", ""))
        if not name or (allowed_tools and name not in allowed_tools):
            continue
        input_schema = (
            tool_data.get("inputSchema") or tool_data.get("input_schema") or {}
        )
        if not isinstance(input_schema, Mapping):
            input_schema = {}
        discovered.append(
            {
                "server_id": server_id,
                "server_name": config[CONF_NAME],
                "name": name,
                "description": tool_data.get("description") or "",
                "input_schema": _jsonable(input_schema),
                "schema_hash": schema_hash(input_schema),
            }
        )
    _LOGGER.info(
        "Discovered %s %s MCP tools for server %s",
        len(discovered),
        "allowed" if apply_allowlist else "available",
        server_id,
    )
    _LOGGER.debug("MCP server config used for discovery: %s", redact_data(config))
    return discovered


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
        try:
            config = mcp_config_from_subentry(subentry)
            validated_url = await async_validate_mcp_url_details(
                hass, config[CONF_MCP_URL]
            )
            config[CONF_MCP_URL] = validated_url.url
        except MCPValidationError as err:
            _LOGGER.warning(
                "Invalid selected MCP server %s for runtime: %s",
                subentry.subentry_id,
                err.message,
            )
            raise
        if not config[CONF_MCP_ALLOWED_TOOLS]:
            raise MCPValidationError(
                "mcp_tools_not_allowlisted",
                "Select at least one allowed MCP tool before enabling this "
                "server for runtime use.",
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
        ) -> Any:
            if tool_name not in allowed_tools:
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
        toolset = toolset.filtered(
            lambda _ctx, tool_def, allowed_tools=allowed_tools: (
                tool_def.name in allowed_tools
            )
        )
        toolset = toolset.prefixed(f"mcp_{slugify(server_id)}")
        if config[CONF_MCP_DEFERRED_LOADING]:
            toolset = toolset.defer_loading()
        toolsets.append(toolset)
    missing_server_ids = selected_servers - configured_server_ids
    if missing_server_ids:
        missing_server_id = sorted(missing_server_ids)[0]
        raise MCPValidationError(
            "mcp_server_not_found",
            "Selected MCP server subentry was not found.",
            server_id=missing_server_id,
        )
    return toolsets
