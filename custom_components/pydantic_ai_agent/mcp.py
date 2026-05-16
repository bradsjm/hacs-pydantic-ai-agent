"""MCP tool discovery helpers for Pydantic AI Agent."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import logging
from typing import Any
from urllib.parse import urlparse, urlunparse

from fastmcp.client import Client as FastMCPClient
from fastmcp.client.transports import StreamableHttpTransport
import httpx
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import PrefixedToolset
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.util import slugify

from .const import (
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_HEADERS,
    CONF_MCP_URL,
    DEFAULT_MCP_TIMEOUT,
    DOMAIN,
    SUBENTRY_TYPE_MCP_SERVER,
)

_LOGGER = logging.getLogger(__name__)

_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "cookie",
    "headers",
    "password",
    "secret",
    "token",
    "x-api-key",
}


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


def _redact_value(key: object, value: object) -> object:
    """Return a log-safe value for a potentially sensitive mapping field."""
    if key == CONF_MCP_URL and isinstance(value, str):
        return redact_mcp_url_password(value)
    key_text = str(key).lower()
    if key_text in _SENSITIVE_KEYS or key_text.endswith(("_token", "-token")):
        return "**REDACTED**"
    if isinstance(value, Mapping):
        return {
            item_key: _redact_value(item_key, item_value)
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(key, item) for item in value]
    return value


def redact_for_log(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a log-safe copy of mapping data."""
    return {key: _redact_value(key, value) for key, value in data.items()}


def redact_mcp_url_password(url: str) -> str:
    """Redact only the password portion of URL userinfo."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    if parsed.password is None or parsed.hostname is None:
        return url

    raw_userinfo, separator, hostport = parsed.netloc.rpartition("@")
    if not separator:
        return url
    raw_username, password_separator, _raw_password = raw_userinfo.partition(":")
    if not password_separator:
        return url
    netloc = f"{raw_username}:**REDACTED**@{hostport}"
    return urlunparse(parsed._replace(netloc=netloc))


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
    normalized = url.strip()
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise MCPValidationError(
            "invalid_mcp_url",
            "Enter an HTTP or HTTPS Streamable HTTP MCP server URL.",
        )
    try:
        hostname = parsed.hostname
        parsed.port
    except ValueError:
        raise MCPValidationError(
            "invalid_mcp_url",
            "Enter an MCP server URL with a valid host and port.",
        )
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
    if parsed.scheme == "http" and (parsed.username or parsed.password):
        raise MCPValidationError(
            "invalid_mcp_url",
            "Do not include credentials in HTTP MCP server URLs.",
        )
    return normalized


def _default_port(scheme: str, port: int | None) -> int:
    """Return the URL port or the scheme default."""
    if port is not None:
        return port
    if scheme == "http":
        return 80
    return 443


async def async_validate_mcp_url(hass: HomeAssistant, url: str) -> str:
    """Validate an MCP URL."""
    return (await async_validate_mcp_url_details(hass, url)).url


async def async_validate_mcp_url_details(
    hass: HomeAssistant, url: str
) -> ValidatedMCPURL:
    """Validate an MCP URL and return its exact origin."""
    del hass
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
    """Parse an optional JSON object of HTTP headers."""
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as err:
            raise vol.Invalid("invalid_mcp_headers") from err
    if not isinstance(value, Mapping):
        raise vol.Invalid("invalid_mcp_headers")
    headers = dict(value)
    if not all(
        isinstance(key, str) and isinstance(item, str) for key, item in headers.items()
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
        CONF_MCP_ALLOWED_TOOLS: parse_allowed_tools(data.get(CONF_MCP_ALLOWED_TOOLS)),
    }


class _OriginGuardTransport(httpx.AsyncBaseTransport):
    """HTTP transport that rejects requests outside the validated MCP origin."""

    def __init__(self, validated_url: ValidatedMCPURL) -> None:
        """Initialize the transport for a validated MCP URL."""
        self._validated_url = validated_url
        self._transport = httpx.AsyncHTTPTransport(trust_env=False)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Reject redirects or requests outside the validated MCP origin."""
        request_port = _default_port(request.url.scheme, request.url.port)
        if (
            request.url.scheme != self._validated_url.scheme
            or request.url.host != self._validated_url.hostname
            or request_port != self._validated_url.port
        ):
            raise httpx.ConnectError(
                "MCP redirects must stay on the validated origin.",
                request=request,
            )
        return await self._transport.handle_async_request(request)

    async def aclose(self) -> None:
        """Close the underlying transport."""
        await self._transport.aclose()


def _mcp_http_client_factory(
    validated_url: ValidatedMCPURL,
) -> Any:
    """Return an HTTP client factory for FastMCP Streamable HTTP connections."""

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
        follow_redirects: bool = False,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers=headers or {},
            timeout=timeout,
            auth=auth,
            follow_redirects=follow_redirects,
            trust_env=False,
            transport=_OriginGuardTransport(validated_url),
        )

    return factory


def _mcp_client(
    validated_url: ValidatedMCPURL, headers: dict[str, str], timeout: float
) -> FastMCPClient[Any]:
    """Return a FastMCP client pinned to Streamable HTTP transport."""
    transport = StreamableHttpTransport(
        validated_url.url,
        headers=headers,
        httpx_client_factory=_mcp_http_client_factory(validated_url),
    )
    return FastMCPClient(transport=transport, init_timeout=timeout, timeout=timeout)


def mcp_catalog_cache(hass: HomeAssistant) -> dict[str, Any]:
    """Return the integration-wide MCP discovery cache."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    return domain_data.setdefault("mcp_tool_cache", {})


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
    timeout: float = DEFAULT_MCP_TIMEOUT,
) -> list[dict[str, Any]]:
    """Discover tools exposed by one remote MCP server configuration."""
    config = _mcp_config_from_data(data, server_id=server_id)
    validated_url = await async_validate_mcp_url_details(hass, config[CONF_MCP_URL])
    config[CONF_MCP_URL] = validated_url.url
    allowed_tools = set(config[CONF_MCP_ALLOWED_TOOLS])
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
        tools = await toolset.list_tools()
    except TimeoutError as err:
        raise MCPValidationError(
            "timeout",
            "Timed out connecting to the MCP server.",
            server_id=server_id,
        ) from err
    except Exception as err:
        status_code = getattr(err, "status_code", None)
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
        "Discovered %s allowed MCP tools for server %s",
        len(discovered),
        server_id,
    )
    _LOGGER.debug("MCP server config used for discovery: %s", redact_for_log(config))
    return discovered


async def async_refresh_mcp_tools(
    hass: HomeAssistant,
    entry: ConfigEntry,
    subentry_id: str,
) -> list[dict[str, Any]]:
    """Refresh and cache tools for one MCP server subentry."""
    subentry = get_mcp_subentry(entry, subentry_id)
    tools = await async_discover_mcp_tools(hass, subentry)
    mcp_catalog_cache(hass)[_cache_key(entry, subentry_id)] = tools
    return tools


def cached_mcp_tools(
    hass: HomeAssistant,
    entry: ConfigEntry,
    subentry_id: str,
) -> list[dict[str, Any]] | None:
    """Return cached MCP tools for one server if available."""
    get_mcp_subentry(entry, subentry_id)
    tools = mcp_catalog_cache(hass).get(_cache_key(entry, subentry_id))
    if tools is None:
        return None
    return list(tools)


async def async_runtime_mcp_toolsets(
    hass: HomeAssistant,
    entry: ConfigEntry,
    selected_server_ids: Sequence[str] | None,
) -> tuple[list[Any], list[Any]]:
    """Return Agent MCP toolsets and HTTP clients for explicitly allowlisted servers."""
    toolsets: list[Any] = []
    http_clients: list[Any] = []
    selected_servers = set(selected_server_ids or [])
    if not selected_servers:
        return toolsets, http_clients
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
                "Select at least one allowed MCP tool before enabling this server for runtime use.",
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
            _LOGGER.info(
                "Calling MCP tool %s on server %s",
                tool_name,
                server_id,
            )
            _LOGGER.debug("MCP tool call arguments: %s", redact_for_log(tool_args))
            result = await call_tool(tool_name, tool_args)
            _LOGGER.debug("MCP tool result: %s", redact_for_log({"result": result}))
            return result

        toolset = MCPToolset(
            _mcp_client(
                validated_url,
                config[CONF_MCP_HEADERS],
                DEFAULT_MCP_TIMEOUT,
            ),
            id=server_id,
            tool_error_behavior="error",
            process_tool_call=process_tool_call,
        )
        toolsets.append(PrefixedToolset(toolset, f"mcp_{slugify(server_id)}"))
    missing_server_ids = selected_servers - configured_server_ids
    if missing_server_ids:
        missing_server_id = sorted(missing_server_ids)[0]
        raise MCPValidationError(
            "mcp_server_not_found",
            "Selected MCP server subentry was not found.",
            server_id=missing_server_id,
        )
    return toolsets, http_clients
