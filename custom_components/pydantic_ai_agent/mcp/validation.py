"""MCP parsing and validation helpers."""

# ruff: noqa: ANN401

import json
import re
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any
from urllib.parse import urlparse

import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv

from ..runtime.header_metadata import parse_header_rows
from .errors import MCPValidationError
from .models import ValidatedMCPURL

_HTTP_HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


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
    """Parse optional HTTP headers from selector rows or a stored mapping."""
    try:
        headers, _secret_header_keys = parse_header_rows(value)
    except ValueError as err:
        raise vol.Invalid(str(err)) from err
    if not all(_HTTP_HEADER_NAME_PATTERN.fullmatch(key) for key in headers):
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
