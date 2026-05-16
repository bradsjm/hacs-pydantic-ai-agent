"""Shared redaction helpers for Pydantic AI Agent."""

from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse, urlunparse

from homeassistant.helpers.redact import async_redact_data

from .const import CONF_MCP_URL

COMMON_SENSITIVE_KEYS = {
    "access_token",
    "api_key",
    "authorization",
    "cookie",
    "extra_headers",
    "headers",
    "password",
    "request_headers",
    "response_headers",
    "secret",
    "token",
    "x-api-key",
}


def redaction_keys(
    data: object, extra_sensitive_keys: Iterable[object] = ()
) -> set[object]:
    """Return nested mapping keys that should be redacted."""
    sensitive_keys = {str(key).lower() for key in COMMON_SENSITIVE_KEYS}
    sensitive_keys.update(str(key).lower() for key in extra_sensitive_keys)
    keys: set[object] = set()
    if isinstance(data, Mapping):
        for key, value in data.items():
            key_text = str(key).lower()
            if key_text in sensitive_keys or key_text.endswith(("_token", "-token")):
                keys.add(key)
            keys.update(redaction_keys(value, sensitive_keys))
    elif isinstance(data, list):
        for item in data:
            keys.update(redaction_keys(item, sensitive_keys))
    return keys


def redact_data(
    data: Any, extra_sensitive_keys: Iterable[object] = ()
) -> Any:
    """Return a copy of data with sensitive values redacted by Home Assistant."""
    keys = redaction_keys(data, extra_sensitive_keys)
    redacted = async_redact_data(data, keys) if keys else _copy_containers(data)
    redact_mcp_urls(redacted)
    return redacted


def _copy_containers(data: Any) -> Any:
    """Copy nested containers before in-place URL redaction."""
    if isinstance(data, Mapping):
        return {key: _copy_containers(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_copy_containers(item) for item in data]
    return data


def redact_mcp_urls(data: object) -> None:
    """Redact MCP URL passwords in-place without modifying query parameters."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key == CONF_MCP_URL and isinstance(value, str):
                data[key] = redact_mcp_url_password(value)
            else:
                redact_mcp_urls(value)
    elif isinstance(data, list):
        for item in data:
            redact_mcp_urls(item)


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
