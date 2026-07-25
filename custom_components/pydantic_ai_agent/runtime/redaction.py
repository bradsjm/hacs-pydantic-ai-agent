"""Shared redaction helpers for Pydantic AI Agent."""

from collections.abc import Iterable, Mapping, Sequence
from typing import cast

from homeassistant.const import CONF_API_KEY, CONF_PASSWORD

from ..const import (
    CONF_LOGFIRE_TOKEN,
    CONF_MCP_HEADERS,
    CONF_MCP_SECRET_HEADER_KEYS,
    CONF_MCP_URL,
    CONF_PROMPT,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_SECRET_HEADER_KEYS,
    CONF_SKILL_CONTENT,
)
from .header_metadata import REDACTED, mask_secret_header_values

TO_REDACT = frozenset(
    {
        CONF_API_KEY,
        CONF_LOGFIRE_TOKEN,
        CONF_MCP_URL,
        CONF_PASSWORD,
        CONF_PROMPT,
        CONF_SKILL_CONTENT,
        "Authorization",
        "Cookie",
        "X-API-Key",
        "access_token",
        "api_key",
        "auth",
        "authorization",
        "client_secret",
        "cookie",
        "extra_headers",
        "headers",
        "password",
        "refresh_token",
        "request_headers",
        "response_headers",
        "secret",
        "token",
        "x-api-key",
    }
)


def redaction_keys(extra_sensitive_keys: Iterable[object] = ()) -> frozenset[object]:
    """Return the integration-wide set of sensitive mapping keys."""
    return TO_REDACT | frozenset(extra_sensitive_keys)


def redact_data[T](data: T, extra_sensitive_keys: Iterable[object] = ()) -> T:
    """Return a copy of data with sensitive values redacted by Home Assistant."""
    return cast(T, _redact_value(data, redaction_keys(extra_sensitive_keys)))


def _redact_value(value: object, sensitive_keys: frozenset[object]) -> object:
    """Return a deeply redacted copy of a value."""
    if isinstance(value, Mapping):
        redacted: dict[object, object] = {}
        for key, item in value.items():
            if key == CONF_PROVIDER_HEADERS:
                redacted[key] = mask_secret_header_values(item, value.get(CONF_PROVIDER_SECRET_HEADER_KEYS))
                continue
            if key == CONF_MCP_HEADERS:
                redacted[key] = mask_secret_header_values(item, value.get(CONF_MCP_SECRET_HEADER_KEYS))
                continue
            if key in sensitive_keys:
                redacted[key] = REDACTED
                continue
            redacted[key] = _redact_value(item, sensitive_keys)
        return redacted
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_redact_value(item, sensitive_keys) for item in value]
    return value
