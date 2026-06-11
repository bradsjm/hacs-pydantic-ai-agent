"""Shared redaction helpers for Pydantic AI Agent."""

from collections.abc import Iterable

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.const import CONF_API_KEY, CONF_PASSWORD

from .const import (
    CONF_LOGFIRE_TOKEN,
    CONF_PROMPT,
    CONF_PROVIDER_HEADERS,
    CONF_SKILL_CONTENT,
)

TO_REDACT = frozenset(
    {
        CONF_API_KEY,
        CONF_LOGFIRE_TOKEN,
        CONF_PASSWORD,
        CONF_PROMPT,
        CONF_PROVIDER_HEADERS,
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
    return async_redact_data(data, redaction_keys(extra_sensitive_keys))
