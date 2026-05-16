"""Shared redaction helpers for Pydantic AI Agent."""

from collections.abc import Iterable, Mapping
from typing import Any

from homeassistant.components.diagnostics import async_redact_data

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
    return async_redact_data(data, keys) if keys else _copy_containers(data)


def _copy_containers(data: Any) -> Any:
    """Copy nested containers before in-place URL redaction."""
    if isinstance(data, Mapping):
        return {key: _copy_containers(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_copy_containers(item) for item in data]
    return data
