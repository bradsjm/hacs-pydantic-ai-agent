"""Shared helpers for header secret metadata and masking."""

from collections.abc import Iterable, Mapping
from typing import Any

from ..const import (
    CONF_KEY_VALUE_IS_SECRET,
    CONF_KEY_VALUE_KEY,
    CONF_KEY_VALUE_VALUE,
)

REDACTED = "**REDACTED**"
HEADER_VALUE_REDACTED = "(redacted)"


def _header_key_name_set(header_keys: object) -> set[str]:
    """Return normalized header names for case-insensitive matching."""
    if isinstance(header_keys, str):
        values: Iterable[object] = (header_keys,)
    elif isinstance(header_keys, Iterable):
        values = header_keys
    else:
        return set()
    return {
        header_name.casefold()
        for item in values
        if isinstance(item, str) and (header_name := item.strip())
    }


def format_header_rows(
    headers: object, secret_header_keys: object = ()
) -> list[dict[str, str | bool]]:
    """Return stored headers in selector-compatible row shape."""
    secrets = _header_key_name_set(secret_header_keys)
    if isinstance(headers, list):
        list_rows: list[dict[str, str | bool]] = []
        for item in headers:
            if not isinstance(item, Mapping):
                continue
            key = item.get(CONF_KEY_VALUE_KEY)
            row_value = item.get(CONF_KEY_VALUE_VALUE)
            if not isinstance(key, str) or not isinstance(row_value, str):
                continue
            is_secret = item.get(CONF_KEY_VALUE_IS_SECRET)
            is_secret = (
                is_secret if isinstance(is_secret, bool) else key.casefold() in secrets
            )
            list_rows.append(
                {
                    CONF_KEY_VALUE_KEY: key,
                    CONF_KEY_VALUE_VALUE: _header_row_value(row_value, is_secret),
                    CONF_KEY_VALUE_IS_SECRET: is_secret,
                }
            )
        return list_rows
    if not isinstance(headers, Mapping):
        return []
    mapping_rows: list[dict[str, str | bool]] = []
    for key in sorted(headers):
        value = headers[key]
        if isinstance(key, str) and isinstance(value, str):
            is_secret = key.casefold() in secrets
            mapping_rows.append(
                {
                    CONF_KEY_VALUE_KEY: key,
                    CONF_KEY_VALUE_VALUE: _header_row_value(value, is_secret),
                    CONF_KEY_VALUE_IS_SECRET: is_secret,
                }
            )
    return mapping_rows


def _header_row_value(value: str, is_secret: bool) -> str:
    """Return the config-flow row value with secrets redacted."""
    return HEADER_VALUE_REDACTED if is_secret else value


def parse_header_rows(
    value: object,
    previous_headers: object = None,
    previous_secret_header_keys: object = (),
) -> tuple[dict[str, str], list[str]]:
    """Return selector header rows as a mapping and secret-key list."""
    previous_secret_values = _previous_secret_values(
        previous_headers, previous_secret_header_keys
    )
    if value in (None, ""):
        return {}, []
    if isinstance(value, Mapping):
        parsed = dict(value)
        if not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in parsed.items()
        ):
            raise ValueError("invalid_key_value")
        return _restore_redacted_headers(parsed, previous_secret_values), []
    if not isinstance(value, list):
        raise ValueError("invalid_key_value")
    return _parse_header_row_list(value, previous_secret_values)


def _parse_header_row_list(
    value: list[object], previous_secret_values: Mapping[str, str]
) -> tuple[dict[str, str], list[str]]:
    """Return selector header row list as a mapping and secret-key list."""
    parsed: dict[str, str] = {}
    secret_keys: list[str] = []
    for item in value:
        key, row_value, is_secret = _parse_header_row(item)
        if not key and row_value in (None, ""):
            continue
        if not key or row_value is None:
            raise ValueError("invalid_key_value")
        if key in parsed:
            raise ValueError("duplicate_key")
        if row_value == HEADER_VALUE_REDACTED:
            row_value = _restore_redacted_header_value(
                key,
                row_value,
                previous_secret_values,
                require_match=is_secret,
            )
        if is_secret:
            secret_keys.append(key)
        parsed[key] = row_value
    return parsed, secret_keys


def _previous_secret_values(
    headers: object, secret_header_keys: object
) -> dict[str, str]:
    """Return previous secret header values keyed by normalized header name."""
    if isinstance(headers, list):
        headers, secret_header_keys = parse_header_rows(headers)
    if not isinstance(headers, Mapping):
        return {}
    secrets = _header_key_name_set(secret_header_keys)
    return {
        key.casefold(): value
        for key, value in headers.items()
        if isinstance(key, str)
        and isinstance(value, str)
        and key.casefold() in secrets
    }


def _restore_redacted_headers(
    headers: dict[str, str], previous_secret_values: Mapping[str, str]
) -> dict[str, str]:
    """Restore unchanged redacted placeholders from previous secret values."""
    return {
        key: _restore_redacted_header_value(key, value, previous_secret_values)
        if value == HEADER_VALUE_REDACTED
        else value
        for key, value in headers.items()
    }


def _restore_redacted_header_value(
    key: str,
    value: str,
    previous_secret_values: Mapping[str, str],
    *,
    require_match: bool = False,
) -> str:
    """Return a previous secret value for a redacted placeholder when possible."""
    if restored_value := previous_secret_values.get(key.casefold()):
        return restored_value
    if require_match and previous_secret_values:
        raise ValueError("invalid_key_value")
    return value


def normalize_secret_header_keys(
    headers: object, secret_header_keys: object
) -> list[str]:
    """Return secret header keys filtered to the current header mapping."""
    if not isinstance(headers, Mapping):
        return []
    secrets = _header_key_name_set(secret_header_keys)
    return [
        key
        for key in headers
        if isinstance(key, str)
        and isinstance(headers[key], str)
        and key.casefold() in secrets
    ]


def _parse_header_row(item: object) -> tuple[str, str | None, bool]:
    """Return one validated header row."""
    if not isinstance(item, Mapping):
        raise ValueError("invalid_key_value")
    key = str(item.get(CONF_KEY_VALUE_KEY, "")).strip()
    row_value = item.get(CONF_KEY_VALUE_VALUE)
    is_secret = item.get(CONF_KEY_VALUE_IS_SECRET, False)
    if row_value is not None and not isinstance(row_value, str):
        raise ValueError("invalid_key_value")
    if not isinstance(is_secret, bool):
        raise ValueError("invalid_key_value")
    return key, row_value, is_secret


def mask_secret_header_values(headers: object, secret_header_keys: object) -> object:
    """Return header mapping with secret-marked values redacted."""
    if not isinstance(headers, Mapping):
        return headers
    secrets = _header_key_name_set(secret_header_keys)
    masked: dict[str, Any] = {}
    for key, value in headers.items():
        if not isinstance(key, str):
            continue
        masked[key] = REDACTED if key.casefold() in secrets else value
    return masked
