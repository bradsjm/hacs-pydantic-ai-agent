"""Shared helpers for header secret metadata and masking."""

from collections.abc import Iterable, Mapping
from typing import Any

from .const import (
    CONF_KEY_VALUE_IS_SECRET,
    CONF_KEY_VALUE_KEY,
    CONF_KEY_VALUE_VALUE,
)

REDACTED = "**REDACTED**"


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
            list_rows.append(
                {
                    CONF_KEY_VALUE_KEY: key,
                    CONF_KEY_VALUE_VALUE: row_value,
                    CONF_KEY_VALUE_IS_SECRET: is_secret
                    if isinstance(is_secret, bool)
                    else key.casefold() in secrets,
                }
            )
        return list_rows
    if not isinstance(headers, Mapping):
        return []
    mapping_rows: list[dict[str, str | bool]] = []
    for key in sorted(headers):
        value = headers[key]
        if isinstance(key, str) and isinstance(value, str):
            mapping_rows.append(
                {
                    CONF_KEY_VALUE_KEY: key,
                    CONF_KEY_VALUE_VALUE: value,
                    CONF_KEY_VALUE_IS_SECRET: key.casefold() in secrets,
                }
            )
    return mapping_rows


def parse_header_rows(value: object) -> tuple[dict[str, str], list[str]]:
    """Return selector header rows as a mapping and secret-key list."""
    if value in (None, ""):
        return {}, []
    if isinstance(value, Mapping):
        parsed = dict(value)
        if not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in parsed.items()
        ):
            raise ValueError("invalid_key_value")
        return parsed, []
    if not isinstance(value, list):
        raise ValueError("invalid_key_value")
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
        parsed[key] = row_value
        if is_secret:
            secret_keys.append(key)
    return parsed, secret_keys


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
