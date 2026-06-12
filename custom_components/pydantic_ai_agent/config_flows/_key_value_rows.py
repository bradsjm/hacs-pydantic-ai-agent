"""Shared key/value row helpers for config-flow object selectors."""

import json
from collections.abc import Mapping
from typing import Any

from ..const import (
    CONF_KEY_VALUE_JSON_VALUE,
    CONF_KEY_VALUE_KEY,
    CONF_KEY_VALUE_VALUE,
)


def _format_key_value_text_rows(value: object) -> list[dict[str, str]]:
    """Return stored key/value text data in selector-compatible shape."""
    if isinstance(value, list):
        formatted: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            key = item.get(CONF_KEY_VALUE_KEY)
            row_value = item.get(CONF_KEY_VALUE_VALUE)
            if isinstance(key, str) and isinstance(row_value, str):
                formatted.append(
                    {
                        CONF_KEY_VALUE_KEY: key,
                        CONF_KEY_VALUE_VALUE: row_value,
                    }
                )
        return formatted
    if not isinstance(value, Mapping):
        return []
    formatted: list[dict[str, str]] = []
    for key in sorted(value):
        item = value[key]
        if isinstance(key, str) and isinstance(item, str):
            formatted.append(
                {
                    CONF_KEY_VALUE_KEY: key,
                    CONF_KEY_VALUE_VALUE: item,
                }
            )
    return formatted


def _format_key_value_json_rows(value: object) -> list[dict[str, str]]:
    """Return stored key/value JSON data in selector-compatible shape."""
    if isinstance(value, list):
        formatted: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            key = item.get(CONF_KEY_VALUE_KEY)
            row_value = item.get(CONF_KEY_VALUE_JSON_VALUE)
            if isinstance(key, str) and isinstance(row_value, str):
                formatted.append(
                    {
                        CONF_KEY_VALUE_KEY: key,
                        CONF_KEY_VALUE_JSON_VALUE: row_value,
                    }
                )
        return formatted
    if not isinstance(value, Mapping):
        return []
    formatted: list[dict[str, str]] = []
    for key in sorted(value):
        if isinstance(key, str):
            formatted.append(
                {
                    CONF_KEY_VALUE_KEY: key,
                    CONF_KEY_VALUE_JSON_VALUE: json.dumps(value[key], sort_keys=True),
                }
            )
    return formatted


def _parse_key_value_text_rows(value: object) -> dict[str, str]:
    """Return selector key/value text rows as a mapping."""
    if value in (None, ""):
        return {}
    if isinstance(value, Mapping):
        parsed = dict(value)
        if not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in parsed.items()
        ):
            raise ValueError("invalid_key_value")
        return parsed
    if not isinstance(value, list):
        raise ValueError("invalid_key_value")
    parsed: dict[str, str] = {}
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("invalid_key_value")
        key = str(item.get(CONF_KEY_VALUE_KEY, "")).strip()
        row_value = item.get(CONF_KEY_VALUE_VALUE)
        if not key and row_value in (None, ""):
            continue
        if not key or not isinstance(row_value, str):
            raise ValueError("invalid_key_value")
        if key in parsed:
            raise ValueError("duplicate_key")
        parsed[key] = row_value
    return parsed


def _parse_key_value_json_rows(value: object) -> dict[str, Any]:
    """Return selector key/value JSON rows as a mapping."""
    if value in (None, ""):
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, list):
        raise ValueError("invalid_key_value")
    parsed: dict[str, Any] = {}
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("invalid_key_value")
        key = str(item.get(CONF_KEY_VALUE_KEY, "")).strip()
        row_value = item.get(CONF_KEY_VALUE_JSON_VALUE)
        if not key and row_value in (None, ""):
            continue
        if not key or not isinstance(row_value, str):
            raise ValueError("invalid_key_value")
        if key in parsed:
            raise ValueError("duplicate_key")
        try:
            parsed[key] = json.loads(row_value.strip())
        except json.JSONDecodeError as err:
            raise ValueError("invalid_json") from err
    return parsed
