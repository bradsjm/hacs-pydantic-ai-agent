"""Shared config-flow schema helpers."""

from collections.abc import Iterable, Mapping
from typing import Any

import voluptuous as vol
from homeassistant.helpers.selector import SelectOptionDict
from homeassistant.helpers.typing import VolDictType


def _sorted_select_options(
    options: Iterable[SelectOptionDict],
) -> list[SelectOptionDict]:
    """Return selector options sorted by label, then value."""
    return sorted(
        options,
        key=lambda option: (
            str(option.get("label", "")).casefold(),
            str(option.get("value", "")).casefold(),
        ),
    )


def _flatten_section_data(
    data: Mapping[str, Any], section_keys: Iterable[str]
) -> dict[str, Any]:
    """Return form data with HA section namespaces flattened."""
    flattened = dict(data)
    for key in section_keys:
        value = flattened.pop(key, None)
        if isinstance(value, Mapping):
            flattened.update(value)
        elif value is not None:
            flattened[key] = value
    return flattened


def _section_defaults(section_schema: VolDictType) -> dict[str, Any]:
    """Return defaults for an expandable section from its nested fields."""
    defaults: dict[str, Any] = {}
    for key in section_schema:
        key_default = getattr(key, "default", vol.Undefined)
        if (
            isinstance(key, vol.Marker)
            and isinstance(key.schema, str)
            and not isinstance(key_default, vol.Undefined)
        ):
            defaults[key.schema] = key_default()
    return defaults


def _section_schema_key(section_name: str, section_schema: VolDictType) -> vol.Optional:
    """Return a section marker whose default mirrors its nested schema values."""
    return vol.Optional(section_name, default=_section_defaults(section_schema))
