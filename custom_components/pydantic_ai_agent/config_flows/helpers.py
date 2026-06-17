"""Shared config-flow schema helpers."""

from collections.abc import Iterable, Mapping
from typing import Any

import voluptuous as vol
from homeassistant.helpers.selector import (
    ObjectSelector,
    ObjectSelectorConfig,
    ObjectSelectorField,
    SelectOptionDict,
)
from homeassistant.helpers.typing import VolDictType

from ..const import CONF_KEY_VALUE_IS_SECRET, CONF_KEY_VALUE_KEY


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


def _flatten_section_data_with_presence(
    data: Mapping[str, Any], section_keys: Iterable[str]
) -> tuple[dict[str, Any], set[str]]:
    """Return flattened form data and keys explicitly present in the submission."""
    flattened = dict(data)
    section_key_set = set(section_keys)
    submitted_keys = set(flattened) - section_key_set
    for key in section_key_set:
        if key not in flattened:
            continue
        value = flattened.pop(key)
        if isinstance(value, Mapping):
            flattened.update(value)
            submitted_keys.update(value)
        elif value is not None:
            flattened[key] = value
            submitted_keys.add(key)
    return flattened, submitted_keys


def _merge_submitted_optional_fields(
    storage_data: dict[str, Any],
    *,
    existing_data: Mapping[str, Any],
    validated_data: Mapping[str, Any],
    submitted_keys: set[str],
    keys: Iterable[str],
    derived_sources: Mapping[str, str] | None = None,
) -> None:
    """Merge optional fields while preserving values absent from a form submission."""
    derived_sources = derived_sources or {}
    for key in keys:
        source_key = derived_sources.get(key, key)
        if key in submitted_keys or source_key in submitted_keys:
            if key in validated_data:
                storage_data[key] = validated_data[key]
            else:
                storage_data.pop(key, None)
            continue
        if key in existing_data:
            storage_data[key] = existing_data[key]
        else:
            storage_data.pop(key, None)


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


def _key_value_rows_selector(
    value_field: str,
    value_selector: dict[str, Any],
    *,
    key_label: str | None = None,
    value_label: str | None = None,
    include_secret_toggle: bool = False,
    secret_default: bool = False,
    label_field: str | None = None,
    description_field: str | None = None,
    translation_key: str | None = None,
) -> ObjectSelector:
    """Return a repeated key/value row selector."""
    key_field = ObjectSelectorField(selector={"text": None}, required=True)
    if key_label is not None:
        key_field["label"] = key_label
    value_field_config = ObjectSelectorField(
        selector=value_selector,
        required=True,
    )
    if value_label is not None:
        value_field_config["label"] = value_label
    fields = {
        CONF_KEY_VALUE_KEY: key_field,
        value_field: value_field_config,
    }
    if include_secret_toggle:
        del secret_default
        fields[CONF_KEY_VALUE_IS_SECRET] = ObjectSelectorField(
            selector={"boolean": {}}, required=False
        )
    config = ObjectSelectorConfig(multiple=True, fields=fields)
    if label_field is not None:
        config["label_field"] = label_field
    if description_field is not None:
        config["description_field"] = description_field
    if translation_key is not None:
        config["translation_key"] = translation_key
    return ObjectSelector(config)
