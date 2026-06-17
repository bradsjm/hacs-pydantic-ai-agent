"""Voluptuous schema helpers for config-flow tests."""

from typing import Any

import voluptuous as vol
import voluptuous_serialize
from homeassistant.helpers import config_validation as cv


def schema_default(data_schema: vol.Schema | None, field: str) -> Any:
    """Return a voluptuous top-level field default from a flow schema."""
    assert data_schema is not None
    for key in data_schema.schema:
        if key.schema == field:
            return key.default()
    raise AssertionError(f"Schema field {field} not found")


def schema_select_options(data_schema: vol.Schema | None, field: str) -> list[Any]:
    """Return selector options for a voluptuous top-level field."""
    assert data_schema is not None
    for key, selector in data_schema.schema.items():
        if key.schema == field:
            return list(selector.config["options"])
    raise AssertionError(f"Schema field {field} not found")


def schema_select_custom_value(data_schema: vol.Schema | None, field: str) -> bool:
    """Return whether a top-level select field allows free-text values."""
    assert data_schema is not None
    for key, selector in data_schema.schema.items():
        if key.schema == field:
            return bool(selector.config.get("custom_value", False))
    raise AssertionError(f"Schema field {field} not found")


def schema_key_names(data_schema: vol.Schema | None) -> set[str]:
    """Return top-level field names from a flow schema."""
    assert data_schema is not None
    return {key.schema for key in data_schema.schema}


def section_key_names(data_schema: vol.Schema | None, section_name: str) -> list[str]:
    """Return section field names from a flow schema in display order."""
    assert data_schema is not None
    for section_key, section_value in data_schema.schema.items():
        if section_key.schema == section_name:
            return [field_key.schema for field_key in section_value.schema.schema]
    raise AssertionError(f"Section {section_name} not found")


def section_field_suggested_value(
    data_schema: vol.Schema | None, section_name: str, field: str
) -> Any:
    """Return a sectioned field suggested value from a flow schema."""
    assert data_schema is not None
    for section_key, section_value in data_schema.schema.items():
        if section_key.schema != section_name:
            continue
        for field_key in section_value.schema.schema:
            if field_key.schema == field:
                assert field_key.description is not None
                return field_key.description["suggested_value"]
    raise AssertionError(f"Section field {section_name}.{field} not found")


def section_default(data_schema: vol.Schema | None, section_name: str) -> Any:
    """Return a section default from a flow schema."""
    assert data_schema is not None
    for section_key in data_schema.schema:
        if section_key.schema == section_name:
            return section_key.default()
    raise AssertionError(f"Section {section_name} not found")


def serialized_section_default(
    data_schema: vol.Schema | None, section_name: str
) -> Any:
    """Return the section default serialized for the config-flow frontend."""
    assert data_schema is not None
    serialized_schema = voluptuous_serialize.convert(
        data_schema, custom_serializer=cv.custom_serializer
    )
    assert isinstance(serialized_schema, list)
    for field in serialized_schema:
        if field["name"] == section_name:
            return field.get("default")
    raise AssertionError(f"Serialized section {section_name} not found")
