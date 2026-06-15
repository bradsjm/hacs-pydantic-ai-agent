"""Shared config-flow helper test utilities."""

from collections.abc import Callable, Mapping
from typing import Any

import voluptuous as vol
from pytest_homeassistant_custom_component.common import MockConfigEntry
from tests.components.pydantic_ai_agent.support.builders import (
    model_profile_data,
    provider_subentry_data,
    workspace_entry,
)


def section_key_names(data_schema: vol.Schema, section_name: str) -> set[str]:
    """Return the field names within a schema section."""
    for section_key, section_value in data_schema.schema.items():
        if section_key.schema == section_name:
            return {key.schema for key in section_value.schema.schema}
    raise AssertionError(f"Section {section_name} not found")


def fallback_test_entry() -> MockConfigEntry:
    """Return a workspace entry with multiple provider profiles."""
    return workspace_entry(
        (
            provider_subentry_data(
                model_profiles={
                    "profile-1": model_profile_data(
                        profile_id="profile-1", name="Fast"
                    ),
                    "profile-2": model_profile_data(
                        profile_id="profile-2", name="Cheap"
                    ),
                    "profile-3": model_profile_data(
                        profile_id="profile-3", name="Backup"
                    ),
                }
            ),
        )
    )


def schema_key_names(data_schema: vol.Schema) -> set[str]:
    """Return the top-level key names for a schema."""
    return {key.schema for key in data_schema.schema}


def section_selector(data_schema: vol.Schema, section_name: str, field: str) -> object:
    """Return a selector from a named schema section."""
    for section_key, section_value in data_schema.schema.items():
        if section_key.schema != section_name:
            continue
        for field_key, selector in section_value.schema.schema.items():
            if field_key.schema == field:
                return selector
    raise AssertionError(f"Section field {section_name}.{field} not found")


def thinking_test_entry(
    provider_mode: str,
    model_name: str,
    *,
    thinking_support: str | None = None,
) -> MockConfigEntry:
    """Return a workspace entry for thinking-support tests."""
    extra_data = {}
    if provider_mode.startswith("openai_compatible") and thinking_support is not None:
        extra_data = {"thinking_support": thinking_support}
    return workspace_entry(
        (
            provider_subentry_data(
                provider_mode=provider_mode,
                model_profiles={
                    "profile-1": model_profile_data(
                        model=model_name,
                        extra_data=extra_data,
                    ),
                },
            ),
        )
    )


type SaveDataHelper = Callable[
    [Mapping[str, Any], Mapping[str, Any], MockConfigEntry | None],
    dict[str, Any],
]
