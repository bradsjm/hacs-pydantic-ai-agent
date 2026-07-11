"""Shared fixtures and builders for the pydantic_ai_agent test suite.

Most unit tests in this suite target pure algorithmic helpers and need no
Home Assistant fixture at all. This conftest provides:

* ``enable_custom_integrations`` autouse so any test that *does* need ``hass``
  together with the custom integration works without per-test boilerplate.
* ``make_profile`` — a factory for ``ResolvedModelProfile`` with sane defaults.
* ``make_subentry`` — a factory for real ``ConfigSubentry`` objects used by the
  subentry-aware model/MCP/skill helpers.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

pytest_plugins = ("pytest_homeassistant_custom_component",)


@pytest.fixture(autouse=True)
def _auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Enable custom integration loading for every test in the suite."""


def _profile_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return constructor kwargs for ``ResolvedModelProfile`` with defaults."""
    base: dict[str, Any] = {
        "ref": "provider-1:default",
        "provider_subentry_id": "provider-1",
        "profile_id": "default",
        "title": "Default Profile",
        "provider_title": "Test Provider",
        "provider_mode": "openai_compatible_completions",
        "model_name": "test-model",
        "model_settings": {},
    }
    base.update(overrides)
    return base


@pytest.fixture
def make_profile() -> Any:
    """Factory that builds a ``ResolvedModelProfile`` from overrides."""
    from custom_components.pydantic_ai_agent.models.model_profiles import (
        ResolvedModelProfile,
    )

    def _factory(**overrides: Any) -> Any:
        return ResolvedModelProfile(**_profile_kwargs(**overrides))

    return _factory


@pytest.fixture
def make_subentry() -> Any:
    """Factory that builds a real ``ConfigSubentry`` for subentry-aware helpers."""
    from homeassistant.config_entries import ConfigSubentry

    def _factory(
        *,
        data: Mapping[str, Any],
        subentry_type: str,
        title: str = "Subentry",
        subentry_id: str | None = None,
        unique_id: str | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "data": MappingProxyType(data),
            "subentry_type": subentry_type,
            "title": title,
            "unique_id": unique_id,
        }
        if subentry_id is not None:
            kwargs["subentry_id"] = subentry_id
        return ConfigSubentry(**kwargs)

    return _factory


@pytest.fixture
def make_config_entry() -> Any:
    """Factory that builds a current-version workspace config entry."""
    from custom_components.pydantic_ai_agent.const import CONF_NAME, DOMAIN
    from homeassistant.config_entries import ConfigEntryState, ConfigSubentry

    def _factory(
        *,
        name: str = "Workspace",
        title: str = "Workspace",
        data: Mapping[str, Any] | None = None,
        subentries: Sequence[ConfigSubentry] = (),
        entry_id: str | None = None,
        state: ConfigEntryState | None = None,
    ) -> MockConfigEntry:
        return MockConfigEntry(
            domain=DOMAIN,
            title=title,
            data={CONF_NAME: name, **(data or {})},
            entry_id=entry_id,
            subentries_data=(
                {
                    "data": subentry.data,
                    "subentry_id": subentry.subentry_id,
                    "subentry_type": subentry.subentry_type,
                    "title": subentry.title,
                    "unique_id": subentry.unique_id,
                }
                for subentry in subentries
            ),
            version=2,
            minor_version=5,
            state=state,
        )

    return _factory
