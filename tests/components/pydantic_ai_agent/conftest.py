"""Test fixtures for Pydantic AI Agent."""

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Enable custom integrations for all tests."""


@pytest.fixture(autouse=True)
async def initialize_homeassistant_component(hass: HomeAssistant) -> None:
    """Initialize exposed-entity storage required by conversation setup."""
    assert await async_setup_component(hass, "homeassistant", {})
