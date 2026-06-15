"""Test fixtures for Pydantic AI Agent."""

from collections.abc import Iterator
from unittest.mock import patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from pydantic_ai.models.test import TestModel
from tests.components.pydantic_ai_agent.support.pydantic_ai import Agent


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Enable custom integrations for all tests."""


@pytest.fixture(autouse=True)
async def initialize_homeassistant_component(hass: HomeAssistant) -> None:
    """Initialize exposed-entity storage required by conversation setup."""
    assert await async_setup_component(hass, "homeassistant", {})


@pytest.fixture
def mock_chat_model_for_profile() -> Iterator[TestModel]:
    """Patch the entity chat model lookup with a realistic fake model."""
    model = TestModel(custom_output_text="fixture response")
    with patch(
        "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
        return_value=model,
    ):
        yield model


@pytest.fixture
def mock_entity_agent() -> Iterator[tuple[Agent, object]]:
    """Patch the entity Agent constructor with a reusable test double."""
    agent = Agent()
    with patch(
        "custom_components.pydantic_ai_agent.entity.Agent",
        return_value=agent,
    ) as agent_class:
        yield agent, agent_class
