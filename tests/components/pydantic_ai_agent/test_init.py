"""Tests for config-entry lifecycle behavior."""

from typing import Any
from unittest.mock import AsyncMock, patch

from custom_components.pydantic_ai_agent import PLATFORMS
from custom_components.pydantic_ai_agent.runtime.types import WorkspaceRuntimeData
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component


async def test_config_entry_setup_and_unload(
    hass: HomeAssistant, make_config_entry: Any
) -> None:
    """Initialize workspace runtime, then release platforms and Logfire."""
    entry = make_config_entry(name="Visible Workspace")
    entry.add_to_hass(hass)
    assert await async_setup_component(hass, "homeassistant", {})

    with (
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new=AsyncMock(),
        ) as forward_setups,
        patch.object(
            hass.config_entries,
            "async_unload_platforms",
            new=AsyncMock(return_value=True),
        ) as unload_platforms,
        patch(
            "custom_components.pydantic_ai_agent.async_configure_logfire",
            new=AsyncMock(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.async_release_logfire",
            new=AsyncMock(),
        ) as release_logfire,
    ):
        assert await hass.config_entries.async_setup(entry.entry_id) is True

        assert isinstance(entry.runtime_data, WorkspaceRuntimeData)
        assert entry.runtime_data.workspace_name == "Visible Workspace"
        forward_setups.assert_awaited_once_with(entry, PLATFORMS)

        assert await hass.config_entries.async_unload(entry.entry_id) is True

    unload_platforms.assert_awaited_once_with(entry, PLATFORMS)
    release_logfire.assert_awaited_once_with(hass, entry)
