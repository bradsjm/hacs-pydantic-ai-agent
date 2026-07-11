"""Tests for registered integration services."""

import json
from typing import Any
from unittest.mock import AsyncMock, patch

from custom_components.pydantic_ai_agent import (
    ATTR_CONFIG_ENTRY_ID,
    SERVICE_LIST_MCP_TOOLS,
)
from custom_components.pydantic_ai_agent.const import DOMAIN
from custom_components.pydantic_ai_agent.runtime.types import WorkspaceRuntimeData
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.setup import async_setup_component
import pytest


async def test_list_mcp_tools_returns_empty_success_for_workspace_without_servers(
    hass: HomeAssistant, make_config_entry: Any
) -> None:
    """Listing tools for an empty loaded workspace succeeds without discovery."""
    entry = make_config_entry()
    entry.add_to_hass(hass)
    entry.runtime_data = WorkspaceRuntimeData(workspace_name="Workspace")
    assert await async_setup_component(hass, "homeassistant", {})
    assert await hass.config_entries.async_setup(entry.entry_id)

    with patch(
        "custom_components.pydantic_ai_agent.async_refresh_mcp_tools",
        new=AsyncMock(),
    ) as refresh_mcp_tools:
        response = await hass.services.async_call(
            DOMAIN,
            SERVICE_LIST_MCP_TOOLS,
            {ATTR_CONFIG_ENTRY_ID: entry.entry_id},
            blocking=True,
            return_response=True,
        )

    assert response == {"success": True, "servers": {}, "tools": []}
    refresh_mcp_tools.assert_not_awaited()
    json.dumps(response)


async def test_list_mcp_tools_rejects_unknown_config_entry(
    hass: HomeAssistant,
) -> None:
    """The service reports a stable validation key for an unknown workspace."""
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, DOMAIN, {})

    with pytest.raises(ServiceValidationError) as exc_info:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_LIST_MCP_TOOLS,
            {ATTR_CONFIG_ENTRY_ID: "unknown-entry"},
            blocking=True,
            return_response=True,
        )

    assert exc_info.value.translation_key == "config_entry_not_found"
