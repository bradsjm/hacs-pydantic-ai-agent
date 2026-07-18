"""Tests for registered integration services."""

import json
from typing import Any
from unittest.mock import AsyncMock, patch

from custom_components.pydantic_ai_agent import (
    ATTR_CONFIG_ENTRY_ID,
    SERVICE_LIST_MCP_TOOLS,
    SERVICE_REFRESH_MCP_TOOLS,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_MCP_URL,
    DOMAIN,
    SUBENTRY_TYPE_MCP_SERVER,
)
from custom_components.pydantic_ai_agent.mcp import MCPValidationError
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.setup import async_setup_component
import pytest


async def test_list_mcp_tools_returns_discovered_tools(
    hass: HomeAssistant, make_config_entry: Any, make_subentry: Any
) -> None:
    """The list service groups and flattens newly discovered MCP tools."""
    mcp = make_subentry(
        subentry_id="mcp-weather",
        subentry_type=SUBENTRY_TYPE_MCP_SERVER,
        title="Weather",
        data={CONF_MCP_URL: "https://example.test/mcp"},
    )
    entry = make_config_entry(subentries=(mcp,))
    entry.add_to_hass(hass)
    assert await async_setup_component(hass, "homeassistant", {})
    assert await hass.config_entries.async_setup(entry.entry_id)

    tool = {
        "server_id": "mcp-weather",
        "server_name": "Weather",
        "name": "weather_forecast",
        "description": "Return a forecast",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        "schema_hash": "forecast-schema",
    }
    with patch(
        "custom_components.pydantic_ai_agent.async_refresh_mcp_tools",
        new=AsyncMock(return_value=[tool]),
    ):
        response = await hass.services.async_call(
            DOMAIN,
            SERVICE_LIST_MCP_TOOLS,
            {ATTR_CONFIG_ENTRY_ID: entry.entry_id},
            blocking=True,
            return_response=True,
        )

    assert response == {
        "success": True,
        "servers": {"mcp-weather": [tool]},
        "tools": [tool],
        "errors": [],
    }
    assert json.loads(json.dumps(response)) == response


async def test_refresh_mcp_tools_returns_structured_validation_failure(
    hass: HomeAssistant, make_config_entry: Any, make_subentry: Any
) -> None:
    """The refresh service returns stable, JSON-safe MCP failure metadata."""
    mcp = make_subentry(
        subentry_id="mcp-weather",
        subentry_type=SUBENTRY_TYPE_MCP_SERVER,
        title="Weather",
        data={CONF_MCP_URL: "https://example.test/mcp"},
    )
    entry = make_config_entry(subentries=(mcp,))
    entry.add_to_hass(hass)
    assert await async_setup_component(hass, "homeassistant", {})
    assert await hass.config_entries.async_setup(entry.entry_id)

    error = MCPValidationError(
        reason="invalid_auth",
        message="Unstable user-facing detail",
        status_code=401,
        server_id="mcp-weather",
        tool_name="weather_forecast",
    )
    with patch(
        "custom_components.pydantic_ai_agent.async_refresh_mcp_tools",
        new=AsyncMock(side_effect=error),
    ):
        response = await hass.services.async_call(
            DOMAIN,
            SERVICE_REFRESH_MCP_TOOLS,
            {ATTR_CONFIG_ENTRY_ID: entry.entry_id},
            blocking=True,
            return_response=True,
        )

    assert response["success"] is False
    assert response["servers"] == {}
    assert response["tools"] == []
    assert len(response["errors"]) == 1
    service_error = response["errors"][0]
    assert service_error["reason"] == "invalid_auth"
    assert service_error["status_code"] == 401
    assert service_error["server_id"] == "mcp-weather"
    assert service_error["tool_name"] == "weather_forecast"
    assert json.loads(json.dumps(response)) == response


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
