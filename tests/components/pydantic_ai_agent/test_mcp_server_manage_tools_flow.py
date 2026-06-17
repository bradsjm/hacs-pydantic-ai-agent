"""Tests for MCP server tool-management flows."""

from unittest.mock import patch

from custom_components.pydantic_ai_agent.const import (
    CONF_MCP_URL,
    CONF_NAME,
    SUBENTRY_TYPE_MCP_SERVER,
)
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from tests.components.pydantic_ai_agent.support.wizard import (
    loaded_workspace_entry,
    subentry_configure_result,
    subentry_init_result,
)


async def test_mcp_server_manage_tools_starts_with_progress(
    hass: HomeAssistant,
) -> None:
    """Test MCP tool management shows progress before discovery."""
    entry = await loaded_workspace_entry(
        hass,
        (
            {
                "subentry_id": "mcp-1",
                "subentry_type": SUBENTRY_TYPE_MCP_SERVER,
                "title": "Echo MCP",
                "unique_id": None,
                "data": {
                    CONF_NAME: "Echo MCP",
                    CONF_MCP_URL: "https://mcp.example.com/mcp",
                },
            },
        ),
    )
    subentry = next(iter(entry.subentries.values()))

    async def discover_tools(
        *_args: object, **_kwargs: object
    ) -> list[dict[str, object]]:
        return [
            {
                "name": "echo",
                "description": "Echo a message",
                "input_schema": {"type": "object"},
            }
        ]

    result = await subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_MCP_SERVER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": subentry.subentry_id,
        },
    )

    with patch(
        "custom_components.pydantic_ai_agent.config_flows.mcp_server_flow.async_discover_mcp_tools_from_config",
        new=discover_tools,
    ):
        result = await subentry_configure_result(
            hass,
            result["flow_id"],
            {"next_step_id": "manage_tools"},
        )
        assert result["type"] is FlowResultType.SHOW_PROGRESS
        assert result["step_id"] == "validate_mcp_server_progress"
        assert result["progress_action"] == "discover_mcp_tools"
