"""Tests for MCP server subentry flows."""

import asyncio
from unittest.mock import patch

import pytest
import voluptuous_serialize
from custom_components.pydantic_ai_agent.config_flows import mcp_server_flow
from custom_components.pydantic_ai_agent.const import (
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_URL,
    CONF_NAME,
    SUBENTRY_TYPE_MCP_SERVER,
)
from custom_components.pydantic_ai_agent.mcp import MCPValidationError
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import config_validation as cv
from tests.components.pydantic_ai_agent.support.wizard import (
    loaded_workspace_entry,
    subentry_configure_result,
    subentry_init_result,
)


async def test_mcp_server_validation_success_creates_entry(
    hass: HomeAssistant,
) -> None:
    """Test MCP validation success creates the server immediately."""
    entry = await loaded_workspace_entry(hass)

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
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM

    with patch(
        "custom_components.pydantic_ai_agent.config_flows.mcp_server_flow.async_discover_mcp_tools_from_config",
        new=discover_tools,
    ):
        result = await subentry_configure_result(
            hass,
            result["flow_id"],
            {CONF_NAME: "Echo MCP", CONF_MCP_URL: "https://mcp.example.com/mcp"},
        )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Echo MCP"
    assert CONF_MCP_ALLOWED_TOOLS not in result["data"]


async def test_mcp_server_validation_known_failure_returns_form_error(
    hass: HomeAssistant,
) -> None:
    """Test MCP validation errors return to the form instead of hanging."""
    entry = await loaded_workspace_entry(hass)

    async def fail_discovery(*_args: object, **_kwargs: object) -> list[dict[str, str]]:
        raise MCPValidationError(
            "cannot_connect",
            "Could not connect to the MCP server.",
            status_code=502,
        )

    result = await subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_MCP_SERVER),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM

    with patch(
        "custom_components.pydantic_ai_agent.config_flows.mcp_server_flow.async_discover_mcp_tools_from_config",
        new=fail_discovery,
    ):
        result = await subentry_configure_result(
            hass,
            result["flow_id"],
            {CONF_NAME: "Echo MCP", CONF_MCP_URL: "https://mcp.example.com/mcp"},
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"
    assert result["errors"] == {"base": "cannot_connect"}
    assert result["description_placeholders"] == {
        "error_message": "Could not connect to the MCP server.",
        "status_code": "502",
    }


async def test_mcp_server_validation_hard_timeout_returns_form_error(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test flow-level MCP validation timeout returns to the form."""
    entry = await loaded_workspace_entry(hass)
    monkeypatch.setattr(mcp_server_flow, "DEFAULT_MCP_TIMEOUT", 0.001)

    async def hang_discovery(*_args: object, **_kwargs: object) -> list[dict[str, str]]:
        await asyncio.sleep(60)
        return []

    result = await subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_MCP_SERVER),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM

    with patch(
        "custom_components.pydantic_ai_agent.config_flows.mcp_server_flow.async_discover_mcp_tools_from_config",
        new=hang_discovery,
    ):
        result = await subentry_configure_result(
            hass,
            result["flow_id"],
            {CONF_NAME: "Echo MCP", CONF_MCP_URL: "https://mcp.example.com/mcp"},
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"
    assert result["errors"] == {"base": "timeout"}


async def test_mcp_server_validation_exception_returns_form_error(
    hass: HomeAssistant,
) -> None:
    """Test MCP validation task failures return to the form instead of hanging."""
    entry = await loaded_workspace_entry(hass)

    async def fail_discovery(*_args: object, **_kwargs: object) -> list[dict[str, str]]:
        raise RuntimeError("boom")

    result = await subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_MCP_SERVER),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM

    with patch(
        "custom_components.pydantic_ai_agent.config_flows.mcp_server_flow.async_discover_mcp_tools_from_config",
        new=fail_discovery,
    ):
        result = await subentry_configure_result(
            hass,
            result["flow_id"],
            {CONF_NAME: "Echo MCP", CONF_MCP_URL: "https://mcp.example.com/mcp"},
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"
    assert result["errors"] == {"base": "unknown"}


async def test_mcp_server_validation_logs_underlying_import_error(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test MCP validation logs the underlying import error type."""
    entry = await loaded_workspace_entry(hass)

    async def fail_discovery(*_args: object, **_kwargs: object) -> list[dict[str, str]]:
        try:
            raise ImportError("FastMCP server support is not installed")
        except ImportError as err:
            raise MCPValidationError(
                "cannot_connect",
                "Could not connect to the MCP server.",
            ) from err

    result = await subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_MCP_SERVER),
        context={"source": config_entries.SOURCE_USER},
    )

    with (
        patch(
            "custom_components.pydantic_ai_agent.config_flows.mcp_server_flow.async_discover_mcp_tools_from_config",
            new=fail_discovery,
        ),
        caplog.at_level("WARNING"),
    ):
        await subentry_configure_result(
            hass,
            result["flow_id"],
            {CONF_NAME: "Echo MCP", CONF_MCP_URL: "https://mcp.example.com/mcp"},
        )

    assert "cause=ImportError" in caplog.text
    assert "FastMCP server support is not installed" in caplog.text


async def test_mcp_server_reconfigure_menu_exposes_tool_management(
    hass: HomeAssistant,
) -> None:
    """Test MCP reconfigure separates server editing from tool selection."""
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

    result = await subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_MCP_SERVER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": subentry.subentry_id,
        },
    )

    assert result["type"] is FlowResultType.MENU
    assert result["menu_options"] == ["edit_server", "manage_tools"]


async def test_mcp_server_manage_tools_defaults_to_all_tools_and_saves_subset(
    hass: HomeAssistant,
) -> None:
    """Test MCP tool management defaults to all discovered tools and saves subset."""
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
            },
            {
                "name": "fetch",
                "description": "Fetch data",
                "input_schema": {"type": "object"},
            },
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

        assert result["type"] is FlowResultType.FORM
        assert result["step_id"] == "manage_tools"
        schema = voluptuous_serialize.convert(
            result["data_schema"], custom_serializer=cv.custom_serializer
        )
        assert isinstance(schema, list)
        tool_field = next(
            field for field in schema if field["name"] == CONF_MCP_ALLOWED_TOOLS
        )
        assert tool_field["default"] == ["echo", "fetch"]

        result = await subentry_configure_result(
            hass,
            result["flow_id"],
            {CONF_MCP_ALLOWED_TOOLS: ["echo"]},
        )

    assert result["type"] is FlowResultType.ABORT
    updated_subentry = entry.subentries[subentry.subentry_id]
    assert updated_subentry.data[CONF_MCP_ALLOWED_TOOLS] == ["echo"]


async def test_mcp_server_manage_tools_requires_at_least_one_tool(
    hass: HomeAssistant,
) -> None:
    """Test MCP tool management rejects saving an empty allowlist."""
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
        result = await subentry_configure_result(
            hass,
            result["flow_id"],
            {CONF_MCP_ALLOWED_TOOLS: []},
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "manage_tools"
    assert result["errors"] == {CONF_MCP_ALLOWED_TOOLS: "mcp_tools_not_allowlisted"}


async def test_mcp_server_reconfigure_validation_uses_discovered_tools_for_no_tools(
    hass: HomeAssistant,
) -> None:
    """Test reconfigure fails when discovery finds no tools."""
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
                    CONF_MCP_ALLOWED_TOOLS: ["stale_tool"],
                },
            },
        ),
    )
    subentry = next(iter(entry.subentries.values()))

    async def discover_no_tools(
        *_args: object, **_kwargs: object
    ) -> list[dict[str, object]]:
        return []

    result = await subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_MCP_SERVER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": subentry.subentry_id,
        },
    )
    result = await subentry_configure_result(
        hass,
        result["flow_id"],
        {"next_step_id": "edit_server"},
    )

    with patch(
        "custom_components.pydantic_ai_agent.config_flows.mcp_server_flow.async_discover_mcp_tools_from_config",
        new=discover_no_tools,
    ):
        result = await subentry_configure_result(
            hass,
            result["flow_id"],
            {CONF_NAME: "Echo MCP", CONF_MCP_URL: "https://mcp.example.com/mcp"},
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "edit_server"
    assert result["errors"] == {"base": "no_mcp_tools"}
