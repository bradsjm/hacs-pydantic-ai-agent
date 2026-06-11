"""Tests for MCP runtime toolset helpers."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from custom_components.pydantic_ai_agent.const import (
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_DEFERRED_LOADING,
    CONF_MCP_HEADERS,
    CONF_MCP_INCLUDE_RETURN_SCHEMA,
    CONF_MCP_URL,
    DOMAIN,
    SUBENTRY_TYPE_MCP_SERVER,
)
from custom_components.pydantic_ai_agent.mcp import (
    MCPValidationError,
    async_runtime_mcp_toolsets,
)
from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry


def _mcp_entry(
    *,
    subentry_id: str = "mcp_server_1",
    allowed_tools: list[str] | None = None,
    include_return_schema: bool | None = None,
    deferred_loading: bool | None = None,
) -> MockConfigEntry:
    """Return a config entry with one MCP server subentry."""
    data: dict[str, object] = {
        CONF_NAME: "Echo MCP",
        CONF_MCP_URL: "https://mcp.example.com/mcp",
        CONF_MCP_HEADERS: {"Authorization": "Bearer secret"},
        CONF_MCP_ALLOWED_TOOLS: allowed_tools or [],
    }
    if include_return_schema is not None:
        data[CONF_MCP_INCLUDE_RETURN_SCHEMA] = include_return_schema
    if deferred_loading is not None:
        data[CONF_MCP_DEFERRED_LOADING] = deferred_loading
    return MockConfigEntry(
        domain=DOMAIN,
        title="Workspace",
        data={CONF_NAME: "Workspace"},
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "subentry_id": subentry_id,
                "data": data,
                "subentry_type": SUBENTRY_TYPE_MCP_SERVER,
                "title": "Echo MCP",
                "unique_id": None,
            },
        ),
        options={},
        unique_id=None,
    )


async def test_runtime_mcp_toolsets_require_selected_allowlisted_servers(
    hass: HomeAssistant,
) -> None:
    assert await async_runtime_mcp_toolsets(hass, _mcp_entry(), []) == []

    with pytest.raises(MCPValidationError) as err:
        await async_runtime_mcp_toolsets(hass, _mcp_entry(), ["missing"])
    assert err.value.reason == "mcp_server_not_found"

    with pytest.raises(MCPValidationError) as err:
        await async_runtime_mcp_toolsets(hass, _mcp_entry(), ["mcp_server_1"])
    assert err.value.reason == "mcp_tools_not_allowlisted"


async def test_runtime_mcp_toolsets_enforce_allowlist_and_deferred_loading(
    hass: HomeAssistant,
) -> None:
    class FakePrefixedToolset:
        def __init__(self, toolset: object, prefix: str) -> None:
            self.toolset = toolset
            self.prefix = prefix
            self.deferred = False

        def defer_loading(self) -> FakePrefixedToolset:
            self.deferred = True
            return self

    class FakeMCPToolset:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            self.process_tool_call = kwargs["process_tool_call"]
            self.include_return_schema = kwargs["include_return_schema"]
            self.filter_func = None

        def filtered(self, filter_func: object) -> FakeMCPToolset:
            self.filter_func = filter_func
            return self

        def prefixed(self, prefix: str) -> FakePrefixedToolset:
            return FakePrefixedToolset(self, prefix)

    entry = _mcp_entry(
        allowed_tools=["echo"],
        include_return_schema=False,
        deferred_loading=True,
    )
    with (
        patch(
            "custom_components.pydantic_ai_agent.mcp.runtime.MCPToolset",
            FakeMCPToolset,
        ),
        patch(
            "custom_components.pydantic_ai_agent.mcp.runtime._mcp_client",
            return_value=object(),
        ),
    ):
        toolsets = await async_runtime_mcp_toolsets(hass, entry, ["mcp_server_1"])

    toolset = cast(Any, toolsets[0])
    assert len(toolsets) == 1
    assert toolset.prefix == "mcp_mcp_server_1"
    assert toolset.deferred is True
    assert toolset.toolset.include_return_schema is False
    assert toolset.toolset.filter_func is not None
    assert toolset.toolset.filter_func(None, SimpleNamespace(name="echo")) is True
    assert toolset.toolset.filter_func(None, SimpleNamespace(name="hidden")) is False

    async def call_tool(
        tool_name: str, tool_args: dict[str, object]
    ) -> dict[str, object]:
        return {"tool": tool_name, "args": tool_args}

    assert await toolset.toolset.process_tool_call(
        None, call_tool, "echo", {"message": "hi"}
    ) == {"tool": "echo", "args": {"message": "hi"}}

    with pytest.raises(MCPValidationError) as err:
        await toolset.toolset.process_tool_call(
            None, call_tool, "read_file", {"path": "/tmp/x"}
        )
    assert err.value.reason == "mcp_tool_not_allowed"
    assert err.value.tool_name == "read_file"
