"""Tests for MCP discovery helpers."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from custom_components.pydantic_ai_agent.const import (
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_URL,
    DOMAIN,
    SUBENTRY_TYPE_MCP_SERVER,
)
from custom_components.pydantic_ai_agent.mcp import (
    MCPValidationError,
    async_discover_mcp_tools_from_config,
    schema_hash,
)
from custom_components.pydantic_ai_agent.mcp.discovery import (
    _cache_key,
    cached_mcp_tools,
    mcp_catalog_cache,
)
from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
import httpx
import pytest
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
        CONF_MCP_ALLOWED_TOOLS: allowed_tools or [],
    }
    if include_return_schema is not None:
        data["mcp_include_return_schema"] = include_return_schema
    if deferred_loading is not None:
        data["mcp_deferred_loading"] = deferred_loading
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


def test_cached_mcp_tools_returns_copy_and_validates_entry_state() -> None:
    entry = _mcp_entry()

    with pytest.raises(MCPValidationError, match="config entry is not loaded"):
        cached_mcp_tools(entry, "mcp_server_1")

    entry.runtime_data = SimpleNamespace(mcp_tool_cache={})
    assert cached_mcp_tools(entry, "mcp_server_1") is None

    cache_key = _cache_key(entry, "mcp_server_1")
    entry.runtime_data.mcp_tool_cache[cache_key] = [{"name": "echo"}]
    cached = cached_mcp_tools(entry, "mcp_server_1")

    assert cached == [{"name": "echo"}]
    assert cached is not entry.runtime_data.mcp_tool_cache[cache_key]


def test_mcp_catalog_cache_uses_entry_runtime_data_storage() -> None:
    entry = _mcp_entry()
    cache: dict[str, object] = {}
    entry.runtime_data = SimpleNamespace(mcp_tool_cache=cache)

    assert mcp_catalog_cache(entry) is cache


async def test_discover_mcp_tools_from_config_shapes_and_filters_tools(
    hass: HomeAssistant,
) -> None:
    class FakeMCPToolset:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakeMCPToolset:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def list_tools(self) -> list[dict[str, object]]:
            return [
                {
                    "name": "echo",
                    "description": "Echo text",
                    "inputSchema": {"type": "object"},
                },
                {"name": "ignored", "inputSchema": {"type": "object"}},
                {"description": "missing name"},
            ]

    with (
        patch(
            "custom_components.pydantic_ai_agent.mcp.discovery.MCPToolset",
            FakeMCPToolset,
        ),
        patch(
            "custom_components.pydantic_ai_agent.mcp.discovery._mcp_client",
            return_value=object(),
        ),
    ):
        tools = await async_discover_mcp_tools_from_config(
            hass,
            {
                CONF_NAME: "Echo MCP",
                CONF_MCP_URL: "https://mcp.example.com/mcp",
                CONF_MCP_ALLOWED_TOOLS: ["echo"],
            },
            server_id="server-1",
        )

    assert tools == [
        {
            "server_id": "server-1",
            "server_name": "Echo MCP",
            "name": "echo",
            "description": "Echo text",
            "input_schema": {"type": "object"},
            "schema_hash": schema_hash({"type": "object"}),
        }
    ]


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (TimeoutError(), "timeout"),
        (
            httpx.HTTPStatusError(
                "Unauthorized",
                request=httpx.Request("GET", "https://mcp.example.com/mcp"),
                response=httpx.Response(401),
            ),
            "invalid_auth",
        ),
        (RuntimeError("down"), "cannot_connect"),
    ],
)
async def test_discover_mcp_tools_from_config_maps_connection_errors(
    hass: HomeAssistant, error: BaseException, reason: str
) -> None:
    class FakeMCPToolset:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakeMCPToolset:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def list_tools(self) -> list[dict[str, object]]:
            raise error

    with (
        patch(
            "custom_components.pydantic_ai_agent.mcp.discovery.MCPToolset",
            FakeMCPToolset,
        ),
        patch(
            "custom_components.pydantic_ai_agent.mcp.discovery._mcp_client",
            return_value=object(),
        ),
        pytest.raises(MCPValidationError) as err,
    ):
        await async_discover_mcp_tools_from_config(
            hass,
            {CONF_NAME: "Echo MCP", CONF_MCP_URL: "https://mcp.example.com/mcp"},
        )

    assert err.value.reason == reason


async def test_discover_mcp_tools_from_config_times_out_cleanly(
    hass: HomeAssistant,
) -> None:
    class FakeMCPToolset:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def list_tools(self) -> list[dict[str, object]]:
            await asyncio.sleep(0.05)
            return []

    with (
        patch(
            "custom_components.pydantic_ai_agent.mcp.discovery.MCPToolset",
            FakeMCPToolset,
        ),
        patch(
            "custom_components.pydantic_ai_agent.mcp.discovery._mcp_client",
            return_value=object(),
        ),
        pytest.raises(MCPValidationError) as err,
    ):
        await async_discover_mcp_tools_from_config(
            hass,
            {CONF_NAME: "Echo MCP", CONF_MCP_URL: "https://mcp.example.com/mcp"},
            request_timeout=0.01,
        )

    assert err.value.reason == "timeout"
