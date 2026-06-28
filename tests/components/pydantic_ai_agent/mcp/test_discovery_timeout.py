"""Tests for MCP discovery timeout configuration and classification."""

import asyncio
from collections.abc import Sequence
from typing import Any

from custom_components.pydantic_ai_agent.const import (
    CONF_MCP_TIMEOUT,
    CONF_MCP_URL,
)
from custom_components.pydantic_ai_agent.mcp import discovery
from custom_components.pydantic_ai_agent.mcp.errors import MCPValidationError
from custom_components.pydantic_ai_agent.mcp.models import ValidatedMCPURL
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
import httpx
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData
import pytest


async def _discover_with_patched_toolset(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tools: Sequence[dict[str, Any]] | None = None,
    list_tools_error: BaseException | None = None,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    client_call: dict[str, object] = {}

    async def fake_validate_mcp_url_details(
        _hass: HomeAssistant, url: str
    ) -> ValidatedMCPURL:
        await asyncio.sleep(0)
        return ValidatedMCPURL(
            url=url,
            scheme="https",
            hostname="example.test",
            port=443,
        )

    def fake_mcp_client(
        validated_url: ValidatedMCPURL,
        headers: dict[str, str],
        timeout: float,
    ) -> object:
        client_call["url"] = validated_url.url
        client_call["headers"] = headers
        client_call["timeout"] = timeout
        return object()

    class FakeMCPToolset:
        def __init__(self, client: object, **kwargs: object) -> None:
            self.client = client
            self.kwargs = kwargs

        async def list_tools(self) -> Sequence[dict[str, Any]]:
            await asyncio.sleep(0)
            if list_tools_error is not None:
                raise list_tools_error
            return tools or []

    monkeypatch.setattr(
        discovery, "async_validate_mcp_url_details", fake_validate_mcp_url_details
    )
    monkeypatch.setattr(discovery, "_mcp_client", fake_mcp_client)
    monkeypatch.setattr(discovery, "MCPToolset", FakeMCPToolset)

    discovered = await discovery.async_discover_mcp_tools_from_config(
        hass,
        {
            CONF_NAME: "Weather MCP",
            CONF_MCP_URL: "https://example.test/mcp",
            CONF_MCP_TIMEOUT: 42,
        },
        server_id="mcp-weather",
    )
    return discovered, client_call


async def test_discovery_uses_configured_mcp_timeout(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MCP discovery uses the per-server timeout for outbound requests."""
    discovered, client_call = await _discover_with_patched_toolset(
        hass,
        monkeypatch,
        tools=[
            {
                "name": "weather.get",
                "description": "Get weather",
                "inputSchema": {"type": "object"},
            }
        ],
    )

    assert client_call["timeout"] == 42
    assert discovered[0]["name"] == "weather.get"


@pytest.mark.parametrize(
    ("err", "expected_reason"),
    [
        (
            McpError(
                ErrorData(
                    code=httpx.codes.REQUEST_TIMEOUT,
                    message="Request timed out. Waited 42 seconds.",
                )
            ),
            "timeout",
        ),
        (McpError(ErrorData(code=500, message="Server error")), "cannot_connect"),
    ],
)
async def test_discovery_classifies_mcp_errors_with_stable_reasons(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    err: BaseException,
    expected_reason: str,
) -> None:
    """Discovery maps timeout and non-timeout MCP errors to stable reasons."""
    with pytest.raises(MCPValidationError) as exc_info:
        await _discover_with_patched_toolset(
            hass,
            monkeypatch,
            list_tools_error=err,
        )

    assert exc_info.value.reason == expected_reason
