"""Tests for runtime MCP tool-call timeout behavior."""

import asyncio
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any, cast

from custom_components.pydantic_ai_agent.const import (
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_TIMEOUT,
    CONF_MCP_TOOL_MODE,
    CONF_MCP_URL,
    MCP_TOOL_MODE_ALL,
    MCP_TOOL_MODE_SPECIFIED,
    SUBENTRY_TYPE_MCP_SERVER,
)
from custom_components.pydantic_ai_agent.mcp import runtime
from custom_components.pydantic_ai_agent.mcp.errors import MCPValidationError
from custom_components.pydantic_ai_agent.mcp.models import ValidatedMCPURL
from custom_components.pydantic_ai_agent.runtime.types import WorkspaceRuntimeData
from homeassistant.core import HomeAssistant
import httpx
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData
from pydantic_ai.exceptions import ModelRetry
import pytest

MakeSubentry = Callable[..., Any]
ProcessToolCall = Callable[
    [object, Callable[[str, dict[str, Any]], Awaitable[object]], str, dict[str, Any]],
    Awaitable[object],
]


class CapturingMCPToolset:
    """Fake MCP toolset that exposes constructor behavior without network I/O."""

    latest: CapturingMCPToolset | None = None

    def __init__(self, client: object, **kwargs: object) -> None:
        self.client = client
        self.kwargs = kwargs
        CapturingMCPToolset.latest = self

    def filtered(self, tool_filter: object) -> CapturingMCPToolset:
        self.tool_filter = tool_filter
        return self

    def prefixed(self, prefix: str) -> CapturingMCPToolset:
        self.prefix = prefix
        return self

    def defer_loading(self) -> CapturingMCPToolset:
        return self


async def _build_runtime_toolset(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    make_subentry: MakeSubentry,
    data: dict[str, object] | None = None,
) -> tuple[CapturingMCPToolset, dict[str, object]]:
    client_call: dict[str, object] = {}
    CapturingMCPToolset.latest = None

    async def fake_validate_mcp_url_details(_hass: HomeAssistant, url: str) -> ValidatedMCPURL:
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

    monkeypatch.setattr(runtime, "MCPToolset", CapturingMCPToolset)
    monkeypatch.setattr(runtime, "async_validate_mcp_url_details", fake_validate_mcp_url_details)
    monkeypatch.setattr(runtime, "_mcp_client", fake_mcp_client)

    subentry = make_subentry(
        title="Weather MCP",
        data={
            CONF_MCP_URL: "https://example.test/mcp",
            CONF_MCP_TIMEOUT: 20.0,
            CONF_MCP_TOOL_MODE: MCP_TOOL_MODE_ALL,
            **(data or {}),
        },
        subentry_type=SUBENTRY_TYPE_MCP_SERVER,
        subentry_id="mcp-weather",
    )
    entry = SimpleNamespace(
        entry_id="workspace-1",
        runtime_data=WorkspaceRuntimeData(workspace_name="Test workspace"),
    )

    await runtime._async_runtime_mcp_toolset_for_subentry(
        hass,
        cast(Any, entry),
        "conversation-1",
        subentry,
    )
    latest = CapturingMCPToolset.latest
    assert latest is not None
    return latest, client_call


@pytest.mark.parametrize(
    "err",
    [
        TimeoutError("timed out"),
        httpx.ReadTimeout("timed out"),
    ],
)
async def test_process_tool_call_converts_client_timeouts_to_model_retry(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    make_subentry: MakeSubentry,
    err: BaseException,
) -> None:
    """Client-side MCP tool-call timeouts are retry feedback for the model."""
    toolset, _client_call = await _build_runtime_toolset(hass, monkeypatch, make_subentry)
    process_tool_call = cast(ProcessToolCall, toolset.kwargs["process_tool_call"])

    async def call_tool(_tool_name: str, _tool_args: dict[str, Any]) -> object:
        await asyncio.sleep(0)
        raise err

    with pytest.raises(ModelRetry) as exc_info:
        await process_tool_call(None, call_tool, "weather.get", {"city": "Paris"})

    assert exc_info.value.__cause__ is err


async def test_process_tool_call_converts_mcp_timeout_error_to_model_retry(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    make_subentry: MakeSubentry,
) -> None:
    """Server-reported MCP request timeouts are retry feedback for the model."""
    toolset, _client_call = await _build_runtime_toolset(hass, monkeypatch, make_subentry)
    process_tool_call = cast(ProcessToolCall, toolset.kwargs["process_tool_call"])

    err = McpError(
        ErrorData(
            code=httpx.codes.REQUEST_TIMEOUT,
            message="Request timed out. Waited 20 seconds.",
        )
    )

    async def call_tool(_tool_name: str, _tool_args: dict[str, Any]) -> object:
        await asyncio.sleep(0)
        raise err

    with pytest.raises(ModelRetry) as exc_info:
        await process_tool_call(None, call_tool, "weather.get", {"city": "Paris"})

    assert exc_info.value.__cause__ is err


async def test_process_tool_call_keeps_non_timeout_mcp_errors_terminal(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    make_subentry: MakeSubentry,
) -> None:
    """Non-timeout MCP errors still terminate the tool call."""
    toolset, _client_call = await _build_runtime_toolset(hass, monkeypatch, make_subentry)
    process_tool_call = cast(ProcessToolCall, toolset.kwargs["process_tool_call"])
    err = McpError(ErrorData(code=500, message="Server error"))

    async def call_tool(_tool_name: str, _tool_args: dict[str, Any]) -> object:
        await asyncio.sleep(0)
        raise err

    with pytest.raises(McpError) as exc_info:
        await process_tool_call(None, call_tool, "weather.get", {"city": "Paris"})

    assert exc_info.value is err
    assert exc_info.value.error.code == 500


async def test_runtime_mcp_toolset_uses_configured_timeout(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    make_subentry: MakeSubentry,
) -> None:
    """Runtime MCP toolsets pass the configured timeout."""
    _toolset, client_call = await _build_runtime_toolset(hass, monkeypatch, make_subentry)

    assert client_call["timeout"] == 20.0


async def test_process_tool_call_rejects_non_allowlisted_tools_terminally(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    make_subentry: MakeSubentry,
) -> None:
    """Allowlist violations are configuration errors, not model retry feedback."""
    toolset, _client_call = await _build_runtime_toolset(
        hass,
        monkeypatch,
        make_subentry,
        {
            CONF_MCP_TOOL_MODE: MCP_TOOL_MODE_SPECIFIED,
            CONF_MCP_ALLOWED_TOOLS: ["weather.allowed"],
        },
    )
    process_tool_call = cast(ProcessToolCall, toolset.kwargs["process_tool_call"])

    async def call_tool(_tool_name: str, _tool_args: dict[str, Any]) -> object:
        await asyncio.sleep(0)
        pytest.fail("non-allowlisted tools must not be called")

    with pytest.raises(MCPValidationError) as exc_info:
        await process_tool_call(None, call_tool, "weather.get", {"city": "Paris"})

    assert exc_info.value.reason == "mcp_tool_not_allowed"
    assert exc_info.value.tool_name == "weather.get"
