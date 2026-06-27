"""Tests for MCP runtime toolset helpers."""

from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from custom_components.pydantic_ai_agent.const import (
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_CALL_CACHE_ENABLED,
    CONF_MCP_CALL_CACHE_TTL,
    CONF_MCP_DEFERRED_LOADING,
    CONF_MCP_HEADERS,
    CONF_MCP_INCLUDE_RETURN_SCHEMA,
    CONF_MCP_TOOL_MODE,
    CONF_MCP_URL,
    DOMAIN,
    MCP_TOOL_MODE_ALL,
    SUBENTRY_TYPE_MCP_SERVER,
)
from custom_components.pydantic_ai_agent.mcp import (
    MCPValidationError,
    async_runtime_mcp_toolsets,
)
from custom_components.pydantic_ai_agent.runtime.types import (
    MCPCallCacheEntry,
    WorkspaceRuntimeData,
)
from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry


class FakePrefixedToolset:
    def __init__(self, toolset: FakeMCPToolset, prefix: str) -> None:
        self.toolset = toolset
        self.prefix = prefix
        self.deferred = False

    def defer_loading(self) -> FakePrefixedToolset:
        self.deferred = True
        return self


class FakeMCPToolset:
    def __init__(self, *_args: object, **kwargs: object) -> None:
        self.process_tool_call: Callable[..., Awaitable[Any]] = cast(
            Callable[..., Awaitable[Any]], kwargs["process_tool_call"]
        )
        self.include_return_schema = kwargs["include_return_schema"]
        self.filter_func: Callable[..., bool] | None = None

    def filtered(self, filter_func: Callable[..., bool]) -> FakeMCPToolset:
        self.filter_func = filter_func
        return self

    def prefixed(self, prefix: str) -> FakePrefixedToolset:
        return FakePrefixedToolset(self, prefix)


def _mcp_subentry_data(
    *,
    subentry_id: str = "mcp_server_1",
    allowed_tools: list[str] | None = None,
    mode: str | None = None,
    store_allowed_tools: bool = False,
    call_cache_enabled: bool | None = None,
    call_cache_ttl: int | None = None,
    include_return_schema: bool | None = None,
    deferred_loading: bool | None = None,
) -> dict[str, object]:
    """Return one MCP server subentry payload."""
    data: dict[str, object] = {
        CONF_NAME: "Echo MCP",
        CONF_MCP_URL: "https://mcp.example.com/mcp",
        CONF_MCP_HEADERS: {"Authorization": "Bearer secret"},
    }
    if allowed_tools is not None or store_allowed_tools:
        data[CONF_MCP_ALLOWED_TOOLS] = allowed_tools or []
    if mode is not None:
        data[CONF_MCP_TOOL_MODE] = mode
    if call_cache_enabled is not None:
        data[CONF_MCP_CALL_CACHE_ENABLED] = call_cache_enabled
    if call_cache_ttl is not None:
        data[CONF_MCP_CALL_CACHE_TTL] = call_cache_ttl
    if include_return_schema is not None:
        data[CONF_MCP_INCLUDE_RETURN_SCHEMA] = include_return_schema
    if deferred_loading is not None:
        data[CONF_MCP_DEFERRED_LOADING] = deferred_loading
    return {
        "subentry_id": subentry_id,
        "data": data,
        "subentry_type": SUBENTRY_TYPE_MCP_SERVER,
        "title": "Echo MCP",
        "unique_id": None,
    }


def _mcp_entry(
    *,
    subentry_id: str = "mcp_server_1",
    allowed_tools: list[str] | None = None,
    mode: str | None = None,
    store_allowed_tools: bool = False,
    call_cache_enabled: bool | None = None,
    call_cache_ttl: int | None = None,
    include_return_schema: bool | None = None,
    deferred_loading: bool | None = None,
) -> MockConfigEntry:
    """Return a config entry with one MCP server subentry."""
    return MockConfigEntry(
        domain=DOMAIN,
        title="Workspace",
        data={CONF_NAME: "Workspace"},
        source=config_entries.SOURCE_USER,
        subentries_data=(
            _mcp_subentry_data(
                subentry_id=subentry_id,
                allowed_tools=allowed_tools,
                mode=mode,
                store_allowed_tools=store_allowed_tools,
                call_cache_enabled=call_cache_enabled,
                call_cache_ttl=call_cache_ttl,
                include_return_schema=include_return_schema,
                deferred_loading=deferred_loading,
            ),
        ),
        options={},
        unique_id=None,
    )


async def _async_runtime_toolsets(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    selected_server_ids: list[str],
    *,
    agent_subentry_id: str = "conversation-1",
) -> list[FakePrefixedToolset]:
    """Build fake runtime toolsets for one entry."""
    entry.runtime_data = WorkspaceRuntimeData(workspace_name="Workspace")
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
        toolsets = await async_runtime_mcp_toolsets(
            hass,
            entry,
            agent_subentry_id,
            selected_server_ids,
        )
    return cast(list[FakePrefixedToolset], toolsets)


async def test_runtime_mcp_toolsets_require_selected_servers(
    hass: HomeAssistant,
) -> None:
    entry = _mcp_entry()
    entry.runtime_data = WorkspaceRuntimeData(workspace_name="Workspace")

    assert await async_runtime_mcp_toolsets(hass, entry, "conversation-1", []) == []

    with pytest.raises(MCPValidationError) as err:
        await async_runtime_mcp_toolsets(hass, entry, "conversation-1", ["missing"])
    assert err.value.reason == "mcp_server_not_found"


async def test_runtime_mcp_toolsets_disabled_mode_contributes_no_tools(
    hass: HomeAssistant,
) -> None:
    toolsets = await _async_runtime_toolsets(
        hass,
        _mcp_entry(mode="disabled", store_allowed_tools=True),
        ["mcp_server_1"],
    )

    assert toolsets == []


async def test_runtime_mcp_toolsets_enforce_allowlist_and_deferred_loading(
    hass: HomeAssistant,
) -> None:
    entry = _mcp_entry(
        allowed_tools=["echo"],
        mode="specified",
        include_return_schema=False,
        deferred_loading=True,
    )
    toolsets = await _async_runtime_toolsets(hass, entry, ["mcp_server_1"])

    toolset = toolsets[0]
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
    assert (
        entry.runtime_data.metrics.record_for("conversation-1").last_mcp_tool_call
        == "echo"
    )

    with pytest.raises(MCPValidationError) as err:
        await toolset.toolset.process_tool_call(
            None, call_tool, "read_file", {"path": "/tmp/x"}
        )
    assert err.value.reason == "mcp_tool_not_allowed"
    assert err.value.tool_name == "read_file"


async def test_runtime_mcp_toolsets_without_allowlist_enable_all_tools(
    hass: HomeAssistant,
) -> None:
    toolsets = await _async_runtime_toolsets(hass, _mcp_entry(), ["mcp_server_1"])

    toolset = toolsets[0]
    assert toolset.toolset.filter_func is None

    async def call_tool(
        tool_name: str, tool_args: dict[str, object]
    ) -> dict[str, object]:
        return {"tool": tool_name, "args": tool_args}

    assert await toolset.toolset.process_tool_call(
        None, call_tool, "any_tool", {"message": "hi"}
    ) == {"tool": "any_tool", "args": {"message": "hi"}}


async def test_runtime_mcp_toolsets_all_mode_ignores_stored_allowlist(
    hass: HomeAssistant,
) -> None:
    entry = _mcp_entry(allowed_tools=["echo"], mode=MCP_TOOL_MODE_ALL)
    toolsets = await _async_runtime_toolsets(hass, entry, ["mcp_server_1"])

    toolset = toolsets[0]
    assert toolset.toolset.filter_func is None

    async def call_tool(
        tool_name: str, tool_args: dict[str, object]
    ) -> dict[str, object]:
        return {"tool": tool_name, "args": tool_args}

    assert await toolset.toolset.process_tool_call(
        None, call_tool, "hidden", {"message": "hi"}
    ) == {"tool": "hidden", "args": {"message": "hi"}}


async def test_runtime_mcp_tool_calls_skip_cache_when_disabled(
    hass: HomeAssistant,
) -> None:
    entry = _mcp_entry(call_cache_enabled=False)
    toolset = (await _async_runtime_toolsets(hass, entry, ["mcp_server_1"]))[0]
    call_count = 0

    async def call_tool(
        tool_name: str, tool_args: dict[str, object]
    ) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        return {"tool": tool_name, "args": tool_args, "call": call_count}

    first = await toolset.toolset.process_tool_call(
        None, call_tool, "echo", {"message": "hi"}
    )
    second = await toolset.toolset.process_tool_call(
        None, call_tool, "echo", {"message": "hi"}
    )

    assert first["call"] == 1
    assert second["call"] == 2
    assert call_count == 2
    assert entry.runtime_data.mcp_call_cache == {}


async def test_runtime_mcp_tool_calls_cache_successful_results_with_normalized_args(
    hass: HomeAssistant,
) -> None:
    entry = _mcp_entry(call_cache_enabled=True, call_cache_ttl=60)
    toolset = (await _async_runtime_toolsets(hass, entry, ["mcp_server_1"]))[0]
    call_count = 0

    async def call_tool(
        tool_name: str, tool_args: dict[str, object]
    ) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        return {"tool": tool_name, "args": tool_args, "call": call_count}

    first = await toolset.toolset.process_tool_call(
        None,
        call_tool,
        "echo",
        {"message": "hi", "count": 1},
    )
    second = await toolset.toolset.process_tool_call(
        None,
        call_tool,
        "echo",
        {"count": 1, "message": "hi"},
    )

    assert first == second
    assert call_count == 1
    assert len(entry.runtime_data.mcp_call_cache) == 1


async def test_runtime_mcp_tool_calls_expire_cached_results(
    hass: HomeAssistant,
) -> None:
    entry = _mcp_entry(call_cache_enabled=True, call_cache_ttl=60)
    toolset = (await _async_runtime_toolsets(hass, entry, ["mcp_server_1"]))[0]
    call_count = 0

    async def call_tool(
        tool_name: str, tool_args: dict[str, object]
    ) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        return {"tool": tool_name, "args": tool_args, "call": call_count}

    await toolset.toolset.process_tool_call(None, call_tool, "echo", {"message": "hi"})
    cached_entry = next(iter(entry.runtime_data.mcp_call_cache.values()))
    cached_entry.expires_at = hass.loop.time() - 1

    result = await toolset.toolset.process_tool_call(
        None, call_tool, "echo", {"message": "hi"}
    )

    assert result["call"] == 2
    assert call_count == 2


async def test_runtime_mcp_tool_calls_prune_expired_entries_from_other_keys(
    hass: HomeAssistant,
) -> None:
    entry = _mcp_entry(call_cache_enabled=True, call_cache_ttl=60)
    toolset = (await _async_runtime_toolsets(hass, entry, ["mcp_server_1"]))[0]
    call_count = 0

    async def call_tool(
        tool_name: str, tool_args: dict[str, object]
    ) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        return {"tool": tool_name, "args": tool_args, "call": call_count}

    cached_result = await toolset.toolset.process_tool_call(
        None, call_tool, "echo", {"message": "cached"}
    )
    entry.runtime_data.mcp_call_cache["expired-key"] = MCPCallCacheEntry(
        expires_at=hass.loop.time() - 1,
        result={"tool": "stale"},
    )

    lookup_result = await toolset.toolset.process_tool_call(
        None, call_tool, "echo", {"message": "cached"}
    )

    assert lookup_result == cached_result
    assert call_count == 1
    assert set(entry.runtime_data.mcp_call_cache) != {"expired-key"}
    assert "expired-key" not in entry.runtime_data.mcp_call_cache

    entry.runtime_data.mcp_call_cache["another-expired-key"] = MCPCallCacheEntry(
        expires_at=hass.loop.time() - 1,
        result={"tool": "older-stale"},
    )

    stored_result = await toolset.toolset.process_tool_call(
        None, call_tool, "echo", {"message": "fresh"}
    )

    assert stored_result["call"] == 2
    assert call_count == 2
    assert "another-expired-key" not in entry.runtime_data.mcp_call_cache
    assert len(entry.runtime_data.mcp_call_cache) == 2


async def test_runtime_mcp_tool_calls_do_not_cache_failures(
    hass: HomeAssistant,
) -> None:
    entry = _mcp_entry(call_cache_enabled=True, call_cache_ttl=60)
    toolset = (await _async_runtime_toolsets(hass, entry, ["mcp_server_1"]))[0]
    call_count = 0

    async def call_tool(_tool_name: str, _tool_args: dict[str, object]) -> object:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await toolset.toolset.process_tool_call(None, call_tool, "explode", {"id": 1})
    with pytest.raises(RuntimeError, match="boom"):
        await toolset.toolset.process_tool_call(None, call_tool, "explode", {"id": 1})

    assert call_count == 2
    assert entry.runtime_data.mcp_call_cache == {}
    assert (
        entry.runtime_data.metrics.record_for("conversation-1").last_mcp_tool_call
        == "explode"
    )


async def test_runtime_mcp_tool_call_cache_separates_args_tools_and_servers(
    hass: HomeAssistant,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Workspace",
        data={CONF_NAME: "Workspace"},
        source=config_entries.SOURCE_USER,
        subentries_data=(
            _mcp_subentry_data(
                subentry_id="mcp_server_1",
                call_cache_enabled=True,
                call_cache_ttl=60,
            ),
            _mcp_subentry_data(
                subentry_id="mcp_server_2",
                call_cache_enabled=True,
                call_cache_ttl=60,
            ),
        ),
        options={},
        unique_id=None,
    )
    toolsets = await _async_runtime_toolsets(
        hass,
        entry,
        ["mcp_server_1", "mcp_server_2"],
    )
    first_toolset, second_toolset = toolsets
    call_count = 0

    async def call_tool(
        tool_name: str, tool_args: dict[str, object]
    ) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        return {"tool": tool_name, "args": tool_args, "call": call_count}

    await first_toolset.toolset.process_tool_call(
        None, call_tool, "echo", {"message": "hi"}
    )
    await first_toolset.toolset.process_tool_call(
        None, call_tool, "echo", {"message": "bye"}
    )
    await first_toolset.toolset.process_tool_call(
        None, call_tool, "list_files", {"message": "hi"}
    )
    await second_toolset.toolset.process_tool_call(
        None, call_tool, "echo", {"message": "hi"}
    )

    assert call_count == 4
    assert len(entry.runtime_data.mcp_call_cache) == 4


async def test_runtime_mcp_tool_calls_skip_cache_for_non_jsonable_args(
    hass: HomeAssistant,
) -> None:
    entry = _mcp_entry(call_cache_enabled=True, call_cache_ttl=60)
    toolset = (await _async_runtime_toolsets(hass, entry, ["mcp_server_1"]))[0]
    call_count = 0

    async def call_tool(
        tool_name: str, tool_args: dict[str, object]
    ) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        return {"tool": tool_name, "args": tool_args, "call": call_count}

    first = await toolset.toolset.process_tool_call(
        None, call_tool, "echo", {"payload": object()}
    )
    second = await toolset.toolset.process_tool_call(
        None, call_tool, "echo", {"payload": object()}
    )

    assert first["call"] == 1
    assert second["call"] == 2
    assert call_count == 2
    assert entry.runtime_data.mcp_call_cache == {}
