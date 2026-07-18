"""Tests for bounded runtime MCP call-result caching."""

from types import SimpleNamespace
from typing import Any, cast

from custom_components.pydantic_ai_agent.mcp import runtime
from custom_components.pydantic_ai_agent.runtime.types import (
    MCPCallCacheEntry,
    WorkspaceRuntimeData,
)
from homeassistant.core import HomeAssistant
import pytest


def _entry() -> Any:
    return SimpleNamespace(
        entry_id="workspace-1",
        runtime_data=WorkspaceRuntimeData(workspace_name="Test workspace"),
    )


def test_mcp_cache_evicts_oldest_entry_at_capacity(hass: HomeAssistant) -> None:
    """A new entry evicts the oldest insertion once the cache reaches capacity."""
    entry = _entry()
    cache = entry.runtime_data.mcp_call_cache
    for index in range(runtime._MAX_MCP_CALL_CACHE_ENTRIES):
        cache[str(index)] = MCPCallCacheEntry(
            expires_at=hass.loop.time() + 60, result=index
        )

    runtime._store_cached_mcp_tool_result(hass, cast(Any, entry), "new", 60, "new")

    assert len(cache) == runtime._MAX_MCP_CALL_CACHE_ENTRIES
    assert "0" not in cache
    assert cache["new"].result == "new"


def test_mcp_cache_update_at_capacity_keeps_unrelated_entries(
    hass: HomeAssistant,
) -> None:
    """Replacing a live key does not evict the oldest unrelated cache entry."""
    entry = _entry()
    cache = entry.runtime_data.mcp_call_cache
    for index in range(runtime._MAX_MCP_CALL_CACHE_ENTRIES):
        cache[str(index)] = MCPCallCacheEntry(
            expires_at=hass.loop.time() + 60, result=index
        )

    runtime._store_cached_mcp_tool_result(hass, cast(Any, entry), "0", 60, "updated")

    assert len(cache) == runtime._MAX_MCP_CALL_CACHE_ENTRIES
    assert cache["0"].result == "updated"
    assert "1" in cache


def test_mcp_cache_prunes_expired_entries_before_eviction(hass: HomeAssistant) -> None:
    """Expired entries are removed before a fresh result is stored."""
    entry = _entry()
    cache = entry.runtime_data.mcp_call_cache
    cache["expired"] = MCPCallCacheEntry(expires_at=hass.loop.time(), result="old")

    runtime._store_cached_mcp_tool_result(hass, cast(Any, entry), "fresh", 60, "new")

    assert list(cache) == ["fresh"]


async def test_mcp_cache_reuses_normalized_successful_calls(hass: HomeAssistant) -> None:
    """Equivalent serializable calls reuse their prior successful result."""
    entry = _entry()
    calls = 0

    async def call_tool(_name: str, _args: dict[str, Any]) -> object:
        nonlocal calls
        calls += 1
        return {"call": calls}

    first = await runtime._process_cached_mcp_tool_call(
        hass,
        cast(Any, entry),
        "agent-1",
        call_tool,
        "server-1",
        "weather",
        {"city": "Paris", "units": "c"},
        cache_enabled=True,
        cache_ttl=60,
    )
    second = await runtime._process_cached_mcp_tool_call(
        hass,
        cast(Any, entry),
        "agent-1",
        call_tool,
        "server-1",
        "weather",
        {"units": "c", "city": "Paris"},
        cache_enabled=True,
        cache_ttl=60,
    )

    assert first == second == {"call": 1}
    assert calls == 1


async def test_mcp_cache_disabled_always_calls_tool(hass: HomeAssistant) -> None:
    """Disabling caching executes identical calls and leaves no cached result."""
    entry = _entry()
    calls = 0

    async def call_tool(_name: str, _args: dict[str, Any]) -> object:
        nonlocal calls
        calls += 1
        return {"call": calls}

    results = [
        await runtime._process_cached_mcp_tool_call(
            hass,
            cast(Any, entry),
            "agent-1",
            call_tool,
            "server-1",
            "weather",
            {"city": "Paris"},
            cache_enabled=False,
            cache_ttl=60,
        )
        for _ in range(2)
    ]

    assert results == [{"call": 1}, {"call": 2}]
    assert calls == 2
    assert entry.runtime_data.mcp_call_cache == {}


async def test_mcp_cache_does_not_cache_failures(hass: HomeAssistant) -> None:
    """A failed call is attempted again and never occupies the result cache."""
    entry = _entry()
    calls = 0

    async def call_tool(_name: str, _args: dict[str, Any]) -> object:
        nonlocal calls
        calls += 1
        raise RuntimeError("failed")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await runtime._process_cached_mcp_tool_call(
                hass,
                cast(Any, entry),
                "agent-1",
                call_tool,
                "server-1",
                "weather",
                {"city": "Paris"},
                cache_enabled=True,
                cache_ttl=60,
            )

    assert calls == 2
    assert entry.runtime_data.mcp_call_cache == {}


async def test_mcp_cache_isolates_server_tool_and_arguments(
    hass: HomeAssistant,
) -> None:
    """Server, tool, and argument changes each identify a distinct cached call."""
    entry = _entry()
    calls = 0

    async def call_tool(_name: str, _args: dict[str, Any]) -> object:
        nonlocal calls
        calls += 1
        return {"call": calls}

    calls_by_key = [
        ("server-1", "weather", {"city": "Paris"}),
        ("server-2", "weather", {"city": "Paris"}),
        ("server-1", "forecast", {"city": "Paris"}),
        ("server-1", "weather", {"city": "London"}),
    ]

    results = [
        await runtime._process_cached_mcp_tool_call(
            hass,
            cast(Any, entry),
            "agent-1",
            call_tool,
            server_id,
            tool_name,
            arguments,
            cache_enabled=True,
            cache_ttl=60,
        )
        for server_id, tool_name, arguments in calls_by_key
    ]

    assert results == [
        {"call": 1},
        {"call": 2},
        {"call": 3},
        {"call": 4},
    ]
    assert calls == 4
    assert len(entry.runtime_data.mcp_call_cache) == 4


async def test_mcp_cache_expiry_and_non_json_values_bypass_reuse(
    hass: HomeAssistant,
) -> None:
    """Expired and non-serializable calls are never reused."""
    entry = _entry()
    calls = 0

    async def call_tool(_name: str, _args: dict[str, Any]) -> object:
        nonlocal calls
        calls += 1
        return calls

    await runtime._process_cached_mcp_tool_call(
        hass,
        cast(Any, entry),
        "agent-1",
        call_tool,
        "server-1",
        "weather",
        {"city": "Paris"},
        cache_enabled=True,
        cache_ttl=60,
    )
    entry.runtime_data.mcp_call_cache[next(iter(entry.runtime_data.mcp_call_cache))].expires_at = (
        hass.loop.time()
    )
    assert (
        await runtime._process_cached_mcp_tool_call(
            hass,
            cast(Any, entry),
            "agent-1",
            call_tool,
            "server-1",
            "weather",
            {"city": "Paris"},
            cache_enabled=True,
            cache_ttl=60,
        )
        == 2
    )
    await runtime._process_cached_mcp_tool_call(
        hass,
        cast(Any, entry),
        "agent-1",
        call_tool,
        "server-1",
        "weather",
        {"value": object()},
        cache_enabled=True,
        cache_ttl=60,
    )
    await runtime._process_cached_mcp_tool_call(
        hass,
        cast(Any, entry),
        "agent-1",
        call_tool,
        "server-1",
        "weather",
        {"value": object()},
        cache_enabled=True,
        cache_ttl=60,
    )

    assert calls == 4
