"""MCP runtime toolset helpers."""

# ruff: noqa: ANN401

import json
import logging
from collections.abc import Sequence
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.util import slugify
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import AbstractToolset

from .._redaction import redact_data
from .._types import MCPCallCacheEntry, WorkspaceRuntimeData
from ..const import (
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_CALL_CACHE_ENABLED,
    CONF_MCP_CALL_CACHE_TTL,
    CONF_MCP_DEFERRED_LOADING,
    CONF_MCP_HEADERS,
    CONF_MCP_INCLUDE_RETURN_SCHEMA,
    CONF_MCP_URL,
    DEFAULT_MCP_TIMEOUT,
)
from ..metrics import record_mcp_tool_call
from .client import _mcp_client
from .entry_helpers import mcp_config_from_subentry, mcp_subentries
from .errors import MCPValidationError
from .validation import async_validate_mcp_url_details

_LOGGER = logging.getLogger(__name__)
_MISSING = object()


async def _validated_runtime_mcp_config(
    hass: HomeAssistant, subentry: Any
) -> tuple[dict[str, Any], Any]:
    """Return validated runtime MCP config and URL details."""
    try:
        config = mcp_config_from_subentry(subentry)
        validated_url = await async_validate_mcp_url_details(hass, config[CONF_MCP_URL])
        config[CONF_MCP_URL] = validated_url.url
    except MCPValidationError as err:
        _LOGGER.warning(
            "Invalid selected MCP server %s for runtime: %s",
            subentry.subentry_id,
            err.message,
        )
        raise
    return config, validated_url


def _runtime_allowed_tools(
    subentry: Any, config: dict[str, Any]
) -> tuple[bool, set[str]]:
    """Return MCP allowlist status and normalized tool names."""
    has_allowlist = CONF_MCP_ALLOWED_TOOLS in subentry.data
    if has_allowlist and not config[CONF_MCP_ALLOWED_TOOLS]:
        raise MCPValidationError(
            "mcp_tools_not_allowlisted",
            "Select at least one allowed MCP tool before enabling this server for "
            "runtime use.",
            server_id=subentry.subentry_id,
        )
    return has_allowlist, set(config[CONF_MCP_ALLOWED_TOOLS])


def _cached_mcp_tool_result(
    hass: HomeAssistant,
    entry: ConfigEntry[WorkspaceRuntimeData],
    cache_key: str,
) -> Any:
    """Return a cached MCP tool result or `_MISSING`."""
    _prune_expired_mcp_tool_results(hass, entry)
    cached_entry = entry.runtime_data.mcp_call_cache.get(cache_key)
    if cached_entry is None:
        return _MISSING
    if cached_entry.expires_at > hass.loop.time():
        return cached_entry.result
    entry.runtime_data.mcp_call_cache.pop(cache_key, None)
    return _MISSING


def _store_cached_mcp_tool_result(
    hass: HomeAssistant,
    entry: ConfigEntry[WorkspaceRuntimeData],
    cache_key: str,
    cache_ttl: int,
    result: Any,
) -> None:
    """Store one successful MCP tool result in the runtime cache."""
    _prune_expired_mcp_tool_results(hass, entry)
    entry.runtime_data.mcp_call_cache[cache_key] = MCPCallCacheEntry(
        expires_at=hass.loop.time() + cache_ttl,
        result=result,
    )


def _prune_expired_mcp_tool_results(
    hass: HomeAssistant, entry: ConfigEntry[WorkspaceRuntimeData]
) -> None:
    """Drop expired MCP tool call cache entries."""
    now = hass.loop.time()
    for cache_key, cached_entry in tuple(entry.runtime_data.mcp_call_cache.items()):
        if cached_entry.expires_at <= now:
            entry.runtime_data.mcp_call_cache.pop(cache_key, None)


async def _process_cached_mcp_tool_call(
    hass: HomeAssistant,
    entry: ConfigEntry[WorkspaceRuntimeData],
    agent_subentry_id: str,
    call_tool: Any,
    server_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    cache_enabled: bool,
    cache_ttl: int,
) -> Any:
    """Process one MCP tool call with optional runtime caching."""
    record_mcp_tool_call(
        hass,
        entry.entry_id,
        entry.runtime_data.metrics,
        agent_subentry_id,
        tool_name=tool_name,
    )
    _LOGGER.info("Calling MCP tool %s on server %s", tool_name, server_id)
    _LOGGER.debug("MCP tool call arguments: %s", redact_data(tool_args))
    cache_key = _mcp_call_cache_key(server_id, tool_name, tool_args)
    if cache_enabled and cache_key is not None:
        cached_result = _cached_mcp_tool_result(hass, entry, cache_key)
        if cached_result is not _MISSING:
            _LOGGER.debug(
                "Using cached MCP tool result for %s on server %s",
                tool_name,
                server_id,
            )
            return cached_result
    result = await call_tool(tool_name, tool_args)
    if cache_enabled and cache_key is not None:
        _store_cached_mcp_tool_result(hass, entry, cache_key, cache_ttl, result)
    _LOGGER.debug("MCP tool result: %s", redact_data({"result": result}))
    return result


async def _async_runtime_mcp_toolset_for_subentry(
    hass: HomeAssistant,
    entry: ConfigEntry[WorkspaceRuntimeData],
    agent_subentry_id: str,
    subentry: Any,
) -> AbstractToolset[Any]:
    """Return one runtime MCP toolset for a configured server subentry."""
    config, validated_url = await _validated_runtime_mcp_config(hass, subentry)
    has_allowlist, allowed_tools = _runtime_allowed_tools(subentry, config)
    cache_enabled = config[CONF_MCP_CALL_CACHE_ENABLED]
    cache_ttl = config[CONF_MCP_CALL_CACHE_TTL]
    server_id = subentry.subentry_id

    async def process_tool_call(
        _ctx: Any,
        call_tool: Any,
        tool_name: str,
        tool_args: dict[str, Any],
        *,
        server_id: str = server_id,
        allowed_tools: set[str] = allowed_tools,
        cache_enabled: bool = cache_enabled,
        cache_ttl: int = cache_ttl,
        has_allowlist: bool = has_allowlist,
    ) -> Any:
        if has_allowlist and tool_name not in allowed_tools:
            raise MCPValidationError(
                "mcp_tool_not_allowed",
                "MCP tool is not allowlisted for this server.",
                server_id=server_id,
                tool_name=tool_name,
            )
        return await _process_cached_mcp_tool_call(
            hass,
            entry,
            agent_subentry_id,
            call_tool,
            server_id,
            tool_name,
            tool_args,
            cache_enabled=cache_enabled,
            cache_ttl=cache_ttl,
        )

    toolset: AbstractToolset[Any] = MCPToolset(
        _mcp_client(
            validated_url,
            config[CONF_MCP_HEADERS],
            DEFAULT_MCP_TIMEOUT,
        ),
        id=server_id,
        tool_error_behavior="error",
        process_tool_call=process_tool_call,
        include_return_schema=config[CONF_MCP_INCLUDE_RETURN_SCHEMA],
    )
    if has_allowlist:
        toolset = toolset.filtered(
            lambda _ctx, tool_def, allowed_tools=allowed_tools: (
                tool_def.name in allowed_tools
            )
        )
    toolset = toolset.prefixed(f"mcp_{slugify(server_id)}")
    if config[CONF_MCP_DEFERRED_LOADING]:
        toolset = toolset.defer_loading()
    return toolset


async def async_runtime_mcp_toolsets(
    hass: HomeAssistant,
    entry: ConfigEntry[WorkspaceRuntimeData],
    agent_subentry_id: str,
    selected_server_ids: Sequence[str] | None,
) -> list[AbstractToolset[Any]]:
    """Return agent MCP toolsets for explicitly allowlisted servers."""
    toolsets: list[AbstractToolset[Any]] = []
    selected_servers = set(selected_server_ids or [])
    if not selected_servers:
        return toolsets
    configured_server_ids: set[str] = set()
    for subentry in mcp_subentries(entry):
        if subentry.subentry_id not in selected_servers:
            continue
        configured_server_ids.add(subentry.subentry_id)
        toolsets.append(
            await _async_runtime_mcp_toolset_for_subentry(
                hass,
                entry,
                agent_subentry_id,
                subentry,
            )
        )
    missing_server_ids = selected_servers - configured_server_ids
    if missing_server_ids:
        missing_server_id = sorted(missing_server_ids)[0]
        raise MCPValidationError(
            "mcp_server_not_found",
            "Selected MCP server subentry was not found.",
            server_id=missing_server_id,
        )
    return toolsets


def _mcp_call_cache_key(
    server_id: str, tool_name: str, tool_args: dict[str, Any]
) -> str | None:
    """Return a stable cache key for one MCP tool call."""
    try:
        return json.dumps(
            {
                "server_id": server_id,
                "tool_name": tool_name,
                "tool_args": tool_args,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except TypeError:
        return None
