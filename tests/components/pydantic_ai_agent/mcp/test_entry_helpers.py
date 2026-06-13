"""Tests for MCP entry helpers."""

from custom_components.pydantic_ai_agent.const import (
    CONF_MCP_CALL_CACHE_ENABLED,
    CONF_MCP_CALL_CACHE_TTL,
    CONF_MCP_DEFERRED_LOADING,
    CONF_MCP_INCLUDE_RETURN_SCHEMA,
    CONF_MCP_URL,
    DEFAULT_MCP_CALL_CACHE_TTL,
    DOMAIN,
    SUBENTRY_TYPE_MCP_SERVER,
)
from custom_components.pydantic_ai_agent.mcp.entry_helpers import (
    mcp_config_from_subentry,
)
from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from pytest_homeassistant_custom_component.common import MockConfigEntry


def _mcp_entry(
    *,
    call_cache_enabled: bool | None = None,
    call_cache_ttl: int | None = None,
    include_return_schema: bool | None = None,
    deferred_loading: bool | None = None,
) -> MockConfigEntry:
    """Return a config entry with one MCP server subentry."""
    data: dict[str, object] = {
        CONF_NAME: "Echo MCP",
        CONF_MCP_URL: "https://mcp.example.com/mcp",
    }
    if call_cache_enabled is not None:
        data[CONF_MCP_CALL_CACHE_ENABLED] = call_cache_enabled
    if call_cache_ttl is not None:
        data[CONF_MCP_CALL_CACHE_TTL] = call_cache_ttl
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
                "subentry_id": "mcp_server_1",
                "data": data,
                "subentry_type": SUBENTRY_TYPE_MCP_SERVER,
                "title": "Echo MCP",
                "unique_id": None,
            },
        ),
        options={},
        unique_id=None,
    )


def test_public_mcp_package_reexports_expected_symbols() -> None:
    from custom_components.pydantic_ai_agent import mcp

    assert callable(mcp.normalise_mcp_url)
    assert callable(mcp.parse_mcp_headers)
    assert callable(mcp.parse_allowed_tools)
    assert callable(mcp.async_runtime_mcp_toolsets)


def test_mcp_config_from_subentry_defaults_return_schema_preference() -> None:
    entry: MockConfigEntry = _mcp_entry()
    subentry = next(iter(entry.subentries.values()))

    assert mcp_config_from_subentry(subentry)[CONF_MCP_INCLUDE_RETURN_SCHEMA] is True

    entry = _mcp_entry(include_return_schema=False)
    subentry = next(iter(entry.subentries.values()))
    assert mcp_config_from_subentry(subentry)[CONF_MCP_INCLUDE_RETURN_SCHEMA] is False


def test_mcp_config_from_subentry_defaults_deferred_loading_preference() -> None:
    entry: MockConfigEntry = _mcp_entry()
    subentry = next(iter(entry.subentries.values()))

    assert mcp_config_from_subentry(subentry)[CONF_MCP_DEFERRED_LOADING] is False

    entry = _mcp_entry(deferred_loading=True)
    subentry = next(iter(entry.subentries.values()))
    assert mcp_config_from_subentry(subentry)[CONF_MCP_DEFERRED_LOADING] is True


def test_mcp_config_from_subentry_defaults_call_cache_settings() -> None:
    entry: MockConfigEntry = _mcp_entry()
    subentry = next(iter(entry.subentries.values()))

    config = mcp_config_from_subentry(subentry)

    assert config[CONF_MCP_CALL_CACHE_ENABLED] is False
    assert config[CONF_MCP_CALL_CACHE_TTL] == DEFAULT_MCP_CALL_CACHE_TTL


def test_mcp_config_from_subentry_preserves_call_cache_settings() -> None:
    entry: MockConfigEntry = _mcp_entry(
        call_cache_enabled=True,
        call_cache_ttl=900,
    )
    subentry = next(iter(entry.subentries.values()))

    config = mcp_config_from_subentry(subentry)

    assert config[CONF_MCP_CALL_CACHE_ENABLED] is True
    assert config[CONF_MCP_CALL_CACHE_TTL] == 900
