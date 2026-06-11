"""Tests for MCP entry helpers."""

from custom_components.pydantic_ai_agent.const import (
    CONF_MCP_DEFERRED_LOADING,
    CONF_MCP_INCLUDE_RETURN_SCHEMA,
)
from custom_components.pydantic_ai_agent.mcp.entry_helpers import (
    mcp_config_from_subentry,
)
from pytest_homeassistant_custom_component.common import MockConfigEntry

from .test_discovery import _mcp_entry


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
