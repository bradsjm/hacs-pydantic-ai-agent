"""Tests for MCP config-entry helper functions."""

from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any

from custom_components.pydantic_ai_agent.const import (
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_CALL_CACHE_ENABLED,
    CONF_MCP_CALL_CACHE_TTL,
    CONF_MCP_DEFERRED_LOADING,
    CONF_MCP_HEADERS,
    CONF_MCP_INCLUDE_RETURN_SCHEMA,
    CONF_MCP_SECRET_HEADER_KEYS,
    CONF_MCP_TOOL_MODE,
    CONF_MCP_URL,
    MCP_TOOL_MODE_ALL,
    MCP_TOOL_MODE_DISABLED,
    MCP_TOOL_MODE_SPECIFIED,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_MCP_SERVER,
)
from custom_components.pydantic_ai_agent.mcp.entry_helpers import (
    effective_mcp_tool_mode,
    get_mcp_subentry,
    mcp_config_from_subentry,
    mcp_subentries,
    stored_mcp_tool_configuration,
)
from custom_components.pydantic_ai_agent.mcp.errors import MCPValidationError
import pytest

MakeSubentry = Callable[..., Any]


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({}, MCP_TOOL_MODE_ALL),
        ({CONF_MCP_ALLOWED_TOOLS: ""}, MCP_TOOL_MODE_DISABLED),
        ({CONF_MCP_ALLOWED_TOOLS: "tool_b, tool_a"}, MCP_TOOL_MODE_SPECIFIED),
        (
            {
                CONF_MCP_TOOL_MODE: MCP_TOOL_MODE_DISABLED,
                CONF_MCP_ALLOWED_TOOLS: "tool_a",
            },
            MCP_TOOL_MODE_DISABLED,
        ),
    ],
)
def test_effective_mcp_tool_mode_handles_current_and_legacy_data(
    data: Mapping[str, Any], expected: str
) -> None:
    """Tool mode is explicit when stored and inferred for legacy allowlists."""
    assert effective_mcp_tool_mode(data) == expected


def test_stored_mcp_tool_configuration_for_each_mode() -> None:
    """Stored tool config contains only fields required by the selected mode."""
    assert stored_mcp_tool_configuration(MCP_TOOL_MODE_ALL, []) == {
        CONF_MCP_TOOL_MODE: MCP_TOOL_MODE_ALL
    }
    assert stored_mcp_tool_configuration(MCP_TOOL_MODE_DISABLED, ["ignored"]) == {
        CONF_MCP_TOOL_MODE: MCP_TOOL_MODE_DISABLED,
        CONF_MCP_ALLOWED_TOOLS: [],
    }
    assert stored_mcp_tool_configuration(MCP_TOOL_MODE_SPECIFIED, ["tool_a"]) == {
        CONF_MCP_TOOL_MODE: MCP_TOOL_MODE_SPECIFIED,
        CONF_MCP_ALLOWED_TOOLS: ["tool_a"],
    }


@pytest.mark.parametrize(
    ("mode", "allowed_tools"),
    [(MCP_TOOL_MODE_SPECIFIED, []), ("unsupported", ["tool_a"])],
)
def test_stored_mcp_tool_configuration_rejects_invalid_modes(
    mode: str, allowed_tools: list[str]
) -> None:
    """Specified mode must include tools, and unknown modes are invalid."""
    with pytest.raises(ValueError, match=r"specified mode requires|unsupported MCP"):
        stored_mcp_tool_configuration(mode, allowed_tools)


def test_mcp_subentries_filters_by_subentry_type(make_subentry: MakeSubentry) -> None:
    """Only MCP server subentries are returned from a workspace entry."""
    mcp = make_subentry(
        data={CONF_MCP_URL: "https://example.test/mcp"},
        subentry_type=SUBENTRY_TYPE_MCP_SERVER,
        subentry_id="mcp-1",
    )
    conversation = make_subentry(
        data={},
        subentry_type=SUBENTRY_TYPE_CONVERSATION,
        subentry_id="conversation-1",
    )
    entry = SimpleNamespace(subentries={"mcp-1": mcp, "conversation-1": conversation})

    assert mcp_subentries(entry) == [mcp]


def test_get_mcp_subentry_returns_matching_subentry(
    make_subentry: MakeSubentry,
) -> None:
    """MCP subentries can be looked up by ID."""
    subentry = make_subentry(
        data={CONF_MCP_URL: "https://example.test/mcp"},
        subentry_type=SUBENTRY_TYPE_MCP_SERVER,
        subentry_id="mcp-1",
    )
    entry = SimpleNamespace(subentries={"mcp-1": subentry})

    assert get_mcp_subentry(entry, "mcp-1") is subentry


@pytest.mark.parametrize(
    ("subentries", "subentry_id"),
    [
        ({}, "missing"),
        ({"conversation-1": SUBENTRY_TYPE_CONVERSATION}, "conversation-1"),
    ],
)
def test_get_mcp_subentry_rejects_missing_or_wrong_type(
    make_subentry: MakeSubentry, subentries: dict[str, str], subentry_id: str
) -> None:
    """Missing and non-MCP subentries raise a stable MCP validation reason."""
    entry_subentries = {
        key: make_subentry(data={}, subentry_type=value, subentry_id=key)
        for key, value in subentries.items()
    }
    entry = SimpleNamespace(subentries=entry_subentries)

    with pytest.raises(MCPValidationError) as exc_info:
        get_mcp_subentry(entry, subentry_id)

    assert exc_info.value.reason == "mcp_server_not_found"
    assert exc_info.value.server_id == subentry_id


def test_mcp_config_from_subentry_normalizes_stored_data(
    make_subentry: MakeSubentry,
) -> None:
    """Subentry config extraction normalizes URL, headers, booleans, and tools."""
    subentry = make_subentry(
        title="Weather MCP",
        data={
            CONF_MCP_URL: " https://example.test:8443/mcp ",
            CONF_MCP_HEADERS: {"Authorization": "Bearer token"},
            CONF_MCP_SECRET_HEADER_KEYS: ["Authorization"],
            CONF_MCP_CALL_CACHE_ENABLED: 1,
            CONF_MCP_CALL_CACHE_TTL: "120",
            CONF_MCP_INCLUDE_RETURN_SCHEMA: 0,
            CONF_MCP_DEFERRED_LOADING: 1,
            CONF_MCP_TOOL_MODE: MCP_TOOL_MODE_SPECIFIED,
            CONF_MCP_ALLOWED_TOOLS: "weather.get, weather.set\nweather.get",
        },
        subentry_type=SUBENTRY_TYPE_MCP_SERVER,
        subentry_id="mcp-1",
    )

    config = mcp_config_from_subentry(subentry)

    assert config["name"] == "Weather MCP"
    assert config[CONF_MCP_URL] == "https://example.test:8443/mcp"
    assert config[CONF_MCP_HEADERS] == {"Authorization": "Bearer token"}
    assert config[CONF_MCP_SECRET_HEADER_KEYS] == ["Authorization"]
    assert config[CONF_MCP_CALL_CACHE_ENABLED] is True
    assert config[CONF_MCP_CALL_CACHE_TTL] == 120
    assert config[CONF_MCP_INCLUDE_RETURN_SCHEMA] is False
    assert config[CONF_MCP_DEFERRED_LOADING] is True
    assert config[CONF_MCP_TOOL_MODE] == MCP_TOOL_MODE_SPECIFIED
    assert config[CONF_MCP_ALLOWED_TOOLS] == ["weather.get", "weather.set"]
