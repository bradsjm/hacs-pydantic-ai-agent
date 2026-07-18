"""Tests for MCP timeout parsing and storage normalization."""

from collections.abc import Mapping
from typing import Any

from custom_components.pydantic_ai_agent.config_flows.mcp_helpers import (
    _mcp_server_data_from_user_input,
    _parse_mcp_timeout,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_MCP_TIMEOUT,
    CONF_MCP_URL,
    DEFAULT_MCP_TIMEOUT,
)
from homeassistant.const import CONF_NAME
import pytest
import voluptuous as vol


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (10, 10.0),
        (10.0, 10.0),
        ("15.5", 15.5),
        (1, 1.0),
        (600, 600.0),
    ],
)
def test_parse_mcp_timeout_accepts_valid_values(value: object, expected: float) -> None:
    """MCP timeout accepts finite numeric values inside the supported range."""
    timeout = _parse_mcp_timeout(value)

    assert timeout == expected
    assert isinstance(timeout, float)


@pytest.mark.parametrize(
    "value",
    [
        0,
        0.5,
        601,
        "-5",
        float("nan"),
        float("inf"),
        True,
        False,
        "notanumber",
        None,
    ],
)
def test_parse_mcp_timeout_rejects_invalid_values(value: object) -> None:
    """Invalid timeout input fails with the stable config-flow reason key."""
    with pytest.raises(vol.Invalid) as exc_info:
        _parse_mcp_timeout(value)

    assert exc_info.value.msg == "invalid_mcp_timeout"


@pytest.mark.parametrize(
    ("extra_input", "expected_timeout"),
    [
        ({CONF_MCP_TIMEOUT: 25}, 25.0),
        ({}, DEFAULT_MCP_TIMEOUT),
    ],
)
def test_mcp_server_data_from_user_input_persists_timeout(
    extra_input: Mapping[str, Any], expected_timeout: float
) -> None:
    """MCP server form data persists custom timeouts and defaults legacy input."""
    data = _mcp_server_data_from_user_input(
        {
            CONF_NAME: "Weather MCP",
            CONF_MCP_URL: "https://example.test/mcp",
            **extra_input,
        }
    )

    assert data[CONF_MCP_TIMEOUT] == expected_timeout
