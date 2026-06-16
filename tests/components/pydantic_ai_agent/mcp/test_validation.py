"""Tests for MCP validation helpers."""

import pytest
import voluptuous as vol
from custom_components.pydantic_ai_agent.const import (
    CONF_KEY_VALUE_KEY,
    CONF_KEY_VALUE_VALUE,
)
from custom_components.pydantic_ai_agent.mcp import (
    MCPValidationError,
    normalise_mcp_url,
    parse_allowed_tools,
    parse_mcp_headers,
    schema_hash,
)
from custom_components.pydantic_ai_agent.runtime.redaction import redact_data


def test_mcp_log_redaction_uses_shared_sensitive_key_handling() -> None:
    redacted = redact_data(
        {
            "mcp_url": "https://mcp.example.com/mcp?token=visible",
            "headers": {"Authorization": "Bearer secret"},
            "result": {"token": "secret", "session_token": "visible", "value": "safe"},
        }
    )

    assert redacted["mcp_url"] == "**REDACTED**"
    assert redacted["headers"] == "**REDACTED**"
    assert redacted["result"]["token"] == "**REDACTED**"
    assert redacted["result"]["session_token"] == "visible"


def test_parse_mcp_headers_accepts_object_selector_rows() -> None:
    assert parse_mcp_headers(
        [
            {
                CONF_KEY_VALUE_KEY: "Authorization",
                CONF_KEY_VALUE_VALUE: "Bearer secret",
            },
            {CONF_KEY_VALUE_KEY: "X-Trace", CONF_KEY_VALUE_VALUE: "value:with:colons"},
        ]
    ) == {
        "Authorization": "Bearer secret",
        "X-Trace": "value:with:colons",
    }


def test_parse_mcp_headers_accepts_stored_mapping() -> None:
    assert parse_mcp_headers({"Authorization": "Bearer secret"}) == {
        "Authorization": "Bearer secret"
    }


@pytest.mark.parametrize(
    "headers",
    [
        "Bad Header: value",
        {"Bad Header": "value"},
        {"X-Test": 1},
        [{CONF_KEY_VALUE_KEY: "Bad Header", CONF_KEY_VALUE_VALUE: "value"}],
        [{CONF_KEY_VALUE_KEY: "X-Test", CONF_KEY_VALUE_VALUE: 1}],
    ],
)
def test_parse_mcp_headers_reject_invalid_values(headers: object) -> None:
    with pytest.raises(vol.Invalid):
        parse_mcp_headers(headers)


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("not a url", "invalid_mcp_url"),
        ("ftp://mcp.example.com/mcp", "invalid_mcp_url"),
        ("https://mcp.example.com/mcp#fragment", "invalid_mcp_url"),
        ("https://user:pass@mcp.example.com/mcp", "invalid_mcp_url"),
    ],
)
def test_normalise_mcp_url_rejects_invalid_inputs(url: str, reason: str) -> None:
    with pytest.raises(MCPValidationError) as err:
        normalise_mcp_url(url)

    assert err.value.reason == reason


def test_parse_allowed_tools_normalizes_strings_and_sequences() -> None:
    assert parse_allowed_tools(" read_file, list_files\nread_file ") == [
        "list_files",
        "read_file",
    ]
    assert parse_allowed_tools(["echo", "  list_files ", "echo", ""]) == [
        "echo",
        "list_files",
    ]
    assert parse_allowed_tools(None) == []

    with pytest.raises(vol.Invalid):
        parse_allowed_tools(123)


def test_schema_hash_is_stable_for_json_equivalent_schemas() -> None:
    schema_a = {"type": "object", "properties": {"name": {"type": "string"}}}
    schema_b = {"properties": {"name": {"type": "string"}}, "type": "object"}
    schema_c = {"type": "object", "properties": {"id": {"type": "integer"}}}

    assert schema_hash(schema_a) == schema_hash(schema_b)
    assert schema_hash(schema_a) != schema_hash(schema_c)
