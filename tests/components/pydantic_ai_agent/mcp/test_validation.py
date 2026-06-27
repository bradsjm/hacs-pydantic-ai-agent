"""Tests for MCP validation helper functions."""

from collections.abc import Mapping
from typing import Any

from custom_components.pydantic_ai_agent.const import (
    CONF_KEY_VALUE_KEY,
    CONF_KEY_VALUE_VALUE,
)
from custom_components.pydantic_ai_agent.mcp.errors import MCPValidationError
from custom_components.pydantic_ai_agent.mcp.validation import (
    normalise_mcp_url,
    parse_allowed_tools,
    parse_mcp_headers,
    schema_hash,
    validate_mcp_url_details,
)
import pytest
import voluptuous as vol


def test_schema_hash_is_canonical_for_mapping_order() -> None:
    """Schema hashes are stable for equivalent JSON schemas."""
    first = {
        "type": "object",
        "properties": {
            "temperature": {"minimum": 0, "type": "number"},
            "mode": {"enum": ["heat", "cool"], "type": "string"},
        },
    }
    second = {
        "properties": {
            "mode": {"type": "string", "enum": ["heat", "cool"]},
            "temperature": {"type": "number", "minimum": 0},
        },
        "type": "object",
    }

    assert schema_hash(first) == schema_hash(second)
    assert len(schema_hash(first)) == 16


def test_schema_hash_uses_jsonable_model_dump() -> None:
    """Schema hashing canonicalizes objects exposing model_dump."""

    class JsonableSchema:
        def model_dump(self, *, mode: str, exclude_none: bool) -> Mapping[str, Any]:
            assert mode == "json"
            assert exclude_none is True
            return {"type": "object", "properties": {"name": {"type": "string"}}}

    assert schema_hash({"schema": JsonableSchema()}) == schema_hash(
        {"schema": {"type": "object", "properties": {"name": {"type": "string"}}}}
    )


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (" http://example.test/mcp ", "http://example.test/mcp"),
        ("https://example.test:8443/mcp", "https://example.test:8443/mcp"),
    ],
)
def test_normalise_mcp_url_accepts_http_and_https(url: str, expected: str) -> None:
    """HTTP and HTTPS Streamable HTTP URLs are accepted and trimmed."""
    assert normalise_mcp_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://example.test/mcp",
        "https://example.test:8443/mcp",
    ],
)
def test_validate_mcp_url_details_returns_origin(url: str) -> None:
    """URL details include normalized URL, scheme, host, and effective port."""
    details = validate_mcp_url_details(url)

    assert details.url == url
    assert details.scheme in {"http", "https"}
    assert details.hostname == "example.test"
    assert details.port in {80, 8443}


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        None,
        123,
        "ftp://example.test/mcp",
        "https://user:pass@example.test/mcp",
        "https://example.test/mcp#tools",
        "https://example.test:bad/mcp",
    ],
)
def test_normalise_mcp_url_rejects_invalid_inputs(url: object) -> None:
    """Unsafe or malformed MCP URLs fail with a stable validation reason."""
    with pytest.raises(MCPValidationError) as exc_info:
        normalise_mcp_url(url)

    assert exc_info.value.reason == "invalid_mcp_url"


def test_parse_mcp_headers_accepts_mapping_and_selector_rows() -> None:
    """Stored mappings and selector row lists parse to HTTP header mappings."""
    assert parse_mcp_headers({"Authorization": "Bearer token"}) == {
        "Authorization": "Bearer token"
    }
    assert parse_mcp_headers(
        [
            {CONF_KEY_VALUE_KEY: "X-API-Key", CONF_KEY_VALUE_VALUE: "secret"},
            {CONF_KEY_VALUE_KEY: "", CONF_KEY_VALUE_VALUE: ""},
        ]
    ) == {"X-API-Key": "secret"}


def test_parse_mcp_headers_rejects_invalid_header_name() -> None:
    """HTTP header names must use valid token characters."""
    with pytest.raises(vol.Invalid):
        parse_mcp_headers({"Bad Header": "value"})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "light.turn_on, climate.set_temperature\n todo.add_item",
            ["climate.set_temperature", "light.turn_on", "todo.add_item"],
        ),
        ([" b ", "a", "b", ""], ["a", "b"]),
        (None, []),
        ("", []),
    ],
)
def test_parse_allowed_tools_sorts_unique_tool_names(
    value: object, expected: list[str]
) -> None:
    """Allowlisted tools parse from text or sequences with stable ordering."""
    assert parse_allowed_tools(value) == expected


def test_parse_allowed_tools_rejects_invalid_type() -> None:
    """Non-text and non-sequence allowlists are rejected."""
    with pytest.raises(vol.Invalid):
        parse_allowed_tools(42)
