"""Test config-flow helper behavior for Pydantic AI Agent."""

import errno
import socket
import ssl

import httpx
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
import pytest

from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.pydantic_ai_agent.config_flow import (
    _format_api_error,
    _format_mcp_headers,
    _map_http_error,
    _mcp_tool_options,
    _mcp_url_already_configured,
    _mcp_url_identity,
    _model_settings_schema,
    _parse_model_settings,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
    CONF_CHAT_TEMPLATE_KWARGS,
    CONF_MAX_ITERATIONS,
    CONF_MCP_URL,
    CONF_MODEL_SETTINGS,
    CONF_PROVIDER_MODE,
    DOMAIN,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_MCP_SERVER,
)
from custom_components.pydantic_ai_agent.mcp import MCPValidationError


def test_http_error_formats_redacted_compact_metadata() -> None:
    """Test provider HTTP errors redact metadata without SDK wrapper noise."""
    err = ModelHTTPError(
        status_code=402,
        model_name="deepseek/deepseek-v4-flash:free",
        body={
            "message": "Provider returned error",
            "metadata": {
                "provider_name": "Crucible",
                "access_token": "secret-token",
                "request_headers": {"Authorization": "Bearer nested-secret"},
            },
        },
    )

    result = _map_http_error(err)

    assert result.reason == "provider_error"
    assert result.status_code == 402
    assert "payment issue" in result.message
    assert "'provider_name': 'Crucible'" in result.message
    assert "'access_token': '**REDACTED**'" in result.message
    assert "'request_headers': '**REDACTED**'" in result.message
    assert "secret-token" not in result.message
    assert "nested-secret" not in result.message
    assert "status_code:" not in result.message
    assert "body:" not in result.message


@pytest.mark.parametrize(
    ("status_code", "expected_reason", "expected_label"),
    [
        (400, "invalid_model", "invalid request"),
        (401, "invalid_auth", "authentication issue"),
        (403, "permission_denied", "permission issue"),
        (404, "invalid_model", "model not found"),
        (408, "timeout", "timeout"),
        (429, "rate_limited", "rate limit"),
        (500, "provider_error", "provider server issue"),
    ],
)
def test_http_error_status_categories(
    status_code: int, expected_reason: str, expected_label: str
) -> None:
    """Test HTTP status codes map to stable reasons and labels."""
    err = ModelHTTPError(status_code=status_code, model_name="gpt-test", body=None)

    result = _map_http_error(err)

    assert result.reason == expected_reason
    assert result.message == (
        f"The provider returned error {status_code} ({expected_label}) "
        'for model "gpt-test".'
    )


@pytest.mark.parametrize(
    ("cause", "expected_reason", "expected_message"),
    [
        (socket.gaierror(), "cannot_connect", "Host not found."),
        (
            OSError(errno.ECONNREFUSED, "refused"),
            "cannot_connect",
            "Connection refused.",
        ),
        (
            OSError(errno.ENETUNREACH, "unreachable"),
            "cannot_connect",
            "Network unreachable.",
        ),
        (ssl.SSLError("certificate verify failed"), "cannot_connect", "TLS error."),
        (TimeoutError(), "timeout", "Request timed out."),
        (httpx.ReadTimeout("timeout"), "timeout", "Request timed out."),
    ],
)
def test_api_error_connection_categories(
    cause: BaseException, expected_reason: str, expected_message: str
) -> None:
    """Test wrapped connection failures use well-defined messages."""
    err = ModelAPIError("gpt-test", "probe failed")
    err.__cause__ = cause

    result = _format_api_error(err)

    assert result.reason == expected_reason
    assert result.message == expected_message


def test_api_error_fallback_is_concise() -> None:
    """Test non-HTTP API errors avoid raw upstream exception dumps."""
    err = ModelAPIError("gpt-test", "status_code: 500, body: {'huge': 'payload'}")

    result = _format_api_error(err)

    assert result.reason == "provider_error"
    assert result.message == 'The provider returned an API error for model "gpt-test".'


def test_model_settings_schema_puts_parallel_tool_calls_first() -> None:
    """Test advanced model settings render parallel tool calls first."""
    data_schema = _model_settings_schema()

    first_key = next(iter(data_schema.schema))

    assert first_key.schema == "parallel_tool_calls"


def test_model_settings_schema_formats_stored_values() -> None:
    """Test stored object settings render as selector suggested/default values."""
    data_schema = _model_settings_schema(
        {
            CONF_MODEL_SETTINGS: {
                "extra_body": {"service_tier": "flex", "nullable": None},
                CONF_CHAT_TEMPLATE_KWARGS: [
                    {
                        CONF_CHAT_TEMPLATE_KWARG_KEY: "enable_thinking",
                        CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ true }}",
                    }
                ],
            }
        }
    )

    extra_body_key = next(
        key
        for key in data_schema.schema
        if getattr(key, "schema", None) == "extra_body"
    )
    defaults = data_schema({})

    assert extra_body_key.description == {
        "suggested_value": 'nullable: null\nservice_tier: "flex"'
    }
    assert defaults[CONF_CHAT_TEMPLATE_KWARGS] == [
        {
            CONF_CHAT_TEMPLATE_KWARG_KEY: "enable_thinking",
            CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ true }}",
        }
    ]


def test_parse_model_settings_validates_advanced_fields(hass: HomeAssistant) -> None:
    """Test advanced model settings parse values and report field errors."""
    settings, errors, cleared = _parse_model_settings(
        hass,
        {
            "max_tokens": "1024",
            CONF_MAX_ITERATIONS: "0",
            "timeout": "30.5",
            "extra_body": 'service_tier: "flex"',
            CONF_CHAT_TEMPLATE_KWARGS: [
                {
                    CONF_CHAT_TEMPLATE_KWARG_KEY: "enable_thinking",
                    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ true }}",
                }
            ],
            "seed": "",
        },
        {
            "max_tokens",
            CONF_MAX_ITERATIONS,
            "timeout",
            "extra_body",
            CONF_CHAT_TEMPLATE_KWARGS,
            "seed",
        },
    )

    assert settings == {
        "max_tokens": 1024,
        "timeout": 30.5,
        "extra_body": {"service_tier": "flex"},
        CONF_CHAT_TEMPLATE_KWARGS: [
            {
                CONF_CHAT_TEMPLATE_KWARG_KEY: "enable_thinking",
                CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ true }}",
            }
        ],
    }
    assert errors == {CONF_MAX_ITERATIONS: "invalid_integer"}
    assert cleared == {"seed"}


def test_format_mcp_headers_uses_multiline_header_syntax() -> None:
    """Test stored MCP headers render as one header per line."""
    assert _format_mcp_headers({"X-Z": "last", "Authorization": "Bearer token"}) == (
        "Authorization: Bearer token\nX-Z: last"
    )
    assert _format_mcp_headers("X-Raw: value") == "X-Raw: value"
    assert _format_mcp_headers(None) == ""


def test_mcp_tool_options_include_truncated_descriptions() -> None:
    """Test MCP tool selector options show descriptions without changing values."""
    options = _mcp_tool_options(
        [
            {"name": "echo", "description": "Return text"},
            {"name": "long_tool", "description": " ".join(["long"] * 30)},
            {"name": "plain"},
        ],
        extra_tool_names=["stale_tool"],
    )

    assert options[0] == {"label": "echo (Return text)", "value": "echo"}
    assert options[1]["value"] == "long_tool"
    assert options[1]["label"].startswith("long_tool (long long")
    assert options[1]["label"].endswith("...)")
    assert options[2] == {"label": "plain", "value": "plain"}
    assert options[3] == {"label": "stale_tool", "value": "stale_tool"}


def test_mcp_url_identity_rejects_userinfo() -> None:
    """Test duplicate MCP URL checks reject URL credentials."""
    with pytest.raises(MCPValidationError):
        _mcp_url_identity("https://alice:one@mcp.example.com/mcp")
    assert _mcp_url_identity("https://mcp.example.com/mcp?a=1&b=2") == (
        _mcp_url_identity("https://mcp.example.com:443/mcp?b=2&a=1")
    )


def test_mcp_duplicate_check_ignores_invalid_stale_urls() -> None:
    """Test stale stored MCP URLs do not break duplicate checks."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Workspace",
        data={CONF_NAME: "Workspace"},
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "data": {
                    CONF_NAME: "Stale MCP",
                    CONF_MCP_URL: "https://user:pass@mcp.example.com/mcp",
                },
                "subentry_type": SUBENTRY_TYPE_MCP_SERVER,
                "title": "Stale MCP",
                "unique_id": None,
            },
        ),
        options={},
        unique_id=None,
        version=2,
    )

    assert not _mcp_url_already_configured(entry, "https://mcp.example.com/mcp")


def test_mcp_url_identity_rejects_invalid_url_values() -> None:
    """Test MCP URL identity rejects invalid URL values."""
    with pytest.raises(MCPValidationError):
        _mcp_url_identity("not a url")


def test_workspace_duplicate_mcp_identity_uses_normalized_url() -> None:
    """Test duplicate detection normalizes URL query and default port."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Workspace",
        data={
            CONF_NAME: "Workspace",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "sk-test",
        },
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "data": {
                    CONF_NAME: "MCP",
                    CONF_MCP_URL: "https://mcp.example.com:443/mcp?b=2&a=1",
                },
                "subentry_type": SUBENTRY_TYPE_MCP_SERVER,
                "title": "MCP",
                "unique_id": None,
            },
        ),
        options={},
        unique_id=None,
        version=2,
    )

    assert _mcp_url_already_configured(entry, "https://mcp.example.com/mcp?a=1&b=2")
