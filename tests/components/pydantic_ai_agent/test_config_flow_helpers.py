"""Test config-flow helper behavior for Pydantic AI Agent."""

import errno
import socket
import ssl
from typing import cast

import httpx
import pytest
import voluptuous as vol
from custom_components.pydantic_ai_agent.config_flows._settings_parsing import (
    _format_thinking_value,
    _normalise_run_settings,
    _parse_thinking_setting,
)
from custom_components.pydantic_ai_agent.config_flows.common import (
    _MODEL_PRICING_CACHE_READ,
    _MODEL_PRICING_INPUT,
    _MODEL_PRICING_OUTPUT,
    _model_pricing_from_options,
    _model_profile_select_options,
    _model_settings_from_options,
    _model_settings_schema,
    _parse_model_pricing,
    _parse_model_settings,
    _provider_profile_options,
)
from custom_components.pydantic_ai_agent.config_flows.mcp_helpers import (
    _format_mcp_headers,
    _mcp_server_select_options,
    _mcp_tool_options,
    _mcp_url_already_configured,
    _mcp_url_identity,
    _selected_mcp_server_error,
)
from custom_components.pydantic_ai_agent.config_flows.skill_helpers import (
    SkillDataValidationError,
    _selected_skill_error,
    _skill_data_from_user_input,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
    CONF_CHAT_TEMPLATE_KWARGS,
    CONF_DESCRIPTION,
    CONF_ENABLED,
    CONF_MAX_ITERATIONS,
    CONF_MAX_TOKENS,
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_URL,
    CONF_MODEL,
    CONF_MODEL_PRICING,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_SKILL_CONTENT,
    CONF_SKILL_REFERENCES,
    CONF_SKILLS,
    CONF_THINKING,
    CONF_TIMEOUT,
    DOMAIN,
    SUBENTRY_TYPE_MCP_SERVER,
    SUBENTRY_TYPE_SKILL,
)
from custom_components.pydantic_ai_agent.mcp import MCPValidationError
from custom_components.pydantic_ai_agent.provider_validation import (
    _format_api_error,
    _map_http_error,
)
from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pytest_homeassistant_custom_component.common import MockConfigEntry
from tests.components.pydantic_ai_agent.support.builders import (
    mcp_server_subentry_data,
    provider_subentry_data,
    skill_subentry_data,
    workspace_entry,
)


def _section_key_names(data_schema: vol.Schema, section_name: str) -> set[str]:
    for section_key, section_value in data_schema.schema.items():
        if section_key.schema == section_name:
            return {key.schema for key in section_value.schema.schema}
    raise AssertionError(f"Section {section_name} not found")


def test_http_error_formats_redacted_compact_metadata() -> None:
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
    assert "access_token" not in result.message
    assert "nested-secret" not in result.message


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
def test_http_error_status_categories(status_code, expected_reason, expected_label):
    err = ModelHTTPError(status_code=status_code, model_name="gpt-test", body=None)
    result = _map_http_error(err)
    assert result.reason == expected_reason


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
def test_api_error_connection_categories(cause, expected_reason, expected_message):
    err = ModelAPIError("gpt-test", "probe failed")
    err.__cause__ = cause
    result = _format_api_error(err)
    assert result.reason == expected_reason


def test_api_error_fallback_is_concise() -> None:
    err = ModelAPIError("gpt-test", "status_code: 500, body: {'huge': 'payload'}")
    result = _format_api_error(err)
    assert result.reason == "provider_error"


def test_model_settings_schema_puts_parallel_tool_calls_first() -> None:
    data_schema = _model_settings_schema()
    first_key = next(iter(data_schema.schema))
    assert first_key.schema == "parallel_tool_calls"


def test_model_settings_schema_formats_stored_values() -> None:
    data_schema = _model_settings_schema(
        {
            CONF_MODEL_SETTINGS: {
                CONF_CHAT_TEMPLATE_KWARGS: [
                    {
                        CONF_CHAT_TEMPLATE_KWARG_KEY: "enable_thinking",
                        CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ true }}",
                    }
                ],
            }
        }
    )
    defaults = cast(dict[str, object], data_schema({}))
    assert defaults[CONF_CHAT_TEMPLATE_KWARGS] == [
        {
            CONF_CHAT_TEMPLATE_KWARG_KEY: "enable_thinking",
            CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ true }}",
        }
    ]


def test_parse_model_settings_validates_advanced_fields(hass: HomeAssistant) -> None:
    settings, errors, cleared = _parse_model_settings(
        hass,
        {
            "top_p": "0.8",
            CONF_CHAT_TEMPLATE_KWARGS: [
                {
                    CONF_CHAT_TEMPLATE_KWARG_KEY: "enable_thinking",
                    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ true }}",
                }
            ],
            "seed": "",
            "frequency_penalty": "invalid",
        },
        {"top_p", CONF_CHAT_TEMPLATE_KWARGS, "seed", "frequency_penalty"},
    )
    assert settings == {
        "top_p": 0.8,
        CONF_CHAT_TEMPLATE_KWARGS: [
            {
                CONF_CHAT_TEMPLATE_KWARG_KEY: "enable_thinking",
                CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ true }}",
            }
        ],
    }
    assert errors == {"frequency_penalty": "invalid_number"}
    assert cleared == {"seed"}


def test_parse_model_pricing_validates_and_clears_fields() -> None:
    pricing, errors, cleared = _parse_model_pricing(
        {
            _MODEL_PRICING_INPUT: "0.4",
            _MODEL_PRICING_OUTPUT: "-1",
            _MODEL_PRICING_CACHE_READ: "",
        },
        {_MODEL_PRICING_INPUT, _MODEL_PRICING_OUTPUT, _MODEL_PRICING_CACHE_READ},
    )
    assert pricing == {"input": 0.4}
    assert errors == {_MODEL_PRICING_OUTPUT: "non_negative_number"}
    assert cleared == {"cache_read"}


def test_model_settings_from_options_sanitizes_persisted_settings() -> None:
    assert _model_settings_from_options(
        {
            CONF_MODEL_SETTINGS: {
                "temperature": 0.2,
                "extra_body": {"old": True},
                "extra_headers": {"X-Old": "value"},
                CONF_MAX_ITERATIONS: 20,
                CONF_MAX_TOKENS: 1024,
                CONF_THINKING: "high",
                CONF_TIMEOUT: 30.0,
            }
        }
    ) == {"temperature": 0.2}


def test_format_thinking_value_formats_selector_defaults() -> None:
    assert _format_thinking_value({CONF_THINKING: True}) == "true"
    assert _format_thinking_value({CONF_THINKING: False}) == "false"
    assert _format_thinking_value({CONF_THINKING: "high"}) == "high"
    assert _format_thinking_value({}) == ""


def test_parse_thinking_setting_parses_bool_and_effort_values() -> None:
    assert _parse_thinking_setting("true") is True
    assert _parse_thinking_setting("false") is False
    assert _parse_thinking_setting("low") == "low"


def test_parse_thinking_setting_rejects_invalid_values() -> None:
    try:
        _parse_thinking_setting("invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid thinking value to raise ValueError")


def test_normalise_run_settings_removes_blank_thinking() -> None:
    data = {CONF_THINKING: "", CONF_MAX_ITERATIONS: 30, CONF_TIMEOUT: 15.0}

    _normalise_run_settings(data)

    assert CONF_THINKING not in data


def test_model_pricing_from_options_sanitizes_persisted_pricing() -> None:
    assert _model_pricing_from_options(
        {
            CONF_MODEL_PRICING: {
                "input": 0.4,
                "output": -1,
                "cache_read": False,
                "ignored": 2,
            },
        }
    ) == {"input": 0.4}


def test_provider_model_profile_picker_options_are_sorted() -> None:
    options = _provider_profile_options(
        {
            CONF_MODEL_PROFILES: {
                "zulu": {CONF_NAME: "beta", CONF_MODEL: "model-z", CONF_ENABLED: True},
                "alpha-disabled": {
                    CONF_NAME: "Alpha",
                    CONF_MODEL: "model-a",
                    CONF_ENABLED: False,
                },
                "alpha": {
                    CONF_NAME: "alpha",
                    CONF_MODEL: "model-b",
                    CONF_ENABLED: True,
                },
            }
        }
    )
    assert [(o["label"], o["value"]) for o in options] == [
        ("alpha", "alpha"),
        ("Alpha (disabled)", "alpha-disabled"),
        ("beta", "zulu"),
    ]


def test_agent_selector_options_are_sorted_with_stale_values_last(
    hass: HomeAssistant,
) -> None:
    entry = workspace_entry(
        (
            provider_subentry_data(
                subentry_id="provider-z",
                title="zeta Provider",
                model_profiles={
                    "profile-b": {
                        CONF_NAME: "Beta",
                        CONF_MODEL: "beta-model",
                        CONF_ENABLED: True,
                    },
                },
            ),
            provider_subentry_data(
                subentry_id="provider-a",
                title="Alpha Provider",
                model_profiles={
                    "profile-a": {
                        CONF_NAME: "alpha",
                        CONF_MODEL: "alpha-model",
                        CONF_ENABLED: True,
                    },
                },
            ),
            skill_subentry_data(subentry_id="skill-z", title="zeta Skill"),
            skill_subentry_data(subentry_id="skill-a", title="Alpha Skill"),
        )
    )
    assert [o["label"] for o in _model_profile_select_options(entry)] == [
        "Alpha Provider / alpha",
        "zeta Provider / Beta",
    ]


def test_mcp_server_selector_options_are_sorted() -> None:
    entry = workspace_entry(
        (
            mcp_server_subentry_data(subentry_id="mcp-z", title="zeta MCP"),
            mcp_server_subentry_data(subentry_id="mcp-a", title="Alpha MCP"),
        )
    )

    assert [option["label"] for option in _mcp_server_select_options(entry)] == [
        "Alpha MCP",
        "zeta MCP",
    ]


def test_selected_skill_error_reports_stale_skill_id() -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_NAME: "Workspace"},
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "subentry_id": "skill-1",
                "subentry_type": SUBENTRY_TYPE_SKILL,
                "title": "Skill",
                "unique_id": None,
                "data": {
                    CONF_NAME: "Skill",
                    CONF_SKILL_CONTENT: "content",
                },
            },
        ),
    )
    assert _selected_skill_error(entry, {CONF_SKILLS: ["skill-1"]}) is None


def test_selected_mcp_server_error_reports_stale_or_unallowlisted_server() -> None:
    entry = workspace_entry((mcp_server_subentry_data(subentry_id="mcp-1"),))
    assert _selected_mcp_server_error(entry, {"mcp_server_ids": ["mcp-1"]}) is None
    assert _selected_mcp_server_error(entry, {"mcp_server_ids": ["missing"]}) == (
        "mcp_server_not_found"
    )

    unallowlisted_entry = workspace_entry(
        (mcp_server_subentry_data(subentry_id="mcp-1", allowed_tools=[]),)
    )
    assert (
        _selected_mcp_server_error(unallowlisted_entry, {"mcp_server_ids": ["mcp-1"]})
        == "mcp_tools_not_allowlisted"
    )


def test_format_mcp_headers_uses_multiline_header_syntax() -> None:
    assert _format_mcp_headers({"X-Z": "last", "Authorization": "Bearer token"}) == (
        "Authorization: Bearer token\nX-Z: last"
    )
    assert _format_mcp_headers("X-Raw: value") == "X-Raw: value"
    assert _format_mcp_headers(None) == ""


def test_mcp_tool_options_include_truncated_descriptions() -> None:
    options = _mcp_tool_options(
        [
            {
                "name": "echo",
                "description": (
                    "Echo text back to the caller with a fairly long description "
                    "that should truncate cleanly."
                ),
            },
            {"name": "list_files", "description": "List files"},
        ]
    )

    assert options[0]["value"] == "echo"
    assert "echo" in options[0]["label"]
    assert options[1] == {"label": "list_files (List files)", "value": "list_files"}


def test_mcp_url_identity_rejects_userinfo() -> None:
    with pytest.raises(MCPValidationError):
        _mcp_url_identity("https://alice:one@mcp.example.com/mcp")
    assert _mcp_url_identity(
        "https://mcp.example.com/mcp?a=1&b=2"
    ) == _mcp_url_identity("https://mcp.example.com:443/mcp?b=2&a=1")


def test_mcp_duplicate_check_ignores_invalid_stale_urls() -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_NAME: "Workspace"},
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "subentry_id": "mcp-stale",
                "data": {
                    CONF_NAME: "Stale MCP",
                    CONF_MCP_URL: "https://user:pass@mcp.example.com/mcp",
                },
                "subentry_type": SUBENTRY_TYPE_MCP_SERVER,
                "title": "Stale MCP",
                "unique_id": None,
            },
        ),
    )

    assert not _mcp_url_already_configured(entry, "https://mcp.example.com/mcp")


def test_workspace_duplicate_mcp_identity_uses_normalized_url() -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_NAME: "Workspace"},
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "subentry_id": "mcp-1",
                "data": {
                    CONF_NAME: "MCP",
                    CONF_MCP_URL: "https://mcp.example.com:443/mcp?b=2&a=1",
                    CONF_MCP_ALLOWED_TOOLS: ["echo"],
                },
                "subentry_type": SUBENTRY_TYPE_MCP_SERVER,
                "title": "MCP",
                "unique_id": None,
            },
        ),
    )

    assert _mcp_url_already_configured(entry, "https://mcp.example.com/mcp?a=1&b=2")
    assert _selected_skill_error(entry, {CONF_SKILLS: ["missing"]}) == "skill_not_found"


def test_skill_data_from_user_input_normalizes_and_validates() -> None:
    data = _skill_data_from_user_input(
        {
            CONF_NAME: "  Kitchen Skill  ",
            CONF_DESCRIPTION: "  Helpful  ",
            CONF_SKILL_CONTENT: "  Be concise.  ",
        }
    )
    assert data[CONF_NAME] == "Kitchen Skill"
    assert data[CONF_SKILL_REFERENCES] == []
    with pytest.raises(SkillDataValidationError) as err:
        _skill_data_from_user_input({CONF_NAME: "", CONF_SKILL_CONTENT: ""})
    assert err.value.errors == {CONF_NAME: "required", CONF_SKILL_CONTENT: "required"}
