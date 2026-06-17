"""Test config-flow provider, MCP, and skill helper behavior."""

from typing import cast

import pytest
import voluptuous as vol
from custom_components.pydantic_ai_agent.config_flows._key_value_rows import (
    _format_key_value_json_rows,
    _format_key_value_text_rows,
    _parse_key_value_text_rows,
)
from custom_components.pydantic_ai_agent.config_flows._provider_data import (
    _format_http_headers,
    _provider_connection_schema,
)
from custom_components.pydantic_ai_agent.config_flows._settings_parsing import (
    _parse_key_value_json_setting,
)
from custom_components.pydantic_ai_agent.config_flows.common import _mcp_server_schema
from custom_components.pydantic_ai_agent.config_flows.mcp_helpers import (
    _format_mcp_headers,
    _mcp_server_data_from_user_input,
    _mcp_server_select_options,
    _mcp_tool_mode,
    _mcp_tool_options,
    _mcp_tools_schema,
    _mcp_url_already_configured,
    _mcp_url_identity,
    _parse_mcp_call_cache_ttl,
    _selected_mcp_server_error,
)
from custom_components.pydantic_ai_agent.config_flows.skill_helpers import (
    SkillDataValidationError,
    _selected_skill_error,
    _skill_data_from_user_input,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_DESCRIPTION,
    CONF_KEY_VALUE_IS_SECRET,
    CONF_KEY_VALUE_JSON_VALUE,
    CONF_KEY_VALUE_KEY,
    CONF_KEY_VALUE_VALUE,
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_CALL_CACHE_ENABLED,
    CONF_MCP_CALL_CACHE_TTL,
    CONF_MCP_HEADERS,
    CONF_MCP_SECRET_HEADER_KEYS,
    CONF_MCP_SERVER_IDS,
    CONF_MCP_URL,
    CONF_NAME,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_SECRET_HEADER_KEYS,
    CONF_SKILL_CONTENT,
    CONF_SKILL_REFERENCES,
    CONF_SKILLS,
    DEFAULT_MCP_CALL_CACHE_TTL,
    DOMAIN,
    SUBENTRY_TYPE_MCP_SERVER,
    SUBENTRY_TYPE_SKILL,
)
from custom_components.pydantic_ai_agent.mcp import MCPValidationError
from custom_components.pydantic_ai_agent.runtime.header_metadata import (
    HEADER_VALUE_REDACTED,
    parse_header_rows,
)
from homeassistant import config_entries
from homeassistant.helpers.selector import (
    ObjectSelector,
    SelectSelector,
    SelectSelectorMode,
)
from pytest_homeassistant_custom_component.common import MockConfigEntry
from tests.components.pydantic_ai_agent.support.builders import (
    mcp_server_subentry_data,
    workspace_entry,
)
from tests.components.pydantic_ai_agent.support.config_flow_helpers import (
    section_selector,
)


def test_format_key_value_rows_return_selector_defaults() -> None:
    assert sorted(
        _format_key_value_text_rows({"X-Z": "last", "Authorization": "Bearer"}),
        key=lambda row: str(row[CONF_KEY_VALUE_KEY]),
    ) == [
        {CONF_KEY_VALUE_KEY: "Authorization", CONF_KEY_VALUE_VALUE: "Bearer"},
        {CONF_KEY_VALUE_KEY: "X-Z", CONF_KEY_VALUE_VALUE: "last"},
    ]
    assert _format_key_value_json_rows({"service_tier": "flex"}) == [
        {CONF_KEY_VALUE_KEY: "service_tier", CONF_KEY_VALUE_JSON_VALUE: '"flex"'}
    ]


def test_parse_key_value_row_helpers_accept_object_selector_rows() -> None:
    assert _parse_key_value_text_rows(
        [
            {CONF_KEY_VALUE_KEY: "Authorization", CONF_KEY_VALUE_VALUE: "Bearer"},
            {CONF_KEY_VALUE_KEY: "", CONF_KEY_VALUE_VALUE: ""},
        ]
    ) == {"Authorization": "Bearer"}
    assert _parse_key_value_json_setting(
        [
            {CONF_KEY_VALUE_KEY: "service_tier", CONF_KEY_VALUE_JSON_VALUE: '"flex"'},
            {
                CONF_KEY_VALUE_KEY: "parallel_tool_calls",
                CONF_KEY_VALUE_JSON_VALUE: "true",
            },
        ]
    ) == {"parallel_tool_calls": True, "service_tier": "flex"}


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (
            [
                {CONF_KEY_VALUE_KEY: "Authorization", CONF_KEY_VALUE_VALUE: "one"},
                {CONF_KEY_VALUE_KEY: "Authorization", CONF_KEY_VALUE_VALUE: "two"},
            ],
            "duplicate_key",
        ),
        (
            [
                {
                    CONF_KEY_VALUE_KEY: "service_tier",
                    CONF_KEY_VALUE_JSON_VALUE: "not-json",
                }
            ],
            "invalid_json",
        ),
    ],
)
def test_key_value_row_helpers_reject_invalid_rows(value: object, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        if error == "duplicate_key":
            _parse_key_value_text_rows(value)
        else:
            _parse_key_value_json_setting(value)


def test_provider_and_mcp_schemas_use_object_selectors_for_structured_rows() -> None:
    provider_schema = _provider_connection_schema(
        {
            CONF_PROVIDER_HEADERS: {"Authorization": "Bearer"},
            CONF_PROVIDER_SECRET_HEADER_KEYS: ["Authorization"],
            CONF_PROVIDER_EXTRA_BODY: {"service_tier": "flex"},
        }
    )
    mcp_schema = _mcp_server_schema(
        {
            CONF_MCP_HEADERS: {"Authorization": "Bearer"},
            CONF_MCP_SECRET_HEADER_KEYS: ["Authorization"],
        }
    )

    provider_headers_selector = section_selector(
        provider_schema, "advanced_options", CONF_PROVIDER_HEADERS
    )
    provider_extra_body_selector = section_selector(
        provider_schema, "advanced_options", CONF_PROVIDER_EXTRA_BODY
    )
    mcp_headers_selector = section_selector(
        mcp_schema, "advanced_mcp", CONF_MCP_HEADERS
    )

    assert isinstance(provider_headers_selector, ObjectSelector)
    assert isinstance(provider_extra_body_selector, ObjectSelector)
    assert isinstance(mcp_headers_selector, ObjectSelector)
    provider_headers_selector = cast(ObjectSelector, provider_headers_selector)
    provider_extra_body_selector = cast(ObjectSelector, provider_extra_body_selector)
    assert provider_headers_selector.config["translation_key"] == CONF_PROVIDER_HEADERS
    assert (
        provider_headers_selector.config["fields"][CONF_KEY_VALUE_KEY]["label"]
        == "header name"
    )
    assert (
        provider_headers_selector.config["fields"][CONF_KEY_VALUE_VALUE]["label"]
        == "header value"
    )
    assert provider_headers_selector.config["fields"][CONF_KEY_VALUE_IS_SECRET][
        "selector"
    ] == {"boolean": {}}
    assert (
        provider_headers_selector.config["fields"][CONF_KEY_VALUE_VALUE]["selector"][
            "text"
        ]["type"]
        == "password"
    )
    assert provider_headers_selector.config["label_field"] == CONF_KEY_VALUE_KEY
    assert provider_headers_selector.config["description_field"] == CONF_KEY_VALUE_VALUE
    assert (
        provider_extra_body_selector.config["translation_key"]
        == CONF_PROVIDER_EXTRA_BODY
    )
    assert (
        provider_extra_body_selector.config["fields"][CONF_KEY_VALUE_KEY]["label"]
        == "parameter name"
    )
    assert (
        provider_extra_body_selector.config["fields"][CONF_KEY_VALUE_JSON_VALUE][
            "label"
        ]
        == "value"
    )
    assert provider_extra_body_selector.config["fields"][CONF_KEY_VALUE_JSON_VALUE][
        "selector"
    ] == {"template": {}}
    assert CONF_KEY_VALUE_IS_SECRET not in provider_extra_body_selector.config["fields"]
    assert "label_field" not in provider_extra_body_selector.config
    assert "description_field" not in provider_extra_body_selector.config
    mcp_headers_selector = cast(ObjectSelector, mcp_headers_selector)
    assert mcp_headers_selector.config["fields"][CONF_KEY_VALUE_IS_SECRET][
        "selector"
    ] == {"boolean": {}}
    assert (
        mcp_headers_selector.config["fields"][CONF_KEY_VALUE_VALUE]["selector"]["text"][
            "type"
        ]
        == "password"
    )
    assert mcp_headers_selector.config["label_field"] == CONF_KEY_VALUE_KEY
    assert mcp_headers_selector.config["description_field"] == CONF_KEY_VALUE_VALUE


def test_mcp_server_selector_options_are_sorted() -> None:
    entry = workspace_entry(
        (
            mcp_server_subentry_data(subentry_id="mcp-z", title="zeta MCP"),
            mcp_server_subentry_data(subentry_id="mcp-a", title="Alpha MCP"),
        )
    )

    assert [option["value"] for option in _mcp_server_select_options(entry)] == [
        "mcp-a",
        "mcp-z",
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
    assert _selected_mcp_server_error(entry, {CONF_MCP_SERVER_IDS: ["mcp-1"]}) is None
    assert _selected_mcp_server_error(entry, {CONF_MCP_SERVER_IDS: ["missing"]}) == (
        "mcp_server_not_found"
    )

    disabled_entry = workspace_entry(
        (mcp_server_subentry_data(subentry_id="mcp-1", mode="disabled"),)
    )
    assert (
        _selected_mcp_server_error(disabled_entry, {CONF_MCP_SERVER_IDS: ["mcp-1"]})
        is None
    )


def test_format_http_headers_defaults_legacy_rows_to_non_secret() -> None:
    assert _format_http_headers(
        [{CONF_KEY_VALUE_KEY: "Authorization", CONF_KEY_VALUE_VALUE: "Bearer token"}]
    ) == [
        {
            CONF_KEY_VALUE_KEY: "Authorization",
            CONF_KEY_VALUE_VALUE: "Bearer token",
            CONF_KEY_VALUE_IS_SECRET: False,
        }
    ]


def test_format_mcp_headers_returns_selector_rows_with_secret_flags() -> None:
    assert _format_mcp_headers(
        {"X-Z": "last", "Authorization": "Bearer token"},
        ["Authorization"],
    ) == [
        {
            CONF_KEY_VALUE_KEY: "Authorization",
            CONF_KEY_VALUE_VALUE: HEADER_VALUE_REDACTED,
            CONF_KEY_VALUE_IS_SECRET: True,
        },
        {
            CONF_KEY_VALUE_KEY: "X-Z",
            CONF_KEY_VALUE_VALUE: "last",
            CONF_KEY_VALUE_IS_SECRET: False,
        },
    ]
    assert _format_mcp_headers(None) == []


def test_header_row_parsing_accepts_rows_without_display_field() -> None:
    assert parse_header_rows(
        [
            {
                CONF_KEY_VALUE_KEY: "Authorization",
                CONF_KEY_VALUE_VALUE: "Bearer token",
                CONF_KEY_VALUE_IS_SECRET: True,
            }
        ]
    ) == ({"Authorization": "Bearer token"}, ["Authorization"])


def test_header_row_parsing_restores_unchanged_redacted_secret() -> None:
    assert parse_header_rows(
        [
            {
                CONF_KEY_VALUE_KEY: "Authorization",
                CONF_KEY_VALUE_VALUE: HEADER_VALUE_REDACTED,
                CONF_KEY_VALUE_IS_SECRET: True,
            }
        ],
        {"Authorization": "Bearer token"},
        ["Authorization"],
    ) == ({"Authorization": "Bearer token"}, ["Authorization"])

    assert parse_header_rows(
        [
            {
                CONF_KEY_VALUE_KEY: "Authorization",
                CONF_KEY_VALUE_VALUE: HEADER_VALUE_REDACTED,
                CONF_KEY_VALUE_IS_SECRET: False,
            }
        ],
        {"Authorization": "Bearer token"},
        ["Authorization"],
    ) == ({"Authorization": "Bearer token"}, [])

    with pytest.raises(ValueError, match="invalid_key_value"):
        parse_header_rows(
            [
                {
                    CONF_KEY_VALUE_KEY: "X-Authorization",
                    CONF_KEY_VALUE_VALUE: HEADER_VALUE_REDACTED,
                    CONF_KEY_VALUE_IS_SECRET: True,
                }
            ],
            {"Authorization": "Bearer token"},
            ["Authorization"],
        )

    assert parse_header_rows(
        [
            {
                CONF_KEY_VALUE_KEY: "X-Literal",
                CONF_KEY_VALUE_VALUE: HEADER_VALUE_REDACTED,
                CONF_KEY_VALUE_IS_SECRET: False,
            }
        ],
        {"Authorization": "Bearer token"},
        ["Authorization"],
    ) == ({"X-Literal": HEADER_VALUE_REDACTED}, [])


def test_mcp_server_data_from_user_input_defaults_and_normalizes_cache_fields() -> None:
    defaults = _mcp_server_data_from_user_input(
        {
            CONF_NAME: "  Echo MCP  ",
            CONF_MCP_URL: "https://mcp.example.com/mcp",
        }
    )

    assert defaults[CONF_NAME] == "Echo MCP"
    assert defaults[CONF_MCP_CALL_CACHE_ENABLED] is False
    assert defaults[CONF_MCP_CALL_CACHE_TTL] == DEFAULT_MCP_CALL_CACHE_TTL

    configured = _mcp_server_data_from_user_input(
        {
            CONF_NAME: "Echo MCP",
            CONF_MCP_URL: "https://mcp.example.com/mcp",
            CONF_MCP_CALL_CACHE_ENABLED: True,
            CONF_MCP_CALL_CACHE_TTL: "600",
        }
    )

    assert configured[CONF_MCP_CALL_CACHE_ENABLED] is True
    assert configured[CONF_MCP_CALL_CACHE_TTL] == 600


@pytest.mark.parametrize("value", [True, 0, "0", object()])
def test_parse_mcp_call_cache_ttl_rejects_invalid_values(value: object) -> None:
    with pytest.raises(vol.Invalid, match="invalid_mcp_call_cache_ttl"):
        _parse_mcp_call_cache_ttl(value)


def test_mcp_tool_options_use_name_only_labels() -> None:
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

    assert options == [
        {"label": "echo", "value": "echo"},
        {"label": "list_files", "value": "list_files"},
    ]


def test_mcp_tools_schema_uses_dropdown_mode() -> None:
    data_schema = _mcp_tools_schema(
        [
            {"label": "echo", "value": "echo"},
            {"label": "list_files", "value": "list_files"},
        ],
        "specified",
        ["echo"],
    )
    selector = cast(SelectSelector, list(data_schema.schema.values())[1])

    assert isinstance(selector, SelectSelector)
    assert selector.config["mode"] == SelectSelectorMode.DROPDOWN.value
    assert selector.config["multiple"] is True


def test_mcp_tool_mode_derives_legacy_states() -> None:
    assert _mcp_tool_mode({}) == "all"
    assert _mcp_tool_mode({CONF_MCP_ALLOWED_TOOLS: ["echo"]}) == "specified"
    assert _mcp_tool_mode({CONF_MCP_ALLOWED_TOOLS: []}) == "disabled"


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
