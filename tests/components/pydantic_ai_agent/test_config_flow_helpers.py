"""Test config-flow helper behavior for Pydantic AI Agent."""

import errno
import socket
import ssl

import httpx
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
import pytest

from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.pydantic_ai_agent.config_flows.common import (
    _SECTION_FALLBACK_MODELS,
    _SECTION_HASS_CONTROL,
    _SECTION_SKILLS,
    SkillDataValidationError,
    _ai_task_data_from_user_input,
    _ai_task_data_schema,
    _conversation_data_from_user_input,
    _conversation_schema,
    _format_mcp_headers,
    _mcp_tool_options,
    _mcp_url_already_configured,
    _mcp_url_identity,
    _model_settings_from_options,
    _model_settings_schema,
    _normalise_provider_model_profiles,
    _parse_model_settings,
    _provider_data_matches,
    _provider_model_profiles_for_discovery_mode,
    _selected_skill_error,
    _skill_data_from_user_input,
    _validate_provider_data,
)
from custom_components.pydantic_ai_agent.provider_validation import (
    ProviderValidationError,
    _format_api_error,
    _map_http_error,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_AI_TASK_NAME,
    CONF_BASE_URL,
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
    CONF_CHAT_TEMPLATE_KWARGS,
    CONF_DESCRIPTION,
    CONF_DISCOVERED,
    CONF_ENABLED,
    CONF_FALLBACK_MODEL_REFS,
    CONF_MAX_ITERATIONS,
    CONF_MCP_URL,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_OUTPUT_MODE,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_MODE,
    CONF_SKILL_CONTENT,
    CONF_SKILL_REFERENCES,
    CONF_SKILLS,
    DOMAIN,
    DEFAULT_OUTPUT_MODE,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE_GEMINI,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_MCP_SERVER,
    SUBENTRY_TYPE_PROVIDER,
    SUBENTRY_TYPE_SKILL,
)
from custom_components.pydantic_ai_agent.mcp import MCPValidationError
from tests.components.pydantic_ai_agent.support.schemas import (
    schema_key_names as _schema_key_names,
)


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
                CONF_CHAT_TEMPLATE_KWARGS: [
                    {
                        CONF_CHAT_TEMPLATE_KWARG_KEY: "enable_thinking",
                        CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ true }}",
                    }
                ],
            }
        }
    )

    defaults = data_schema({})

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
            CONF_CHAT_TEMPLATE_KWARGS,
            "seed",
        },
    )

    assert settings == {
        "max_tokens": 1024,
        "timeout": 30.5,
        CONF_CHAT_TEMPLATE_KWARGS: [
            {
                CONF_CHAT_TEMPLATE_KWARG_KEY: "enable_thinking",
                CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ true }}",
            }
        ],
    }
    assert errors == {CONF_MAX_ITERATIONS: "invalid_integer"}
    assert cleared == {"seed"}


def test_model_settings_from_options_sanitizes_persisted_settings() -> None:
    """Test model profile edits keep only supported persisted model settings."""
    assert _model_settings_from_options(
        {
            CONF_MODEL_SETTINGS: {
                "temperature": 0.2,
                "extra_body": {"old": True},
                "extra_headers": {"X-Old": "value"},
            }
        }
    ) == {"temperature": 0.2}


def test_agent_schemas_group_fallbacks_and_hass_control(
    hass: HomeAssistant,
) -> None:
    """Test per-agent schemas expose requested controls as sections."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_NAME: "Workspace"},
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "subentry_id": "provider",
                "subentry_type": SUBENTRY_TYPE_PROVIDER,
                "title": "Provider",
                "unique_id": None,
                "data": {
                    CONF_NAME: "Provider",
                    CONF_MODEL_PROFILES: {
                        "primary": {
                            CONF_NAME: "Primary",
                            CONF_MODEL: "gpt-test",
                            CONF_ENABLED: True,
                        }
                    },
                },
            },
        ),
    )
    conversation_schema = _conversation_schema(hass, entry=entry)
    ai_task_schema = _ai_task_data_schema(hass, entry=entry)

    assert _SECTION_FALLBACK_MODELS in _schema_key_names(conversation_schema)
    assert _SECTION_HASS_CONTROL in _schema_key_names(conversation_schema)
    assert CONF_FALLBACK_MODEL_REFS not in _schema_key_names(conversation_schema)
    assert CONF_LLM_HASS_API not in _schema_key_names(conversation_schema)
    assert _SECTION_FALLBACK_MODELS in _schema_key_names(ai_task_schema)
    assert _SECTION_HASS_CONTROL in _schema_key_names(ai_task_schema)
    assert CONF_FALLBACK_MODEL_REFS not in _schema_key_names(ai_task_schema)
    assert CONF_LLM_HASS_API not in _schema_key_names(ai_task_schema)


def test_sectioned_conversation_input_flattens_and_prunes_legacy_skills() -> None:
    """Test sectioned conversation form input drops legacy skill fields."""
    data = _conversation_data_from_user_input(
        {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_PRIMARY_MODEL_REF: "provider:primary",
            _SECTION_FALLBACK_MODELS: {CONF_FALLBACK_MODEL_REFS: ["provider:fallback"]},
            _SECTION_HASS_CONTROL: {CONF_LLM_HASS_API: ["assist"]},
            _SECTION_SKILLS: {
                "enable_skills": False,
                "skills_folder": "/tmp/skills",
                CONF_SKILLS: ["skill-1", "skill-1", ""],
            },
        },
        {},
    )

    assert data[CONF_FALLBACK_MODEL_REFS] == ["provider:fallback"]
    assert data[CONF_LLM_HASS_API] == ["assist"]
    assert data[CONF_SKILLS] == ["skill-1"]
    assert "enable_skills" not in data
    assert "skills_folder" not in data
    assert _SECTION_FALLBACK_MODELS not in data
    assert _SECTION_HASS_CONTROL not in data
    assert _SECTION_SKILLS not in data


def test_sectioned_ai_task_input_preserves_existing_skills_when_field_omitted() -> None:
    """Test sectioned AI task input preserves selected Skill IDs on partial saves."""
    data = _ai_task_data_from_user_input(
        {
            CONF_AI_TASK_NAME: "Summary Task",
            CONF_PRIMARY_MODEL_REF: "provider:primary",
            CONF_OUTPUT_MODE: DEFAULT_OUTPUT_MODE,
            _SECTION_FALLBACK_MODELS: {CONF_FALLBACK_MODEL_REFS: ["provider:fallback"]},
            _SECTION_HASS_CONTROL: {CONF_LLM_HASS_API: ["assist"]},
            _SECTION_SKILLS: {},
        },
        {CONF_SKILLS: ["skill-1"]},
    )

    assert data[CONF_FALLBACK_MODEL_REFS] == ["provider:fallback"]
    assert data[CONF_LLM_HASS_API] == ["assist"]
    assert data[CONF_SKILLS] == ["skill-1"]
    assert _SECTION_FALLBACK_MODELS not in data
    assert _SECTION_HASS_CONTROL not in data
    assert _SECTION_SKILLS not in data


def test_sectioned_ai_task_input_prunes_empty_llm_api() -> None:
    """Test clearing AI task Home Assistant control removes the persisted key."""
    data = _ai_task_data_from_user_input(
        {
            CONF_AI_TASK_NAME: "Summary Task",
            CONF_PRIMARY_MODEL_REF: "provider:primary",
            CONF_OUTPUT_MODE: DEFAULT_OUTPUT_MODE,
            _SECTION_HASS_CONTROL: {CONF_LLM_HASS_API: []},
        },
        {},
    )

    assert CONF_LLM_HASS_API not in data
    assert _SECTION_HASS_CONTROL not in data


def test_selected_skill_error_reports_stale_skill_id() -> None:
    """Test selected Skill IDs must reference current Skill subentries."""
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
                "data": {CONF_NAME: "Skill", CONF_SKILL_CONTENT: "content"},
            },
        ),
    )

    assert _selected_skill_error(entry, {CONF_SKILLS: ["skill-1"]}) is None
    assert _selected_skill_error(entry, {CONF_SKILLS: ["missing"]}) == "skill_not_found"


def test_skill_data_from_user_input_normalizes_and_validates() -> None:
    """Test native Skill data is raw text with bounded fields."""
    data = _skill_data_from_user_input(
        {
            CONF_NAME: "  Kitchen Skill  ",
            CONF_DESCRIPTION: "  Helpful guidance  ",
            CONF_SKILL_CONTENT: "  Use short responses.  ",
        }
    )

    assert data == {
        CONF_NAME: "Kitchen Skill",
        CONF_DESCRIPTION: "Helpful guidance",
        CONF_SKILL_CONTENT: "Use short responses.",
        CONF_SKILL_REFERENCES: [],
    }

    with pytest.raises(SkillDataValidationError) as err:
        _skill_data_from_user_input({CONF_NAME: "", CONF_SKILL_CONTENT: ""})
    assert err.value.errors == {CONF_NAME: "required", CONF_SKILL_CONTENT: "required"}


def test_provider_base_url_rejects_endpoint_suffix(hass: HomeAssistant) -> None:
    """Test provider base URLs cannot point at generated API endpoints."""
    with pytest.raises(ProviderValidationError) as err:
        _validate_provider_data(
            hass,
            {
                CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
                CONF_BASE_URL: "https://api.example.com/openai/chat/completions",
            },
        )

    assert err.value.reason == "invalid_base_url_endpoint"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.anthropic.com/v1/messages",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini:generateContent",
    ],
)
def test_provider_base_url_rejects_non_openai_endpoint_suffixes(
    hass: HomeAssistant, base_url: str
) -> None:
    """Test endpoint URL validation covers native provider request endpoints."""
    with pytest.raises(ProviderValidationError) as err:
        _validate_provider_data(
            hass,
            {
                CONF_PROVIDER_MODE: PROVIDER_GOOGLE_GEMINI,
                CONF_BASE_URL: base_url,
            },
        )

    assert err.value.reason == "invalid_base_url_endpoint"


def test_provider_base_url_allows_non_v1_base(hass: HomeAssistant) -> None:
    """Test endpoint validation does not require a v1 suffix."""
    _validate_provider_data(
        hass,
        {
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_BASE_URL: "https://api.example.com/openai/deployments/gpt-test",
        },
    )


def test_provider_extra_body_rejects_gemini_provider(hass: HomeAssistant) -> None:
    """Test provider extra body cannot be configured for unsupported providers."""
    with pytest.raises(ProviderValidationError) as err:
        _validate_provider_data(
            hass,
            {
                CONF_PROVIDER_MODE: PROVIDER_GOOGLE_GEMINI,
                CONF_PROVIDER_EXTRA_BODY: {"service_tier": "flex"},
            },
        )

    assert err.value.reason == "provider_extra_body_unsupported"


def test_provider_extra_body_allows_anthropic_provider(hass: HomeAssistant) -> None:
    """Test Anthropic can use provider extra body fields."""
    _validate_provider_data(
        hass,
        {
            CONF_PROVIDER_MODE: PROVIDER_ANTHROPIC,
            CONF_PROVIDER_EXTRA_BODY: {"anthropic_beta": ["feature-test"]},
        },
    )


def test_normalise_provider_model_profiles_adds_new_profiles_disabled() -> None:
    """Test newly discovered model profiles require explicit enablement."""
    profiles = _normalise_provider_model_profiles({}, ["gpt-test"], ["gpt-test"])

    profile = next(iter(profiles.values()))
    assert profile[CONF_MODEL] == "gpt-test"
    assert profile[CONF_ENABLED] is False
    assert profile[CONF_DISCOVERED] is True


def test_normalise_provider_model_profiles_uses_catalog_display_name() -> None:
    """Test catalog names replace default identifier-derived profile names."""
    profiles = _normalise_provider_model_profiles(
        {
            "profile-1": {
                "id": "profile-1",
                CONF_NAME: "deepseek-v4-pro",
                CONF_MODEL: "deepseek-v4-pro",
                CONF_ENABLED: False,
                CONF_DISCOVERED: True,
            },
            "profile-2": {
                "id": "profile-2",
                CONF_NAME: "Custom Display Name",
                CONF_MODEL: "deepseek-v4-flash",
                CONF_ENABLED: False,
                CONF_DISCOVERED: True,
            },
        },
        ["deepseek-v4-pro", "deepseek-v4-flash"],
        ["deepseek-v4-pro", "deepseek-v4-flash"],
        model_labels={
            "deepseek-v4-pro": "Deepseek V4 Pro",
            "deepseek-v4-flash": "Deepseek V4 Flash",
        },
    )

    assert profiles["profile-1"][CONF_NAME] == "Deepseek V4 Pro"
    assert profiles["profile-2"][CONF_NAME] == "Custom Display Name"


def test_normalise_provider_model_profiles_keeps_referenced_missing_profile() -> None:
    """Test refresh pruning keeps disappeared models still referenced by agents."""
    profiles = _normalise_provider_model_profiles(
        {
            "referenced": {
                "id": "referenced",
                CONF_MODEL: "gpt-old",
                CONF_ENABLED: True,
                CONF_DISCOVERED: True,
            },
            "unreferenced": {
                "id": "unreferenced",
                CONF_MODEL: "gpt-removed",
                CONF_ENABLED: True,
                CONF_DISCOVERED: True,
            },
        },
        ["gpt-new"],
        ["gpt-new"],
        keep_profile_ids={"referenced"},
    )

    assert "referenced" in profiles
    assert profiles["referenced"][CONF_MODEL] == "gpt-old"
    assert "unreferenced" not in profiles
    assert any(profile[CONF_MODEL] == "gpt-new" for profile in profiles.values())


def test_provider_data_identity_includes_provider_extra_body() -> None:
    """Test provider-level body settings distinguish provider subentries."""
    base_data = {
        CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        CONF_API_KEY: "sk-test",
    }

    assert not _provider_data_matches(
        base_data | {CONF_PROVIDER_EXTRA_BODY: {"service_tier": "flex"}},
        base_data | {CONF_PROVIDER_EXTRA_BODY: {"service_tier": "default"}},
    )


def test_discovery_mode_profiles_drop_unreferenced_custom_profiles() -> None:
    """Test clearing custom names removes old custom profiles unless referenced."""
    profiles = _provider_model_profiles_for_discovery_mode(
        {
            "discovered": {
                "id": "discovered",
                CONF_MODEL: "gpt-listed",
                CONF_ENABLED: True,
                CONF_DISCOVERED: True,
            },
            "referenced-custom": {
                "id": "referenced-custom",
                CONF_MODEL: "gpt-custom-used",
                CONF_ENABLED: True,
                CONF_DISCOVERED: False,
            },
            "removed-custom": {
                "id": "removed-custom",
                CONF_MODEL: "gpt-custom-removed",
                CONF_ENABLED: True,
                CONF_DISCOVERED: False,
            },
        },
        keep_profile_ids={"referenced-custom"},
    )

    assert set(profiles) == {"discovered", "referenced-custom"}


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
