"""Test config-flow settings and selector helper behavior."""

from unittest.mock import patch

from custom_components.pydantic_ai_agent.config_flows._ai_task_schema_helpers import (
    _ai_task_data_from_user_input,
)
from custom_components.pydantic_ai_agent.config_flows._profile_helpers import (
    RunSettingsVisibility,
    _run_settings_visibility,
)
from custom_components.pydantic_ai_agent.config_flows._schema_helpers import (
    _conversation_data_from_user_input,
    _run_settings_schema,
)
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
    _parse_model_pricing,
    _parse_model_settings,
    _provider_profile_options,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_AI_TASK_NAME,
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
    CONF_ENABLED,
    CONF_FALLBACK_MODEL_REFS,
    CONF_MAX_ITERATIONS,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_MODEL_PRICING,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_NAME,
    CONF_PRIMARY_MODEL_REF,
    CONF_TEMPLATED_EXTRA_BODY,
    CONF_THINKING,
    CONF_TIMEOUT,
    CONF_TOOL_RETRIES,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE_GEMINI,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
)
from homeassistant.core import HomeAssistant
import pytest
from tests.components.pydantic_ai_agent.support.builders import (
    provider_subentry_data,
    skill_subentry_data,
    workspace_entry,
)
from tests.components.pydantic_ai_agent.support.config_flow_helpers import (
    SaveDataHelper,
    schema_key_names,
    thinking_test_entry,
)
from tests.components.pydantic_ai_agent.support.schemas import schema_select_options


def test_parse_model_settings_validates_advanced_fields(hass: HomeAssistant) -> None:
    settings, errors, cleared = _parse_model_settings(
        hass,
        {
            "top_p": "0.8",
            CONF_TEMPLATED_EXTRA_BODY: [
                {
                    CONF_CHAT_TEMPLATE_KWARG_KEY: (
                        "chat_template_kwargs.enable_thinking"
                    ),
                    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ true }}",
                }
            ],
            "seed": "",
            "frequency_penalty": "invalid",
        },
        {"top_p", CONF_TEMPLATED_EXTRA_BODY, "seed", "frequency_penalty"},
    )
    assert settings == {
        "top_p": 0.8,
        CONF_TEMPLATED_EXTRA_BODY: [
            {
                CONF_CHAT_TEMPLATE_KWARG_KEY: "chat_template_kwargs.enable_thinking",
                CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ true }}",
            }
        ],
    }
    assert errors == {"frequency_penalty": "invalid_number"}
    assert cleared == {"seed"}


def test_parse_model_settings_rejects_templated_extra_body_path_conflicts(
    hass: HomeAssistant,
) -> None:
    settings, errors, cleared = _parse_model_settings(
        hass,
        {
            CONF_TEMPLATED_EXTRA_BODY: [
                {
                    CONF_CHAT_TEMPLATE_KWARG_KEY: "a",
                    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ 1 }}",
                },
                {
                    CONF_CHAT_TEMPLATE_KWARG_KEY: "a.b",
                    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ 2 }}",
                },
            ]
        },
        {CONF_TEMPLATED_EXTRA_BODY},
    )
    assert CONF_TEMPLATED_EXTRA_BODY not in settings
    assert errors == {CONF_TEMPLATED_EXTRA_BODY: "templated_extra_body_path_conflict"}
    assert cleared == set()


def test_parse_model_settings_rejects_non_json_templated_extra_body_output(
    hass: HomeAssistant,
) -> None:
    with patch(
        "custom_components.pydantic_ai_agent.config_flows._settings_parsing.Template.async_render",
        return_value=object(),
    ):
        settings, errors, cleared = _parse_model_settings(
            hass,
            {
                CONF_TEMPLATED_EXTRA_BODY: [
                    {
                        CONF_CHAT_TEMPLATE_KWARG_KEY: "generated_at",
                        CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ now() }}",
                    }
                ]
            },
            {CONF_TEMPLATED_EXTRA_BODY},
        )
    assert settings == {}
    assert errors == {CONF_TEMPLATED_EXTRA_BODY: "invalid_chat_template"}
    assert cleared == set()


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
    with pytest.raises(ValueError, match="invalid thinking setting"):
        _parse_thinking_setting("invalid")


def test_normalise_run_settings_removes_blank_thinking() -> None:
    data = {CONF_THINKING: "", CONF_MAX_ITERATIONS: 30, CONF_TIMEOUT: 15.0}

    _normalise_run_settings(data)

    assert CONF_THINKING not in data


def test_normalise_run_settings_keeps_non_negative_tool_retries() -> None:
    data = {
        CONF_MAX_ITERATIONS: 30,
        CONF_TIMEOUT: 15.0,
        CONF_TOOL_RETRIES: 0,
    }

    _normalise_run_settings(data)

    assert data[CONF_TOOL_RETRIES] == 0


def test_run_settings_schema_omits_thinking_when_visibility_disables_it() -> None:
    data_schema = _run_settings_schema(
        default_max_iterations=10,
        visibility=RunSettingsVisibility(supports_thinking=False),
    )

    assert CONF_THINKING not in schema_key_names(data_schema)


def test_run_settings_schema_includes_thinking_when_visibility_enables_it() -> None:
    data_schema = _run_settings_schema(
        default_max_iterations=10,
        visibility=RunSettingsVisibility(supports_thinking=True),
    )

    assert CONF_THINKING in schema_key_names(data_schema)


def test_run_settings_schema_includes_tool_retries_default() -> None:
    data_schema = _run_settings_schema(default_max_iterations=10)

    assert CONF_TOOL_RETRIES in schema_key_names(data_schema)
    tool_retries_key = next(
        key for key in data_schema.schema if key.schema == CONF_TOOL_RETRIES
    )
    assert tool_retries_key.default() == 3


def test_run_settings_schema_excludes_false_thinking_option_when_not_disablable() -> (
    None
):
    data_schema = _run_settings_schema(
        default_max_iterations=10,
        visibility=RunSettingsVisibility(
            supports_thinking=True,
            can_disable_thinking=False,
        ),
    )

    assert "false" not in schema_select_options(data_schema, CONF_THINKING)


def test_run_settings_schema_keeps_false_thinking_option_when_disablable() -> None:
    data_schema = _run_settings_schema(
        default_max_iterations=10,
        visibility=RunSettingsVisibility(
            supports_thinking=True,
            can_disable_thinking=True,
        ),
    )

    assert "false" in schema_select_options(data_schema, CONF_THINKING)


@pytest.mark.parametrize(
    ("provider_mode", "model_name", "thinking_support", "supports_thinking"),
    [
        (
            PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            "deepseek-v4-flash",
            "none",
            False,
        ),
        (
            PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            "deepseek-v4-flash",
            "supported",
            True,
        ),
        (PROVIDER_ANTHROPIC, "claude-sonnet-4", None, True),
    ],
)
def test_run_settings_visibility_reflects_effective_profile_thinking_support(
    provider_mode: str,
    model_name: str,
    thinking_support: str | None,
    supports_thinking: bool,
) -> None:
    entry = thinking_test_entry(
        provider_mode,
        model_name,
        thinking_support=thinking_support,
    )

    visibility = _run_settings_visibility(entry)

    assert visibility.supports_thinking is supports_thinking


@pytest.mark.parametrize(
    ("helper", "user_input"),
    [
        (
            _conversation_data_from_user_input,
            {
                CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
                CONF_THINKING: "high",
                CONF_TOOL_RETRIES: 5,
            },
        ),
        (
            _ai_task_data_from_user_input,
            {
                CONF_AI_TASK_NAME: "Report",
                CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
                CONF_THINKING: "high",
                CONF_TOOL_RETRIES: 5,
            },
        ),
    ],
)
def test_save_time_helpers_prune_unsupported_thinking(
    helper: SaveDataHelper,
    user_input: dict[str, object],
) -> None:
    entry = thinking_test_entry(
        PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        "deepseek-v4-flash",
        thinking_support="none",
    )

    result = helper(user_input, {}, entry)

    assert result[CONF_PRIMARY_MODEL_REF] == "provider-1:profile-1"
    assert CONF_THINKING not in result
    assert result[CONF_TOOL_RETRIES] == 5


@pytest.mark.parametrize(
    ("helper", "user_input"),
    [
        (
            _conversation_data_from_user_input,
            {
                CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
                CONF_FALLBACK_MODEL_REFS: [],
                CONF_THINKING: "false",
            },
        ),
        (
            _ai_task_data_from_user_input,
            {
                CONF_AI_TASK_NAME: "Report",
                CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
                CONF_FALLBACK_MODEL_REFS: [],
                CONF_THINKING: "false",
            },
        ),
    ],
)
def test_save_time_helpers_prune_undisablable_false_thinking(
    helper: SaveDataHelper,
    user_input: dict[str, object],
) -> None:
    entry = thinking_test_entry(PROVIDER_GOOGLE_GEMINI, "gemini-2.5-pro")

    result = helper(user_input, {}, entry)

    assert result[CONF_PRIMARY_MODEL_REF] == "provider-1:profile-1"
    assert CONF_THINKING not in result


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
    assert [option["value"] for option in options] == [
        "alpha",
        "alpha-disabled",
        "zulu",
    ]
    assert any(
        option["value"] == "alpha-disabled"
        and str(option["label"]).endswith("(disabled)")
        for option in options
    )


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
    assert [option["value"] for option in _model_profile_select_options(entry)] == [
        "provider-a:profile-a",
        "provider-z:profile-b",
    ]
