"""Test config-flow schema helper behavior."""

from collections.abc import Callable, Mapping
from typing import Any, cast

import pytest
import voluptuous as vol
from custom_components.pydantic_ai_agent.config_flows._ai_task_schema_helpers import (
    _ai_task_data_from_user_input,
    _ai_task_data_schema,
)
from custom_components.pydantic_ai_agent.config_flows._constants import (
    _MODEL_PRICING_CACHE_READ,
    _MODEL_PRICING_INPUT,
    _SECTION_MODEL_PRICING,
)
from custom_components.pydantic_ai_agent.config_flows._profile_helpers import (
    _fallback_model_profile_select_options,
    _model_profile_edit_schema,
    _selected_model_profile_error,
)
from custom_components.pydantic_ai_agent.config_flows._schema_helpers import (
    _conversation_data_from_user_input,
    _conversation_schema,
)
from custom_components.pydantic_ai_agent.config_flows.common import (
    _SECTION_FALLBACK_MODELS,
    _SECTION_RUN_SETTINGS,
    _model_settings_schema,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_AI_TASK_NAME,
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
    CONF_FALLBACK_MODEL_REFS,
    CONF_MCP_SERVER_IDS,
    CONF_MODEL_PRICING,
    CONF_MODEL_SETTINGS,
    CONF_PRIMARY_MODEL_REF,
    CONF_SKILLS,
    CONF_STREAMING_ENABLED,
    CONF_TEMPLATED_EXTRA_BODY,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
)
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers.selector import SelectSelector, SelectSelectorMode
from pytest_homeassistant_custom_component.common import MockConfigEntry
from tests.components.pydantic_ai_agent.support.builders import (
    mcp_server_subentry_data,
    model_profile_data,
    provider_subentry_data,
    skill_subentry_data,
    workspace_entry,
)
from tests.components.pydantic_ai_agent.support.config_flow_helpers import (
    SaveDataHelper,
    fallback_test_entry,
    schema_key_names,
    section_key_names,
    section_selector,
)
from tests.components.pydantic_ai_agent.support.schemas import (
    serialized_section_default,
)


def test_model_settings_schema_puts_parallel_tool_calls_first() -> None:
    data_schema = _model_settings_schema()
    assert "parallel_tool_calls" in schema_key_names(data_schema)


def test_model_settings_schema_formats_stored_values() -> None:
    data_schema = _model_settings_schema(
        {
            CONF_MODEL_SETTINGS: {
                CONF_TEMPLATED_EXTRA_BODY: [
                    {
                        CONF_CHAT_TEMPLATE_KWARG_KEY: (
                            "chat_template_kwargs.enable_thinking"
                        ),
                        CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ true }}",
                    }
                ],
            }
        }
    )
    defaults = cast(dict[str, object], data_schema({}))
    assert defaults[CONF_TEMPLATED_EXTRA_BODY] == [
        {
            CONF_CHAT_TEMPLATE_KWARG_KEY: "chat_template_kwargs.enable_thinking",
            CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ true }}",
        }
    ]


@pytest.mark.parametrize(
    ("profile", "expected_default"),
    [
        (model_profile_data(), {}),
        (
            model_profile_data(
                extra_data={CONF_MODEL_PRICING: {"input": 0.4, "cache_read": 0.0}}
            ),
            {_MODEL_PRICING_INPUT: 0.4, _MODEL_PRICING_CACHE_READ: 0.0},
        ),
    ],
)
def test_model_profile_edit_schema_serializes_pricing_defaults_only_when_present(
    profile: dict[str, object], expected_default: dict[str, float]
) -> None:
    data_schema = _model_profile_edit_schema(
        profile, PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS
    )

    assert serialized_section_default(data_schema, _SECTION_MODEL_PRICING) == (
        expected_default
    )


@pytest.mark.parametrize(
    "schema_builder",
    [_conversation_schema, _ai_task_data_schema],
)
def test_agent_schema_preserves_ordered_fallback_refs_in_serialized_defaults(
    hass: HomeAssistant,
    schema_builder: Callable[
        [HomeAssistant, Mapping[str, Any], MockConfigEntry], vol.Schema
    ],
) -> None:
    entry = fallback_test_entry()

    data_schema = schema_builder(
        hass,
        {
            CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
            CONF_FALLBACK_MODEL_REFS: [
                "provider-1:missing-profile",
                "provider-1:missing-profile",
                "provider-1:profile-2",
                "provider-1:profile-2",
            ],
        },
        entry,
    )

    assert serialized_section_default(data_schema, _SECTION_FALLBACK_MODELS) == {
        CONF_FALLBACK_MODEL_REFS: [
            "provider-1:missing-profile",
            "provider-1:profile-2",
        ]
    }


@pytest.mark.parametrize(
    ("schema_builder", "section_name", "field"),
    [
        (_conversation_schema, "hass_control", CONF_LLM_HASS_API),
        (_ai_task_data_schema, "hass_control", CONF_LLM_HASS_API),
        (_conversation_schema, "fallback_models", CONF_FALLBACK_MODEL_REFS),
        (_ai_task_data_schema, "fallback_models", CONF_FALLBACK_MODEL_REFS),
        (_conversation_schema, "skill_settings", CONF_SKILLS),
        (_conversation_schema, "external_tools", CONF_MCP_SERVER_IDS),
    ],
)
def test_agent_schema_multi_selectors_use_dropdown_mode(
    hass: HomeAssistant,
    schema_builder: Callable[
        [HomeAssistant, Mapping[str, Any], MockConfigEntry], vol.Schema
    ],
    section_name: str,
    field: str,
) -> None:
    entry = workspace_entry(
        (
            provider_subentry_data(),
            skill_subentry_data(),
            mcp_server_subentry_data(),
        )
    )

    selector = cast(
        SelectSelector,
        section_selector(schema_builder(hass, {}, entry), section_name, field),
    )

    assert isinstance(selector, SelectSelector)
    assert selector.config["mode"] == SelectSelectorMode.DROPDOWN.value
    assert selector.config["multiple"] is True


def test_conversation_schema_streaming_toggle_defaults_true(
    hass: HomeAssistant,
) -> None:
    entry = workspace_entry((provider_subentry_data(),))
    data_schema = _conversation_schema(hass, {}, entry)

    assert CONF_STREAMING_ENABLED in section_key_names(
        data_schema, _SECTION_RUN_SETTINGS
    )
    assert (
        serialized_section_default(data_schema, _SECTION_RUN_SETTINGS)[
            CONF_STREAMING_ENABLED
        ]
        is True
    )


def test_conversation_schema_streaming_toggle_defaults_false_when_saved(
    hass: HomeAssistant,
) -> None:
    entry = workspace_entry((provider_subentry_data(),))
    data_schema = _conversation_schema(hass, {CONF_STREAMING_ENABLED: False}, entry)

    assert CONF_STREAMING_ENABLED in section_key_names(
        data_schema, _SECTION_RUN_SETTINGS
    )
    assert (
        serialized_section_default(data_schema, _SECTION_RUN_SETTINGS)[
            CONF_STREAMING_ENABLED
        ]
        is False
    )


def test_fallback_model_profile_select_options_include_unavailable_selected_ref(
    hass: HomeAssistant,
) -> None:
    entry = fallback_test_entry()

    options = _fallback_model_profile_select_options(
        hass,
        entry,
        [
            "provider-1:missing-profile",
            "provider-1:missing-profile",
            "provider-1:profile-2",
        ],
    )

    assert any(option["value"] == "provider-1:missing-profile" for option in options)
    assert [option["value"] for option in options].count(
        "provider-1:missing-profile"
    ) == 1


@pytest.mark.parametrize(
    "schema_builder",
    [_conversation_schema, _ai_task_data_schema],
)
def test_agent_schema_fallback_options_exclude_primary_and_use_display_labels(
    hass: HomeAssistant,
    schema_builder: Callable[
        [HomeAssistant, Mapping[str, Any], MockConfigEntry], vol.Schema
    ],
) -> None:
    entry = fallback_test_entry()

    selector = cast(
        SelectSelector,
        section_selector(
            schema_builder(
                hass,
                {
                    CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
                    CONF_FALLBACK_MODEL_REFS: [
                        "provider-1:missing-profile",
                        "provider-1:profile-2",
                    ],
                },
                entry,
            ),
            "fallback_models",
            CONF_FALLBACK_MODEL_REFS,
        ),
    )

    option_values = [option["value"] for option in selector.config["options"]]
    assert "provider-1:profile-1" not in option_values
    assert "provider-1:profile-2" in option_values
    assert "provider-1:missing-profile" in option_values


@pytest.mark.parametrize(
    ("helper", "user_input"),
    [
        (
            _conversation_data_from_user_input,
            {
                CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
                CONF_FALLBACK_MODEL_REFS: [
                    "provider-1:profile-3",
                    "provider-1:profile-2",
                ],
            },
        ),
        (
            _ai_task_data_from_user_input,
            {
                CONF_AI_TASK_NAME: "Report",
                CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
                CONF_FALLBACK_MODEL_REFS: [
                    "provider-1:profile-3",
                    "provider-1:profile-2",
                ],
            },
        ),
    ],
)
def test_save_time_helpers_preserve_fallback_row_order(
    helper: SaveDataHelper,
    user_input: dict[str, object],
) -> None:
    entry = fallback_test_entry()

    result = helper(user_input, {}, entry)

    assert result[CONF_FALLBACK_MODEL_REFS] == [
        "provider-1:profile-3",
        "provider-1:profile-2",
    ]


def test_conversation_data_from_user_input_prunes_default_streaming_true() -> None:
    entry = fallback_test_entry()

    result = _conversation_data_from_user_input(
        {
            CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
            _SECTION_RUN_SETTINGS: {CONF_STREAMING_ENABLED: True},
        },
        {},
        entry,
    )

    assert CONF_STREAMING_ENABLED not in result


def test_conversation_data_from_user_input_persists_explicit_streaming_false() -> None:
    entry = fallback_test_entry()

    result = _conversation_data_from_user_input(
        {
            CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
            _SECTION_RUN_SETTINGS: {CONF_STREAMING_ENABLED: False},
        },
        {},
        entry,
    )

    assert result[CONF_STREAMING_ENABLED] is False


@pytest.mark.parametrize(
    ("helper", "user_input", "expected_error"),
    [
        (
            _conversation_data_from_user_input,
            {
                CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
                CONF_FALLBACK_MODEL_REFS: [
                    "provider-1:profile-2",
                    "provider-1:profile-2",
                ],
            },
            "duplicate_fallback_model",
        ),
        (
            _conversation_data_from_user_input,
            {
                CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
                CONF_FALLBACK_MODEL_REFS: [
                    "provider-1:profile-1",
                    "provider-1:profile-2",
                ],
            },
            "primary_model_in_fallbacks",
        ),
        (
            _ai_task_data_from_user_input,
            {
                CONF_AI_TASK_NAME: "Report",
                CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
                CONF_FALLBACK_MODEL_REFS: [
                    "provider-1:profile-2",
                    "provider-1:profile-2",
                ],
            },
            "duplicate_fallback_model",
        ),
        (
            _ai_task_data_from_user_input,
            {
                CONF_AI_TASK_NAME: "Report",
                CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
                CONF_FALLBACK_MODEL_REFS: [
                    "provider-1:profile-1",
                    "provider-1:profile-2",
                ],
            },
            "primary_model_in_fallbacks",
        ),
    ],
)
def test_selected_model_profile_error_validates_fallback_rows(
    hass: HomeAssistant,
    helper: SaveDataHelper,
    user_input: dict[str, object],
    expected_error: str,
) -> None:
    entry = fallback_test_entry()

    data = helper(user_input, {}, entry)

    assert _selected_model_profile_error(hass, entry, data) == expected_error
