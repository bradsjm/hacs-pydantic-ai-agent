"""Tests for profile templated extra-body request settings."""

from unittest.mock import patch

import pytest
from custom_components.pydantic_ai_agent.const import (
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
    CONF_TEMPLATED_EXTRA_BODY,
)
from custom_components.pydantic_ai_agent.models.model_profiles import ModelProfile
from custom_components.pydantic_ai_agent.models.model_request_settings import (
    _model_settings_with_templated_extra_body,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from pydantic_ai.settings import ModelSettings


def test_model_settings_with_templated_extra_body_renders_without_mutation(
    hass: HomeAssistant,
) -> None:
    profile = ModelProfile(
        ref="p:1",
        provider_subentry_id="p",
        profile_id="1",
        title="Fast",
        provider_title="P",
        provider_mode="openai_compatible_completions",
        model_name="gpt-test",
        model_settings={
            CONF_TEMPLATED_EXTRA_BODY: [
                {
                    CONF_CHAT_TEMPLATE_KWARG_KEY: (
                        "chat_template_kwargs.enable_thinking"
                    ),
                    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ true }}",
                },
                {
                    CONF_CHAT_TEMPLATE_KWARG_KEY: "metadata.profile",
                    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: '{{ "rendered" }}',
                },
            ]
        },
    )
    settings = ModelSettings(
        extra_body={"service_tier": "flex", "metadata": {"provider": "base"}}
    )
    result = _model_settings_with_templated_extra_body(hass, profile, settings)
    extra_body = result.get("extra_body", {})
    assert isinstance(extra_body, dict)
    assert extra_body == {
        "service_tier": "flex",
        "metadata": {"provider": "base", "profile": "rendered"},
        "chat_template_kwargs": {"enable_thinking": True},
    }
    assert settings == {
        "extra_body": {"service_tier": "flex", "metadata": {"provider": "base"}}
    }


def test_model_settings_with_templated_extra_body_rejects_invalid_rows(
    hass: HomeAssistant,
) -> None:
    profile = ModelProfile(
        ref="p:1",
        provider_subentry_id="p",
        profile_id="1",
        title="Fast",
        provider_title="P",
        provider_mode="openai_compatible_completions",
        model_name="gpt-test",
        model_settings={
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
    )
    with pytest.raises(HomeAssistantError):
        _model_settings_with_templated_extra_body(hass, profile, ModelSettings())


def test_model_settings_with_templated_extra_body_rejects_non_json_render(
    hass: HomeAssistant,
) -> None:
    profile = ModelProfile(
        ref="p:1",
        provider_subentry_id="p",
        profile_id="1",
        title="Fast",
        provider_title="P",
        provider_mode="openai_compatible_completions",
        model_name="gpt-test",
        model_settings={
            CONF_TEMPLATED_EXTRA_BODY: [
                {
                    CONF_CHAT_TEMPLATE_KWARG_KEY: "generated_at",
                    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ now() }}",
                }
            ]
        },
    )
    with (
        patch(
            "custom_components.pydantic_ai_agent.models.templated_extra_body.Template.async_render",
            return_value=object(),
        ),
        pytest.raises(HomeAssistantError),
    ):
        _model_settings_with_templated_extra_body(hass, profile, ModelSettings())


def test_model_settings_with_templated_extra_body_preserves_legacy_dotted_keys(
    hass: HomeAssistant,
) -> None:
    profile = ModelProfile(
        ref="p:1",
        provider_subentry_id="p",
        profile_id="1",
        title="Fast",
        provider_title="P",
        provider_mode="openai_compatible_completions",
        model_name="gpt-test",
        model_settings={
            CONF_TEMPLATED_EXTRA_BODY: [
                {
                    CONF_CHAT_TEMPLATE_KWARG_KEY: "chat_template_kwargs.foo.bar",
                    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ true }}",
                }
            ]
        },
    )
    result = _model_settings_with_templated_extra_body(hass, profile, ModelSettings())
    assert result.get("extra_body") == {"chat_template_kwargs": {"foo.bar": True}}


def test_model_settings_with_templated_extra_body_rejects_shape_conflicts(
    hass: HomeAssistant,
) -> None:
    profile = ModelProfile(
        ref="p:1",
        provider_subentry_id="p",
        profile_id="1",
        title="Fast",
        provider_title="P",
        provider_mode="openai_compatible_completions",
        model_name="gpt-test",
        model_settings={
            CONF_TEMPLATED_EXTRA_BODY: [
                {
                    CONF_CHAT_TEMPLATE_KWARG_KEY: "metadata.profile",
                    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: '{{ "rendered" }}',
                }
            ]
        },
    )
    settings = ModelSettings(extra_body={"metadata": "base"})
    with pytest.raises(HomeAssistantError):
        _model_settings_with_templated_extra_body(hass, profile, settings)
