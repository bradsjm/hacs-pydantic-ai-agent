"""Tests for config entry migration helpers."""

from collections.abc import Callable

from custom_components.pydantic_ai_agent import async_migrate_entry
from custom_components.pydantic_ai_agent.const import (
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
    CONF_CONTEXT_MANAGEMENT_MODE,
    CONF_CONTEXT_SUMMARIZATION_MODEL_REF,
    CONF_CONTEXT_WINDOW_SOURCE,
    CONF_CONTEXT_WINDOW_TOKENS,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_TEMPLATED_EXTRA_BODY,
    CONF_THINKING,
    CONF_THINKING_SUPPORT,
    CONTEXT_MANAGEMENT_CONTEXT_MANAGER,
    CONTEXT_MANAGEMENT_SLIDING_WINDOW,
    CONTEXT_WINDOW_SOURCE_DEFAULT,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_PROVIDER,
)
from custom_components.pydantic_ai_agent.runtime.migration import (
    _migrated_provider_thinking_support,
    _migrated_runtime_thinking,
)
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry


def test_migrated_provider_thinking_support_maps_old_modes_to_boolean() -> None:
    """Legacy profile thinking support modes migrate to model-level booleans."""
    data, changed = _migrated_provider_thinking_support(
        {
            CONF_MODEL_PROFILES: {
                "disabled": {CONF_THINKING_SUPPORT: "none"},
                "optional": {CONF_THINKING_SUPPORT: "supported"},
                "always": {CONF_THINKING_SUPPORT: "always"},
            }
        }
    )

    assert changed is True
    assert data[CONF_MODEL_PROFILES] == {
        "disabled": {CONF_THINKING_SUPPORT: False},
        "optional": {CONF_THINKING_SUPPORT: True},
        "always": {CONF_THINKING_SUPPORT: True},
    }


def test_migrated_runtime_thinking_maps_legacy_values() -> None:
    """Legacy runtime thinking values migrate to explicit effort selections."""
    assert _migrated_runtime_thinking(False) == "none"
    assert _migrated_runtime_thinking(True) == "medium"
    assert _migrated_runtime_thinking("minimal") == "low"
    assert _migrated_runtime_thinking("low") is None


async def test_migrate_entry_updates_persisted_subentries_from_minor_one(
    hass: HomeAssistant,
    make_subentry: Callable[..., ConfigSubentry],
    make_config_entry: Callable[..., MockConfigEntry],
) -> None:
    """Public migration updates legacy provider, conversation, and AI task data."""
    provider = make_subentry(
        subentry_id="provider-1",
        subentry_type=SUBENTRY_TYPE_PROVIDER,
        data={
            CONF_MODEL_PROFILES: {
                "default": {
                    CONF_MODEL_SETTINGS: {
                        "chat_template_kwargs": [
                            {
                                CONF_CHAT_TEMPLATE_KWARG_KEY: "temperature",
                                CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ temperature }}",
                            }
                        ]
                    },
                    CONF_THINKING_SUPPORT: "supported",
                }
            }
        },
    )
    conversation = make_subentry(
        subentry_id="conversation-1",
        subentry_type=SUBENTRY_TYPE_CONVERSATION,
        data={
            CONF_CONTEXT_MANAGEMENT_MODE: "invalid",
            CONF_CONTEXT_SUMMARIZATION_MODEL_REF: "",
            CONF_THINKING: True,
        },
    )
    ai_task = make_subentry(
        subentry_id="ai-task-1",
        subentry_type=SUBENTRY_TYPE_AI_TASK,
        data={"output_mode": "legacy", CONF_THINKING: False},
    )
    entry = make_config_entry(subentries=(provider, conversation, ai_task))
    entry.add_to_hass(hass)
    hass.config_entries.async_update_entry(entry, minor_version=1)

    assert await async_migrate_entry(hass, entry) is True

    assert entry.minor_version == 5
    migrated_profile = entry.subentries["provider-1"].data[CONF_MODEL_PROFILES][
        "default"
    ]
    assert migrated_profile[CONF_MODEL_SETTINGS] == {
        CONF_TEMPLATED_EXTRA_BODY: [
            {
                CONF_CHAT_TEMPLATE_KWARG_KEY: "chat_template_kwargs.temperature",
                CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ temperature }}",
            }
        ]
    }
    assert migrated_profile[CONF_CONTEXT_WINDOW_TOKENS] == DEFAULT_CONTEXT_WINDOW_TOKENS
    assert migrated_profile[CONF_CONTEXT_WINDOW_SOURCE] == CONTEXT_WINDOW_SOURCE_DEFAULT
    assert migrated_profile[CONF_THINKING_SUPPORT] is True

    migrated_conversation = entry.subentries["conversation-1"].data
    assert (
        migrated_conversation[CONF_CONTEXT_MANAGEMENT_MODE]
        == CONTEXT_MANAGEMENT_CONTEXT_MANAGER
    )
    assert CONF_CONTEXT_SUMMARIZATION_MODEL_REF not in migrated_conversation
    assert migrated_conversation[CONF_THINKING] == "medium"

    migrated_ai_task = entry.subentries["ai-task-1"].data
    assert "output_mode" not in migrated_ai_task
    assert (
        migrated_ai_task[CONF_CONTEXT_MANAGEMENT_MODE]
        == CONTEXT_MANAGEMENT_SLIDING_WINDOW
    )
    assert migrated_ai_task[CONF_THINKING] == "none"
