"""Test basic setup lifecycle for Pydantic AI Agent."""

from unittest.mock import AsyncMock, patch

from custom_components.pydantic_ai_agent import (
    PLATFORMS,
    async_setup_entry,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_MAX_TOKENS,
    CONF_THINKING,
    CONF_TIMEOUT,
)
from custom_components.pydantic_ai_agent.model_profiles import model_profile_ref
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry
from tests.components.pydantic_ai_agent.support.builders import (
    ai_task_subentry_data,
    conversation_subentry_data,
    provider_subentry_data,
    workspace_entry,
)


def _provider_subentry(
    *,
    subentry_id: str = "provider-1",
    profile_id: str = "profile-1",
    model: str = "gpt-test",
    model_settings: dict[str, object] | None = None,
    model_profiles: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    return provider_subentry_data(
        subentry_id=subentry_id,
        profile_id=profile_id,
        model=model,
        model_settings=model_settings,
        model_profiles=model_profiles,
        discovered=True,
    )


def _conversation_subentry(
    profile_ref: str, extra_data: dict[str, object] | None = None
) -> dict[str, object]:
    return conversation_subentry_data(
        profile_ref, subentry_id="conversation-1", extra_data=extra_data
    )


def _ai_task_subentry(
    profile_ref: str, extra_data: dict[str, object] | None = None
) -> dict[str, object]:
    return ai_task_subentry_data(
        profile_ref,
        subentry_id="ai-task-1",
        extra_data=extra_data,
    )


def _workspace_entry(
    subentries_data: tuple[dict[str, object], ...] = (),
    data: dict[str, object] | None = None,
) -> MockConfigEntry:
    return workspace_entry(subentries_data, data=data)


async def test_setup_entry_stores_workspace_runtime_data(hass: HomeAssistant) -> None:
    """Test setup stores workspace runtime data from provider subentries."""
    profile_ref = model_profile_ref("provider-1", "profile-1")
    entry = _workspace_entry(
        (
            _provider_subentry(),
            _conversation_subentry(profile_ref),
            _ai_task_subentry(profile_ref),
        )
    )
    entry.add_to_hass(hass)
    with patch.object(
        hass.config_entries,
        "async_forward_entry_setups",
        new_callable=AsyncMock,
    ) as forward_setups:
        assert await async_setup_entry(hass, entry)

    assert entry.runtime_data.workspace_name == "Workspace"
    assert entry.runtime_data.providers["provider-1"].api_key == "sk-test"
    assert entry.runtime_data.providers["provider-1"].name == "OpenAI-compatible"
    assert entry.runtime_data.model_profiles[profile_ref].model_name == "gpt-test"
    forward_setups.assert_awaited_once_with(entry, PLATFORMS)


async def test_setup_entry_does_not_probe_selected_model_setting_combinations(
    hass: HomeAssistant,
) -> None:
    """Test setup succeeds without validating run-specific model combinations."""
    first_ref = model_profile_ref("provider-1", "first-profile")
    second_ref = model_profile_ref("provider-2", "second-profile")
    entry = _workspace_entry(
        (
            _provider_subentry(
                profile_id="first-profile",
                model="shared-model",
                model_settings={CONF_TIMEOUT: 99.0, CONF_MAX_TOKENS: 99},
            ),
            _provider_subentry(
                subentry_id="provider-2",
                profile_id="second-profile",
                model="shared-model",
                model_settings={CONF_TIMEOUT: 99.0, CONF_THINKING: "low"},
            ),
            _conversation_subentry(
                first_ref,
                extra_data={CONF_TIMEOUT: 20.0, CONF_MAX_TOKENS: 512},
            ),
            _ai_task_subentry(
                second_ref,
                extra_data={CONF_TIMEOUT: 30.0, CONF_THINKING: "high"},
            ),
        )
    )
    entry.add_to_hass(hass)

    with patch.object(
        hass.config_entries, "async_forward_entry_setups", new_callable=AsyncMock
    ) as forward_setups:
        assert await async_setup_entry(hass, entry)

    forward_setups.assert_awaited_once_with(entry, PLATFORMS)


async def test_setup_entry_does_not_probe_distinct_subentry_run_settings(
    hass: HomeAssistant,
) -> None:
    """Test setup no longer validates per-subentry run settings at load time."""
    profile_ref = model_profile_ref("provider-1", "profile-1")
    entry = _workspace_entry(
        (
            _provider_subentry(model="shared-model"),
            conversation_subentry_data(
                profile_ref,
                subentry_id="conversation-1",
                extra_data={CONF_TIMEOUT: 20.0},
            ),
            conversation_subentry_data(
                profile_ref,
                subentry_id="conversation-2",
                title="Second Agent",
                agent_name="Second Agent",
                extra_data={CONF_THINKING: "high", CONF_TIMEOUT: 20.0},
            ),
        )
    )
    entry.add_to_hass(hass)

    with patch.object(
        hass.config_entries, "async_forward_entry_setups", new_callable=AsyncMock
    ) as forward_setups:
        assert await async_setup_entry(hass, entry)

    forward_setups.assert_awaited_once_with(entry, PLATFORMS)
