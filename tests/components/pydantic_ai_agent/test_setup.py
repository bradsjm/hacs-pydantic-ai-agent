"""Test basic setup lifecycle for Pydantic AI Agent."""

from unittest.mock import AsyncMock, call, patch

from custom_components.pydantic_ai_agent import (
    PLATFORMS,
    async_setup_entry,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_MAX_TOKENS,
    CONF_THINKING,
    CONF_TIMEOUT,
    OUTPUT_MODE_TOOL,
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
        output_mode=OUTPUT_MODE_TOOL,
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
    provider_data = entry.subentries["provider-1"].data

    with (
        patch(
            "custom_components.pydantic_ai_agent._model_validation.async_probe_model",
            new_callable=AsyncMock,
        ) as probe_model,
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new_callable=AsyncMock,
        ) as forward_setups,
    ):
        assert await async_setup_entry(hass, entry)

    assert entry.runtime_data.workspace_name == "Workspace"
    assert entry.runtime_data.providers["provider-1"].api_key == "sk-test"
    assert entry.runtime_data.providers["provider-1"].name == "OpenAI-compatible"
    assert entry.runtime_data.model_profiles[profile_ref].model_name == "gpt-test"
    forward_setups.assert_awaited_once_with(entry, PLATFORMS)
    probe_model.assert_has_awaits(
        [
            call(
                hass,
                provider_data,
                "gpt-test",
                {},
                profile_id="profile-1",
                stream=True,
            ),
            call(
                hass,
                provider_data,
                "gpt-test",
                {},
                profile_id="profile-1",
                structured_output_mode=OUTPUT_MODE_TOOL,
                stream=True,
            ),
        ],
        any_order=True,
    )


async def test_setup_entry_validates_selected_model_setting_combinations(
    hass: HomeAssistant,
) -> None:
    """Test setup probes selected model/settings/output combinations."""
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

    with (
        patch(
            "custom_components.pydantic_ai_agent._model_validation.async_probe_model",
            new_callable=AsyncMock,
        ) as probe_model,
        patch.object(
            hass.config_entries, "async_forward_entry_setups", new_callable=AsyncMock
        ),
    ):
        assert await async_setup_entry(hass, entry)

    probe_model.assert_has_awaits(
        [
            call(
                hass,
                entry.subentries["provider-1"].data,
                "shared-model",
                {CONF_MAX_TOKENS: 512, CONF_TIMEOUT: 20.0},
                profile_id="first-profile",
                stream=True,
            ),
            call(
                hass,
                entry.subentries["provider-2"].data,
                "shared-model",
                {CONF_TIMEOUT: 30.0},
                profile_id="second-profile",
                structured_output_mode=OUTPUT_MODE_TOOL,
                stream=True,
            ),
        ],
        any_order=True,
    )


async def test_setup_entry_probes_distinct_subentry_run_settings(
    hass: HomeAssistant,
) -> None:
    """Test setup keeps same-model probes distinct by subentry run settings."""
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
    provider_data = entry.subentries["provider-1"].data

    with (
        patch(
            "custom_components.pydantic_ai_agent._model_validation.async_probe_model",
            new_callable=AsyncMock,
        ) as probe_model,
        patch.object(
            hass.config_entries, "async_forward_entry_setups", new_callable=AsyncMock
        ),
    ):
        assert await async_setup_entry(hass, entry)

    probe_model.assert_has_awaits(
        [
            call(
                hass,
                provider_data,
                "shared-model",
                {CONF_TIMEOUT: 20.0},
                profile_id="profile-1",
                stream=True,
            ),
        ],
        any_order=True,
    )
    assert probe_model.await_count == 1
