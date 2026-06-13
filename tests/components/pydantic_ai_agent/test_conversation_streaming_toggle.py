"""Test conversation streaming toggle behavior."""

from unittest.mock import AsyncMock, patch

from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from tests.components.pydantic_ai_agent.support.builders import (
    conversation_subentry_data,
    provider_runtime_data,
    provider_subentry_data,
    workspace_entry,
    workspace_runtime_data,
)
from tests.components.pydantic_ai_agent.support.pydantic_ai import Agent as _Agent

_PROVIDER_SUBENTRY_ID = "provider-1"
_MODEL_PROFILE_ID = "model-profile-1"
_MODEL_PROFILE_REF = f"{_PROVIDER_SUBENTRY_ID}:{_MODEL_PROFILE_ID}"


def _entry(*, streaming_enabled: bool):
    entry = workspace_entry(
        (
            conversation_subentry_data(
                _MODEL_PROFILE_REF,
                streaming_enabled=streaming_enabled,
            ),
            provider_subentry_data(
                subentry_id=_PROVIDER_SUBENTRY_ID,
                title="Hosted OpenAI",
                profile_id=_MODEL_PROFILE_ID,
            ),
        )
    )
    entry.runtime_data = workspace_runtime_data(
        providers={
            _PROVIDER_SUBENTRY_ID: provider_runtime_data(
                subentry_id=_PROVIDER_SUBENTRY_ID,
                name="Hosted OpenAI",
            )
        },
    )
    return entry


async def test_conversation_runtime_uses_non_streaming_run_when_disabled(
    hass: HomeAssistant,
) -> None:
    """Test disabled streaming uses the non-streaming Agent.run path."""
    entry = _entry(streaming_enabled=False)
    entry.add_to_hass(hass)
    agent = _Agent(output="non-stream response")

    with patch(
        "custom_components.pydantic_ai_agent._model_validation.async_probe_model",
        new_callable=AsyncMock,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    entity_id = next(
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id != "conversation.home_assistant"
    )
    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            return_value=agent,
        ),
    ):
        result = await conversation.async_converse(
            hass,
            "hello",
            None,
            Context(),
            agent_id=entity_id,
        )

    assert result.response.speech["plain"]["speech"] == "non-stream response"
    assert agent.run_calls == 1
    assert agent.run_stream_events_calls == 0
