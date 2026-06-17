"""Test Pydantic AI Agent conversation virtual workspace and fallback."""

from types import SimpleNamespace
from unittest.mock import patch

from custom_components.pydantic_ai_agent.const import (
    CONF_FALLBACK_MODEL_REFS,
)
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry
from tests.components.pydantic_ai_agent.support.builders import (
    conversation_subentry_data,
    model_profile_data,
    provider_runtime_data,
    provider_subentry_data,
    workspace_entry,
    workspace_runtime_data,
)
from tests.components.pydantic_ai_agent.support.pydantic_ai import (
    Agent as _Agent,
)

_PROVIDER_SUBENTRY_ID = "provider-1"
_MODEL_PROFILE_ID = "model-profile-1"
_MODEL_PROFILE_REF = f"{_PROVIDER_SUBENTRY_ID}:{_MODEL_PROFILE_ID}"


def _entry(
    *,
    virtual_workspace_enabled: bool = False,
    web_fetch_enabled: bool = False,
    extra_data: dict[str, object] | None = None,
) -> MockConfigEntry:
    """Return a config entry with one conversation subentry."""
    entry = workspace_entry(
        (
            conversation_subentry_data(
                _MODEL_PROFILE_REF,
                virtual_workspace_enabled=virtual_workspace_enabled,
                web_fetch_enabled=web_fetch_enabled,
                extra_data=extra_data,
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
                subentry_id=_PROVIDER_SUBENTRY_ID, name="Hosted OpenAI"
            )
        },
    )
    return entry


async def test_conversation_runtime_adds_virtual_workspace_tools(
    hass: HomeAssistant,
) -> None:
    """Test virtual workspace-enabled conversations add tools/instructions."""
    entry = _entry(virtual_workspace_enabled=True)
    entry.add_to_hass(hass)
    fake_toolset = object()

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    entity_id = next(
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id != "conversation.home_assistant"
    )
    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.virtual_workspace_parts",
            return_value=SimpleNamespace(
                toolsets=(fake_toolset,), instructions="virtual instructions"
            ),
        ) as workspace_parts,
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            return_value=_Agent(),
        ) as agent_class,
    ):
        result = await conversation.async_converse(
            hass,
            "use a workspace",
            None,
            Context(),
            agent_id=entity_id,
        )

    assert result.response.speech["plain"]["speech"]
    assert workspace_parts.call_count == 1
    assert agent_class.call_args.kwargs["toolsets"]
    assert agent_class.call_args.kwargs["toolsets"][0] is fake_toolset
    assert agent_class.call_args.kwargs["instructions"] == "virtual instructions"


async def test_conversation_runtime_recreates_virtual_workspace_for_fallback(
    hass: HomeAssistant,
) -> None:
    """Test fallback model attempts do not reuse a failed workspace."""
    primary_ref = f"{_PROVIDER_SUBENTRY_ID}:primary-model"
    fallback_ref = f"{_PROVIDER_SUBENTRY_ID}:fallback-model"
    entry = workspace_entry(
        (
            conversation_subentry_data(
                primary_ref,
                virtual_workspace_enabled=True,
                extra_data={CONF_FALLBACK_MODEL_REFS: [fallback_ref]},
            ),
            provider_subentry_data(
                subentry_id=_PROVIDER_SUBENTRY_ID,
                title="Hosted OpenAI",
                default_model_profile_id="primary-model",
                model_profiles={
                    "primary-model": model_profile_data(
                        profile_id="primary-model",
                        name="Primary",
                        model="primary-model",
                    ),
                    "fallback-model": model_profile_data(
                        profile_id="fallback-model",
                        name="Fallback",
                        model="fallback-model",
                    ),
                },
            ),
        )
    )
    entry.runtime_data = workspace_runtime_data(
        providers={
            _PROVIDER_SUBENTRY_ID: provider_runtime_data(
                subentry_id=_PROVIDER_SUBENTRY_ID, name="Hosted OpenAI"
            )
        },
    )
    entry.add_to_hass(hass)

    class FailingAgent:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def run_stream_events(self, *_args, **_kwargs):
            class FailingEvents:
                async def __aenter__(self):
                    raise TimeoutError

                async def __aexit__(self, *_args):
                    return None

            return FailingEvents()

    first_toolset = object()
    second_toolset = object()

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    entity_id = next(
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id != "conversation.home_assistant"
    )
    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.virtual_workspace_parts",
            side_effect=[
                SimpleNamespace(toolsets=(first_toolset,), instructions="first"),
                SimpleNamespace(toolsets=(second_toolset,), instructions="second"),
            ],
        ) as workspace_parts,
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            side_effect=[FailingAgent(), _Agent()],
        ) as agent_class,
    ):
        result = await conversation.async_converse(
            hass,
            "use a workspace",
            None,
            Context(),
            agent_id=entity_id,
        )

    assert result.response.speech["plain"]["speech"]
    assert workspace_parts.call_count == 2
    first_attempt_toolsets = agent_class.call_args_list[0].kwargs["toolsets"]
    second_attempt_toolsets = agent_class.call_args_list[1].kwargs["toolsets"]
    assert first_attempt_toolsets != second_attempt_toolsets
    assert first_attempt_toolsets[0] is first_toolset
    assert second_attempt_toolsets[0] is second_toolset
    assert agent_class.call_args_list[0].kwargs["instructions"] == "first"
    assert agent_class.call_args_list[1].kwargs["instructions"] == "second"
