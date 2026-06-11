"""Test Pydantic AI Agent conversation Logfire instrumentation."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from custom_components.pydantic_ai_agent.const import (
    DOMAIN,
    OUTPUT_MODE_TOOL,
)
from homeassistant.components import conversation
from homeassistant.const import __version__
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
    ConversationAgent as _ConversationAgent,
)

_PROVIDER_SUBENTRY_ID = "provider-1"


class _Span:
    """Synchronous context manager returned by the Logfire span mock."""

    def __init__(self) -> None:
        self.attributes: dict[str, int] = {}

    def __enter__(self) -> None:
        pass

    def __exit__(self, *_args: object) -> None:
        pass

    def set_attributes(self, attributes: dict[str, int]) -> None:
        self.attributes.update(attributes)


class _FailingSetAttributesSpan(_Span):
    """Span that fails when usage attributes are copied."""

    def set_attributes(self, attributes: dict[str, int]) -> None:
        del attributes
        raise RuntimeError("set attributes failed")


def _entry_with_conversation_subentries(logfire: bool = False) -> MockConfigEntry:
    """Return a config entry with two conversation subentries."""
    kitchen_profile_id = "kitchen-model"
    garage_profile_id = "garage-model"
    data: dict[str, object] = {"name": "Workspace"}
    if logfire:
        data["logfire_token"] = "lf-token"
        data["logfire_include_content"] = True
    entry = workspace_entry(
        (
            conversation_subentry_data(
                f"{_PROVIDER_SUBENTRY_ID}:{kitchen_profile_id}",
                title="Kitchen Agent",
                agent_name="Kitchen Agent",
            ),
            conversation_subentry_data(
                f"{_PROVIDER_SUBENTRY_ID}:{garage_profile_id}",
                title="Garage Agent",
                agent_name="Garage Agent",
            ),
            provider_subentry_data(
                subentry_id=_PROVIDER_SUBENTRY_ID,
                title="Hosted OpenAI",
                model_profiles={
                    kitchen_profile_id: model_profile_data(
                        profile_id=kitchen_profile_id,
                        name="Kitchen Model",
                        model="gpt-kitchen",
                    ),
                    garage_profile_id: model_profile_data(
                        profile_id=garage_profile_id,
                        name="Garage Model",
                        model="gpt-garage",
                    ),
                },
                default_model_profile_id=kitchen_profile_id,
            ),
        ),
        data=data,
    )
    entry.runtime_data = workspace_runtime_data(
        providers={
            _PROVIDER_SUBENTRY_ID: provider_runtime_data(
                subentry_id=_PROVIDER_SUBENTRY_ID, name="Hosted OpenAI"
            )
        },
        logfire_enabled=logfire,
        logfire_include_content=logfire,
    )
    return entry


async def test_conversation_logfire_instruments_agent_with_ha_metadata(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test active Logfire entries instrument agents and add HA trace metadata."""
    instrument = Mock()
    span = Mock(return_value=_Span())
    monkeypatch.setitem(
        sys.modules,
        "logfire",
        SimpleNamespace(configure=Mock(), instrument_pydantic_ai=instrument, span=span),
    )
    entry = _entry_with_conversation_subentries(logfire=True)
    entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent._model_validation.async_probe_model",
        new_callable=AsyncMock,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    entity_ids = sorted(
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id != "conversation.home_assistant"
    )
    kitchen_entity_id = next(
        entity_id for entity_id in entity_ids if entity_id.endswith("kitchen_agent")
    )
    agent = _ConversationAgent()
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
        await conversation.async_converse(
            hass,
            "hello",
            "conversation-test",
            Context(),
            agent_id=kitchen_entity_id,
        )

    instrument.assert_called_once_with(agent, include_content=True)
    span.assert_called_once()
    assert span.call_args.args == ("Run Pydantic AI agent",)
    assert span.call_args.kwargs["ha.domain"] == DOMAIN
    assert span.call_args.kwargs["ha.version"] == __version__
    assert span.call_args.kwargs["ha.entry_id"] == entry.entry_id
    assert span.call_args.kwargs["ha.subentry_title"] == "Kitchen Agent"
    assert span.call_args.kwargs["ha.model"] == "gpt-kitchen"
    assert span.call_args.kwargs["ha.structured_output_mode"] == OUTPUT_MODE_TOOL
    assert "ha.output_mode" not in span.call_args.kwargs
    assert span.call_args.kwargs["ha.entity_id"] == kitchen_entity_id
    assert span.call_args.kwargs["ha.conversation_id"] == "conversation-test"
    assert span.call_args.kwargs["ha.logfire_include_content"] is True
    assert span.return_value.attributes == {
        "gen_ai.usage.input_tokens": 10,
        "gen_ai.usage.output_tokens": 2,
    }


async def test_conversation_logfire_usage_failures_do_not_block_agent_run(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test optional wrapper usage attributes do not block provider responses."""
    monkeypatch.setitem(
        sys.modules,
        "logfire",
        SimpleNamespace(
            configure=Mock(),
            instrument_pydantic_ai=Mock(),
            span=Mock(return_value=_FailingSetAttributesSpan()),
        ),
    )
    entry = _entry_with_conversation_subentries(logfire=True)
    entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent._model_validation.async_probe_model",
        new_callable=AsyncMock,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    entity_id = next(
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id.endswith("kitchen_agent")
    )
    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            return_value=_ConversationAgent(),
        ),
    ):
        result = await conversation.async_converse(
            hass,
            "hello",
            "conversation-test",
            Context(),
            agent_id=entity_id,
        )

    assert result.response.speech["plain"]["speech"] == "runtime response"


async def test_conversation_logfire_failures_do_not_block_agent_run(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test optional Logfire runtime failures do not block the provider call."""
    monkeypatch.setitem(
        sys.modules,
        "logfire",
        SimpleNamespace(
            configure=Mock(),
            instrument_pydantic_ai=Mock(side_effect=RuntimeError("instrument failed")),
            span=Mock(side_effect=RuntimeError("span failed")),
        ),
    )
    entry = _entry_with_conversation_subentries(logfire=True)
    entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent._model_validation.async_probe_model",
        new_callable=AsyncMock,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    entity_ids = sorted(
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id != "conversation.home_assistant"
    )
    kitchen_entity_id = next(
        entity_id for entity_id in entity_ids if entity_id.endswith("kitchen_agent")
    )
    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            return_value=_ConversationAgent(),
        ),
    ):
        result = await conversation.async_converse(
            hass,
            "hello",
            "conversation-test",
            Context(),
            agent_id=kitchen_entity_id,
        )

    assert result.response.speech["plain"]["speech"] == "runtime response"
