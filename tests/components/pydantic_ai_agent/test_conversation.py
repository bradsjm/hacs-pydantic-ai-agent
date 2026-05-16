"""Test Pydantic AI Agent conversation entities."""

from collections.abc import AsyncGenerator, Iterable
from contextlib import asynccontextmanager
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from homeassistant import config_entries
from homeassistant.components import conversation
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity import Entity
from homeassistant.helpers import llm
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pydantic_ai import ModelResponse, TextPart

from custom_components.pydantic_ai_agent import PydanticAIAgentRuntimeData
from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_LOGFIRE_INCLUDE_CONTENT,
    CONF_LOGFIRE_TOKEN,
    CONF_MODEL,
    CONF_PROVIDER_MODE,
    CONF_SKILLS,
    DEFAULT_SKILLS_FOLDER,
    DOMAIN,
    PROVIDER_OPENAI,
    SUBENTRY_TYPE_CONVERSATION,
)
from custom_components.pydantic_ai_agent.conversation import (
    PydanticAIConversationEntity,
    async_setup_entry,
)


class _TextStream:
    """Async iterator over text chunks."""

    def __init__(self, text: str) -> None:
        """Initialize the event stream."""
        self._events = iter((text,))

    def __aiter__(self) -> "_TextStream":
        """Return the async iterator."""
        return self

    async def __anext__(self) -> object:
        """Return the next stream event."""
        try:
            return next(self._events)
        except StopIteration as err:
            raise StopAsyncIteration from err


class _StreamResult:
    """Minimal Agent streamed result for conversation tests."""

    output = "runtime response"

    def stream_text(self, *, delta: bool = False) -> _TextStream:
        """Return streamed text chunks."""
        del delta
        return _TextStream("runtime response")

    def get_output(self) -> str:
        """Return final output."""
        return "runtime response"

    def new_messages(self) -> list[ModelResponse]:
        """Return final Agent messages."""
        return [ModelResponse(parts=[TextPart(content="runtime response")])]


class _Agent:
    """Minimal async-context Agent test double."""

    async def __aenter__(self) -> "_Agent":
        """Enter the agent context."""
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Exit the agent context."""

    @asynccontextmanager
    async def run_stream(
        self, *_args: object, **_kwargs: object
    ) -> AsyncGenerator[_StreamResult]:
        """Return a deterministic streamed result."""
        yield _StreamResult()

    async def run(self, *_args: object, **_kwargs: object) -> _StreamResult:
        """Return a deterministic run result."""
        return _StreamResult()


def _entry(
    llm_hass_api: list[str] | None, skills: list[str] | None = None
) -> MockConfigEntry:
    """Return a config entry with one conversation subentry."""
    subentry_data: dict[str, object] = {
        CONF_AGENT_NAME: "Kitchen Agent",
        CONF_MODEL: "gpt-test",
    }
    if llm_hass_api is not None:
        subentry_data[CONF_LLM_HASS_API] = llm_hass_api
    if skills is not None:
        subentry_data[CONF_SKILLS] = skills

    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Hosted OpenAI",
        data={
            CONF_NAME: "Hosted OpenAI",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI,
            CONF_API_KEY: "sk-test",
        },
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "data": subentry_data,
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
            },
        ),
        options={},
        unique_id=None,
    )
    entry.runtime_data = PydanticAIAgentRuntimeData(
        provider_mode=PROVIDER_OPENAI,
        name="Hosted OpenAI",
        api_key="sk-test",
        base_url=None,
        logfire_enabled=False,
        logfire_include_content=False,
        skills_folder=DEFAULT_SKILLS_FOLDER,
        enable_skill_script_execution=False,
    )
    return entry


def _entry_with_conversation_subentries(*, logfire: bool = False) -> MockConfigEntry:
    """Return a config entry with two conversation subentries."""
    data: dict[str, object] = {
        CONF_NAME: "Hosted OpenAI",
        CONF_PROVIDER_MODE: PROVIDER_OPENAI,
        CONF_API_KEY: "sk-test",
    }
    if logfire:
        data[CONF_LOGFIRE_TOKEN] = "lf-token"
        data[CONF_LOGFIRE_INCLUDE_CONTENT] = True
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Hosted OpenAI",
        data=data,
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "data": {
                    CONF_AGENT_NAME: "Kitchen Agent",
                    CONF_MODEL: "gpt-kitchen",
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
            },
            {
                "data": {
                    CONF_AGENT_NAME: "Garage Agent",
                    CONF_MODEL: "gpt-garage",
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Garage Agent",
                "unique_id": None,
            },
        ),
        options={},
        unique_id=None,
    )
    entry.runtime_data = PydanticAIAgentRuntimeData(
        provider_mode=PROVIDER_OPENAI,
        name="Hosted OpenAI",
        api_key="sk-test",
        base_url=None,
        logfire_enabled=logfire,
        logfire_include_content=logfire,
        skills_folder=DEFAULT_SKILLS_FOLDER,
        enable_skill_script_execution=False,
    )
    return entry


class _Span:
    """Synchronous context manager returned by the Logfire span mock."""

    def __enter__(self) -> None:
        """Enter the mocked span."""

    def __exit__(self, *_args: object) -> None:
        """Exit the mocked span."""


def test_conversation_entity_controls_home_assistant_with_llm_api() -> None:
    """Test LLM API selection enables Home Assistant control support."""
    entry = _entry([llm.LLM_API_ASSIST])
    subentry = next(iter(entry.subentries.values()))

    entity = PydanticAIConversationEntity(entry, subentry)

    assert entity.supported_features == conversation.ConversationEntityFeature.CONTROL
    assert entity.extra_state_attributes["ha_tools_enabled"] is True
    assert entity.extra_state_attributes["ha_llm_api"] == [llm.LLM_API_ASSIST]


def test_conversation_entity_does_not_advertise_streaming() -> None:
    """Test Agent-backed conversation entities do not advertise streaming."""
    entry = _entry(None)
    subentry = next(iter(entry.subentries.values()))

    entity = PydanticAIConversationEntity(entry, subentry)

    assert entity.supports_streaming is False


def test_conversation_entity_without_llm_api_has_no_control() -> None:
    """Test missing LLM API leaves Home Assistant control support disabled."""
    entry = _entry(None)
    subentry = next(iter(entry.subentries.values()))

    entity = PydanticAIConversationEntity(entry, subentry)

    assert entity.supported_features == 0
    assert entity.extra_state_attributes["ha_tools_enabled"] is False
    assert entity.extra_state_attributes["ha_llm_api"] is None


async def test_conversation_subentries_add_separate_entity_agents(
    hass: HomeAssistant,
) -> None:
    """Test each conversation subentry is exposed as a separate entity agent."""
    entry = _entry_with_conversation_subentries()
    added_entities: list[tuple[list[Entity], str | None]] = []

    def async_add_entities(
        new_entities: Iterable[Entity],
        update_before_add: bool = False,
        *,
        config_subentry_id: str | None = None,
    ) -> None:
        del update_before_add
        added_entities.append((list(new_entities), config_subentry_id))

    await async_setup_entry(hass, entry, async_add_entities)

    subentries = list(entry.subentries.values())
    assert [item[1] for item in added_entities] == [
        subentries[0].subentry_id,
        subentries[1].subentry_id,
    ]
    assert [item[0][0].unique_id for item in added_entities] == [
        subentries[0].subentry_id,
        subentries[1].subentry_id,
    ]
    assert [item[0][0].device_info for item in added_entities] == [
        {
            "identifiers": {(DOMAIN, subentries[0].subentry_id)},
            "name": "Kitchen Agent",
            "manufacturer": "Pydantic AI",
            "model": "gpt-kitchen",
            "entry_type": dr.DeviceEntryType.SERVICE,
        },
        {
            "identifiers": {(DOMAIN, subentries[1].subentry_id)},
            "name": "Garage Agent",
            "manufacturer": "Pydantic AI",
            "model": "gpt-garage",
            "entry_type": dr.DeviceEntryType.SERVICE,
        },
    ]


async def test_conversation_entity_id_dispatches_assist_agent(
    hass: HomeAssistant,
) -> None:
    """Test conversation entity IDs are valid Assist agent IDs."""
    entry = _entry_with_conversation_subentries()
    entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
        new_callable=AsyncMock,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    entity_ids = sorted(
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id != "conversation.home_assistant"
    )

    assert len(entity_ids) == 2
    assert all(
        conversation.async_get_agent(hass, entity_id) for entity_id in entity_ids
    )

    kitchen_entity_id = next(
        entity_id for entity_id in entity_ids if entity_id.endswith("kitchen_agent")
    )
    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.openai_chat_model",
            return_value=object(),
        ) as chat_model,
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            return_value=_Agent(),
        ) as agent_class,
    ):
        result = await conversation.async_converse(
            hass,
            "hello",
            None,
            Context(),
            agent_id=kitchen_entity_id,
        )

    assert result.response.speech["plain"]["speech"] == "runtime response"
    assert chat_model.call_args is not None
    assert chat_model.call_args.kwargs["model_name"] == "gpt-kitchen"
    assert agent_class.call_args.kwargs["output_type"] is str


async def test_conversation_runtime_passes_selected_skills_capabilities(
    hass: HomeAssistant,
) -> None:
    """Test selected conversation skills become Agent capabilities."""
    entry = _entry(None, skills=["kitchen-skill"])
    entry.add_to_hass(hass)
    capability = object()

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
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
            "custom_components.pydantic_ai_agent.entity.openai_chat_model",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.async_skills_capabilities",
            new_callable=AsyncMock,
            return_value=[capability],
        ) as skills_capabilities,
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            return_value=_Agent(),
        ) as agent_class,
    ):
        await conversation.async_converse(
            hass,
            "hello",
            None,
            Context(),
            agent_id=entity_id,
        )

    skills_capabilities.assert_awaited_once_with(
        hass, entry, ["kitchen-skill"]
    )
    assert agent_class.call_args.kwargs["capabilities"] == [capability]


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
        "custom_components.pydantic_ai_agent.async_probe_model",
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
    agent = _Agent()
    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.openai_chat_model",
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
    assert span.call_args.kwargs["ha.entry_id"] == entry.entry_id
    assert span.call_args.kwargs["ha.subentry_title"] == "Kitchen Agent"
    assert span.call_args.kwargs["ha.model"] == "gpt-kitchen"
    assert span.call_args.kwargs["ha.entity_id"] == kitchen_entity_id
    assert span.call_args.kwargs["ha.conversation_id"] == "conversation-test"
    assert span.call_args.kwargs["ha.logfire_include_content"] is True


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
        "custom_components.pydantic_ai_agent.async_probe_model",
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
            "custom_components.pydantic_ai_agent.entity.openai_chat_model",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            return_value=_Agent(),
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
