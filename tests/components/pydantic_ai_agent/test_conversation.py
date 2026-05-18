"""Test Pydantic AI Agent conversation entities."""

from collections.abc import AsyncGenerator, Iterable
from contextlib import asynccontextmanager
import sys
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from homeassistant import config_entries
from homeassistant.components import conversation
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME, __version__
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity import Entity
from homeassistant.helpers import llm
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pydantic_ai import (
    AgentRunResultEvent,
    FunctionToolset,
    ModelResponse,
    PartStartEvent,
    TextPart,
)
from pydantic_ai.capabilities import ToolSearch

from custom_components.pydantic_ai_agent import PydanticAIAgentRuntimeData
from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_LOGFIRE_INCLUDE_CONTENT,
    CONF_LOGFIRE_TOKEN,
    CONF_MAX_ITERATIONS,
    CONF_MODEL,
    CONF_MODEL_SETTINGS,
    CONF_MODEL_SUBENTRY_ID,
    CONF_PROVIDER_MODE,
    CONF_SKILLS,
    CONF_WEB_FETCH_ENABLED,
    DEFAULT_SKILLS_FOLDER,
    DOMAIN,
    OUTPUT_MODE_TOOL,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_MODEL,
)
from custom_components.pydantic_ai_agent.context_management import (
    SlidingWindowContextCapability,
)
from custom_components.pydantic_ai_agent.conversation import (
    PydanticAIConversationEntity,
    async_setup_entry,
)
from custom_components.pydantic_ai_agent.metrics import EVENT_AGENT_RUN_COMPLETED


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


class _EventStream:
    """Async iterator over Pydantic AI stream events."""

    def __init__(self, events: Iterable[object]) -> None:
        """Initialize the event stream."""
        self._events = iter(events)

    def __aiter__(self) -> "_EventStream":
        """Return the async iterator."""
        return self

    async def __anext__(self) -> object:
        """Return the next stream event."""
        try:
            return next(self._events)
        except StopIteration as err:
            raise StopAsyncIteration from err


class _Usage:
    """Minimal Pydantic AI usage test double."""

    input_tokens = 10
    output_tokens = 2
    total_tokens = 12
    requests = 1
    tool_calls = 3

    def opentelemetry_attributes(self) -> dict[str, int]:
        """Return deterministic token usage attributes."""
        return {
            "gen_ai.usage.input_tokens": 10,
            "gen_ai.usage.output_tokens": 2,
        }


class _StreamResult:
    """Minimal Agent streamed result for conversation tests."""

    output = "runtime response"
    usage = _Usage()

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

    def __init__(self) -> None:
        """Initialize recorded run state."""
        self.run_kwargs: dict[str, object] = {}

    async def __aenter__(self) -> "_Agent":
        """Enter the agent context."""
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Exit the agent context."""

    @asynccontextmanager
    async def run_stream_events(
        self, *_args: object, **kwargs: object
    ) -> AsyncGenerator[_EventStream]:
        """Return deterministic streamed Agent events."""
        self.run_kwargs = kwargs
        result = _StreamResult()
        yield _EventStream(
            (
                PartStartEvent(index=0, part=TextPart(content="runtime response")),
                AgentRunResultEvent(cast(Any, result)),
            )
        )

    @asynccontextmanager
    async def run_stream(
        self, *_args: object, **_kwargs: object
    ) -> AsyncGenerator[_StreamResult]:
        """Return a deterministic streamed result."""
        yield _StreamResult()

    async def run(self, *_args: object, **kwargs: object) -> _StreamResult:
        """Return a deterministic run result."""
        self.run_kwargs = kwargs
        return _StreamResult()


def _entry(
    llm_hass_api: list[str] | None,
    skills: list[str] | None = None,
    *,
    web_fetch_enabled: bool = False,
    model_settings: dict[str, object] | None = None,
) -> MockConfigEntry:
    """Return a config entry with one conversation subentry."""
    subentry_data: dict[str, object] = {
        CONF_AGENT_NAME: "Kitchen Agent",
        CONF_MODEL_SUBENTRY_ID: "model_profile_1",
    }
    if llm_hass_api is not None:
        subentry_data[CONF_LLM_HASS_API] = llm_hass_api
    if skills is not None:
        subentry_data[CONF_SKILLS] = skills
    if web_fetch_enabled:
        subentry_data[CONF_WEB_FETCH_ENABLED] = True

    model_subentry_data: dict[str, object] = {
        CONF_NAME: "Fast GPT",
        CONF_MODEL: "gpt-test",
    }
    if model_settings is not None:
        model_subentry_data[CONF_MODEL_SETTINGS] = model_settings

    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Hosted OpenAI",
        data={
            CONF_NAME: "Hosted OpenAI",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
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
            {
                "subentry_id": "model_profile_1",
                "data": model_subentry_data,
                "subentry_type": SUBENTRY_TYPE_MODEL,
                "title": "Fast GPT",
                "unique_id": None,
            },
        ),
        options={},
        unique_id=None,
    )
    entry.runtime_data = PydanticAIAgentRuntimeData(
        provider_mode=PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
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
        CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
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
                    CONF_MODEL_SUBENTRY_ID: "kitchen_model",
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
            },
            {
                "data": {
                    CONF_AGENT_NAME: "Garage Agent",
                    CONF_MODEL_SUBENTRY_ID: "garage_model",
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Garage Agent",
                "unique_id": None,
            },
            {
                "subentry_id": "kitchen_model",
                "data": {CONF_NAME: "Kitchen Model", CONF_MODEL: "gpt-kitchen"},
                "subentry_type": SUBENTRY_TYPE_MODEL,
                "title": "Kitchen Model",
                "unique_id": None,
            },
            {
                "subentry_id": "garage_model",
                "data": {CONF_NAME: "Garage Model", CONF_MODEL: "gpt-garage"},
                "subentry_type": SUBENTRY_TYPE_MODEL,
                "title": "Garage Model",
                "unique_id": None,
            },
        ),
        options={},
        unique_id=None,
    )
    entry.runtime_data = PydanticAIAgentRuntimeData(
        provider_mode=PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
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

    def __init__(self) -> None:
        """Initialize recorded attributes."""
        self.attributes: dict[str, int] = {}

    def __enter__(self) -> None:
        """Enter the mocked span."""

    def __exit__(self, *_args: object) -> None:
        """Exit the mocked span."""

    def set_attributes(self, attributes: dict[str, int]) -> None:
        """Record attributes set on the mocked span."""
        self.attributes.update(attributes)


class _FailingSetAttributesSpan(_Span):
    """Span that fails when usage attributes are copied."""

    def set_attributes(self, attributes: dict[str, int]) -> None:
        """Raise while setting attributes."""
        del attributes
        raise RuntimeError("set attributes failed")


def _assert_context_management_capability(capabilities: list[object]) -> None:
    """Assert the automatic context management capability is attached."""
    assert any(
        isinstance(capability, SlidingWindowContextCapability)
        for capability in capabilities
    )


def _state(hass: HomeAssistant, entity_id: str) -> str:
    """Return a state value for an expected entity."""
    state = hass.states.get(entity_id)
    assert state is not None
    return state.state


def test_conversation_entity_controls_home_assistant_with_llm_api() -> None:
    """Test LLM API selection enables Home Assistant control support."""
    entry = _entry([llm.LLM_API_ASSIST])
    subentry = next(iter(entry.subentries.values()))

    entity = PydanticAIConversationEntity(entry, subentry)

    assert entity.supported_features == conversation.ConversationEntityFeature.CONTROL
    assert entity.extra_state_attributes["ha_tools_enabled"] is True
    assert entity.extra_state_attributes["ha_llm_api"] == [llm.LLM_API_ASSIST]
    assert entity.extra_state_attributes["web_fetch_enabled"] is False


def test_conversation_entity_advertises_streaming() -> None:
    """Test Agent-backed conversation entities advertise streaming."""
    entry = _entry(None)
    subentry = next(iter(entry.subentries.values()))

    entity = PydanticAIConversationEntity(entry, subentry)

    assert entity.supports_streaming is True


def test_conversation_entity_without_llm_api_has_no_control() -> None:
    """Test missing LLM API leaves Home Assistant control support disabled."""
    entry = _entry(None)
    subentry = next(iter(entry.subentries.values()))

    entity = PydanticAIConversationEntity(entry, subentry)

    assert entity.supported_features == 0
    assert entity.extra_state_attributes["ha_tools_enabled"] is False
    assert entity.extra_state_attributes["ha_llm_api"] is None
    assert entity.extra_state_attributes["web_fetch_enabled"] is False


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
    events: list[dict[str, object]] = []
    hass.bus.async_listen(
        f"{DOMAIN}_{EVENT_AGENT_RUN_COMPLETED}",
        lambda event: events.append(dict(event.data)),
    )
    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
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
        await hass.async_block_till_done()

    assert result.response.speech["plain"]["speech"] == "runtime response"
    assert chat_model.call_args is not None
    assert chat_model.call_args.args[2].model_name == "gpt-kitchen"
    assert agent_class.call_args.kwargs["output_type"] is str
    _assert_context_management_capability(agent_class.call_args.kwargs["capabilities"])
    assert (
        _state(hass, "sensor.kitchen_agent_last_run_model_profile") == "Kitchen Model"
    )
    assert _state(hass, "sensor.kitchen_agent_last_run_input_tokens") == "10"
    assert _state(hass, "sensor.kitchen_agent_last_run_output_tokens") == "2"
    assert _state(hass, "sensor.kitchen_agent_last_run_total_tokens") == "12"
    assert _state(hass, "sensor.kitchen_agent_last_run_model_request_count") == "1"
    assert _state(hass, "sensor.kitchen_agent_last_run_tool_use_count") == "3"
    assert _state(hass, "sensor.kitchen_agent_cumulative_input_tokens") == "10"
    assert _state(hass, "sensor.kitchen_agent_cumulative_output_tokens") == "2"
    assert _state(hass, "sensor.kitchen_agent_cumulative_total_tokens") == "12"
    assert _state(hass, "sensor.kitchen_agent_consecutive_failures") == "0"
    assert _state(hass, "sensor.kitchen_agent_last_error_type") == "unknown"
    assert _state(hass, "binary_sensor.kitchen_agent_provider_healthy") == "on"
    assert _state(hass, "binary_sensor.kitchen_agent_last_run_succeeded") == "on"
    assert events == [
        {
            "config_entry_id": entry.entry_id,
            "subentry_id": next(iter(entry.subentries.values())).subentry_id,
            "entity_id": kitchen_entity_id,
            "model_profile": "Kitchen Model",
        }
    ]


async def test_conversation_runtime_uses_configured_max_iterations(
    hass: HomeAssistant,
) -> None:
    """Test conversation runs use the model profile iteration limit."""
    entry = _entry(None, model_settings={CONF_MAX_ITERATIONS: 24})
    entry.add_to_hass(hass)
    agent = _Agent()

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
            None,
            Context(),
            agent_id=entity_id,
        )

    assert getattr(agent.run_kwargs["usage_limits"], "request_limit") == 24


async def test_conversation_runtime_defaults_max_iterations(
    hass: HomeAssistant,
) -> None:
    """Test conversation runs keep the default iteration limit when unset."""
    entry = _entry(None)
    entry.add_to_hass(hass)
    agent = _Agent()

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
            None,
            Context(),
            agent_id=entity_id,
        )

    assert getattr(agent.run_kwargs["usage_limits"], "request_limit") == 10


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
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
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

    skills_capabilities.assert_awaited_once_with(hass, entry, ["kitchen-skill"])
    capabilities = agent_class.call_args.kwargs["capabilities"]
    assert capability in capabilities
    _assert_context_management_capability(capabilities)


async def test_conversation_runtime_adds_web_fetch_capability(
    hass: HomeAssistant,
) -> None:
    """Test WebFetch-enabled conversation agents get the WebFetch capability."""
    entry = _entry(None, web_fetch_enabled=True)
    entry.add_to_hass(hass)
    web_fetch_capability = object()

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
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.WebFetch",
            return_value=web_fetch_capability,
        ) as web_fetch,
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            return_value=_Agent(),
        ) as agent_class,
    ):
        await conversation.async_converse(
            hass,
            "fetch https://example.com",
            None,
            Context(),
            agent_id=entity_id,
        )

    web_fetch.assert_called_once_with(local=True)
    capabilities = agent_class.call_args.kwargs["capabilities"]
    assert web_fetch_capability in capabilities
    _assert_context_management_capability(capabilities)


async def test_conversation_runtime_adds_keyword_tool_search_for_deferred_mcp(
    hass: HomeAssistant,
) -> None:
    """Test deferred MCP toolsets force local keyword tool search."""
    entry = _entry(None)
    entry.add_to_hass(hass)
    deferred_toolset = FunctionToolset().defer_loading()

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
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.async_runtime_mcp_toolsets",
            new_callable=AsyncMock,
            return_value=[deferred_toolset],
        ),
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

    capabilities = agent_class.call_args.kwargs["capabilities"]
    assert any(
        isinstance(capability, ToolSearch) and capability.strategy == "keywords"
        for capability in capabilities
    )
    _assert_context_management_capability(capabilities)


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
        "custom_components.pydantic_ai_agent.async_probe_model",
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
            return_value=_Agent(),
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
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
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
