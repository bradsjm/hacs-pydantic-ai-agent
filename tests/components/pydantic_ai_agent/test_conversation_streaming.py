"""Test Pydantic AI Agent conversation streaming behavior."""

from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import patch

from custom_components.pydantic_ai_agent.const import (
    CONF_MAX_ITERATIONS,
    CONF_TOOL_RETRIES,
    DOMAIN,
)
from custom_components.pydantic_ai_agent.metrics import (
    EVENT_AGENT_RUN_FAILED,
)
from homeassistant.components import conversation
from homeassistant.components.conversation.chat_log import (
    DATA_CHAT_LOGS,
    AssistantContent,
)
from homeassistant.core import Context, HomeAssistant
from pydantic_ai import (
    AgentRunResultEvent,
    FunctionToolResultEvent,
    ModelResponse,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
)
from pydantic_ai.exceptions import (
    ModelRetry,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)
from tests.components.pydantic_ai_agent.support.pydantic_ai import (
    Agent as _Agent,
)
from tests.components.pydantic_ai_agent.support.pydantic_ai import (
    CallbackStreamAgent,
    RunResultWithMessages,
)
from tests.components.pydantic_ai_agent.support.runtime import (
    first_non_default_conversation_entity_id,
    loaded_conversation_entry,
)

_PROVIDER_SUBENTRY_ID = "provider-1"
_MODEL_PROFILE_ID = "model-profile-1"
_MODEL_PROFILE_REF = f"{_PROVIDER_SUBENTRY_ID}:{_MODEL_PROFILE_ID}"


async def test_streaming_iteration_failure_updates_chat_and_sensors(
    hass: HomeAssistant,
    mock_chat_model_for_profile: object,
) -> None:
    """Test streaming usage-limit failures stay actionable after partial output."""
    del mock_chat_model_for_profile
    entry = loaded_conversation_entry(extra_data={CONF_MAX_ITERATIONS: 24})
    entry.add_to_hass(hass)

    async def stream() -> AsyncIterator[object]:
        yield PartStartEvent(index=0, part=TextPart(content="partial"))
        raise UsageLimitExceeded(
            "The next request would exceed the request_limit of 24"
        )

    agent = CallbackStreamAgent(stream_factory=stream)
    events: list[dict[str, object]] = []
    hass.bus.async_listen(
        f"{DOMAIN}_{EVENT_AGENT_RUN_FAILED}",
        lambda event: events.append(dict(event.data)),
    )

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    entity_id = first_non_default_conversation_entity_id(hass)
    with patch("custom_components.pydantic_ai_agent.entity.Agent", return_value=agent):
        result = await conversation.async_converse(
            hass,
            "hello",
            None,
            Context(),
            agent_id=entity_id,
        )
        await hass.async_block_till_done()

    speech = result.response.speech["plain"]["speech"]
    assert "configured maximum of 24 iterations" in speech
    assert events[-1]["error_type"] == "UsageLimitExceeded"
    assert events[-1]["partial_response"] is True


async def test_streaming_backfills_final_text_after_thinking_only_events(
    hass: HomeAssistant,
    mock_chat_model_for_profile: object,
) -> None:
    """Test final result text is used when live events only stream thinking."""
    del mock_chat_model_for_profile
    entry = loaded_conversation_entry()
    entry.add_to_hass(hass)
    result = RunResultWithMessages(
        "Hello. How can I help you?",
        [
            ModelResponse(
                parts=[
                    ThinkingPart(content='"Hello! How can I help'),
                    TextPart(content="Hello. How can I help you?"),
                ]
            )
        ],
    )

    async def stream() -> AsyncIterator[object]:
        yield PartStartEvent(
            index=0, part=ThinkingPart(content='"Hello! How can I help')
        )
        yield AgentRunResultEvent(cast(Any, result))

    agent = CallbackStreamAgent(stream_factory=stream)

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    entity_id = first_non_default_conversation_entity_id(hass)
    with patch("custom_components.pydantic_ai_agent.entity.Agent", return_value=agent):
        result = await conversation.async_converse(
            hass,
            "hello",
            None,
            Context(),
            agent_id=entity_id,
        )

    assert result.response.speech["plain"]["speech"] == "Hello. How can I help you?"
    assert result.conversation_id is not None
    assistant_messages = [
        content
        for content in hass.data[DATA_CHAT_LOGS][result.conversation_id].content
        if isinstance(content, AssistantContent)
    ]
    assert len(assistant_messages) == 1
    assert assistant_messages[-1].content == "Hello. How can I help you?"
    assert assistant_messages[-1].thinking_content == '"Hello! How can I help'


async def test_streaming_tool_retry_exhaustion_reports_tool_context(
    hass: HomeAssistant,
    mock_chat_model_for_profile: object,
) -> None:
    """Test streamed exhausted tool retries surface actionable tool context."""
    del mock_chat_model_for_profile
    entry = loaded_conversation_entry(extra_data={CONF_TOOL_RETRIES: 2})
    entry.add_to_hass(hass)

    async def stream() -> AsyncIterator[object]:
        yield PartStartEvent(index=0, part=TextPart(content="partial"))
        yield FunctionToolResultEvent(
            RetryPromptPart(
                tool_name="turn_on",
                tool_call_id="tool-1",
                content='Home Assistant tool "turn_on" failed: device lookup failed',
            )
        )
        failure = UnexpectedModelBehavior("tool retries exhausted")
        failure.__cause__ = ModelRetry(
            'Home Assistant tool "turn_on" failed after retries were exhausted.'
        )
        raise failure

    agent = CallbackStreamAgent(stream_factory=stream)
    events: list[dict[str, object]] = []
    hass.bus.async_listen(
        f"{DOMAIN}_{EVENT_AGENT_RUN_FAILED}",
        lambda event: events.append(dict(event.data)),
    )

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    entity_id = first_non_default_conversation_entity_id(hass)
    with patch(
        "custom_components.pydantic_ai_agent.entity.Agent",
        return_value=agent,
    ) as agent_class:
        result = await conversation.async_converse(
            hass,
            "hello",
            None,
            Context(),
            agent_id=entity_id,
        )
        await hass.async_block_till_done()

    assert agent_class.call_args.kwargs["tool_retries"] == 2
    speech = result.response.speech["plain"]["speech"]
    assert "turn_on" in speech
    assert "unexpected response" not in speech.lower()
    assert events[-1]["error_type"] == "UnexpectedModelBehavior"
    assert events[-1]["partial_response"] is True
    assert events[-1]["tool_name"] == "turn_on"
    assert events[-1]["tool_call_id"] == "tool-1"


async def test_streaming_backfills_missing_final_text_suffix(
    hass: HomeAssistant,
    mock_chat_model_for_profile: object,
) -> None:
    """Test final speech is complete when live text misses the final suffix."""
    del mock_chat_model_for_profile
    entry = loaded_conversation_entry()
    entry.add_to_hass(hass)
    result = RunResultWithMessages(
        "Hello. How can I help you?",
        [ModelResponse(parts=[TextPart(content="Hello. How can I help you?")])],
    )

    async def stream() -> AsyncIterator[object]:
        yield PartStartEvent(index=0, part=TextPart(content="Hello. How can I help"))
        yield AgentRunResultEvent(cast(Any, result))

    agent = CallbackStreamAgent(stream_factory=stream)

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    entity_id = first_non_default_conversation_entity_id(hass)
    with patch("custom_components.pydantic_ai_agent.entity.Agent", return_value=agent):
        result = await conversation.async_converse(
            hass,
            "hello",
            None,
            Context(),
            agent_id=entity_id,
        )

    assert result.response.speech["plain"]["speech"] == "Hello. How can I help you?"
    assert result.conversation_id is not None
    assistant_messages = [
        content
        for content in hass.data[DATA_CHAT_LOGS][result.conversation_id].content
        if isinstance(content, AssistantContent)
    ]
    assert len(assistant_messages) == 1
    assert assistant_messages[0].content == "Hello. How can I help you?"


async def test_streaming_does_not_duplicate_already_streamed_final_text(
    hass: HomeAssistant,
    mock_chat_model_for_profile: object,
) -> None:
    """Test final result text is not replayed when live events streamed it."""
    del mock_chat_model_for_profile
    entry = loaded_conversation_entry()
    entry.add_to_hass(hass)
    agent = _Agent(stream_text="Hello. How can I help you?")

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    entity_id = first_non_default_conversation_entity_id(hass)
    with patch("custom_components.pydantic_ai_agent.entity.Agent", return_value=agent):
        result = await conversation.async_converse(
            hass,
            "hello",
            None,
            Context(),
            agent_id=entity_id,
        )

    assert result.response.speech["plain"]["speech"] == "Hello. How can I help you?"
    assert result.conversation_id is not None
    assistant_messages = [
        content
        for content in hass.data[DATA_CHAT_LOGS][result.conversation_id].content
        if isinstance(content, AssistantContent)
    ]
    assert len(assistant_messages) == 1
