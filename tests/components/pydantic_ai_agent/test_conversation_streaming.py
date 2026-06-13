"""Test Pydantic AI Agent conversation streaming behavior."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast
from unittest.mock import AsyncMock, patch

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
from pytest_homeassistant_custom_component.common import MockConfigEntry
from tests.components.pydantic_ai_agent.support.builders import (
    conversation_subentry_data,
    provider_runtime_data,
    provider_subentry_data,
    workspace_entry,
    workspace_runtime_data,
)
from tests.components.pydantic_ai_agent.support.pydantic_ai import (
    Agent as _Agent,
)
from tests.components.pydantic_ai_agent.support.pydantic_ai import (
    EventStream,
    Usage,
)

_PROVIDER_SUBENTRY_ID = "provider-1"
_MODEL_PROFILE_ID = "model-profile-1"
_MODEL_PROFILE_REF = f"{_PROVIDER_SUBENTRY_ID}:{_MODEL_PROFILE_ID}"


def _entry(
    extra_data: dict[str, object] | None = None,
) -> MockConfigEntry:
    """Return a config entry with one conversation subentry."""
    entry = workspace_entry(
        (
            conversation_subentry_data(
                _MODEL_PROFILE_REF,
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


class _ResultWithMessages:
    """Minimal Agent result with explicit final messages."""

    def __init__(self, output: str, messages: list[ModelResponse]) -> None:
        """Initialize the result."""
        self.output = output
        self.usage = Usage()
        self._messages = messages

    def new_messages(self) -> list[ModelResponse]:
        """Return final Agent messages."""
        return self._messages


async def test_streaming_iteration_failure_updates_chat_and_sensors(
    hass: HomeAssistant,
) -> None:
    """Test streaming usage-limit failures stay actionable after partial output."""
    entry = _entry(extra_data={CONF_MAX_ITERATIONS: 24})
    entry.add_to_hass(hass)

    class FailingAfterPartialAgent(_Agent):
        @asynccontextmanager
        async def run_stream_events(
            self, *_args: object, **kwargs: object
        ) -> AsyncIterator[EventStream]:
            self.run_stream_events_calls += 1
            self.run_kwargs = kwargs

            async def stream() -> AsyncIterator[object]:
                yield PartStartEvent(index=0, part=TextPart(content="partial"))
                raise UsageLimitExceeded(
                    "The next request would exceed the request_limit of 24"
                )

            yield cast(EventStream, stream())

    agent = FailingAfterPartialAgent()
    events: list[dict[str, object]] = []
    hass.bus.async_listen(
        f"{DOMAIN}_{EVENT_AGENT_RUN_FAILED}",
        lambda event: events.append(dict(event.data)),
    )

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
        await hass.async_block_till_done()

    speech = result.response.speech["plain"]["speech"]
    assert "configured maximum of 24 iterations" in speech
    assert events[-1]["error_type"] == "UsageLimitExceeded"
    assert events[-1]["partial_response"] is True


async def test_streaming_backfills_final_text_after_thinking_only_events(
    hass: HomeAssistant,
) -> None:
    """Test final result text is used when live events only stream thinking."""
    entry = _entry()
    entry.add_to_hass(hass)

    class ThinkingOnlyAgent(_Agent):
        @asynccontextmanager
        async def run_stream_events(
            self, *_args: object, **kwargs: object
        ) -> AsyncIterator[EventStream]:
            self.run_stream_events_calls += 1
            self.run_kwargs = kwargs
            result = _ResultWithMessages(
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
                    index=0,
                    part=ThinkingPart(content='"Hello! How can I help'),
                )
                yield AgentRunResultEvent(cast(Any, result))

            yield cast(EventStream, stream())

    agent = ThinkingOnlyAgent()

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
) -> None:
    """Test streamed exhausted tool retries surface actionable tool context."""
    entry = _entry(extra_data={CONF_TOOL_RETRIES: 2})
    entry.add_to_hass(hass)

    class FailingAfterRetryPromptAgent(_Agent):
        @asynccontextmanager
        async def run_stream_events(
            self, *_args: object, **kwargs: object
        ) -> AsyncIterator[EventStream]:
            self.run_stream_events_calls += 1
            self.run_kwargs = kwargs

            async def stream() -> AsyncIterator[object]:
                yield PartStartEvent(index=0, part=TextPart(content="partial"))
                yield FunctionToolResultEvent(
                    RetryPromptPart(
                        tool_name="turn_on",
                        tool_call_id="tool-1",
                        content=(
                            'Home Assistant tool "turn_on" failed: device lookup failed'
                        ),
                    )
                )
                failure = UnexpectedModelBehavior("tool retries exhausted")
                failure.__cause__ = ModelRetry(
                    'Home Assistant tool "turn_on" failed after retries were exhausted.'
                )
                raise failure

            yield cast(EventStream, stream())

    agent = FailingAfterRetryPromptAgent()
    events: list[dict[str, object]] = []
    hass.bus.async_listen(
        f"{DOMAIN}_{EVENT_AGENT_RUN_FAILED}",
        lambda event: events.append(dict(event.data)),
    )

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
        ) as agent_class,
    ):
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
) -> None:
    """Test final speech is complete when live text misses the final suffix."""
    entry = _entry()
    entry.add_to_hass(hass)

    class PartialTextAgent(_Agent):
        @asynccontextmanager
        async def run_stream_events(
            self, *_args: object, **kwargs: object
        ) -> AsyncIterator[EventStream]:
            self.run_stream_events_calls += 1
            self.run_kwargs = kwargs
            result = _ResultWithMessages(
                "Hello. How can I help you?",
                [ModelResponse(parts=[TextPart(content="Hello. How can I help you?")])],
            )

            async def stream() -> AsyncIterator[object]:
                yield PartStartEvent(
                    index=0,
                    part=TextPart(content="Hello. How can I help"),
                )
                yield AgentRunResultEvent(cast(Any, result))

            yield cast(EventStream, stream())

    agent = PartialTextAgent()

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
) -> None:
    """Test final result text is not replayed when live events streamed it."""
    entry = _entry()
    entry.add_to_hass(hass)
    agent = _Agent(stream_text="Hello. How can I help you?")

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

    assert result.response.speech["plain"]["speech"] == "Hello. How can I help you?"
    assert result.conversation_id is not None
    assistant_messages = [
        content
        for content in hass.data[DATA_CHAT_LOGS][result.conversation_id].content
        if isinstance(content, AssistantContent)
    ]
    assert len(assistant_messages) == 1
