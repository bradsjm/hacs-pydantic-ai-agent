"""Test streamed tool-resume separator preservation in conversation flows."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from custom_components.pydantic_ai_agent.conversation import _merged_assistant_speech
from homeassistant.components import conversation
from homeassistant.components.conversation.chat_log import (
    DATA_CHAT_LOGS,
    AssistantContent,
)
from homeassistant.components.conversation.const import DATA_COMPONENT
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers.chat_session import async_get_chat_session
from pydantic_ai import (
    AgentRunResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelResponse,
    PartStartEvent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)
from tests.components.pydantic_ai_agent.support.pydantic_ai import Agent as _Agent
from tests.components.pydantic_ai_agent.support.pydantic_ai import EventStream
from tests.components.pydantic_ai_agent.test_conversation_streaming import (
    _entry,
    _ResultWithMessages,
)


def test_merged_assistant_speech_prefers_full_resumed_answer() -> None:
    """Test a resumed full answer replaces an earlier prefix fragment."""
    assert (
        _merged_assistant_speech(
            [
                AssistantContent(agent_id="agent", content="Turning on "),
                AssistantContent(
                    agent_id="agent",
                    content="\n\nTurning on the kitchen light. Done!",
                ),
            ]
        )
        == "Turning on the kitchen light. Done!"
    )


async def test_streaming_backfill_preserves_tool_resume_separator(
    hass: HomeAssistant,
) -> None:
    """Test final reconciliation keeps the streamed separator after tool results."""
    entry = _entry()
    entry.add_to_hass(hass)

    class ToolResumeAgent(_Agent):
        @asynccontextmanager
        async def run_stream_events(
            self, *_args: object, **kwargs: object
        ) -> AsyncIterator[EventStream]:
            self.run_stream_events_calls += 1
            self.run_kwargs = kwargs
            result = _ResultWithMessages(
                "turning \n\ndone!",
                [ModelResponse(parts=[TextPart(content="done!")])],
            )

            async def stream() -> AsyncIterator[object]:
                yield PartStartEvent(index=0, part=TextPart(content="turning "))
                yield FunctionToolCallEvent(
                    ToolCallPart(
                        tool_name="HassTurnOn",
                        args={"name": "Kitchen"},
                        tool_call_id="tool-1",
                    )
                )
                yield FunctionToolResultEvent(
                    ToolReturnPart(
                        tool_name="HassTurnOn",
                        content={"success": True},
                        tool_call_id="tool-1",
                    )
                )
                yield PartStartEvent(index=0, part=TextPart(content="done"))
                yield AgentRunResultEvent(cast(Any, result))

            yield cast(EventStream, stream())

    agent = ToolResumeAgent()

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

    assert result.response.speech["plain"]["speech"] == "turning \n\ndone!"
    assert result.conversation_id is not None
    assistant_messages = [
        content
        for content in hass.data[DATA_CHAT_LOGS][result.conversation_id].content
        if isinstance(content, AssistantContent)
    ]
    assert [content.content for content in assistant_messages] == [
        "turning ",
        "\n\ndone!",
    ]


async def test_streaming_backfill_appends_tool_resume_separator_after_tool_result_only(
    hass: HomeAssistant,
) -> None:
    """Test backfill appends the separator after a trailing tool result."""
    entry = _entry()
    entry.add_to_hass(hass)

    class ToolResultOnlyBackfillAgent(_Agent):
        @asynccontextmanager
        async def run_stream_events(
            self, *_args: object, **kwargs: object
        ) -> AsyncIterator[EventStream]:
            self.run_stream_events_calls += 1
            self.run_kwargs = kwargs
            result = _ResultWithMessages(
                "turning \n\ndone!",
                [ModelResponse(parts=[TextPart(content="done!")])],
            )

            async def stream() -> AsyncIterator[object]:
                yield PartStartEvent(index=0, part=TextPart(content="turning "))
                yield FunctionToolCallEvent(
                    ToolCallPart(
                        tool_name="HassTurnOn",
                        args={"name": "Kitchen"},
                        tool_call_id="tool-1",
                    )
                )
                yield FunctionToolResultEvent(
                    ToolReturnPart(
                        tool_name="HassTurnOn",
                        content={"success": True},
                        tool_call_id="tool-1",
                    )
                )
                yield AgentRunResultEvent(cast(Any, result))

            yield cast(EventStream, stream())

    agent = ToolResultOnlyBackfillAgent()

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

    assert result.response.speech["plain"]["speech"] == "turning \n\ndone!"
    assert result.conversation_id is not None
    assistant_messages = [
        content
        for content in hass.data[DATA_CHAT_LOGS][result.conversation_id].content
        if isinstance(content, AssistantContent)
    ]
    assert [content.content for content in assistant_messages] == [
        "turning ",
        "\n\ndone!",
    ]


async def test_streaming_backfill_preserves_separator_after_resumed_thinking(
    hass: HomeAssistant,
) -> None:
    """Test post-tool thinking-only resume still backfills text with the separator."""
    entry = _entry()
    entry.add_to_hass(hass)

    class ToolResumeThinkingOnlyAgent(_Agent):
        @asynccontextmanager
        async def run_stream_events(
            self, *_args: object, **kwargs: object
        ) -> AsyncIterator[EventStream]:
            self.run_stream_events_calls += 1
            self.run_kwargs = kwargs
            result = _ResultWithMessages(
                "turning \n\ndone!",
                [
                    ModelResponse(
                        parts=[
                            ThinkingPart(content="thinking about confirmation"),
                            TextPart(content="done!"),
                        ]
                    )
                ],
            )

            async def stream() -> AsyncIterator[object]:
                yield PartStartEvent(index=0, part=TextPart(content="turning "))
                yield FunctionToolCallEvent(
                    ToolCallPart(
                        tool_name="HassTurnOn",
                        args={"name": "Kitchen"},
                        tool_call_id="tool-1",
                    )
                )
                yield FunctionToolResultEvent(
                    ToolReturnPart(
                        tool_name="HassTurnOn",
                        content={"success": True},
                        tool_call_id="tool-1",
                    )
                )
                yield PartStartEvent(
                    index=0,
                    part=ThinkingPart(content="thinking about confirmation"),
                )
                yield AgentRunResultEvent(cast(Any, result))

            yield cast(EventStream, stream())

    agent = ToolResumeThinkingOnlyAgent()

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

    streamed_deltas: list[dict[str, object]] = []

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
        entity = cast(Any, hass.data[DATA_COMPONENT].get_entity(entity_id))
        user_input = conversation.ConversationInput(
            text="hello",
            context=Context(),
            conversation_id=None,
            device_id=None,
            satellite_id=None,
            language=hass.config.language,
            agent_id=entity_id,
        )
        with (
            async_get_chat_session(hass, None) as session,
            conversation.async_get_chat_log(
                hass,
                session,
                user_input,
                chat_log_delta_listener=lambda _chat_log, delta: streamed_deltas.append(
                    dict(delta)
                ),
            ) as chat_log,
        ):
            result = await entity._async_handle_message(user_input, chat_log)

    assert result.response.speech["plain"]["speech"] == "turning \n\ndone!"
    assert streamed_deltas[-1] == {"content": "\n\ndone!"}
    assert result.conversation_id is not None
    assistant_messages = [
        content
        for content in hass.data[DATA_CHAT_LOGS][result.conversation_id].content
        if isinstance(content, AssistantContent)
    ]
    assert [content.content for content in assistant_messages] == [
        "turning ",
        "\n\ndone!",
    ]
    assert assistant_messages[-1].thinking_content == "thinking about confirmation"
