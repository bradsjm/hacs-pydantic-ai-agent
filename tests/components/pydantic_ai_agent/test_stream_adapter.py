"""Test Pydantic AI stream conversion."""

from collections.abc import AsyncGenerator

import pytest
from pydantic_ai import (
    FinalResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
)
from pydantic_ai.messages import ModelResponseStreamEvent

from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

from custom_components.pydantic_ai_agent.stream_adapter import (
    model_stream_to_chat_deltas,
)


async def _events(
    *events: ModelResponseStreamEvent,
) -> AsyncGenerator[ModelResponseStreamEvent]:
    """Yield stream events."""
    for event in events:
        yield event


async def test_text_stream_converts_to_content_deltas() -> None:
    """Test text stream events convert to assistant content deltas."""
    deltas = [
        delta
        async for delta in model_stream_to_chat_deltas(
            _events(
                PartStartEvent(index=0, part=TextPart(content="Hello")),
                FinalResultEvent(tool_name=None, tool_call_id=None),
                PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=" world")),
                PartEndEvent(index=0, part=TextPart(content="Hello world")),
            )
        )
    ]

    assert deltas == [
        {"role": "assistant"},
        {"content": "Hello"},
        {"content": " world"},
    ]


async def test_thinking_stream_converts_to_thinking_deltas() -> None:
    """Test thinking stream events convert to thinking content deltas."""
    deltas = [
        delta
        async for delta in model_stream_to_chat_deltas(
            _events(
                PartStartEvent(index=0, part=ThinkingPart(content="I")),
                PartDeltaEvent(
                    index=0,
                    delta=ThinkingPartDelta(content_delta=" think"),
                ),
                PartEndEvent(index=0, part=ThinkingPart(content="I think")),
            )
        )
    ]

    assert deltas == [
        {"role": "assistant"},
        {"thinking_content": "I"},
        {"thinking_content": " think"},
    ]


async def test_tool_call_stream_converts_json_args() -> None:
    """Test completed tool calls convert to Home Assistant tool inputs."""
    deltas = [
        delta
        async for delta in model_stream_to_chat_deltas(
            _events(
                PartStartEvent(
                    index=0,
                    part=ToolCallPart(
                        tool_name="HassTurnOn",
                        args=None,
                        tool_call_id="tool-1",
                    ),
                ),
                PartEndEvent(
                    index=0,
                    part=ToolCallPart(
                        tool_name="HassTurnOn",
                        args='{"name":"Kitchen"}',
                        tool_call_id="tool-1",
                    ),
                ),
            )
        )
    ]

    assert deltas == [
        {"role": "assistant"},
        {
            "tool_calls": [
                llm.ToolInput(
                    tool_name="HassTurnOn",
                    tool_args={"name": "Kitchen"},
                    id="tool-1",
                )
            ]
        },
    ]


async def test_output_tool_stream_converts_to_content() -> None:
    """Test completed output tool calls convert to assistant content."""
    deltas = [
        delta
        async for delta in model_stream_to_chat_deltas(
            _events(
                PartEndEvent(
                    index=0,
                    part=ToolCallPart(
                        tool_name="structured_task",
                        args={"name": "Kitchen"},
                        tool_call_id="tool-1",
                    ),
                )
            ),
            output_tool_names={"structured_task"},
        )
    ]

    assert deltas == [
        {"role": "assistant"},
        {"content": '{"name":"Kitchen"}'},
    ]


async def test_malformed_tool_call_json_raises() -> None:
    """Test malformed tool call JSON raises a Home Assistant error."""
    with pytest.raises(HomeAssistantError, match="malformed tool call arguments"):
        _ = [
            delta
            async for delta in model_stream_to_chat_deltas(
                _events(
                    PartEndEvent(
                        index=0,
                        part=ToolCallPart(
                            tool_name="HassTurnOn",
                            args="{",
                            tool_call_id="tool-1",
                        ),
                    )
                )
            )
        ]
