"""Pydantic AI stream event adapters for Home Assistant ChatLog."""

from collections.abc import AsyncGenerator, AsyncIterable
import json
from typing import Any

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
    ToolCallPartDelta,
)
from pydantic_ai.messages import ModelResponseStreamEvent

from homeassistant.components import conversation
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm


async def model_stream_to_chat_deltas(
    stream: AsyncIterable[ModelResponseStreamEvent],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Convert a Pydantic AI model stream into Home Assistant deltas."""
    yield {"role": "assistant"}

    async for event in stream:
        if isinstance(event, PartStartEvent):
            if isinstance(event.part, TextPart) and event.part.content:
                yield {"content": event.part.content}
            elif isinstance(event.part, ThinkingPart) and event.part.content:
                yield {"thinking_content": event.part.content}
            elif isinstance(event.part, ToolCallPart):
                continue
            elif not isinstance(event.part, TextPart | ThinkingPart | ToolCallPart):
                raise HomeAssistantError("Provider returned unsupported response content")
        elif isinstance(event, PartDeltaEvent):
            if isinstance(event.delta, TextPartDelta):
                yield {"content": event.delta.content_delta}
            elif isinstance(event.delta, ThinkingPartDelta):
                if event.delta.content_delta:
                    yield {"thinking_content": event.delta.content_delta}
            elif isinstance(event.delta, ToolCallPartDelta):
                continue
        elif isinstance(event, PartEndEvent):
            if isinstance(event.part, ToolCallPart):
                yield {"tool_calls": [_tool_input_from_part(event.part)]}
            elif not isinstance(event.part, TextPart | ThinkingPart):
                raise HomeAssistantError("Provider returned unsupported response content")
        elif isinstance(event, FinalResultEvent):
            continue


def _tool_input_from_part(part: ToolCallPart) -> llm.ToolInput:
    """Convert a Pydantic AI tool call part into Home Assistant tool input."""
    return llm.ToolInput(
        id=part.tool_call_id,
        tool_name=part.tool_name,
        tool_args=_tool_args_as_dict(part.args),
    )


def _tool_args_as_dict(args: str | dict[str, Any] | None) -> dict[str, Any]:
    """Return tool call arguments as a dictionary."""
    if not args:
        return {}
    if isinstance(args, dict):
        return args
    try:
        parsed = json.loads(args)
    except json.JSONDecodeError as err:
        raise HomeAssistantError(
            "Provider returned malformed tool call arguments"
        ) from err
    if not isinstance(parsed, dict):
        raise HomeAssistantError("Provider returned non-object tool call arguments")
    return parsed
