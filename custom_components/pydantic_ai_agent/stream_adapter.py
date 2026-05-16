"""Pydantic AI stream event adapters for Home Assistant ChatLog."""

from collections.abc import AsyncGenerator, AsyncIterable, Mapping
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
    *,
    output_tool_names: set[str] | None = None,
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Convert Pydantic AI stream events into Home Assistant ChatLog deltas."""
    output_tool_names = output_tool_names or set()
    # HA ChatLog streams start with the assistant role before content deltas.
    yield {"role": "assistant"}

    async for event in stream:
        if isinstance(event, PartStartEvent):
            if isinstance(event.part, TextPart) and event.part.content:
                yield {"content": event.part.content}
            elif isinstance(event.part, ThinkingPart) and event.part.content:
                yield {"thinking_content": event.part.content}
            elif isinstance(event.part, ToolCallPart):
                # HA accepts complete tool inputs only, so defer tool-call
                # emission until Pydantic AI closes the part.
                continue
            elif not isinstance(event.part, TextPart | ThinkingPart | ToolCallPart):
                raise HomeAssistantError(
                    "Provider returned unsupported response content"
                )
        elif isinstance(event, PartDeltaEvent):
            if isinstance(event.delta, TextPartDelta):
                yield {"content": event.delta.content_delta}
            elif isinstance(event.delta, ThinkingPartDelta):
                if event.delta.content_delta:
                    yield {"thinking_content": event.delta.content_delta}
            elif isinstance(event.delta, ToolCallPartDelta):
                # Tool-call deltas can be partial JSON; wait for PartEndEvent.
                continue
        elif isinstance(event, PartEndEvent):
            if isinstance(event.part, ToolCallPart):
                if event.part.tool_name in output_tool_names:
                    yield {"content": _tool_args_as_json(event.part.args)}
                    continue
                yield {"tool_calls": [_tool_input_from_part(event.part)]}
            elif not isinstance(event.part, TextPart | ThinkingPart):
                raise HomeAssistantError(
                    "Provider returned unsupported response content"
                )
        elif isinstance(event, FinalResultEvent):
            continue


def _tool_input_from_part(part: ToolCallPart) -> llm.ToolInput:
    """Convert a completed Pydantic AI tool call into Home Assistant input."""
    return llm.ToolInput(
        id=part.tool_call_id,
        tool_name=part.tool_name,
        tool_args=_tool_args_as_dict(part.args),
    )


def _tool_args_as_json(args: object) -> str:
    """Return output tool arguments as JSON assistant content."""
    parsed = _tool_args_as_dict(args)
    return json.dumps(parsed, separators=(",", ":"))


def _tool_args_as_dict(args: object) -> dict[str, Any]:
    """Return tool call arguments as a dictionary."""
    if not args:
        return {}
    if isinstance(args, Mapping):
        return dict(args)
    if not isinstance(args, str):
        raise HomeAssistantError("Provider returned non-object tool call arguments")
    try:
        # Pydantic AI can expose provider tool arguments as JSON text depending
        # on how the model reports function-call arguments.
        parsed = json.loads(args)
    except json.JSONDecodeError as err:
        raise HomeAssistantError(
            "Provider returned malformed tool call arguments"
        ) from err
    if not isinstance(parsed, dict):
        raise HomeAssistantError("Provider returned non-object tool call arguments")
    return parsed
