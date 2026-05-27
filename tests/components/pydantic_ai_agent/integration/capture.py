"""Capture helpers for provider integration tests."""

from pydantic_ai import (
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
)


def tool_part_names(messages: list[object]) -> list[str]:
    """Return tool call and return names from Pydantic AI messages."""
    names: list[str] = []
    for message in messages:
        for part in getattr(message, "parts", ()):
            if isinstance(part, ToolCallPart | ToolReturnPart):
                names.append(part.tool_name)
    return names


def append_text_event(text_parts: list[str], event: object) -> None:
    """Append display text from a Pydantic AI stream event."""
    if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
        text_parts.append(event.part.content)
    elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
        text_parts.append(event.delta.content_delta)
    elif isinstance(event, PartEndEvent) and isinstance(event.part, TextPart):
        text_parts.append(event.part.content)
