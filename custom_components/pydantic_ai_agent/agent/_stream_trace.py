"""Bounded stream trace recorder and summary helpers for Pydantic AI streams."""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from homeassistant.components import conversation
from pydantic_ai import AgentRunResultEvent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    OutputToolCallEvent,
    OutputToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)

_TRACE_MAX_ITEMS = 200
_TRACE_HEAD_ITEMS = 100
_TRACE_TAIL_ITEMS = _TRACE_MAX_ITEMS - _TRACE_HEAD_ITEMS
_TRACE_PREVIEW_CHARS = 4096


@dataclass
class _StreamTraceRecorder:
    """Collect a bounded, JSON-safe summary of one Pydantic AI stream."""

    run_recorder: Any | None = None
    events_total: int = 0
    chat_deltas_total: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    chat_deltas: list[dict[str, Any]] = field(default_factory=list)
    events_tail: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=_TRACE_TAIL_ITEMS)
    )
    chat_deltas_tail: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=_TRACE_TAIL_ITEMS)
    )

    def record_event(self, event: object) -> None:
        """Record one Pydantic AI stream event summary."""
        self.events_total += 1
        summary = {
            "order": self.events_total,
            **_stream_event_summary(event, True),
        }
        if len(self.events) < _TRACE_HEAD_ITEMS:
            self.events.append(summary)
        else:
            self.events_tail.append(summary)
        if self.run_recorder is not None:
            diagnostics_summary = _stream_event_summary(event, True)
            self.run_recorder.record(
                phase="llm_stream",
                source="provider",
                event=diagnostics_summary["event_type"],
                data={"summary": diagnostics_summary},
            )

    def record_chat_delta(self, delta: Mapping[str, Any]) -> None:
        """Record one Home Assistant ChatLog delta summary."""
        self.chat_deltas_total += 1
        summary = {
            "order": self.chat_deltas_total,
            **_chat_delta_summary(delta, True),
        }
        if len(self.chat_deltas) < _TRACE_HEAD_ITEMS:
            self.chat_deltas.append(summary)
        else:
            self.chat_deltas_tail.append(summary)
        if self.run_recorder is not None:
            diagnostics_summary = _chat_delta_summary(delta, True)
            self.run_recorder.record(
                phase="chat_delta",
                source="home_assistant",
                event="delta_emitted",
                data={"summary": diagnostics_summary},
            )

    def payload(
        self,
        *,
        final_messages: list[ModelMessage],
        backfill: Mapping[str, Any],
        final_chat_content: conversation.Content | None,
    ) -> dict[str, Any]:
        """Return the trace payload for HA conversation traces and diagnostics."""
        events_tail = list(self.events_tail)
        chat_deltas_tail = list(self.chat_deltas_tail)
        events_truncated = self.events_total > len(self.events) + len(events_tail)
        chat_deltas_truncated = self.chat_deltas_total > len(self.chat_deltas) + len(
            chat_deltas_tail
        )
        return {
            "schema_version": 1,
            "limits": {
                "max_items": _TRACE_MAX_ITEMS,
                "head_items": _TRACE_HEAD_ITEMS,
                "tail_items": _TRACE_TAIL_ITEMS,
                "preview_chars": _TRACE_PREVIEW_CHARS,
            },
            "events_total": self.events_total,
            "events_truncated": events_truncated,
            "events_omitted_middle_count": max(
                self.events_total - len(self.events) - len(events_tail),
                0,
            ),
            "events": self.events + ([] if events_truncated else events_tail),
            "events_tail": events_tail if events_truncated else [],
            "chat_deltas_total": self.chat_deltas_total,
            "chat_deltas_truncated": chat_deltas_truncated,
            "chat_deltas_omitted_middle_count": max(
                self.chat_deltas_total - len(self.chat_deltas) - len(chat_deltas_tail),
                0,
            ),
            "chat_deltas": self.chat_deltas
            + ([] if chat_deltas_truncated else chat_deltas_tail),
            "chat_deltas_tail": chat_deltas_tail if chat_deltas_truncated else [],
            "final_new_messages": _messages_summary(final_messages, True),
            "backfill": dict(backfill),
            "final_chat_content": _chat_content_summary(final_chat_content, True),
        }


def _stream_event_summary(event: object, include_preview: bool) -> dict[str, Any]:
    """Return a safe summary of a Pydantic AI stream event."""
    summary: dict[str, Any] = {"event_type": type(event).__name__}
    if isinstance(event, PartStartEvent | PartEndEvent):
        summary["part_index"] = event.index
        summary["part"] = _part_summary(event.part, include_preview)
        if isinstance(event, PartEndEvent):
            summary["complete_snapshot"] = True
            summary["next_part_kind"] = event.next_part_kind
        return summary
    if isinstance(event, PartDeltaEvent):
        summary["part_index"] = event.index
        summary["delta"] = _part_delta_summary(event.delta, include_preview)
        return summary
    if isinstance(event, FunctionToolCallEvent | OutputToolCallEvent):
        summary["part"] = _part_summary(event.part, include_preview)
        return summary
    if isinstance(event, FunctionToolResultEvent | OutputToolResultEvent):
        summary["part"] = _part_summary(event.part, include_preview)
        return summary
    if isinstance(event, AgentRunResultEvent):
        output = getattr(event.result, "output", None)
        result_summary: dict[str, Any] = {"output_type": type(output).__name__}
        if isinstance(output, str):
            result_summary["output_chars"] = len(output)
            _add_preview(result_summary, "output", output, include_preview)
        summary["result"] = result_summary
        return summary
    return summary


def _part_summary(part: object, include_preview: bool) -> dict[str, Any]:
    """Return a safe summary of one Pydantic AI message part."""
    summary: dict[str, Any] = {
        "type": type(part).__name__,
        "part_kind": getattr(part, "part_kind", None),
    }
    if isinstance(part, TextPart):
        _add_text_summary(summary, "content", part.content, include_preview)
    elif isinstance(part, ThinkingPart):
        _add_text_summary(summary, "content", part.content, include_preview)
        summary["signature_present"] = bool(part.signature)
    elif isinstance(part, ToolCallPart):
        summary["tool_name"] = part.tool_name
        summary["tool_call_id_present"] = bool(part.tool_call_id)
    elif isinstance(part, ToolReturnPart | RetryPromptPart):
        summary["tool_name"] = part.tool_name
        summary["tool_call_id_present"] = bool(part.tool_call_id)
        summary["content_type"] = type(part.content).__name__
        if isinstance(part.content, str):
            summary["content_chars"] = len(part.content)
    return summary


def _part_delta_summary(delta: object, include_preview: bool) -> dict[str, Any]:
    """Return a safe summary of one Pydantic AI part delta."""
    summary: dict[str, Any] = {
        "type": type(delta).__name__,
        "part_delta_kind": getattr(delta, "part_delta_kind", None),
    }
    if isinstance(delta, TextPartDelta):
        _add_text_summary(
            summary, "content_delta", delta.content_delta, include_preview
        )
    elif isinstance(delta, ThinkingPartDelta):
        _add_text_summary(
            summary, "content_delta", delta.content_delta, include_preview
        )
        summary["signature_delta_present"] = bool(delta.signature_delta)
    return summary


def _chat_delta_summary(
    delta: Mapping[str, Any], include_preview: bool
) -> dict[str, Any]:
    """Return a safe summary of a ChatLog delta."""
    summary: dict[str, Any] = {"keys": sorted(delta)}
    if role := delta.get("role"):
        summary["role"] = role
    if isinstance(content := delta.get("content"), str):
        _add_text_summary(summary, "content", content, include_preview)
    if isinstance(thinking := delta.get("thinking_content"), str):
        _add_text_summary(summary, "thinking_content", thinking, include_preview)
    if tool_calls := delta.get("tool_calls"):
        summary["tool_calls"] = [
            {
                "tool_name": getattr(tool_call, "tool_name", None),
                "tool_call_id_present": bool(getattr(tool_call, "id", None)),
                "external": getattr(tool_call, "external", None),
            }
            for tool_call in tool_calls
        ]
    if "tool_result" in delta:
        tool_result = delta["tool_result"]
        tool_result_summary: dict[str, Any] = {"type": type(tool_result).__name__}
        if isinstance(tool_result, str):
            tool_result_summary["chars"] = len(tool_result)
        summary["tool_result"] = tool_result_summary
    return summary


def _messages_summary(
    messages: list[ModelMessage], include_preview: bool
) -> list[dict[str, Any]]:
    """Return a safe summary of final Agent messages."""
    summaries: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        summary: dict[str, Any] = {
            "index": index,
            "type": type(message).__name__,
        }
        if isinstance(message, (ModelResponse, ModelRequest)):
            summary["parts"] = [
                _part_summary(part, include_preview) for part in message.parts
            ]
        summaries.append(summary)
    return summaries


def _chat_content_summary(
    content: conversation.Content | None, include_preview: bool
) -> dict[str, Any] | None:
    """Return a safe summary of final ChatLog content."""
    if content is None:
        return None
    summary: dict[str, Any] = {
        "type": type(content).__name__,
        "role": content.role,
    }
    if isinstance(content, conversation.AssistantContent):
        if content.content is not None:
            _add_text_summary(summary, "content", content.content, include_preview)
        if content.thinking_content is not None:
            _add_text_summary(
                summary,
                "thinking_content",
                content.thinking_content,
                include_preview,
            )
        summary["tool_call_count"] = len(content.tool_calls or [])
    return summary


def _add_text_summary(
    summary: dict[str, Any], field_name: str, text: str | None, include_preview: bool
) -> None:
    """Add text length and optional preview fields to a summary."""
    text = text or ""
    summary[f"{field_name}_chars"] = len(text)
    _add_preview(summary, field_name, text, include_preview)


def _add_preview(
    summary: dict[str, Any], field_name: str, text: str, include_preview: bool
) -> None:
    """Add a bounded text preview when requested."""
    if include_preview and text:
        summary[f"{field_name}_preview"] = text[:_TRACE_PREVIEW_CHARS]
