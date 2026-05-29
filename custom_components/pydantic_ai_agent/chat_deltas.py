"""Convert Pydantic AI messages and events to Home Assistant ChatLog deltas."""

from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
import json
import logging
from typing import Any, cast

from pydantic_ai import AgentRunResultEvent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    OutputToolCallEvent,
    OutputToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)

from homeassistant.components import conversation
from homeassistant.helpers import llm

from .run_failures import _ToolProblem
from .run_state import _StreamRunState
from .virtual_workspace.const import TOOL_RETURN_METADATA_SOURCE

_LOGGER = logging.getLogger(__name__)

_TRACE_MAX_ITEMS = 200
_TRACE_HEAD_ITEMS = 100
_TRACE_TAIL_ITEMS = _TRACE_MAX_ITEMS - _TRACE_HEAD_ITEMS
_TRACE_PREVIEW_CHARS = 4096


async def _append_agent_messages(
    chat_log: conversation.ChatLog,
    agent_id: str,
    messages: list[ModelMessage],
    output_tool_names: set[str] | None = None,
) -> None:
    """Append Agent-produced assistant/tool messages to the Home Assistant log."""
    async for _content in chat_log.async_add_delta_content_stream(
        agent_id,
        cast(
            AsyncIterable[Any],
            _agent_messages_to_chat_deltas(messages, output_tool_names or set()),
        ),
    ):
        pass


async def _append_text(
    chat_log: conversation.ChatLog,
    agent_id: str,
    text: str,
) -> None:
    """Append one assistant text response to the Home Assistant log."""
    async for _content in chat_log.async_add_delta_content_stream(
        agent_id,
        cast(AsyncIterable[Any], _text_stream_to_chat_deltas(_single_text(text))),
    ):
        pass


async def _append_missing_final_text(
    chat_log: conversation.ChatLog,
    agent_id: str,
    messages: list[ModelMessage],
) -> dict[str, Any]:
    """Append final Agent text when live events did not stream it completely."""
    final_text = _final_text_from_messages(messages)
    if not final_text:
        return {
            "attempted": False,
            "changed": False,
            "reason": "no_final_text",
            "final_text_chars": 0,
        }

    last_content = chat_log.content[-1]
    if not isinstance(last_content, conversation.AssistantContent):
        await _append_text(chat_log, agent_id, final_text)
        return {
            "attempted": True,
            "changed": True,
            "reason": "last_content_not_assistant",
            "mode": "append_assistant_message",
            "final_text_chars": len(final_text),
            "backfill_chars": len(final_text),
        }

    streamed_text = last_content.content or ""
    if streamed_text == final_text:
        return {
            "attempted": True,
            "changed": False,
            "reason": "already_complete",
            "streamed_text_chars": len(streamed_text),
            "final_text_chars": len(final_text),
            "backfill_chars": 0,
        }

    missing_text: str | None = None
    if streamed_text and final_text.startswith(streamed_text):
        missing_text = final_text.removeprefix(streamed_text)
    elif not streamed_text:
        missing_text = final_text

    chat_log.content[-1] = replace(last_content, content=final_text)
    if missing_text and chat_log.delta_listener:
        chat_log.delta_listener(chat_log, {"content": missing_text})
    return {
        "attempted": True,
        "changed": True,
        "reason": "missing_or_divergent_final_text",
        "mode": "replace_assistant_content",
        "streamed_text_chars": len(streamed_text),
        "final_text_chars": len(final_text),
        "backfill_chars": len(missing_text or ""),
        "notified_delta_listener": bool(missing_text and chat_log.delta_listener),
    }


def _stream_trace_without_previews(value: Any) -> Any:
    """Return a stream trace copy with debug-only previews removed."""
    if isinstance(value, Mapping):
        return {
            key: _stream_trace_without_previews(item)
            for key, item in value.items()
            if not str(key).endswith("_preview")
        }
    if isinstance(value, list):
        return [_stream_trace_without_previews(item) for item in value]
    return value


def _final_text_from_messages(messages: list[ModelMessage]) -> str:
    """Return text from the final assistant response in Agent messages."""
    for message in reversed(messages):
        if isinstance(message, ModelResponse):
            return "".join(
                part.content for part in message.parts if isinstance(part, TextPart)
            )
    return ""


async def _text_stream_to_chat_deltas(
    text_stream: AsyncIterable[str],
) -> AsyncIterator[dict[str, str]]:
    """Yield ChatLog text deltas from a Pydantic AI text stream."""
    async for chunk in text_stream:
        if chunk:
            yield {"content": chunk}


async def _single_text(text: str) -> AsyncIterator[str]:
    """Yield one text chunk as an async iterator."""
    yield text


async def _agent_events_to_chat_deltas(
    events: AsyncIterable[Any],
    output_tool_names: set[str],
    state: _StreamRunState,
    trace_recorder: "_StreamTraceRecorder | None" = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield HA ChatLog deltas from live Pydantic AI Agent events."""
    assistant_open = False
    emitted_tool_call_ids: set[str] = set()
    async for event in events:
        if trace_recorder is not None:
            trace_recorder.record_event(event)
        if isinstance(event, AgentRunResultEvent):
            state.result = event.result
            continue
        if isinstance(event, PartStartEvent):
            if event.index == 0:
                state.emitted_deltas = True
                delta = {"role": "assistant"}
                if trace_recorder is not None:
                    trace_recorder.record_chat_delta(delta)
                yield delta
                assistant_open = True
            async for delta in _part_start_to_chat_deltas(event, output_tool_names):
                state.emitted_deltas = True
                if trace_recorder is not None:
                    trace_recorder.record_chat_delta(delta)
                yield delta
            continue
        if isinstance(event, PartDeltaEvent):
            if not assistant_open:
                state.emitted_deltas = True
                delta = {"role": "assistant"}
                if trace_recorder is not None:
                    trace_recorder.record_chat_delta(delta)
                yield delta
                assistant_open = True
            async for delta in _part_delta_to_chat_deltas(event):
                state.emitted_deltas = True
                if trace_recorder is not None:
                    trace_recorder.record_chat_delta(delta)
                yield delta
            continue
        if isinstance(event, FunctionToolCallEvent | OutputToolCallEvent):
            if not assistant_open:
                state.emitted_deltas = True
                delta = {"role": "assistant"}
                if trace_recorder is not None:
                    trace_recorder.record_chat_delta(delta)
                yield delta
                assistant_open = True
            async for delta in _tool_call_event_to_chat_deltas(
                event.part,
                output_tool_names,
                emitted_tool_call_ids,
            ):
                state.emitted_deltas = True
                if trace_recorder is not None:
                    trace_recorder.record_chat_delta(delta)
                yield delta
            continue
        if isinstance(event, FunctionToolResultEvent | OutputToolResultEvent):
            if event.part is None:
                continue
            tool_problem = _tool_problem_from_part(event.part)
            if tool_problem is not None:
                state.latest_tool_problem = tool_problem
                _log_tool_problem(tool_problem)
            state.emitted_deltas = True
            delta = {
                "role": "tool_result",
                "tool_call_id": event.part.tool_call_id,
                "tool_name": event.part.tool_name,
                "tool_result": event.part.content,
            }
            if trace_recorder is not None:
                trace_recorder.record_chat_delta(delta)
            yield delta
            assistant_open = False


@dataclass
class _StreamTraceRecorder:
    """Collect a bounded, JSON-safe summary of one Pydantic AI stream."""

    include_previews: bool = False
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
            **_stream_event_summary(event, self.include_previews),
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
            **_chat_delta_summary(delta, self.include_previews),
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
        chat_deltas_truncated = self.chat_deltas_total > len(
            self.chat_deltas
        ) + len(chat_deltas_tail)
        return {
            "schema_version": 1,
            "include_previews": self.include_previews,
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
            "final_new_messages": _messages_summary(
                final_messages, self.include_previews
            ),
            "backfill": dict(backfill),
            "final_chat_content": _chat_content_summary(
                final_chat_content,
                self.include_previews,
            ),
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
        if isinstance(message, ModelResponse):
            summary["parts"] = [
                _part_summary(part, include_preview) for part in message.parts
            ]
        elif isinstance(message, ModelRequest):
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
    """Add a bounded text preview when debug previews are enabled."""
    if include_preview and text:
        summary[f"{field_name}_preview"] = text[:_TRACE_PREVIEW_CHARS]


def _tool_problem_from_part(
    part: ToolReturnPart | RetryPromptPart,
) -> _ToolProblem | None:
    """Return a safe tool problem summary from a Pydantic AI tool result part."""
    if isinstance(part, RetryPromptPart):
        return _ToolProblem(
            tool_name=part.tool_name,
            tool_call_id=part.tool_call_id,
            outcome="retry",
            reason=_safe_tool_result_reason(
                part.content, getattr(part, "metadata", None)
            ),
        )
    outcome = getattr(part, "outcome", "success")
    reason = _safe_tool_result_reason(part.content, part.metadata)
    if outcome != "success":
        return _ToolProblem(
            tool_name=part.tool_name,
            tool_call_id=part.tool_call_id,
            outcome=outcome,
            reason=reason,
        )
    if isinstance(part.content, Mapping) and part.content.get("success") is False:
        return _ToolProblem(
            tool_name=part.tool_name,
            tool_call_id=part.tool_call_id,
            outcome="failed",
            reason=reason,
        )
    return None


def _safe_tool_result_reason(content: object, metadata: object) -> str | None:
    """Extract a short safe reason from structured tool failure content."""
    if not (
        isinstance(metadata, Mapping)
        and metadata.get("source") == TOOL_RETURN_METADATA_SOURCE
    ):
        return None
    reason: object | None = None
    if isinstance(content, Mapping):
        errors = content.get("errors")
        if isinstance(errors, Sequence) and not isinstance(errors, str | bytes):
            reason = next((item for item in errors if isinstance(item, str)), None)
        if reason is None:
            for key in ("error", "message"):
                value = content.get(key)
                if isinstance(value, str):
                    reason = value
                    break
    elif isinstance(content, str):
        reason = content
    if not isinstance(reason, str) or not reason:
        return None
    return reason[:200]


def _log_tool_problem(problem: _ToolProblem) -> None:
    """Log a non-terminal tool problem without exposing tool arguments."""
    _LOGGER.warning(
        'Pydantic AI tool "%s" returned %s for call "%s": %s',
        problem.tool_name or "unknown",
        problem.outcome,
        problem.tool_call_id or "unknown",
        problem.reason or "no safe detail provided",
    )


async def _part_start_to_chat_deltas(
    event: PartStartEvent,
    output_tool_names: set[str],
) -> AsyncIterator[dict[str, Any]]:
    """Yield initial HA deltas for a Pydantic AI part-start event."""
    part = event.part
    if isinstance(part, TextPart) and part.content:
        yield {"content": part.content}
    elif isinstance(part, ThinkingPart) and part.content:
        yield {"thinking_content": part.content}
    elif isinstance(part, ToolCallPart) and part.tool_name in output_tool_names:
        yield {"content": json.dumps(part.args_as_dict())}


async def _part_delta_to_chat_deltas(
    event: PartDeltaEvent,
) -> AsyncIterator[dict[str, Any]]:
    """Yield incremental HA deltas for a Pydantic AI part-delta event."""
    delta = event.delta
    if isinstance(delta, TextPartDelta) and delta.content_delta:
        yield {"content": delta.content_delta}
    elif isinstance(delta, ThinkingPartDelta) and delta.content_delta:
        yield {"thinking_content": delta.content_delta}


async def _tool_call_event_to_chat_deltas(
    part: ToolCallPart,
    output_tool_names: set[str],
    emitted_tool_call_ids: set[str],
) -> AsyncIterator[dict[str, Any]]:
    """Yield a HA tool-call delta when Pydantic AI starts executing a tool."""
    if part.tool_name in output_tool_names:
        yield {"content": json.dumps(part.args_as_dict())}
        return
    if part.tool_call_id in emitted_tool_call_ids:
        return
    emitted_tool_call_ids.add(part.tool_call_id)
    yield {
        "tool_calls": [
            llm.ToolInput(
                id=part.tool_call_id,
                tool_name=part.tool_name,
                tool_args=part.args_as_dict(),
                external=True,
            )
        ]
    }


def _json_output(output: object) -> str:
    """Return a JSON string for structured Agent output."""
    if isinstance(output, str):
        return output
    return json.dumps(output)


async def _agent_messages_to_chat_deltas(
    messages: list[ModelMessage],
    output_tool_names: set[str],
) -> AsyncIterator[dict[str, Any]]:
    """Yield ChatLog deltas from Agent messages without re-executing tools."""
    for message in messages:
        if isinstance(message, ModelResponse):
            content = ""
            thinking_content = ""
            tool_calls: list[llm.ToolInput] = []
            for part in message.parts:
                if isinstance(part, TextPart):
                    content += part.content
                elif isinstance(part, ThinkingPart):
                    thinking_content += part.content
                elif isinstance(part, ToolCallPart):
                    if part.tool_name in output_tool_names:
                        content += json.dumps(part.args_as_dict())
                        continue
                    tool_calls.append(
                        llm.ToolInput(
                            id=part.tool_call_id,
                            tool_name=part.tool_name,
                            tool_args=part.args_as_dict(),
                            external=True,
                        )
                    )
            if content or thinking_content or tool_calls:
                yield {
                    "role": "assistant",
                    "content": content,
                    "thinking_content": thinking_content,
                    "tool_calls": tool_calls,
                }
        elif isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, ToolReturnPart | RetryPromptPart):
                    tool_problem = _tool_problem_from_part(part)
                    if tool_problem is not None:
                        _log_tool_problem(tool_problem)
                    yield {
                        "role": "tool_result",
                        "tool_call_id": part.tool_call_id,
                        "tool_name": part.tool_name,
                        "tool_result": part.content,
                    }
