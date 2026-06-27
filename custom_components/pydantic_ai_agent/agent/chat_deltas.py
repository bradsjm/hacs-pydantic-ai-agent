"""Convert Pydantic AI messages and events to Home Assistant ChatLog deltas."""

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import replace
import json
import logging
from typing import Any, cast

from homeassistant.components import conversation
from homeassistant.helpers import llm
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
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)

from ._stream_trace import _StreamTraceRecorder
from ._tool_utils import _log_tool_problem, _tool_problem_from_part
from .multimodal_tool_result import serialize_multimodal_tool_result
from .run_state import _StreamRunState

_LOGGER = logging.getLogger(__name__)
_TOOL_RESUME_SEPARATOR = "\n\n"


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

    separator_prefix = _tool_resume_separator_prefix(
        chat_log.content,
        trailing_assistant_present=isinstance(
            chat_log.content[-1], conversation.AssistantContent
        ),
    )
    final_text_with_separator = f"{separator_prefix}{final_text}"

    last_content = chat_log.content[-1]
    if not isinstance(last_content, conversation.AssistantContent):
        await _append_text(chat_log, agent_id, final_text_with_separator)
        return {
            "attempted": True,
            "changed": True,
            "reason": "last_content_not_assistant",
            "mode": "append_assistant_message",
            "final_text_chars": len(final_text),
            "backfill_chars": len(final_text_with_separator),
        }

    streamed_text = last_content.content or ""
    normalized_streamed_text = streamed_text.removeprefix(separator_prefix)

    if streamed_text == final_text_with_separator:
        return {
            "attempted": True,
            "changed": False,
            "reason": "already_complete",
            "streamed_text_chars": len(streamed_text),
            "final_text_chars": len(final_text),
            "backfill_chars": 0,
        }

    missing_text: str | None = None
    if normalized_streamed_text and final_text.startswith(normalized_streamed_text):
        missing_text = final_text.removeprefix(normalized_streamed_text)
    elif not normalized_streamed_text:
        missing_text = final_text_with_separator

    last_content = replace(last_content, content=final_text_with_separator)
    chat_log.content[-1] = last_content
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


def _maybe_record_event(
    event: object, trace_recorder: _StreamTraceRecorder | None
) -> None:
    """Record an event if a trace recorder is active."""
    if trace_recorder is not None:
        trace_recorder.record_event(event)


async def _agent_events_to_chat_deltas(
    events: AsyncIterable[Any],
    output_tool_names: set[str],
    state: _StreamRunState,
    trace_recorder: _StreamTraceRecorder | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield HA ChatLog deltas from live Pydantic AI Agent events."""
    flags: dict[str, bool] = {
        "assistant_open": False,
        "needs_separator": False,
        "assistant_text_seen": False,
    }
    emitted_tool_call_ids: set[str] = set()
    async for event in events:
        _maybe_record_event(event, trace_recorder)
        async for delta in _process_agent_event(
            event,
            output_tool_names,
            state,
            trace_recorder,
            flags,
            emitted_tool_call_ids,
        ):
            yield delta


async def _process_agent_event(
    event: object,
    output_tool_names: set[str],
    state: _StreamRunState,
    trace_recorder: _StreamTraceRecorder | None,
    flags: dict[str, bool],
    emitted_tool_call_ids: set[str],
) -> AsyncIterator[dict[str, Any]]:
    """Dispatch one agent event to the appropriate handler."""
    if isinstance(event, AgentRunResultEvent):
        state.result = event.result
        return
    if isinstance(event, PartStartEvent):
        async for delta in _handle_events_part_start(
            event, output_tool_names, state, trace_recorder, flags
        ):
            yield delta
        return
    if isinstance(event, PartDeltaEvent):
        async for delta in _handle_events_part_delta(
            event, state, trace_recorder, flags
        ):
            yield delta
        return
    if isinstance(event, FunctionToolCallEvent | OutputToolCallEvent):
        async for delta in _handle_events_tool_call(
            event,
            output_tool_names,
            emitted_tool_call_ids,
            state,
            trace_recorder,
            flags,
        ):
            yield delta
        return
    if isinstance(event, FunctionToolResultEvent | OutputToolResultEvent):
        async for delta in _handle_events_tool_result(
            event, state, trace_recorder, flags
        ):
            yield delta


async def _handle_events_part_start(
    event: PartStartEvent,
    output_tool_names: set[str],
    state: _StreamRunState,
    trace_recorder: _StreamTraceRecorder | None,
    flags: dict[str, bool],
) -> AsyncIterator[dict[str, Any]]:
    """Yield assistant header and initial part content for a part start event."""
    if event.index == 0:
        state.emitted_deltas = True
        delta = {"role": "assistant"}
        if trace_recorder is not None:
            trace_recorder.record_chat_delta(delta)
        yield delta
        flags["assistant_open"] = True
    async for delta in _part_start_to_chat_deltas(event, output_tool_names):
        delta = _maybe_prepend_resumed_text_separator(delta, flags)
        state.emitted_deltas = True
        if trace_recorder is not None:
            trace_recorder.record_chat_delta(delta)
        yield delta


async def _handle_events_part_delta(
    event: PartDeltaEvent,
    state: _StreamRunState,
    trace_recorder: _StreamTraceRecorder | None,
    flags: dict[str, bool],
) -> AsyncIterator[dict[str, Any]]:
    """Yield assistant header if needed and part delta content."""
    if not flags["assistant_open"]:
        state.emitted_deltas = True
        delta = {"role": "assistant"}
        if trace_recorder is not None:
            trace_recorder.record_chat_delta(delta)
        yield delta
    flags["assistant_open"] = True
    async for delta in _part_delta_to_chat_deltas(event):
        delta = _maybe_prepend_resumed_text_separator(delta, flags)
        state.emitted_deltas = True
        if trace_recorder is not None:
            trace_recorder.record_chat_delta(delta)
        yield delta


async def _handle_events_tool_call(
    event: FunctionToolCallEvent | OutputToolCallEvent,
    output_tool_names: set[str],
    emitted_tool_call_ids: set[str],
    state: _StreamRunState,
    trace_recorder: _StreamTraceRecorder | None,
    flags: dict[str, bool],
) -> AsyncIterator[dict[str, Any]]:
    """Yield assistant header if needed and tool call deltas."""
    if not flags["assistant_open"]:
        state.emitted_deltas = True
        delta = {"role": "assistant"}
        if trace_recorder is not None:
            trace_recorder.record_chat_delta(delta)
        yield delta
    flags["assistant_open"] = True
    async for delta in _tool_call_event_to_chat_deltas(
        event.part,
        output_tool_names,
        emitted_tool_call_ids,
    ):
        state.emitted_deltas = True
        if trace_recorder is not None:
            trace_recorder.record_chat_delta(delta)
        yield delta


async def _handle_events_tool_result(
    event: FunctionToolResultEvent | OutputToolResultEvent,
    state: _StreamRunState,
    trace_recorder: _StreamTraceRecorder | None,
    flags: dict[str, bool],
) -> AsyncIterator[dict[str, Any]]:
    """Yield a tool result delta and update stream state."""
    tool_problem = _tool_problem_from_part(event.part)
    if tool_problem is not None:
        state.latest_tool_problem = tool_problem
        _log_tool_problem(tool_problem)
    state.emitted_deltas = True
    delta = {
        "role": "tool_result",
        "tool_call_id": event.part.tool_call_id,
        "tool_name": event.part.tool_name,
        "tool_result": serialize_multimodal_tool_result(event.part.content),
    }
    if trace_recorder is not None:
        trace_recorder.record_chat_delta(delta)
    yield delta
    flags["assistant_open"] = False
    flags["needs_separator"] = flags["needs_separator"] or flags["assistant_text_seen"]
    flags["assistant_text_seen"] = False


def _maybe_prepend_resumed_text_separator(
    delta: dict[str, Any], flags: dict[str, bool]
) -> dict[str, Any]:
    """Prefix resumed assistant text after tool results once per resume."""
    content = delta.get("content")
    if not isinstance(content, str) or not content:
        return delta
    flags["assistant_text_seen"] = True
    if not flags["needs_separator"]:
        return delta
    flags["needs_separator"] = False
    return {**delta, "content": f"\n\n{content}"}


def _tool_resume_separator_prefix(
    content: list[conversation.Content], *, trailing_assistant_present: bool
) -> str:
    """Return the separator when the current tail is resumed text after tools."""
    index = len(content) - 2 if trailing_assistant_present else len(content) - 1
    saw_tool_result = False
    while index >= 0 and isinstance(content[index], conversation.ToolResultContent):
        saw_tool_result = True
        index -= 1
    if not saw_tool_result or index < 0:
        return ""
    prior_content = content[index]
    if (
        isinstance(prior_content, conversation.AssistantContent)
        and prior_content.content
    ):
        return _TOOL_RESUME_SEPARATOR
    return ""


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
            async for delta in _response_message_to_deltas(message, output_tool_names):
                yield delta
        elif isinstance(message, ModelRequest):
            async for delta in _request_message_to_deltas(message):
                yield delta


async def _response_message_to_deltas(
    message: ModelResponse,
    output_tool_names: set[str],
) -> AsyncIterator[dict[str, Any]]:
    """Yield assistant deltas from one ModelResponse message."""
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


async def _request_message_to_deltas(
    message: ModelRequest,
) -> AsyncIterator[dict[str, Any]]:
    """Yield tool result deltas from one ModelRequest message."""
    for part in message.parts:
        if isinstance(part, ToolReturnPart | RetryPromptPart):
            tool_problem = _tool_problem_from_part(part)
            if tool_problem is not None:
                _log_tool_problem(tool_problem)
            yield {
                "role": "tool_result",
                "tool_call_id": part.tool_call_id,
                "tool_name": part.tool_name,
                "tool_result": serialize_multimodal_tool_result(part.content),
            }
