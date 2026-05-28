"""Convert Pydantic AI messages and events to Home Assistant ChatLog deltas."""

from collections.abc import AsyncIterable, AsyncIterator, Mapping, Sequence
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
) -> AsyncIterator[dict[str, Any]]:
    """Yield HA ChatLog deltas from live Pydantic AI Agent events."""
    assistant_open = False
    emitted_tool_call_ids: set[str] = set()
    async for event in events:
        if isinstance(event, AgentRunResultEvent):
            state.result = event.result
            continue
        if isinstance(event, PartStartEvent):
            if event.index == 0:
                state.emitted_deltas = True
                yield {"role": "assistant"}
                assistant_open = True
            async for delta in _part_start_to_chat_deltas(event, output_tool_names):
                state.emitted_deltas = True
                yield delta
            continue
        if isinstance(event, PartDeltaEvent):
            if not assistant_open:
                state.emitted_deltas = True
                yield {"role": "assistant"}
                assistant_open = True
            async for delta in _part_delta_to_chat_deltas(event):
                state.emitted_deltas = True
                yield delta
            continue
        if isinstance(event, FunctionToolCallEvent | OutputToolCallEvent):
            if not assistant_open:
                state.emitted_deltas = True
                yield {"role": "assistant"}
                assistant_open = True
            async for delta in _tool_call_event_to_chat_deltas(
                event.part,
                output_tool_names,
                emitted_tool_call_ids,
            ):
                state.emitted_deltas = True
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
            yield {
                "role": "tool_result",
                "tool_call_id": event.part.tool_call_id,
                "tool_name": event.part.tool_name,
                "tool_result": event.part.content,
            }
            assistant_open = False


def _tool_problem_from_part(
    part: ToolReturnPart | RetryPromptPart,
) -> _ToolProblem | None:
    """Return a safe tool problem summary from a Pydantic AI tool result part."""
    if isinstance(part, RetryPromptPart):
        return _ToolProblem(
            tool_name=part.tool_name,
            tool_call_id=part.tool_call_id,
            outcome="retry",
            reason=_safe_tool_result_reason(part.content, getattr(part, "metadata", None)),
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
