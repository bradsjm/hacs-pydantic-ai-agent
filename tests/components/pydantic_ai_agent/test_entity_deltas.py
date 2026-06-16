"""Test entity chat delta streaming/mapping."""

import base64
import logging
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from custom_components.pydantic_ai_agent.agent.chat_deltas import (
    _agent_events_to_chat_deltas,
    _agent_messages_to_chat_deltas,
)
from custom_components.pydantic_ai_agent.agent.run_state import _StreamRunState
from custom_components.pydantic_ai_agent.virtual_workspace.const import (
    TOOL_RETURN_METADATA_SOURCE,
)
from pydantic_ai import (
    AgentRunResultEvent,
    BinaryContent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)


async def _collect_deltas(
    messages: list[Any], output_tool_names: set[str]
) -> list[dict[str, Any]]:
    """Collect chat deltas from the async generator."""
    return [
        delta
        async for delta in _agent_messages_to_chat_deltas(messages, output_tool_names)
    ]


async def _collect_event_deltas(events: list[Any]) -> tuple[list[dict[str, Any]], Any]:
    """Collect chat deltas from live Agent events."""

    async def stream() -> AsyncIterator[Any]:
        for event in events:
            yield event

    state = _StreamRunState()
    deltas = [
        delta async for delta in _agent_events_to_chat_deltas(stream(), set(), state)
    ]
    return deltas, state.result


async def test_agent_messages_to_chat_deltas_preserves_assistant_parts() -> None:
    """Test assistant text, thinking, and external tool calls become deltas."""
    deltas = await _collect_deltas(
        [
            ModelResponse(
                parts=[
                    TextPart(content="hello "),
                    ThinkingPart(content="reasoning"),
                    TextPart(content="world"),
                    ToolCallPart(
                        tool_name="HassTurnOn",
                        args={"name": "Kitchen"},
                        tool_call_id="tool-1",
                    ),
                ]
            )
        ],
        output_tool_names=set(),
    )

    assert len(deltas) == 1
    assert deltas[0]["role"] == "assistant"
    assert deltas[0]["content"] == "hello world"
    assert deltas[0]["thinking_content"] == "reasoning"
    tool_call = deltas[0]["tool_calls"][0]
    assert tool_call.tool_name == "HassTurnOn"
    assert tool_call.tool_args == {"name": "Kitchen"}
    assert tool_call.id == "tool-1"


async def test_agent_events_to_chat_deltas_does_not_replay_final_result() -> None:
    """Test live stream deltas are not duplicated by the final run result."""
    result = object()
    deltas, final_result = await _collect_event_deltas(
        [
            PartStartEvent(index=0, part=TextPart(content="hel")),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="lo")),
            AgentRunResultEvent(cast(Any, result)),
        ]
    )

    assert deltas == [{"role": "assistant"}, {"content": "hel"}, {"content": "lo"}]
    assert final_result is result


async def test_agent_events_to_chat_deltas_streams_tool_call_sequence() -> None:
    """Test live tool call and result events are forwarded in stream order."""
    result = object()
    deltas, final_result = await _collect_event_deltas(
        [
            PartStartEvent(index=0, part=TextPart(content="turning ")),
            FunctionToolCallEvent(
                ToolCallPart(
                    tool_name="HassTurnOn",
                    args={"name": "Kitchen"},
                    tool_call_id="tool-1",
                )
            ),
            FunctionToolResultEvent(
                ToolReturnPart(
                    tool_name="HassTurnOn",
                    content={"success": True},
                    tool_call_id="tool-1",
                )
            ),
            PartStartEvent(index=0, part=TextPart(content="done")),
            AgentRunResultEvent(cast(Any, result)),
        ]
    )

    tool_call = deltas[2]["tool_calls"][0]
    assert deltas[0] == {"role": "assistant"}
    assert deltas[1] == {"content": "turning "}
    assert tool_call.tool_name == "HassTurnOn"
    assert tool_call.tool_args == {"name": "Kitchen"}
    assert tool_call.id == "tool-1"
    assert tool_call.external is True
    assert deltas[3] == {
        "role": "tool_result",
        "tool_call_id": "tool-1",
        "tool_name": "HassTurnOn",
        "tool_result": {"success": True},
    }
    assert deltas[4:] == [{"role": "assistant"}, {"content": "\n\ndone"}]
    assert final_result is result


async def test_agent_events_to_chat_deltas_injects_resume_separator() -> None:
    """Test resumed assistant text gets one separator after one or more tool results."""
    deltas, _ = await _collect_event_deltas(
        [
            PartStartEvent(index=0, part=TextPart(content="turning ")),
            FunctionToolResultEvent(
                ToolReturnPart(
                    tool_name="HassTurnOn",
                    content={"success": True},
                    tool_call_id="tool-1",
                )
            ),
            FunctionToolResultEvent(
                ToolReturnPart(
                    tool_name="HassSetBrightness",
                    content={"success": True},
                    tool_call_id="tool-2",
                )
            ),
            PartStartEvent(index=0, part=TextPart(content="do")),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="ne")),
        ]
    )

    assert deltas == [
        {"role": "assistant"},
        {"content": "turning "},
        {
            "role": "tool_result",
            "tool_call_id": "tool-1",
            "tool_name": "HassTurnOn",
            "tool_result": {"success": True},
        },
        {
            "role": "tool_result",
            "tool_call_id": "tool-2",
            "tool_name": "HassSetBrightness",
            "tool_result": {"success": True},
        },
        {"role": "assistant"},
        {"content": "\n\ndo"},
        {"content": "ne"},
    ]


async def test_agent_events_to_chat_deltas_logs_tool_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test failed tool result payloads are logged and tracked as context."""

    async def stream() -> AsyncIterator[Any]:
        yield FunctionToolResultEvent(
            ToolReturnPart(
                tool_name="applyPatch",
                content={
                    "success": False,
                    "errors": ["patch must start with *** Begin Patch"],
                },
                tool_call_id="tool-1",
                metadata={"source": TOOL_RETURN_METADATA_SOURCE},
            )
        )

    state = _StreamRunState()

    with caplog.at_level(logging.WARNING):
        deltas = [
            delta
            async for delta in _agent_events_to_chat_deltas(stream(), set(), state)
        ]

    assert deltas == [
        {
            "role": "tool_result",
            "tool_call_id": "tool-1",
            "tool_name": "applyPatch",
            "tool_result": {
                "success": False,
                "errors": ["patch must start with *** Begin Patch"],
            },
        }
    ]
    assert state.latest_tool_problem is not None
    assert state.latest_tool_problem.tool_name == "applyPatch"
    assert state.latest_tool_problem.reason == "patch must start with *** Begin Patch"
    assert 'Pydantic AI tool "applyPatch" returned failed' in caplog.text


async def test_agent_events_to_chat_deltas_redacts_untrusted_tool_failure_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test arbitrary tool error payloads are not copied into HA logs."""

    async def stream() -> AsyncIterator[Any]:
        yield FunctionToolResultEvent(
            ToolReturnPart(
                tool_name="applyPatch",
                content={
                    "success": False,
                    "error": "raw provider body with private data",
                },
                tool_call_id="tool-1",
            )
        )

    state = _StreamRunState()

    with caplog.at_level(logging.WARNING):
        deltas = [
            delta
            async for delta in _agent_events_to_chat_deltas(stream(), set(), state)
        ]

    assert deltas[0]["tool_name"] == "applyPatch"
    assert state.latest_tool_problem is not None
    assert state.latest_tool_problem.reason is None
    assert "applyPatch" in caplog.text
    assert "no safe detail provided" in caplog.text
    assert "raw provider body" not in caplog.text


async def test_agent_events_to_chat_deltas_tracks_retry_prompts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test retry prompt parts are logged as non-terminal tool problems."""

    async def stream() -> AsyncIterator[Any]:
        yield FunctionToolResultEvent(
            RetryPromptPart(
                tool_name="applyPatch",
                tool_call_id="tool-1",
                content="patch content must follow a file header",
            )
        )

    state = _StreamRunState()

    with caplog.at_level(logging.WARNING):
        deltas = [
            delta
            async for delta in _agent_events_to_chat_deltas(stream(), set(), state)
        ]

    assert deltas[0]["tool_name"] == "applyPatch"
    assert state.latest_tool_problem is not None
    assert state.latest_tool_problem.outcome == "retry"
    assert state.latest_tool_problem.reason is None
    assert "returned retry" in caplog.text
    assert "patch content must follow a file header" not in caplog.text


async def test_agent_messages_to_chat_deltas_converts_output_tool_to_content() -> None:
    """Test structured-output tool calls are rendered as assistant JSON content."""
    deltas = await _collect_deltas(
        [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="generated_data",
                        args={"summary": "ok"},
                        tool_call_id="output-1",
                    )
                ]
            )
        ],
        output_tool_names={"generated_data"},
    )

    assert deltas == [
        {
            "role": "assistant",
            "content": '{"summary": "ok"}',
            "thinking_content": "",
            "tool_calls": [],
        }
    ]


async def test_agent_messages_to_chat_deltas_preserves_tool_returns() -> None:
    """Test tool result request parts become HA tool-result deltas."""
    deltas = await _collect_deltas(
        [
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="HassTurnOn",
                        content={"success": True},
                        tool_call_id="tool-1",
                    )
                ]
            )
        ],
        output_tool_names=set(),
    )

    assert deltas == [
        {
            "role": "tool_result",
            "tool_call_id": "tool-1",
            "tool_name": "HassTurnOn",
            "tool_result": {"success": True},
        }
    ]


async def test_agent_messages_to_chat_deltas_serializes_multimodal_tool_returns() -> (
    None
):
    """Test multimodal tool result parts persist as JSON-safe sentinel data."""
    deltas = await _collect_deltas(
        [
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="camera_snapshot",
                        content=[
                            "Snapshot",
                            BinaryContent(data=b"jpeg-bytes", media_type="image/jpeg"),
                        ],
                        tool_call_id="tool-1",
                    )
                ]
            )
        ],
        output_tool_names=set(),
    )

    assert deltas == [
        {
            "role": "tool_result",
            "tool_call_id": "tool-1",
            "tool_name": "camera_snapshot",
            "tool_result": {
                "_type": "ha_multimodal_tool_result",
                "text": "Snapshot",
                "attachments": [
                    {
                        "kind": "inline_image",
                        "mime_type": "image/jpeg",
                        "base64": base64.b64encode(b"jpeg-bytes").decode(),
                    }
                ],
            },
        }
    ]
