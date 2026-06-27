"""Test OpenAI-compatible Chat Completions streaming and tool follow-up."""

import json
from typing import cast

from custom_components.pydantic_ai_agent.openai_compatible_adapter import (
    OpenAICompatibleChatModel,
    OpenAICompatibleProvider,
)
import httpx
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import (
    ModelRequest,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition
import pytest


def _model_with_transport(
    handler: httpx.MockTransport,
) -> tuple[OpenAICompatibleChatModel, httpx.AsyncClient]:
    """Return a model backed by a mock HTTP transport."""
    http_client = httpx.AsyncClient(transport=handler)
    provider = OpenAICompatibleProvider(
        api_key="sk-test",
        base_url="https://provider.test/v1",
        http_client=http_client,
    )
    return OpenAICompatibleChatModel("test-model", provider=provider), http_client


async def test_streamed_tool_follow_up_preserves_reasoning_content() -> None:
    """Test streamed tool-call history preserves provider reasoning metadata."""
    captured_bodies: list[dict[str, object]] = []
    stream_body = "".join(
        [
            'data: {"id":"chatcmpl-1","model":"test-model",'
            '"choices":[{"index":0,'
            '"delta":{"reasoning_content":"I should call the echo tool."}}]}\n\n',
            'data: {"id":"chatcmpl-1","model":"test-model",'
            '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"echo","arguments":"{\\"token\\":\\"ok\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n',
            "data: [DONE]\n\n",
        ]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content))
        if len(captured_bodies) == 1:
            return httpx.Response(
                200,
                content=stream_body,
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-2",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    model, http_client = _model_with_transport(httpx.MockTransport(handler))
    request_parameters = ModelRequestParameters(
        function_tools=[
            ToolDefinition(
                name="echo",
                parameters_json_schema={"type": "object", "properties": {}},
            )
        ]
    )
    async with model.request_stream(
        [ModelRequest(parts=[UserPromptPart("Call echo")])], {}, request_parameters
    ) as stream:
        _ = [event async for event in stream]
        first_response = stream.get()

    follow_up = await model.request(
        [
            ModelRequest(parts=[UserPromptPart("Call echo")]),
            first_response,
            ModelRequest(parts=[ToolReturnPart("echo", "ok", tool_call_id="call-1")]),
        ],
        {},
        request_parameters,
    )

    assert any(
        isinstance(part, ThinkingPart) and part.id == "reasoning_content"
        for part in first_response.parts
    )
    messages = cast(list[dict[str, object]], captured_bodies[1]["messages"])
    assistant_message = messages[1]
    assert isinstance(assistant_message, dict)
    assert assistant_message["role"] == "assistant"
    assert assistant_message["content"] == ""
    assert assistant_message["reasoning_content"] == "I should call the echo tool."
    assert "reasoning" not in assistant_message
    assert follow_up.text == "done"
    await http_client.aclose()


async def test_tool_result_follow_up_uses_empty_assistant_content() -> None:
    """Test tool-call history uses content accepted by stricter providers."""
    captured_bodies: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_bodies.append(body)
        if len(captured_bodies) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "reasoning_content": "I should call the echo tool.",
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "echo",
                                            "arguments": '{"token":"ok"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-2",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    model, http_client = _model_with_transport(httpx.MockTransport(handler))
    first_response = await model.request(
        [ModelRequest(parts=[UserPromptPart("Call echo")])],
        {},
        ModelRequestParameters(
            function_tools=[
                ToolDefinition(
                    name="echo",
                    parameters_json_schema={"type": "object", "properties": {}},
                )
            ]
        ),
    )
    follow_up = await model.request(
        [
            ModelRequest(parts=[UserPromptPart("Call echo")]),
            first_response,
            ModelRequest(parts=[ToolReturnPart("echo", "ok", tool_call_id="call-1")]),
        ],
        {},
        ModelRequestParameters(
            function_tools=[
                ToolDefinition(
                    name="echo",
                    parameters_json_schema={"type": "object", "properties": {}},
                )
            ]
        ),
    )

    messages = cast(list[dict[str, object]], captured_bodies[1]["messages"])
    assistant_message = messages[1]
    assert isinstance(assistant_message, dict)
    assert assistant_message["role"] == "assistant"
    assert assistant_message["content"] == ""
    assert assistant_message["reasoning_content"] == "I should call the echo tool."
    assert "tool_calls" in assistant_message
    assert follow_up.text == "done"
    await http_client.aclose()


async def test_request_stream_yields_text_and_usage() -> None:
    """Test streamed text deltas are exposed as Pydantic AI stream events."""
    body = "".join(
        [
            'data: {"id":"1","model":"test-model",'
            '"choices":[{"index":0,"delta":{"content":"O"}}]}\n\n',
            'data: {"id":"1","model":"test-model",'
            '"choices":[{"index":0,"delta":{"content":"K"},"finish_reason":"stop"}]}\n\n',
            'data: {"id":"1","model":"test-model",'
            '"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}\n\n',
            "data: [DONE]\n\n",
        ]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )

    model, http_client = _model_with_transport(httpx.MockTransport(handler))
    async with model.request_stream(
        [ModelRequest(parts=[UserPromptPart("Hi")])],
        {},
        ModelRequestParameters(),
    ) as stream:
        events = [event async for event in stream]
        response = stream.get()

    assert any(
        isinstance(event, PartStartEvent) and isinstance(event.part, TextPart)
        for event in events
    )
    assert any(isinstance(event, PartDeltaEvent) for event in events)
    assert any(isinstance(event, PartEndEvent) for event in events)
    assert response.text == "OK"
    assert response.usage.input_tokens == 2
    assert response.usage.output_tokens == 3
    await http_client.aclose()


async def test_request_stream_accumulates_tool_call_deltas() -> None:
    """Test streamed function tool call argument fragments are accumulated."""
    body = "".join(
        [
            'data: {"id":"1","model":"test-model",'
            '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"turn_on","arguments":"{\\"entity"}}]}}]}\n\n',
            'data: {"id":"1","model":"test-model",'
            '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\":\\"light.kitchen\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n',
            "data: [DONE]\n\n",
        ]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )

    model, http_client = _model_with_transport(httpx.MockTransport(handler))
    async with model.request_stream(
        [ModelRequest(parts=[UserPromptPart("Turn on kitchen")])],
        {},
        ModelRequestParameters(
            function_tools=[
                ToolDefinition(
                    name="turn_on",
                    parameters_json_schema={"type": "object", "properties": {}},
                )
            ]
        ),
    ) as stream:
        events = [event async for event in stream]
        response = stream.get()

    tool_call = next(part for part in response.parts if isinstance(part, ToolCallPart))
    assert tool_call.tool_name == "turn_on"
    assert tool_call.args == '{"entity":"light.kitchen"}'
    assert any(
        isinstance(event, PartEndEvent) and isinstance(event.part, ToolCallPart)
        for event in events
    )
    await http_client.aclose()


async def test_request_stream_maps_status_errors() -> None:
    """Test streamed status errors map to Pydantic AI HTTP errors."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "unknown model"}})

    model, http_client = _model_with_transport(httpx.MockTransport(handler))

    with pytest.raises(ModelHTTPError) as exc_info:
        async with model.request_stream(
            [ModelRequest(parts=[UserPromptPart("Hi")])],
            {},
            ModelRequestParameters(),
        ):
            pass

    assert exc_info.value.status_code == 404
    assert exc_info.value.body == {"error": {"message": "unknown model"}}
    await http_client.aclose()


async def test_request_stream_maps_connection_errors() -> None:
    """Test stream-opening transport errors map to Pydantic AI API errors."""

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connect failed", request=request)

    model, http_client = _model_with_transport(httpx.MockTransport(handler))

    with pytest.raises(ModelAPIError):
        async with model.request_stream(
            [ModelRequest(parts=[UserPromptPart("Hi")])],
            {},
            ModelRequestParameters(),
        ):
            pass

    await http_client.aclose()
