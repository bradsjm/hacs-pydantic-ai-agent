"""Test the Pydantic AI OpenAI-compatible adapter."""

from pathlib import Path
import json
import re
from typing import cast

import httpx
import pytest
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import (
    BinaryContent,
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
from pydantic_ai.output import OutputObjectDefinition
from pydantic_ai.tools import ToolDefinition

from custom_components.pydantic_ai_agent.pydantic_ai_openai_compatible import (
    OpenAICompatibleChatModel,
    OpenAICompatibleProvider,
    OpenAICompatibleResponsesModel,
)

_REPO_ROOT = Path(__file__).parents[3]


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


def _responses_model_with_transport(
    handler: httpx.MockTransport,
    model_name: str = "test-model",
) -> tuple[OpenAICompatibleResponsesModel, httpx.AsyncClient]:
    """Return a Responses model backed by a mock HTTP transport."""
    http_client = httpx.AsyncClient(transport=handler)
    provider = OpenAICompatibleProvider(
        api_key="sk-test",
        base_url="https://provider.test/v1",
        http_client=http_client,
        name="openai-compatible-responses",
    )
    return OpenAICompatibleResponsesModel(model_name, provider=provider), http_client


async def test_request_returns_text_model_response() -> None:
    """Test non-streamed text responses map to ModelResponse."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 5,
                    "total_tokens": 9,
                },
            },
        )

    model, http_client = _model_with_transport(httpx.MockTransport(handler))
    response = await model.request(
        [ModelRequest(parts=[UserPromptPart("Hi")])],
        {},
        ModelRequestParameters(),
    )

    assert response.text == "Hello"
    assert response.provider_name == "openai-compatible-completions"
    assert response.usage.input_tokens == 4
    assert response.usage.output_tokens == 5
    await http_client.aclose()


async def test_responses_request_returns_text_model_response() -> None:
    """Test non-streamed Responses text output maps to ModelResponse."""
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "created_at": 1,
                "model": "test-model",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg-1",
                        "content": [{"type": "output_text", "text": "Hello"}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 5, "total_tokens": 9},
            },
        )

    model, http_client = _responses_model_with_transport(httpx.MockTransport(handler))
    response = await model.request(
        [ModelRequest(parts=[UserPromptPart("Hi")])],
        {},
        ModelRequestParameters(),
    )

    assert response.text == "Hello"
    assert response.provider_name == "openai-compatible-responses"
    assert response.provider_response_id == "resp-1"
    assert isinstance(response.parts[0], TextPart)
    assert response.parts[0].id == "msg-1"
    assert response.usage.input_tokens == 4
    assert response.usage.output_tokens == 5
    body = captured["body"]
    assert isinstance(body, dict)
    assert "instructions" not in body
    await http_client.aclose()


async def test_responses_request_maps_tools_structured_output_and_reasoning() -> None:
    """Test Responses request/response protocol-specific mapping."""
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "model": "test-model",
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs-1",
                        "encrypted_content": "sig",
                        "summary": [{"type": "summary_text", "text": "I should call"}],
                    },
                    {
                        "type": "function_call",
                        "id": "fc-1",
                        "call_id": "call-1",
                        "name": "turn_on",
                        "arguments": '{"entity":"light.kitchen"}',
                    },
                ],
            },
        )

    model, http_client = _responses_model_with_transport(
        httpx.MockTransport(handler), "gpt-5"
    )
    response = await model.request(
        [ModelRequest(parts=[UserPromptPart("Turn on the kitchen")])],
        {"max_tokens": 20},
        ModelRequestParameters(
            function_tools=[
                ToolDefinition(
                    name="turn_on",
                    description="Turn on a light",
                    parameters_json_schema={"type": "object", "properties": {}},
                    strict=True,
                )
            ],
            output_mode="native",
            output_object=OutputObjectDefinition(
                name="probe_response",
                json_schema={"type": "object"},
                strict=True,
            ),
            thinking=True,
        ),
    )

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["tools"] == [
        {
            "type": "function",
            "name": "turn_on",
            "description": "Turn on a light",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
                "required": [],
            },
            "strict": True,
        }
    ]
    assert body["tool_choice"] == "auto"
    assert body["max_output_tokens"] == 20
    assert body["reasoning"] == {"effort": "medium"}
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["text"] == {
        "format": {
                "type": "json_schema",
                "name": "probe_response",
                "schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                    "required": [],
                },
                "strict": True,
            }
        }
    assert isinstance(response.parts[0], ThinkingPart)
    assert response.parts[0].id == "rs-1"
    assert response.parts[0].signature == "sig"
    assert isinstance(response.parts[1], ToolCallPart)
    assert response.parts[1].id == "fc-1"
    assert response.parts[1].tool_call_id == "call-1"
    await http_client.aclose()


async def test_request_maps_tools_and_binary_content() -> None:
    """Test tools and binary image/PDF content are serialized for requests."""
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
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
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "turn_on",
                                        "arguments": '{"entity":"light.kitchen"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )

    model, http_client = _model_with_transport(httpx.MockTransport(handler))
    response = await model.request(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            "Inspect this",
                            BinaryContent(data=b"image", media_type="image/png"),
                            BinaryContent(data=b"%PDF", media_type="application/pdf"),
                        ]
                    )
                ]
            )
        ],
        {},
        ModelRequestParameters(
            function_tools=[
                ToolDefinition(
                    name="turn_on",
                    description="Turn on a light",
                    parameters_json_schema={"type": "object", "properties": {}},
                    strict=True,
                )
            ]
        ),
    )

    assert isinstance(response.parts[0], ToolCallPart)
    assert response.parts[0].tool_name == "turn_on"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["tools"][0]["function"]["strict"] is True
    content = body["messages"][0]["content"]
    assert content[1]["type"] == "image_url"
    assert content[2]["type"] == "file"
    await http_client.aclose()


async def test_output_tool_uses_auto_tool_choice() -> None:
    """Test output tools do not require provider-specific tool_choice support."""
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "{}"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    model, http_client = _model_with_transport(httpx.MockTransport(handler))
    await model.request(
        [ModelRequest(parts=[UserPromptPart("Use the output tool")])],
        {},
        ModelRequestParameters(
            function_tools=[
                ToolDefinition(
                    name="structured_output",
                    parameters_json_schema={"type": "object", "properties": {}},
                    kind="output",
                )
            ],
            allow_text_output=False,
        ),
    )

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["tool_choice"] == "auto"
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
            'data: {"id":"1","model":"test-model","choices":[{"index":0,"delta":{"content":"O"}}]}\n\n',
            'data: {"id":"1","model":"test-model","choices":[{"index":0,"delta":{"content":"K"},"finish_reason":"stop"}]}\n\n',
            'data: {"id":"1","model":"test-model","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}\n\n',
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
            'data: {"id":"1","model":"test-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"turn_on","arguments":"{\\"entity"}}]}}]}\n\n',
            'data: {"id":"1","model":"test-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\":\\"light.kitchen\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n',
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


def test_integration_does_not_import_openai_sdk_surfaces() -> None:
    """Guard against reintroducing OpenAI SDK imports."""
    forbidden = re.compile(
        r"(^|\n)\s*(import\s+openai\b|from\s+openai\b)|"
        r"pydantic_ai\.(models|providers)\.openai\b"
    )
    for path in (_REPO_ROOT / "custom_components" / "pydantic_ai_agent").rglob("*.py"):
        source = path.read_text()
        assert forbidden.search(source) is None, path
