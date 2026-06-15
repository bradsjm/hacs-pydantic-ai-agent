"""Test the Pydantic AI OpenAI-compatible Responses protocol adapter."""

import json
from typing import cast

import httpx
import pytest
from custom_components.pydantic_ai_agent.openai_compatible_adapter import (
    OpenAICompatibleProvider,
    OpenAICompatibleResponsesModel,
)
from custom_components.pydantic_ai_agent.openai_compatible_adapter import (
    _responses_message_mapping as responses_message_mapping,
)
from custom_components.pydantic_ai_agent.provider import openai_compatible_model_profile
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
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
from pydantic_ai.settings import ModelSettings, ThinkingLevel
from pydantic_ai.tools import ToolDefinition


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
    return (
        OpenAICompatibleResponsesModel(
            model_name,
            provider=provider,
            profile=openai_compatible_model_profile(
                {
                    "thinking_support": "supported",
                    "structured_output_support": "json_schema",
                    "supports_tools": True,
                    "openai_supports_strict_tool_definition": True,
                    "openai_supports_encrypted_reasoning_content": True,
                }
            ),
        ),
        http_client,
    )


async def _unused_handler(request: httpx.Request) -> httpx.Response:
    """Fail if a mapping-only test unexpectedly issues an HTTP request."""
    raise AssertionError(f"Unexpected request during mapping test: {request.url}")


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
                "parameters": {"type": "object", "properties": {}},
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
            "schema": {"type": "object"},
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


@pytest.mark.parametrize(
    ("thinking", "expected_reasoning"),
    [
        pytest.param(False, {"effort": "none"}, id="disabled"),
        pytest.param(None, {"effort": "high"}, id="unset"),
        pytest.param("low", {"effort": "low"}, id="explicit-level"),
    ],
)
async def test_responses_request_maps_run_level_thinking(
    thinking: bool | ThinkingLevel | None,
    expected_reasoning: dict[str, str],
) -> None:
    """Test run-level thinking controls Responses reasoning effort."""
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
                        "type": "message",
                        "id": "msg-1",
                        "content": [{"type": "output_text", "text": "Done"}],
                    }
                ],
            },
        )

    model, http_client = _responses_model_with_transport(httpx.MockTransport(handler))
    model_settings = cast(ModelSettings, {"openai_reasoning_effort": "high"})
    request_parameters = ModelRequestParameters()
    if thinking is not None:
        request_parameters = ModelRequestParameters(thinking=thinking)

    await model.request(
        [ModelRequest(parts=[UserPromptPart("Hi")])],
        model_settings,
        request_parameters,
    )

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["reasoning"] == expected_reasoning
    await http_client.aclose()


async def test_responses_text_like_binary_content_decodes_invalid_utf8() -> None:
    """Test Responses text attachments tolerate invalid UTF-8."""
    model, http_client = _responses_model_with_transport(
        httpx.MockTransport(_unused_handler)
    )

    _, mapped_messages = await responses_message_mapping.map_messages(
        model,
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            BinaryContent(data=b"bad\xffutf8", media_type="text/plain")
                        ]
                    )
                ]
            )
        ],
        ModelRequestParameters(),
    )

    assert mapped_messages == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "bad\ufffdutf8"}],
        }
    ]
    await http_client.aclose()


async def test_responses_tool_return_with_multimodal_content_maps_text_and_image() -> (
    None
):
    """Test Responses tool returns forward both text and images."""
    model, http_client = _responses_model_with_transport(
        httpx.MockTransport(_unused_handler)
    )

    _, mapped_messages = await responses_message_mapping.map_messages(
        model,
        [
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="camera_snapshot",
                        content=[
                            "Snapshot",
                            BinaryContent(data=b"image", media_type="image/png"),
                        ],
                        tool_call_id="call-1",
                    )
                ]
            )
        ],
        ModelRequestParameters(),
    )

    assert mapped_messages[0]["type"] == "function_call_output"
    assert "Snapshot" in mapped_messages[0]["output"]
    assert "See file" in mapped_messages[0]["output"]
    assert mapped_messages[1]["role"] == "user"
    assert mapped_messages[1]["content"][1]["type"] == "input_image"
    await http_client.aclose()


async def test_responses_request_stream_yields_text_reasoning_and_usage() -> None:
    """Test streamed Responses text and reasoning map to Pydantic AI events."""
    response_events = [
        {
            "type": "response.created",
            "response": {
                "id": "resp-1",
                "model": "test-model",
                "status": "in_progress",
                "output": [],
            },
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs-1",
            "summary_index": 0,
            "delta": "I should answer.",
        },
        {"type": "response.output_text.delta", "item_id": "msg-1", "delta": "O"},
        {"type": "response.output_text.delta", "item_id": "msg-1", "delta": "K"},
        {
            "type": "response.completed",
            "response": {
                "id": "resp-1",
                "model": "test-model",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
            },
        },
    ]
    body = "".join(
        [
            *(f"data: {json.dumps(event)}\n\n" for event in response_events),
            "data: [DONE]\n\n",
        ]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )

    model, http_client = _responses_model_with_transport(httpx.MockTransport(handler))
    async with model.request_stream(
        [ModelRequest(parts=[UserPromptPart("Hi")])],
        {},
        ModelRequestParameters(),
    ) as stream:
        events = [event async for event in stream]
        response = stream.get()

    assert any(
        isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart)
        for event in events
    )
    assert any(
        isinstance(event, PartStartEvent) and isinstance(event.part, TextPart)
        for event in events
    )
    assert response.text == "OK"
    assert any(
        isinstance(part, ThinkingPart) and part.content == "I should answer."
        for part in response.parts
    )
    assert response.provider_response_id == "resp-1"
    assert response.usage.input_tokens == 2
    assert response.usage.output_tokens == 3
    await http_client.aclose()


async def test_responses_request_stream_accumulates_tool_call_deltas() -> None:
    """Test streamed Responses function-call argument fragments are accumulated."""
    response_events = [
        {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "fc-1",
                "call_id": "call-1",
                "name": "turn_on",
                "arguments": "",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc-1",
            "delta": '{"entity"',
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc-1",
            "delta": ':"light.kitchen"}',
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "id": "fc-1",
                "call_id": "call-1",
                "name": "turn_on",
                "arguments": '{"entity":"light.kitchen"}',
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp-1",
                "model": "test-model",
                "status": "completed",
                "output": [],
            },
        },
    ]
    body = "".join(
        [
            *(f"data: {json.dumps(event)}\n\n" for event in response_events),
            "data: [DONE]\n\n",
        ]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )

    model, http_client = _responses_model_with_transport(httpx.MockTransport(handler))
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
