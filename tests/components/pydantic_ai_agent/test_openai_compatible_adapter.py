"""Test the Pydantic AI OpenAI-compatible Chat Completions adapter."""

import json
import re
from pathlib import Path

import httpx
from custom_components.pydantic_ai_agent.openai_compatible_adapter import (
    OpenAICompatibleChatModel,
    OpenAICompatibleProvider,
)
from custom_components.pydantic_ai_agent.openai_compatible_adapter import (
    _message_mapping as chat_message_mapping,
)
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

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


async def _unused_handler(request: httpx.Request) -> httpx.Response:
    """Fail if a mapping-only test unexpectedly issues an HTTP request."""
    raise AssertionError(f"Unexpected request during mapping test: {request.url}")


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


async def test_request_maps_disabled_thinking_to_reasoning_effort_none() -> None:
    """Test explicit disabled thinking sends reasoning_effort='none'."""
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
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    model, http_client = _model_with_transport(httpx.MockTransport(handler))
    await model.request(
        [ModelRequest(parts=[UserPromptPart("Hi")])],
        {},
        ModelRequestParameters(thinking=False),
    )

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["reasoning_effort"] == "none"
    await http_client.aclose()


async def test_request_omits_reasoning_effort_when_thinking_is_unset() -> None:
    """Test unset thinking leaves reasoning_effort absent."""
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
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    model, http_client = _model_with_transport(httpx.MockTransport(handler))
    await model.request(
        [ModelRequest(parts=[UserPromptPart("Hi")])],
        {},
        ModelRequestParameters(),
    )

    body = captured["body"]
    assert isinstance(body, dict)
    assert "reasoning_effort" not in body
    await http_client.aclose()


async def test_request_passes_explicit_thinking_level_to_reasoning_effort() -> None:
    """Test explicit thinking levels pass through to reasoning_effort."""
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
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    model, http_client = _model_with_transport(httpx.MockTransport(handler))
    await model.request(
        [ModelRequest(parts=[UserPromptPart("Hi")])],
        {},
        ModelRequestParameters(thinking="low"),
    )

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["reasoning_effort"] == "low"
    await http_client.aclose()


async def test_chat_history_keeps_reasoning_only_assistant_message() -> None:
    """Test Chat Completions history preserves reasoning-only assistant items."""
    model, http_client = _model_with_transport(httpx.MockTransport(_unused_handler))

    messages = await chat_message_mapping.map_messages(
        model,
        [ModelResponse(parts=[ThinkingPart(id="reasoning_content", content="think")])],
        ModelRequestParameters(),
    )

    assert messages == [
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "think",
        }
    ]
    await http_client.aclose()


async def test_chat_text_like_binary_content_decodes_invalid_utf8() -> None:
    """Test Chat Completions text attachments tolerate invalid UTF-8."""
    model, http_client = _model_with_transport(httpx.MockTransport(_unused_handler))

    mapped_messages = await chat_message_mapping.map_messages(
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
            "content": [{"type": "text", "text": "bad\ufffdutf8"}],
        }
    ]
    await http_client.aclose()


async def test_chat_tool_return_with_multimodal_content_maps_text_and_image() -> None:
    """Test Chat Completions tool returns forward both text and images."""
    model, http_client = _model_with_transport(httpx.MockTransport(_unused_handler))

    mapped_messages = await chat_message_mapping.map_messages(
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

    assert mapped_messages[0]["role"] == "tool"
    assert "Snapshot" in mapped_messages[0]["content"]
    assert "See file" in mapped_messages[0]["content"]
    assert mapped_messages[1]["role"] == "user"
    assert mapped_messages[1]["content"][1]["type"] == "image_url"
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


def test_integration_does_not_import_openai_sdk_surfaces() -> None:
    """Guard against reintroducing OpenAI SDK imports."""
    forbidden = re.compile(
        r"(^|\n)\s*(import\s+openai\b|from\s+openai\b)|"
        r"pydantic_ai\.(models|providers)\.openai\b"
    )
    for path in (_REPO_ROOT / "custom_components" / "pydantic_ai_agent").rglob("*.py"):
        source = path.read_text()
        assert forbidden.search(source) is None, path
