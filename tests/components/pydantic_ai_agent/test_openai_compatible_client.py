"""Test the lightweight OpenAI-compatible client."""

import json

import httpx
import pytest
from custom_components.pydantic_ai_agent.openai_compatible_client import (
    NOT_GIVEN,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAICompatible,
    omit,
)


async def test_chat_completion_serializes_payload_and_headers() -> None:
    """Test request serialization omits sentinels and preserves None."""
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["provider"] = request.headers.get("x-provider")
        captured["custom"] = request.headers.get("x-custom")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                },
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncOpenAICompatible(
        api_key="secret",
        base_url="https://provider.test/v1/",
        headers={"x-provider": "configured"},
        http_client=http_client,
    )

    response = await client.chat.completions.create(
        model="test-model",
        messages=[{"role": "user", "content": "Hi"}],
        stream=False,
        temperature=NOT_GIVEN,
        response_format=omit,
        extra_body={"nullable": None, "omitted": omit},
        extra_headers={"x-custom": "value"},
    )

    assert response.choices[0].message.content == "OK"
    assert captured["url"] == "https://provider.test/v1/chat/completions"
    assert captured["authorization"] == "Bearer secret"
    assert captured["provider"] == "configured"
    assert captured["custom"] == "value"
    assert captured["body"] == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
        "nullable": None,
    }
    await http_client.aclose()


async def test_models_list_parses_ids_and_sends_headers() -> None:
    """Test models list uses provider headers and returns sorted unique IDs."""
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["organization"] = request.headers.get("openai-organization")
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "z-model"},
                    {"id": "a-model"},
                    {"id": "z-model"},
                    {"object": "model"},
                ]
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncOpenAICompatible(
        api_key="secret",
        base_url="https://provider.test/v1/",
        headers={"OpenAI-Organization": "org_123"},
        http_client=http_client,
    )

    models = await client.models.list()

    assert models == ["a-model", "z-model"]
    assert captured == {
        "url": "https://provider.test/v1/models",
        "authorization": "Bearer secret",
        "organization": "org_123",
    }
    await http_client.aclose()


async def test_responses_create_serializes_payload_and_headers() -> None:
    """Test Responses requests use the /responses endpoint."""
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["custom"] = request.headers.get("x-custom")
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
                        "content": [{"type": "output_text", "text": "OK"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncOpenAICompatible(
        api_key="secret",
        base_url="https://provider.test/v1/",
        http_client=http_client,
    )

    response = await client.responses.create(
        model="test-model",
        input=[{"role": "user", "content": "Hi"}],
        text=omit,
        extra_body={"nullable": None, "omitted": omit},
        extra_headers={"x-custom": "value"},
    )

    assert response is not None
    assert response.output[0]["content"][0]["text"] == "OK"
    assert captured["url"] == "https://provider.test/v1/responses"
    assert captured["authorization"] == "Bearer secret"
    assert captured["custom"] == "value"
    assert captured["body"] == {
        "model": "test-model",
        "input": [{"role": "user", "content": "Hi"}],
        "stream": False,
        "nullable": None,
    }
    await http_client.aclose()


async def test_models_list_rejects_invalid_response_shape() -> None:
    """Test models list rejects malformed provider responses."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {}})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncOpenAICompatible(
        api_key=None,
        base_url="https://provider.test/v1",
        http_client=http_client,
    )

    with pytest.raises(APIConnectionError, match="Invalid models response JSON"):
        await client.models.list()
    await http_client.aclose()


async def test_chat_completion_stream_parses_sse_chunks() -> None:
    """Test streaming SSE parsing including a final usage-only chunk."""
    body = "".join(
        [
            'data: {"id":"1","model":"m",'
            '"choices":[{"index":0,"delta":{"content":"O"}}]}\n\n',
            'data: {"id":"1","model":"m",'
            '"choices":[{"index":0,"delta":{"content":"K"},"finish_reason":"stop"}]}\n\n',
            'data: {"id":"1","model":"m",'
            '"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n\n',
            "data: [DONE]\n\n",
        ]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncOpenAICompatible(
        api_key=None, base_url="https://provider.test/v1", http_client=http_client
    )

    stream = await client.chat.completions.create(model="m", messages=[], stream=True)
    chunks = [chunk async for chunk in stream]

    deltas = [chunk.choices[0].delta for chunk in chunks[:2] if chunk.choices]
    contents: list[str | None] = []
    for delta in deltas:
        assert delta is not None
        contents.append(delta.content)
    assert contents == ["O", "K"]
    assert chunks[2].choices == []
    assert chunks[2].usage is not None
    assert chunks[2].usage.prompt_tokens == 1
    await http_client.aclose()


async def test_chat_completion_stream_status_errors_read_body() -> None:
    """Test streamed status errors expose the provider body."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad model"}})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncOpenAICompatible(
        api_key=None, base_url="https://provider.test/v1", http_client=http_client
    )

    stream = await client.chat.completions.create(model="m", messages=[], stream=True)
    with pytest.raises(APIStatusError) as exc_info:
        async with stream:
            _ = [chunk async for chunk in stream]

    assert exc_info.value.status_code == 400
    assert exc_info.value.body == {"error": {"message": "bad model"}}
    assert exc_info.value.message == "bad model"
    await http_client.aclose()


async def test_responses_stream_parses_sse_events() -> None:
    """Test Responses streaming SSE parsing."""
    body = "".join(
        [
            'data: {"type":"response.created",'
            '"response":{"id":"resp-1","model":"m","status":"in_progress","output":[]}}\n\n',
            'data: {"type":"response.output_text.delta",'
            '"item_id":"msg-1","delta":"OK"}\n\n',
            'data: {"type":"response.completed",'
            '"response":{"id":"resp-1","model":"m","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}}\n\n',
            "data: [DONE]\n\n",
        ]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncOpenAICompatible(
        api_key=None, base_url="https://provider.test/v1", http_client=http_client
    )

    stream = await client.responses.create(model="m", input=[], stream=True)
    events = [event async for event in stream]

    assert [event.type for event in events] == [
        "response.created",
        "response.output_text.delta",
        "response.completed",
    ]
    assert events[1].delta == "OK"
    assert events[2].response is not None
    assert events[2].response.usage is not None
    assert events[2].response.usage.output_tokens == 2
    await http_client.aclose()


async def test_responses_stream_status_errors_read_body() -> None:
    """Test streamed Responses status errors expose the provider body."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad response"}})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncOpenAICompatible(
        api_key=None, base_url="https://provider.test/v1", http_client=http_client
    )

    stream = await client.responses.create(model="m", input=[], stream=True)
    with pytest.raises(APIStatusError) as exc_info:
        async with stream:
            _ = [event async for event in stream]

    assert exc_info.value.status_code == 400
    assert exc_info.value.body == {"error": {"message": "bad response"}}
    assert exc_info.value.message == "bad response"
    await http_client.aclose()


async def test_status_errors_include_status_and_body() -> None:
    """Test HTTP status errors expose useful response details."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncOpenAICompatible(
        api_key=None, base_url="https://provider.test/v1", http_client=http_client
    )

    with pytest.raises(APIStatusError) as exc_info:
        await client.chat.completions.create(model="m", messages=[])

    assert exc_info.value.status_code == 429
    assert exc_info.value.body == {"error": {"message": "rate limited"}}
    assert exc_info.value.message == "rate limited"
    await http_client.aclose()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (httpx.ConnectError("connect failed"), APIConnectionError),
        (httpx.ReadTimeout("timeout"), APITimeoutError),
    ],
)
async def test_network_errors_are_mapped(
    error: httpx.RequestError, expected: type[Exception]
) -> None:
    """Test httpx network failures are mapped to client exceptions."""

    async def handler(request: httpx.Request) -> httpx.Response:
        raise error

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncOpenAICompatible(
        api_key=None, base_url="https://provider.test/v1", http_client=http_client
    )

    with pytest.raises(expected):
        await client.chat.completions.create(model="m", messages=[])
    await http_client.aclose()
