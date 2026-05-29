# OpenAI-Compatible Client

This package is a lightweight async HTTP client for the OpenAI-compatible API
subset used by the integration. It mirrors the small SDK shape needed by the
adapter while avoiding an OpenAI SDK dependency.

## Public API

- `AsyncOpenAICompatible` - top-level client with `chat`, `models`, and
  `responses` resources.
- `client.chat.completions.create()` - sends `/chat/completions` requests and
  returns `ChatCompletion` or `ChatCompletionStream`.
- `client.responses.create()` - sends `/responses` requests and returns
  `Response` or `ResponseStream`.
- `client.models.list()` - sends `/models` requests and returns sorted model
  ids.
- `NOT_GIVEN` and `omit` - sentinels for omitted request values.
- `APIConnectionError`, `APITimeoutError`, and `APIStatusError` - lightweight
  exceptions consumed by the adapter.

## Modules

- `_client.py` - base URL normalization, auth header construction, and resource
  construction.
- `_chat.py` - recursive payload serialization, Chat Completions POST handling,
  and stream construction.
- `_responses.py` - Responses POST handling and stream construction.
- `_models.py` - model list retrieval.
- `_streaming.py` - HTTP status error extraction and typed SSE iterators.
- `_types.py` - Pydantic models for chat, responses, chunks, stream events, and
  usage objects.
- `_exceptions.py` - client exception classes.
- `_sentinels.py` - omitted-value sentinel classes and helpers.

## Request Handling

The client receives an `httpx.AsyncClient` from its caller. It does not own Home
Assistant lifecycle hooks, retries, or model fallback. Resource methods merge
auth headers with per-request extra headers, strip omitted sentinels from JSON
payloads, preserve `None`, and validate JSON responses with permissive Pydantic
models.

## Streaming

`ChatCompletionStream` and `ResponseStream` wrap `httpx.AsyncClient.stream()`.
They enter the HTTP stream lazily on first iteration, validate HTTP status, parse
SSE `data:` lines, stop at `[DONE]`, and close the stream through `close()` or
the async context manager.

## Error Model

- HTTP status codes 400 and above raise `APIStatusError` with parsed JSON or
  text body data.
- `httpx.TimeoutException` maps to `APITimeoutError`.
- `httpx.RequestError`, invalid JSON, and response validation failures map to
  `APIConnectionError`.

## Testing

- `scripts/test -k test_openai_compatible_client`
- `scripts/test -k test_openai_compatible_adapter`

The main client tests live in
`tests/components/pydantic_ai_agent/test_openai_compatible_client.py` and use
`httpx.MockTransport`.
