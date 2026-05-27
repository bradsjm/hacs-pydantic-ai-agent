# OpenAI-Compatible Provider Design

## Status

This document describes the implemented in-repo OpenAI-compatible provider path
for `custom_components/pydantic_ai_agent`. Current source, tests, manifests, and
lockfiles remain authoritative when they differ from this design note.

This document is scoped to the `openai_compatible_completions` and
`openai_compatible_responses` provider modes. Native Anthropic and Google Gemini
provider modes are separate runtime paths built in `provider.py` with Pydantic
AI's public provider/model classes and explicit Home Assistant config-entry
credentials.

Implemented source areas:

| Area                               | Source                                                           |
| ---------------------------------- | ---------------------------------------------------------------- |
| Low-level HTTP client              | `custom_components/pydantic_ai_agent/openai_compatible_client/`  |
| Pydantic AI model/provider adapter | `custom_components/pydantic_ai_agent/openai_compatible_adapter/` |
| Home Assistant model factory       | `custom_components/pydantic_ai_agent/provider.py`                |
| Config-flow provider probe         | `custom_components/pydantic_ai_agent/config_flow.py`             |
| Runtime agent construction         | `custom_components/pydantic_ai_agent/entity.py`                  |
| Real-provider tests                | `tests/components/pydantic_ai_agent/test_real_server.py`         |

## Rationale

The integration needs OpenAI-compatible Completions and Responses providers
without the OpenAI SDK dependency. Home Assistant installs custom integration
requirements into the same runtime as Core, so dependency size, dependency
conflicts, async client ownership, diagnostics, and lifecycle behavior matter.

The in-repo adapter exists to:

- use Home Assistant's managed `httpx.AsyncClient` configuration;
- avoid adding the OpenAI SDK dependency;
- support OpenAI-compatible endpoints through small Completions and Responses
  surfaces;
- preserve Pydantic AI's public `Model` and `Provider` contracts;
- map provider/network errors to Pydantic AI exceptions consistently;
- preserve provider-specific reasoning metadata needed by reasoning models;
- keep request and response shaping testable with local mocked HTTP transports.

## Goals

- Support the `openai_compatible_completions` provider mode through Chat
  Completions.
- Support the `openai_compatible_responses` provider mode through the Responses
  API.
- Use `https://api.openai.com/v1` when no custom `base_url` is configured.
- Share Home Assistant's async HTTP client when constructed from integration
  runtime code.
- Support plain text, tool calls, tool results, structured output modes, model
  settings, and usage mapping for both provider modes.
- Support SSE streaming for both Chat Completions and Responses provider modes,
  including provider probing and Home Assistant conversation runtime streaming.
- Keep diagnostics and tests free of API keys, headers, `.env` values, and raw
  provider payloads.

## Non-Goals

- Do not provide an OpenAI SDK compatibility shim.
- Do not reintroduce `pydantic_ai.models.openai.OpenAIChatModel` or
  `pydantic_ai.providers.openai.OpenAIProvider` as runtime dependencies.
- Do not route native Anthropic or Google Gemini support through the in-repo
  OpenAI-compatible adapter.
- Do not simulate streaming by chunking completed non-streamed responses.
- Do not add stdio, SSE, or local process MCP support as part of provider work.

## Dependency Contract

Runtime dependencies are declared in both `pyproject.toml` and
`custom_components/pydantic_ai_agent/manifest.json`.

Relevant dependency decisions:

- `pydantic-ai-slim==1.97.0` supplies Pydantic AI core APIs without provider SDK
  extras.
- `anthropic>=0.97.0` and `google-genai>=1.70.0` are declared explicitly for
  native Anthropic and Google Gemini provider modes.
- `fastmcp-slim[client,server]>=3.3.0` is required for remote MCP runtime use
  because Pydantic AI's `MCPToolset` imports FastMCP symbols that require the
  server extra, even when connecting to remote Streamable HTTP servers.
- `markdownify>=1.2` supports WebFetch content conversion.
- The OpenAI SDK is intentionally absent.

## Architecture

### Low-Level Client

`openai_compatible_client.AsyncOpenAICompatible` is a minimal async client with:

- `api_key` and `base_url` fields;
- bearer `Authorization` header generation;
- `url_for()` path joining;
- a `chat.completions.create()` resource compatible with the subset the adapter
  needs.

`ChatCompletionsResource.create()` builds a JSON request body, omits sentinel
values, merges `extra_body`, applies `extra_headers`, and calls
`/chat/completions` using the provided `httpx.AsyncClient`.

Non-streamed responses are validated into Pydantic models in `_types.py`.
Streamed responses return `ChatCompletionStream`, which parses SSE `data:` lines
into `ChatCompletionChunk` models and closes the underlying response stream on
exit.

### Pydantic AI Provider

`OpenAICompatibleProvider` implements `Provider[AsyncOpenAICompatible]`.

Construction modes:

- pass an existing `AsyncOpenAICompatible` client;
- pass `api_key`, optional `base_url`, and optional `http_client`.

When `base_url` is omitted, it defaults to `https://api.openai.com/v1`. When no
HTTP client is supplied, the provider creates one through Pydantic AI's
`create_async_http_client()`. Home Assistant runtime construction instead passes
`get_async_client(hass)` through `provider.py`, so HA owns SSL, proxy, and
connection-pooling behavior.

### Pydantic AI Model

`OpenAICompatibleChatModel` implements Pydantic AI's `Model` contract.

It provides:

- `model_name`, `system`, and `base_url` properties;
- `request()` for non-streamed Chat Completions;
- `request_stream()` for streamed Chat Completions;
- request preparation through Pydantic AI `prepare_request()`;
- error mapping from lightweight client exceptions to `ModelHTTPError` and
  `ModelAPIError`;
- conversion from Pydantic AI model settings to Chat Completions fields.

## Request Mapping

`_message_mapping.py` maps Pydantic AI messages into Chat Completions messages.

Supported request parts:

- `SystemPromptPart` to a system/developer role based on the model profile;
- `UserPromptPart` to user text or multimodal content;
- `ToolReturnPart` to `role: tool` with `tool_call_id` and text content;
- `RetryPromptPart` to user or tool retry content.

Supported response history parts:

- `TextPart` to assistant `content`;
- `ThinkingPart` with `id` of `reasoning` or `reasoning_content` back to the
  same provider-specific assistant field;
- other `ThinkingPart` values to displayable `<think>...</think>` text;
- `ToolCallPart` to assistant `tool_calls`.

Assistant messages with tool calls and no text use `content: ""`. Some
OpenAI-compatible servers reject `content: null`, while the empty string keeps the
assistant message valid without inventing user-visible text.

## Reasoning Metadata

Reasoning-capable OpenAI-compatible providers can require prior assistant
reasoning metadata to be replayed with tool-call follow-up requests. A live
DeepSeek-style endpoint returned a 400 error until the adapter preserved
`reasoning_content` from the assistant tool-call response and included it in the
next request's assistant history message.

Implementation rules:

- Preserve `reasoning` and `reasoning_content` as provider-specific assistant
  fields in history.
- Do not convert those fields into normal assistant `content`.
- Do not expose raw chain-of-thought beyond provider-approved fields already
  returned by the provider and represented as Pydantic AI `ThinkingPart` data.
- Keep mocked regression coverage for tool-result follow-up payloads that include
  reasoning metadata.

## Tool Calling

Pydantic AI `ToolDefinition` objects are mapped to Chat Completions function
tools. Strict schemas are included only when the active OpenAI model profile says
strict tool definitions are supported.

Tool choice mapping:

- string `none`, `auto`, and `required` pass through;
- a one-item list becomes a forced function tool choice;
- output-only structured requests force `required`;
- otherwise tool use defaults to `auto`.

Tool results use Pydantic AI's `ToolReturnPart.model_response_str_and_user_content()`
helper so non-text tool return data is serialized as text for Chat Completions.
Any file content extracted by that helper is appended as a trailing user message.

## Structured Output

AI task structured output supports the Pydantic AI output modes used by the
integration:

- `tool` output through output tools;
- `native` output through Chat Completions `response_format` with JSON schema;
- `prompted` output through `response_format: {"type": "json_object"}` when the
  active model profile supports JSON object output.

Structured output support is provider/model dependent. Real-server tests skip
structured tool-output E2E cases when the configured model rejects that mode with
a controlled validation error.

## Streaming

The low-level client and Pydantic AI adapter support SSE streaming because the
provider validation probe and direct adapter tests use Pydantic AI's streamed
request path.

Streaming implementation details:

- status errors read the streamed response body before raising so provider error
  details are preserved;
- stream entry and iteration errors are mapped through `_map_api_errors()`;
- `ChatCompletionStream.close()` and `ResponseStream.close()` close the SSE line
  iterator and response context;
- Home Assistant plain-conversation runtime consumes Pydantic AI
  `run_stream_events(...)` so visible assistant deltas are emitted live while the
  final `AgentRunResultEvent` is used only for usage and health metrics;
- conversations that can call HA LLM tools, MCP tools, Web fetch, or skills use
  non-streamed requests for provider-compatible tool-result follow-up handling;
- streamed conversation handling does not append final `new_messages()` after
  live deltas, preventing duplicate final assistant text;
- the real-server probe test drains the event loop after validation to avoid
  racing async-generator finalization.

## Error Handling

The lightweight client raises:

- `APIStatusError` for HTTP status failures;
- `APITimeoutError` for `httpx.TimeoutException`;
- `APIConnectionError` for request errors, invalid JSON, and invalid response
  shapes.

The Pydantic AI model maps those to:

- `ModelHTTPError` for HTTP 4xx/5xx status errors;
- `ModelAPIError` for connection, timeout, invalid JSON, and invalid response
  errors.

`config_flow.py` then maps Pydantic AI errors to provider validation errors and
Home Assistant form/repair behavior.
