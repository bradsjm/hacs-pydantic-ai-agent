# OpenAI-Compatible Adapter

This package adapts the in-repo `openai_compatible_client` to Pydantic AI model
interfaces. It supports OpenAI-compatible Chat Completions and Responses APIs
without depending on the OpenAI SDK.

## Public API

- `OpenAICompatibleProvider` wraps an `AsyncOpenAICompatible` client and exposes
  Pydantic AI provider metadata.
- `OpenAICompatibleChatModel` implements Pydantic AI `Model` for
  `/chat/completions`.
- `OpenAICompatibleResponsesModel` implements Pydantic AI `Model` for
  `/responses`.

These names are exported by `__init__.py` and are constructed by the integration
provider factory code.

## Modules

- `_provider.py` - provider wrapper, default base URL, low-level client access,
  and HTTP client replacement hook.
- `_chat_model.py` - non-streamed and streamed Chat Completions requests,
  response processing, tool choice, structured output, and error mapping.
- `_responses_model.py` - non-streamed and streamed Responses requests,
  response processing, reasoning settings, tool choice, and structured output.
- `_message_mapping.py` - Chat Completions message, tool, multimodal, retry, and
  structured-output mapping.
- `_responses_message_mapping.py` - Responses input item, instructions,
  reasoning, function call, multimodal, and structured-output mapping.
- `_streamed_response.py` - Chat Completions SSE chunk to Pydantic AI stream
  event conversion.
- `_responses_streamed_response.py` - Responses SSE event to Pydantic AI stream
  event conversion.
- `_usage.py` - `CompletionUsage` and `ResponseUsage` to `RequestUsage` mapping.

## Data Flow

- Pydantic AI calls `request()` or `request_stream()` on one of the model
  classes.
- The model prepares settings and request parameters through Pydantic AI.
- Message mapping converts Pydantic AI messages into provider payloads.
- The low-level client sends the HTTP request and returns typed response models
  or typed SSE streams.
- The adapter converts responses back into Pydantic AI `ModelResponse` or stream
  events.
- Usage data, finish reasons, provider response ids, provider names, and provider
  URLs are attached to the Pydantic AI response objects.

## Reasoning Preservation

Reasoning metadata is intentionally preserved because some providers require
prior assistant reasoning to be sent back with tool-call follow-up requests.
Chat Completions uses `reasoning` and `reasoning_content` fields. Responses uses
reasoning output items, item ids, encrypted content, summaries, raw reasoning
content, and provider names.

## Unsupported Features

- Native Pydantic AI tools are not supported by these adapters.
- Chat Completions rejects audio and video URL inputs.
- Responses rejects video URL inputs and only accepts uploaded files from the
  same provider system.

## Testing

- `scripts/test -k test_openai_compatible_adapter`
- `scripts/test -k test_provider`

The main adapter tests live in
`tests/components/pydantic_ai_agent/test_openai_compatible_adapter.py`.
