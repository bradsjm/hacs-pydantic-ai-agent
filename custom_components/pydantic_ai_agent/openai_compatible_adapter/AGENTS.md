# OpenAI-Compatible Adapter Agent Instructions

## Scope

These instructions apply to
`custom_components/pydantic_ai_agent/openai_compatible_adapter`.

## Agent Focus

- Treat this package as the Pydantic AI adapter boundary for the in-repo
  lightweight OpenAI-compatible client.
- Do not import or wrap the OpenAI SDK here.
- Use public Pydantic AI APIs only.
- Preserve both API modes: Chat Completions and Responses.
- Keep message mapping, streaming events, usage mapping, and error mapping in
  sync with tests and runtime provider construction.

## Read First

- `_provider.py` - `OpenAICompatibleProvider` and low-level client ownership.
- `_chat_model.py` - Chat Completions `Model` implementation.
- `_responses_model.py` - Responses API `Model` implementation.
- `_message_mapping.py` - Pydantic AI message to Chat Completions payload
  mapping.
- `_responses_message_mapping.py` - Pydantic AI message to Responses input
  mapping.
- `_streamed_response.py` - Chat Completions stream mapping.
- `_responses_streamed_response.py` - Responses stream event mapping.
- `_usage.py` - provider usage to `RequestUsage` conversion.

## Invariants

- `OpenAICompatibleChatModel` uses `client.chat.completions.create()`.
- `OpenAICompatibleResponsesModel` uses `client.responses.create()`.
- Both models share `OpenAICompatibleProvider`, but their payload and stream
  formats are intentionally separate.
- Always call `check_allow_model_requests()` and `prepare_request()` in model
  request paths.
- Native Pydantic AI tools are unsupported and must raise
  `UnexpectedModelBehavior` unless a full design changes that contract.
- Tool definitions and structured-output schemas must honor the OpenAI model
  profile strict-tool support flag.
- Preserve provider reasoning metadata. Chat history must keep `reasoning` and
  `reasoning_content`; Responses history must keep item ids, encrypted content,
  summaries, raw reasoning content, and provider names where available.
- Preserve function-call ids. Responses tool-call ids may combine call id and
  item id with `|` and must round-trip through `_split_combined_tool_call_id()`.
- Map low-level client exceptions through `_map_api_errors()` so Pydantic AI sees
  `ModelHTTPError` or `ModelAPIError`.
- Stream implementations must close the underlying client stream and emit
  Pydantic AI `ModelResponseStreamEvent` parts without dropping usage.

## High-Risk Changes

- Thinking and reasoning changes can break provider follow-up requests after
  tool calls. Keep non-streamed and streamed behavior aligned.
- Multimodal mapping differs by API shape. Chat Completions uses `image_url` and
  file payloads; Responses uses `input_image`, `input_file`, and instructions.
- Prompted JSON output for Responses inserts an input system message and omits
  `instructions`. Keep this behavior unless Pydantic AI or provider behavior
  changes.
- Finish reason mapping affects metrics, diagnostics, and agent control flow.
- The adapter allows provider-specific response fields through the client models;
  downstream code must tolerate unknown extras.

## Validation

- Run `scripts/test -k test_openai_compatible_adapter` for adapter changes.
- Run `scripts/test -k test_provider` when provider construction changes.
- Run `scripts/type-check` for changes to Pydantic AI model signatures.
- Run `scripts/lint-check` when changing imports or public exports.
