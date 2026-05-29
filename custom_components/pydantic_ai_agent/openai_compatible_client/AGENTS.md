# OpenAI-Compatible Client Agent Instructions

## Scope

These instructions apply to
`custom_components/pydantic_ai_agent/openai_compatible_client`.

## Agent Focus

- Treat this package as a lightweight async HTTP client, not a Pydantic AI
  adapter.
- Keep it independent from Home Assistant and from the OpenAI SDK.
- Preserve the small OpenAI SDK-like surface used by `openai_compatible_adapter`
  and `provider.py`.
- Keep request serialization, SSE parsing, response models, and exceptions
  minimal and predictable.

## Read First

- `__init__.py` - public exports.
- `_client.py` - `AsyncOpenAICompatible`, base URL handling, auth headers, and
  resource wiring.
- `_chat.py` - Chat Completions payload serialization and request handling.
- `_responses.py` - Responses API payload serialization and request handling.
- `_models.py` - `/models` listing.
- `_streaming.py` - typed SSE stream parser and HTTP status handling.
- `_types.py` - permissive Pydantic response models.
- `_exceptions.py` - lightweight client exception hierarchy.
- `_sentinels.py` - omitted-value sentinels.

## Invariants

- Do not add an `openai` package dependency.
- Do not create or own an event loop. The caller supplies the `httpx.AsyncClient`.
- Do not add Home Assistant imports here. HA lifecycle belongs in provider
  construction code outside this package.
- Preserve `None` values during request serialization. Strip only `NOT_GIVEN`,
  `NotGiven`, `omit`, and `Omit` sentinels.
- `AsyncOpenAICompatible.base_url` is normalized by stripping a trailing slash;
  resource paths are joined through `url_for()`.
- Auth headers use a bearer token when `api_key` is present and merge configured
  headers afterward.
- Streaming responses are lazy async context managers. They must call
  `raise_for_status()` after entering and close the underlying HTTP stream.
- SSE parsing currently consumes one `data:` JSON object per line and stops on
  `[DONE]`.
- Response Pydantic models use `extra="allow"` so provider-specific fields are
  preserved for adapter code.
- The client does not retry. Retry, backoff, and provider selection decisions
  belong to callers.

## High-Risk Changes

- Changing sentinel serialization can send unsupported keys to providers or drop
  intentional `null` values.
- Changing exception types can break provider validation and Pydantic AI adapter
  error mapping.
- Changing stream close behavior can leak HTTP connections in Home Assistant.
- Tightening `_types.py` models can drop provider-specific fields used by the
  adapter for reasoning, refusal, or usage details.
- Adding request logging here can expose API keys, headers, prompts, or provider
  response bodies.

## Validation

- Run `scripts/test -k test_openai_compatible_client` for client changes.
- Run `scripts/test -k test_openai_compatible_adapter` when response models,
  streaming, sentinels, or exceptions change.
- Run `scripts/type-check` for public type or overload changes.
- Run `scripts/lint-check` when changing imports or exports.
