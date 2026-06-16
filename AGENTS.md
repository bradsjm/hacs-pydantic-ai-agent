# Repository Instructions

## Project Overview

- This repository is the `pydantic_ai_agent` Home Assistant custom integration,
  shipped through HACS from `custom_components/pydantic_ai_agent`.
- Treat this as Home Assistant integration code, not a standalone Python app.
  Home Assistant owns lifecycle, config entries, registries, entity platforms,
  repairs, diagnostics, translations, and service actions.
- Custom components run inside the Home Assistant process. Avoid blocking I/O,
  leaked tasks, unsafe inputs, and other behavior that can destabilize HA.
- Treat `docs/` as product direction, not proof that a feature exists. Verify
  behavior against source.
- Keep behavior and metadata aligned across `manifest.json`, `hacs.json`,
  translations, diagnostics, repairs, README status, and tests.
- Runtime dependencies are pinned in both `pyproject.toml` and
  `manifest.json`.
- Supported provider modes are `openai_compatible_completions`,
  `openai_compatible_responses`, `anthropic`, and `google_gemini`.
- For OpenAI-compatible modes, use the in-repo
  `OpenAICompatibleChatModel`, `OpenAICompatibleResponsesModel`, and
  `OpenAICompatibleProvider` unless explicitly told otherwise.

## Working Rules

- Follow current Home Assistant Core patterns. Start from the smallest relevant
  example in `open_router`, `openai_conversation`, `anthropic`,
  `google_generative_ai_conversation`, `conversation`, or `ai_task`.
- Aim for current Home Assistant quality standards: async I/O, clean unload,
  typed runtime data, clear errors, redacted diagnostics, and tests for
  user-visible behavior.
- Store per-entry runtime state on typed `entry.runtime_data`. Use
  `hass.data[DOMAIN]` only for intentional process-global coordination.
- Do not introduce global singleton agents or shared conversation memory across
  entries. `ChatLog` is the canonical conversation history.
- Register Home Assistant service actions in `async_setup()`, not
  `async_setup_entry()`.
- Use HA lifecycle and network helpers first: `hass.async_create_task`,
  `entry.async_create_background_task`, `entry.async_on_unload`,
  `get_async_client`, `async_get_clientsession`, and executor jobs for blocking
  file I/O.
- Do not create ad-hoc event loops, orphan tasks, unmanaged client globals,
  module-level mutable runtime state, or `threading.Lock`-based coordination.
- Keep exposed attributes, diagnostics, events, and service responses JSON
  serializable. Use `dt_util.now()` or `dt_util.utcnow()` for timestamps and
  convert exposed timestamps to strings.
- Use public Pydantic AI APIs only. For HA LLM tool conversion, use documented
  surfaces such as `Tool.from_schema`; do not import `pydantic_ai._*`.
- Build provider models through `provider.py` and `model_profiles.py` so HA-owned
  credentials, async clients, base URLs, headers, and extra body data stay
  consistent.
- Model profiles are owned by provider subentries. Runtime refs use
  `<provider_subentry_id>:<model_profile_id>`.
- `model_settings()` must pass only supported Pydantic AI settings. Strip
  integration-only settings before provider requests.
- Structured output modes are `tool`, `native`, and `prompted`; use
  `structured_output.py` helpers for HA schema conversion.
- Preserve provider reasoning metadata in message history for tool-call
  follow-up flows.
- Classify provider and network failures with typed exceptions such as
  `ModelHTTPError`, `ModelAPIError`, `httpx` errors, `TimeoutError`,
  `OSError.errno`, and `ssl.SSLError`. Do not branch on exception names or
  message text.
- Reject unsafe user input early in config flows. Do not add arbitrary command
  execution, local MCP processes, runtime package installation, or runtime Git
  behavior without explicit design and approval.
- This repo targets Python `>=3.14.2` and Ruff `py314`; do not add
  compatibility workarounds for older Python.
- When Home Assistant schemas guarantee a key exists, prefer direct access like
  `data[CONF_MODEL]` over masking contract violations with `.get()`.
- Keep Logfire process-global behavior explicit and HA-owned. Prompt or
  completion capture must remain controlled by the entry setting.
- Keep changes small and source-grounded. Do not add compatibility shims,
  fallback paths, or comments that merely restate the code.

## Architecture Map

- `custom_components/pydantic_ai_agent/__init__.py` owns setup, unload,
  migration, runtime data, repair coordination, platform forwarding, and shared
  response services.
- `custom_components/pydantic_ai_agent/config_flow.py` and
  `custom_components/pydantic_ai_agent/config_flows/` own provider,
  conversation, AI task, and skill subentry flows.
- `custom_components/pydantic_ai_agent/provider.py`,
  `custom_components/pydantic_ai_agent/model_profiles.py`,
  `custom_components/pydantic_ai_agent/provider_validation.py`, and
  `custom_components/pydantic_ai_agent/structured_output.py` own provider model
  construction, settings, probing, and structured output.
- `custom_components/pydantic_ai_agent/entity.py`,
  `custom_components/pydantic_ai_agent/history.py`,
  `custom_components/pydantic_ai_agent/context_management.py`,
  `custom_components/pydantic_ai_agent/ha_toolset.py`,
  `custom_components/pydantic_ai_agent/ha_todo_tools.py`, and
  `custom_components/pydantic_ai_agent/skills.py` own the shared agent runtime,
  history conversion, context trimming, HA tools, todo tools, and native skills.
- `custom_components/pydantic_ai_agent/conversation.py` and
  `custom_components/pydantic_ai_agent/ai_task.py` expose the user-facing entity
  platforms.
- `custom_components/pydantic_ai_agent/metrics.py`, `sensor.py`,
  `binary_sensor.py`, `diagnostics.py`, `system_health.py`,
  `repair_issues.py`, and `logfire_support.py` own observability,
  diagnostics, repairs, and Logfire support.
- `custom_components/pydantic_ai_agent/openai_compatible_client/` and
  `custom_components/pydantic_ai_agent/openai_compatible_adapter/` own the
  lightweight OpenAI-compatible HTTP client and Pydantic AI adapter layers.

## Development And Validation

- Set up with `scripts/setup` and run full validation with `scripts/check`.
- Use focused commands while iterating: `scripts/lint-check`,
  `scripts/type-check`, `scripts/yaml-check`, `scripts/markdown-check`,
  `scripts/format`, and `scripts/test -k <pattern>`.
- Run the standard test suite with `scripts/test`.
- Live provider tests are opt-in, networked, and live in
  `tests/components/pydantic_ai_agent/integration/`; run them with
  `scripts/test-provider-integration`.
- Root `conftest.py` loads `pytest_homeassistant_custom_component`, and shared
  helpers live under `tests/components/pydantic_ai_agent/support/`.
- Mock provider model-list responses only when the test explicitly covers
  provider discovery behavior.
- Annotate test function parameters with concrete types where practical; avoid
  `Any` in tests when a real fixture type is available.
- Prefer parametrized tests over branching and duplicate test bodies.
- Assert user-visible behavior, stable error keys, persisted data, emitted
  events, diagnostics redaction, and cleanup over private wiring, mock call
  choreography, import paths, or exact English copy.
- For config flows, assert result types, translated error keys, placeholders,
  created subentry data, and reconfigure behavior; do not over-specify selector
  structure, field ordering, or incidental UI text.
- For runtime changes, cover metrics, repairs, diagnostics redaction, system
  health, entity attributes, and unload/cleanup when relevant.
- For diagnostics, tracing, and observability tests, assert stable metadata,
  redaction, classification, and presence of key attributes rather than prose,
  span titles, or incidental ordering.
- Avoid exact English string assertions for UI copy, translated errors,
  diagnostics prose, and provider-facing text when a stable reason key,
  placeholder, classification, or behavior can be asserted instead.
- Prefer shared builders, fixtures, and fakes from
  `tests/components/pydantic_ai_agent/support/` over ad-hoc stubs,
  `SimpleNamespace`, or sentinel objects when they reduce coupling to internals.
- When removing behavior, delete obsolete tests and docs in the same change.
- Pre-commit runs Ruff, JSON/YAML checks, `yamllint`, `markdownlint`, and
  whitespace fixers. CI also runs HACS validation and Hassfest.

## References

- Use the repo skills when they fit the task, especially
  `home-assistant-custom-component-dev`, `building-pydantic-ai-agents`,
  `pydantic-ai-model-integration`, `pydantic-ai-tool-system`, and
  `pydantic-ai-testing`.
- Primary external references:
  <https://developers.home-assistant.io/>,
  <https://developers.home-assistant.io/docs/core/integration-quality-scale/>,
  and <https://github.com/home-assistant/core/tree/dev/homeassistant/components/open_router>.

## Tests

- Tests must only be used for meaningful behavioral verification, asserting the expected **critical path** behavior of the component, including edge cases and error conditions.
- Tests must not be used for non-functional behavior (e.g., UI copy, translated errors, diagnostics prose, or provider-facing text).
- Avoid overly brittle tests that depend on implementation details or specific error messages.
- Reserve mocking for external dependencies (databases, networks, third-party APIs) that are slow, non-deterministic, or outside your control.
- Never create tests that pass simply because the mock is configured to return expected values.

## Workflow Constraints

- Treat manifests, translations, services/actions, diagnostics, and tests as
  first-class validation surfaces.
- Do not amend, squash, or rebase commits that have already been pushed to a PR
  branch.
