# Repository Instructions

## Project Shape

- This is the `pydantic_ai_agent` Home Assistant custom integration, distributed through HACS from `custom_components/pydantic_ai_agent`.
- Treat `docs/pydantic_ai_agent_spec.md` as the product/architecture direction, not proof that a feature is implemented. Verify against source before documenting or relying on behavior.
- Runtime dependencies are pinned in both `pyproject.toml` and `manifest.json`: `pydantic-ai-slim==1.97.0`, `anthropic>=0.97.0`, `google-genai>=1.70.0`, `logfire==4.33.0`, `pydantic-ai-skills==0.10.0`, `tiktoken>=0.12.0`, `fastmcp-slim[client,server]>=3.3.0`, and `markdownify>=1.2`; use the in-repo `OpenAICompatibleChatModel`/`OpenAICompatibleResponsesModel`/`OpenAICompatibleProvider` for OpenAI-compatible modes, not the OpenAI SDK-backed Pydantic AI classes. Do not add the OpenAI SDK dependency unless explicitly requested.
- The implemented provider modes are `openai_compatible_completions`, `openai_compatible_responses`, `anthropic`, and `google_gemini`; when `base_url` is omitted the in-repo OpenAI-compatible provider defaults to `https://api.openai.com/v1` without using the OpenAI SDK. Google Gemini support is Gemini Developer API only, not Vertex AI or Google Cloud IAM.

## Home Assistant Patterns

- Although this is a custom component, implement to official Home Assistant integration quality and lifecycle patterns, not one-off HACS shortcuts.
- Aim for Silver/Gold Integration Quality Scale when generating code: type hints, async I/O, clear error handling, device info, config-flow validation, reauth/repair paths when applicable, and diagnostics that redact secrets with `async_redact_data()`.
- Before coding a feature area, inspect Home Assistant core components with similar functionality and adapt the smallest correct pattern. For this repo, start with `homeassistant/components/open_router`, `openai_conversation`, `anthropic`, `google_generative_ai_conversation`, `conversation`, and `ai_task` as applicable.
- For config entries, subentries, translations, diagnostics, repairs, entities, and tests, prefer current Home Assistant core examples and Platinum/Gold integrations over memory or generic Python patterns.
- If adding Home Assistant service actions, define the schema/strings and register handlers in `async_setup()`, not `async_setup_entry()`.
- Provider credentials belong on the parent config entry; per-agent settings belong in `conversation` config subentries. Do not introduce global singleton agents or shared conversation memory across entries.
- Store per-entry runtime objects, caches, clients, and discovered data on typed `entry.runtime_data`. Use `hass.data[DOMAIN]` only for intentional integration process-global state that cannot belong to one config entry, such as Logfire's process-global configuration.
- Keep HACS/Core metadata current when behavior changes: `manifest.json`, `hacs.json`, translations, diagnostics, repairs, and README status should not drift from source.
- Always use Home Assistant-native and Pydantic AI helpers where available instead of custom code workarounds especially for async or network related activities.
- Always plan to work within the Home Assistant and Pydantic AI ecosystem instead of around it.
- For network clients, start with HA helper-managed configuration and lifecycle (`get_async_client`, `async_get_clientsession`, HA SSL context, HA cleanup hooks). If a third-party library must own a per-session client, wrap the HA helper configuration in the smallest closeable adapter and test that the library's context manager closes it.
- Use HA lifecycle helpers for background work and cleanup (`hass.async_create_task`, `entry.async_create_background_task`, `entry.async_on_unload`, executor jobs for blocking file I/O). Do not create ad-hoc event loops, orphan tasks, or unmanaged client/session globals.
- Keep Logfire and other process-global integrations explicit and HA-owned. Do not add module-level mutable globals or `threading.Lock` state for Home Assistant runtime coordination.
- Use public Pydantic AI APIs only. For HA LLM tool conversion use documented surfaces such as `Tool.from_schema`; do not import private modules such as `pydantic_ai._*`.
- Classify provider and network failures with typed exceptions (`ModelHTTPError`, `ModelAPIError`, `httpx` errors, `OSError.errno`, `ssl.SSLError`, etc.). Do not branch on exception class-name strings or localized message text.
- Use `hass.config.path(...)` for Home Assistant config-relative paths and then enforce the integration's containment rules. For skills, all configured folders must stay under `/config/skills`.
- Do not keep production code, tests, or docs for streaming paths unless streaming is actually wired into runtime behavior.
- Preserve provider reasoning metadata in Pydantic AI message history. DeepSeek-style OpenAI-compatible endpoints require prior assistant `reasoning` / `reasoning_content` fields to be passed back with tool-call follow-up requests.

## MCP Server Rules

- MCP support is remote Streamable HTTP only. Do not add stdio, SSE, local command, or arbitrary process execution MCP support unless explicitly requested and designed through HA lifecycle/security patterns.
- MCP server URLs must not contain username, password, or other URL userinfo. Reject URL credentials during validation; authentication belongs in configured HTTP headers.
- Treat persisted or stale MCP URLs as sensitive. Redact the full `mcp_url` value in diagnostics and logs rather than trying to preserve part of the URL.
- If older stored MCP subentries contain invalid URLs, skip them during duplicate checks and surface controlled validation errors when the user edits or uses that server. Do not let stale invalid data crash unrelated config flows.
- MCP tool discovery caches are scoped to the provider config entry through `entry.runtime_data.mcp_tool_cache`; do not reintroduce domain-global MCP caches in `hass.data`.
- Runtime MCP tools must be explicitly allowlisted per server before they are exposed to an agent.

## Commands

- Install/update dev environment: `scripts/setup`.
- Run all local validation: `scripts/check`.
- Lint only: `scripts/lint-check`.
- Type check only: `scripts/type-check`.
- Format before committing: `scripts/format`.
- Full test suite for this integration: `scripts/test`.
- Focused test example: `scripts/test -k provider_validation`.

## Current Architecture

- `__init__.py` owns setup/unload, typed `entry.runtime_data`, update reloads, setup-time validation of configured subentry models, repair issue creation/cleanup for reconfigurable validation failures, forwarding of both `conversation` and `ai_task` platforms, and response actions for MCP tool listing/refresh.
- `config_flow.py` owns parent provider config, reauth/reconfigure, `conversation`, `ai_task_data`, and `mcp_server` subentry flows, model settings parsing, skill selection, MCP validation/discovery, structured output configuration, and the async Pydantic AI provider probe.
- `conversation.py` registers one Home Assistant conversation entity per `conversation` subentry, advertises streaming only for tool-free conversations, and advertises `CONTROL` only when `CONF_LLM_HASS_API` is configured.
- `ai_task.py` registers one Home Assistant AI task entity per `ai_task_data` subentry for data generation and attachment input; image generation is not implemented.
- `entity.py` owns the shared Pydantic AI `Agent` runtime, Home Assistant LLM API tool conversion, remote MCP toolsets, selected `pydantic-ai-skills` capabilities, usage limits, live conversation streaming through `run_stream_events()`, and cleanup of MCP HTTP clients.
- `openai_compatible_client/` owns the lightweight async Completions/Responses HTTP client and SSE parsers; `openai_compatible_adapter/` owns the Pydantic AI `Model`/`Provider` adapter and message/usage/error mapping for OpenAI-compatible modes; native Anthropic and Google Gemini models are constructed in `provider.py` with Pydantic AI's public provider/model classes and HA-owned credentials.
- `mcp.py` supports remote Streamable HTTP MCP server subentries only; stdio, SSE, and local command MCP servers are not implemented.
- `skills.py` discovers local `pydantic-ai-skills` from `/config/skills` or subfolders and excludes skill script execution unless explicitly enabled on the provider entry.

## Testing

- Root `conftest.py` loads `pytest_homeassistant_custom_component`; `tests/components/pydantic_ai_agent/conftest.py` autoloads custom integrations and initializes the `homeassistant` component for conversation tests.
- Mock provider probes in flow/setup tests unless the test explicitly covers `async_probe_model`; do not hit real providers in unit tests.
- When writing or modifying tests, annotate all test function parameters and prefer concrete types such as `HomeAssistant`, `MockConfigEntry`, and `LogCaptureFixture` over `Any`.
- Avoid branching in tests. Split cases or use `pytest.mark.parametrize`; merge duplicated test bodies with parametrization.

## Python And Style

- This repo targets Python `>=3.14.2` and Ruff `py314`; do not add compatibility workarounds for older Python versions.
- Python 3.14 lazy annotations mean forward references do not need quoting or `from __future__ import annotations` solely for annotations.
- Python 3.14 permits `except TypeA, TypeB:`; do not flag it as invalid syntax in this repo.
- When Home Assistant schemas guarantee a key exists, prefer direct access like `data[CONF_MODEL]` over masking contract violations with `.get()`.
- Do not add comments that restate the next line; comments should explain non-obvious constraints or Home Assistant/Pydantic AI edge cases.

## Workflow Constraints

- Pre-commit runs Ruff check, Ruff format, JSON/YAML checks, yamllint, markdownlint, end-of-file fixing, and trailing-whitespace cleanup.
- CI also runs HACS validation and Hassfest; treat manifest, translations, services, and integration structure as first-class validation surfaces, not just Python tests.
- Do not create new permanent docs or instruction files unless they are clearly needed; update `AGENTS.md`, `README.md` or other existing documents in place when possible.
- Do not amend, squash, or rebase commits that have already been pushed to a PR branch; reviewers need incremental history.

## References

- HACS integration blueprint patterns: <https://github.com/jpawlowski/hacs.integration_blueprint>
- In-repo OpenAI-compatible provider design: `docs/openai_compatible_provider_design.md`
- Home Assistant Integration Quality Scale: <https://developers.home-assistant.io/docs/core/integration-quality-scale/>
- Home Assistant developer docs: <https://developers.home-assistant.io/>
- OpenRouter core integration reference: <https://github.com/home-assistant/core/tree/dev/homeassistant/components/open_router>
