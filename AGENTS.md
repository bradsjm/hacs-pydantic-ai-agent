# Repository Instructions

## Project Shape

- This is the `pydantic_ai_agent` Home Assistant custom integration, distributed through HACS from `custom_components/pydantic_ai_agent`.
- Treat `docs/pydantic_ai_agent_spec.md` as the product/architecture direction, not proof that a feature is implemented. Verify against source before documenting or relying on behavior.
- Runtime dependencies are pinned in both `pyproject.toml` and `manifest.json`: `pydantic-ai-slim[openai,mcp]==1.97.0`, `fastmcp==3.3.1`, `logfire==4.33.0`, and `pydantic-ai-skills==0.10.0`; use `OpenAIChatModel`/`OpenAIProvider`, not deprecated aliases.

## Home Assistant Patterns

- Although this is a custom component, implement to official Home Assistant integration quality and lifecycle patterns, not one-off HACS shortcuts.
- Aim for Silver/Gold Integration Quality Scale when generating code: type hints, async I/O, clear error handling, device info, config-flow validation, reauth/repair paths when applicable, and diagnostics that redact secrets with `async_redact_data()`.
- Before coding a feature area, inspect Home Assistant core components with similar functionality and adapt the smallest correct pattern. For this repo, start with `homeassistant/components/open_router`, `openai_conversation`, `anthropic`, `google_generative_ai_conversation`, `conversation`, and `ai_task` as applicable.
- For config entries, subentries, translations, diagnostics, repairs, entities, and tests, prefer current Home Assistant core examples and Platinum/Gold integrations over memory or generic Python patterns.
- If adding Home Assistant service actions, define the schema/strings and register handlers in `async_setup()`, not `async_setup_entry()`.
- Provider credentials belong on the parent config entry; per-agent settings belong in `conversation` config subentries. Do not introduce global singleton agents or shared conversation memory across entries.
- Keep HACS/Core metadata current when behavior changes: `manifest.json`, `hacs.json`, translations, diagnostics, repairs, and README status should not drift from source.

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
- `conversation.py` registers one Home Assistant conversation entity per `conversation` subentry, does not advertise streaming, and advertises `CONTROL` only when `CONF_LLM_HASS_API` is configured.
- `ai_task.py` registers one Home Assistant AI task entity per `ai_task_data` subentry for data generation and attachment input; image generation is not implemented.
- `entity.py` owns the shared Pydantic AI `Agent` runtime, Home Assistant LLM API tool conversion, remote MCP toolsets, selected `pydantic-ai-skills` capabilities, usage limits, and cleanup of MCP HTTP clients.
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
- Do not create new permanent docs or instruction files unless they are clearly needed; update `AGENTS.md`, `README.md`, or `docs/pydantic_ai_agent_spec.md` in place when possible.
- Do not amend, squash, or rebase commits that have already been pushed to a PR branch; reviewers need incremental history.

## References

- HACS integration blueprint patterns: <https://github.com/jpawlowski/hacs.integration_blueprint>
- Home Assistant Integration Quality Scale: <https://developers.home-assistant.io/docs/core/integration-quality-scale/>
- Home Assistant developer docs: <https://developers.home-assistant.io/>
- OpenRouter core integration reference: <https://github.com/home-assistant/core/tree/dev/homeassistant/components/open_router>
