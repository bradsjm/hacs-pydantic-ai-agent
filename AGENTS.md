# Repository Instructions

## Project Overview

- This repository is the `pydantic_ai_agent` Home Assistant custom
  integration, distributed through HACS from
  `custom_components/pydantic_ai_agent`.
- Treat this as Home Assistant integration code, not a standalone Python app.
  Home Assistant owns startup, shutdown, the event loop, registries, entity
  platforms, config entries, repairs, diagnostics, translations, and service
  actions.
- Custom components run inside the same Python process as Home Assistant Core.
  Bugs, blocking I/O, leaked tasks, or unsafe inputs affect the whole HA
  instance.
- Treat files in `docs` folder as product and architecture direction,
  not proof that a feature is implemented. Verify behavior against source before
  documenting it or using it as a constraint.
- Keep HACS and Home Assistant metadata current when behavior changes:
  `manifest.json`, `hacs.json`, translations, service/action strings,
  diagnostics, repairs, README status, and tests must not drift from source.
- Runtime dependencies are pinned in both `pyproject.toml` and `manifest.json`.
- The implemented provider modes are `openai_compatible_completions`,
  `openai_compatible_responses`, `anthropic`, and `google_gemini`.
- Pydantic and Pydantic AI are foundational technologies here: Pydantic AI
  provides `Agent`, model adapters, public tool APIs, toolsets, structured
  output helpers, usage data, and model request/stream behavior.
- Home Assistant patterns still take precedence over generic Pydantic AI or
  library patterns. Use HA async clients, lifecycle hooks, config flows,
  diagnostics, and repair paths before inventing custom workarounds.
- Use the in-repo `OpenAICompatibleChatModel`,
  `OpenAICompatibleResponsesModel`, and `OpenAICompatibleProvider` for
  OpenAI-compatible modes. Do not add or use the OpenAI SDK-backed Pydantic AI
  classes unless explicitly requested.

## Home Assistant Custom Component Rules

- Implement to official Home Assistant integration quality and lifecycle
  patterns, not one-off HACS shortcuts.
- Aim for Silver/Gold Integration Quality Scale behavior when generating code:
  type hints, async I/O, clean unload, clear errors, device info, config-flow
  validation, reauth/reconfigure/repair paths when applicable, diagnostics that
  redact secrets, and tests around user-visible behavior.
- Before coding a feature area, inspect Home Assistant Core components with
  similar behavior and adapt the smallest correct pattern. Start with
  `homeassistant/components/open_router`, `openai_conversation`, `anthropic`,
  `google_generative_ai_conversation`, `conversation`, and `ai_task` as
  applicable then consultant the Home Assistant developers blog <https://developers.home-assistant.io/blog/feed.json> for SDK changes that might impact the design.
- For config entries, subentries, translations, diagnostics, repairs, entities,
  and tests, prefer current Home Assistant Core examples and Platinum/Gold
  integrations over memory or generic Python patterns.
- Custom component development differs from a typical codebase because UI forms,
  translations, `manifest.json`, `hacs.json`, device/entity registries,
  diagnostics, system health, repair issues, and Hassfest/HACS validation are
  first-class product surfaces.
- Do not introduce global singleton agents or shared conversation memory across
  entries. Home Assistant `ChatLog` is the canonical conversation history.
- Store per-entry runtime objects, caches, clients, metrics, and discovered data
  on typed `entry.runtime_data`.
- Use `hass.data[DOMAIN]` only for intentional process-global state that cannot
  belong to one config entry, such as Logfire process-global coordination,
  provider wizard catalog cache, or cross-entry todo workspace lock registries.
- If adding Home Assistant service actions, define the schema and strings and
  register handlers in `async_setup()`, not `async_setup_entry()`.
- Use HA lifecycle helpers for background work and cleanup:
  `hass.async_create_task`, `entry.async_create_background_task`,
  `entry.async_on_unload`, platform unloads, and executor jobs for blocking file
  I/O.
- Do not create ad-hoc event loops, orphan tasks, unmanaged client/session
  globals, module-level mutable runtime state, or `threading.Lock` state for HA
  runtime coordination.
- For network clients, start with HA helper-managed configuration and lifecycle:
  `get_async_client`, `async_get_clientsession`, HA SSL context, and HA cleanup
  hooks.
- If a third-party library must own a per-session client, wrap HA helper
  configuration in the smallest closeable adapter and test that the library's
  context manager closes it.
- Entity description dataclasses should use modern HA patterns such as
  `frozen=True, kw_only=True`. Entity state attributes must be JSON-serializable
  primitives, lists, and mappings only.
- Use `dt_util.now()` or `dt_util.utcnow()` for timestamps. Convert timestamps
  to strings before exposing them in attributes, diagnostics, or tool output.

## Home Assistant Development Server

- If `ha-dev` tools are available, use them to investigate and debug this
  custom integration against the development Home Assistant environment when
  runtime state, registries, config entries, repairs, diagnostics, traces, logs,
  services, or UI-created resources matter. You should assume this is a ephemeral development instance that you can modify (add, remove, change services, devices and entities) to meet the goal needs and any credentials and keys are also ephemeral demo keys and do not need redacting or protecting.
- When using `ha-dev` tools for Home Assistant custom component development,
  load and follow the relevant Home Assistant development skills and references
  from the progressive reference map below.
- Prefer targeted response services over large diagnostics blobs when debugging
  with `ha-dev`: `get_workspace_status` for entry/subentry/runtime inventory,
  `list_model_profiles` for configured provider-owned profiles,
  `get_agent_metrics` for runtime metric snapshots,
  `get_tool_source_status` for cached MCP and Skill tool sources,
  `list_mcp_tools` / `refresh_mcp_tools` for MCP catalogs, and
  `get_agent_run_diagnostics` for latest conversation or AI task run slices.
- For Home Semantic Index validation, call `refresh_home_semantic_index` first,
  then use `trace_home_semantic_resolution`, `plan_home_semantic_control`,
  `get_home_semantic_document`, or `benchmark_home_semantic_resolution` instead
  of executing live controls. `control_home` remains an LLM-only tool.
- `ha-dev` tools may be available even when the development Home Assistant server
  is not running. If a connection/read-only health call fails because the server
  is unavailable, ask the user to start the development Home Assistant server so
  the tools can connect before continuing runtime investigation.

## Pydantic AI And Provider Rules

- Use public Pydantic AI APIs only. For HA LLM tool conversion use documented
  surfaces such as `Tool.from_schema`; do not import private modules such as
  `pydantic_ai._*`.
- Build provider models through `provider.py` and `model_profiles.py` so HA-owned
  credentials, async clients, base URLs, provider headers, and extra body data
  are applied consistently.
- Model profiles are provider-subentry-owned. Runtime refs use
  `<provider_subentry_id>:<model_profile_id>`, and conversations or AI tasks may
  specify primary and fallback profile refs.
- `model_settings()` should only pass supported Pydantic AI settings to models.
  Strip integration-only settings such as max iterations, chat-template kwargs,
  and unsupported provider extra body before provider requests.
- Structured output modes are `tool`, `native`, and `prompted`; use
  `structured_output.py` helpers to convert HA voluptuous schemas and construct
  Pydantic AI output types.
- Preserve provider reasoning metadata in Pydantic AI message history.
  DeepSeek-style OpenAI-compatible endpoints require prior assistant `reasoning`
  or `reasoning_content` fields to be passed back with tool-call follow-up
  requests.
- Classify provider and network failures with typed exceptions such as
  `ModelHTTPError`, `ModelAPIError`, `httpx` errors, `TimeoutError`,
  `OSError.errno`, and `ssl.SSLError`. Do not branch on exception class-name
  strings or localized message text.

## Current Architecture

- `__init__.py` owns setup/unload/remove/migrate, typed `entry.runtime_data`,
  update reloads, setup-time validation of configured subentry models,
  repair issue creation/cleanup, platform forwarding for `conversation`,
  `ai_task`, `sensor`, and `binary_sensor`, and response actions for MCP tool
  listing/refresh.
- `config_flow.py` re-exports the split flow handlers in `config_flows/`.
- `config_flows/provider_flow.py` owns provider subentry setup/reconfigure,
  guided provider/model selection, model discovery, `models.dev` catalog sync,
  model profile management, model settings, pricing preservation, and provider
  probing.
- `config_flows/conversation_flow.py`, `ai_task_flow.py`, `mcp_server_flow.py`,
  and `skill_flow.py` own their corresponding config subentry flows and
  validation.
- `config_flows/common.py` owns shared selectors, schema helpers, model profile
  selection validation, MCP URL/header/tool parsing, Skill selection, todo
  workspace validation, provider data normalization, and model cache helpers.
- `model_profiles.py` resolves enabled provider-owned model profiles, builds
  Pydantic AI model settings, carries pricing, chooses the provider model class,
  and returns max-iteration limits.
- `provider.py` constructs native Anthropic, Gemini Developer API, and in-repo
  OpenAI-compatible models with HA-managed async HTTP clients and provider
  configuration.
- `provider_validation.py` probes configured models through the same Pydantic AI
  request path used at runtime and maps provider, structured-output, and network
  failures to stable translated reasons.
- `conversation.py` registers one Home Assistant conversation entity per valid
  `conversation` subentry, uses HA `ChatLog`, exposes Assist control only when
  `CONF_LLM_HASS_API` is configured, and delegates model execution to
  `entity.py`.
- `ai_task.py` registers one AI task entity per valid `ai_task_data` subentry,
  supports data generation and attachments, validates structured output against
  the HA-provided schema, and optionally uses a todo workspace toolset.
- `entity.py` owns the shared Pydantic AI `Agent` runtime, fallback profile
  execution, HA LLM API tools, MCP toolsets, native Skill capabilities,
  WebFetch, context trimming, thinking/reasoning preservation, Logfire spans,
  runtime metrics, and streaming/non-streaming response handling.
- `ha_toolset.py` converts HA LLM API tools to Pydantic AI tools through
  `Tool.from_schema` and executes them through HA's LLM API instance.
- `ha_todo_tools.py` owns AI Task todo workspace tooling through HA `todo.*`
  services and a process-global weak lock registry keyed by todo entity.
- `history.py` converts HA `ChatLog` messages and attachments to Pydantic AI
  messages and reads attachment files off the event loop.
- `context_management.py` owns zero-cost sliding-window request trimming.
- `structured_output.py` owns HA schema to Pydantic AI structured output mapping.
- `metrics.py`, `sensor.py`, and `binary_sensor.py` own runtime run metrics,
  pricing-derived monetary sensors, provider health, last-run success, and
  config diagnostic entities.
- `mcp.py` owns remote Streamable HTTP MCP validation, discovery, entry-scoped
  caches, runtime allowlist enforcement, and MCP toolset construction.
- `skills.py` owns native no-script Skill capabilities and exposes only
  `list_skills` and `load_skill` tools for selected Skill subentries.
- `diagnostics.py` and `system_health.py` own redacted config/device diagnostics
  and aggregate non-secret system health data.
- `repairs.py` owns model validation and Logfire conflict repair issue helpers.
- `logfire_support.py` owns HA-managed process-global Logfire configuration,
  conflict handling, instrumentation, and safe run-span metadata.
- `openai_compatible_client/` owns the lightweight async Chat Completions and
  Responses HTTP clients plus SSE parsing.
- `openai_compatible_adapter/` owns the Pydantic AI model/provider adapter,
  message mapping, usage mapping, streaming mapping, error mapping, and
  reasoning/thinking metadata preservation for OpenAI-compatible modes.

## Build And Test Commands

- Install or update the development environment: `scripts/setup`.
- Run all local validation: `scripts/check`.
- Lint Python only: `scripts/lint-check`.
- Type check only: `scripts/type-check`.
- YAML validation only: `scripts/yaml-check`.
- Markdown validation only: `scripts/markdown-check`.
- Format before committing: `scripts/format`.
- Run the normal integration test suite: `scripts/test`.
- Run a focused unit-test selection: `scripts/test -k provider_validation`.
- Run live provider integration tests: `scripts/test-provider-integration`.
- Live provider integration tests are opt-in, networked, and serialized with
  `-n 0`; do not fold them into normal unit-test expectations.

## Testing Instructions

- Root `conftest.py` loads `pytest_homeassistant_custom_component`.
- `tests/components/pydantic_ai_agent/conftest.py` autoloads custom integrations
  and initializes the `homeassistant` component for conversation tests.
- Use shared helpers in `tests/components/pydantic_ai_agent/support/` for
  config-entry/subentry/runtime builders, Pydantic AI fake agents/results/streams,
  and voluptuous schema assertions.
- Mock provider probes in flow/setup tests unless the test explicitly covers
  `async_probe_model`. Do not hit real providers in unit tests.
- Live provider tests live in `tests/components/pydantic_ai_agent/integration/`,
  use the `provider_integration` marker, require provider/MCP environment
  variables, and should run through `scripts/test-provider-integration`.
- When writing or modifying tests, annotate all test function parameters and
  prefer concrete types such as `HomeAssistant`, `MockConfigEntry`, and
  `LogCaptureFixture` over `Any`.
- Avoid branching in tests. Split cases or use `pytest.mark.parametrize`; merge
  duplicated test bodies with parametrization.
- When changing config flows, assert flow result types, translated error keys,
  placeholders, created subentry data, and reconfigure behavior.
- When changing runtime behavior, cover metrics, repairs, diagnostics redaction,
  system health counts, entity state attributes, and cleanup/unload when relevant.
- When removing behavior, delete obsolete tests and docs that only validate the
  removed path. Do not preserve compatibility shims without a concrete need.

## Code Style Guidelines

- This repo targets Python `>=3.14.2` and Ruff `py314`; do not add compatibility
  workarounds for older Python versions.
- Python 3.14 lazy annotations mean forward references do not need quoting or
  `from __future__ import annotations` solely for annotations.
- Python 3.14 permits `except TypeA, TypeB:`; do not flag it as invalid syntax in
  this repo.
- When Home Assistant schemas guarantee a key exists, prefer direct access like
  `data[CONF_MODEL]` over masking contract violations with `.get()`.
- Keep changes small and source-grounded. Prefer one local function or direct
  implementation unless a helper is clearly reusable.
- Do not add backward-compatibility code unless there is persisted data, shipped
  behavior, external consumers, or an explicit user requirement.
- Add comments only for non-obvious Home Assistant or Pydantic AI constraints.
  Do not add comments that restate the next line.
- Use Home Assistant-native and Pydantic AI helpers where available, especially
  for async, network, schema, tool, and lifecycle behavior.
- Keep production attributes, diagnostics, events, and service responses JSON
  serializable and secret-safe.
- Do not introduce new permanent documentation or instruction files unless they
  are clearly needed. Update `AGENTS.md`, `README.md`, or existing docs in place
  when possible.

## Security Considerations

- Home Assistant does not sandbox custom integrations. Code here has filesystem,
  memory, network, service-call, and event-loop access inside the user's HA
  process.
- Reject unsafe user input early in config flows using selectors, voluptuous,
  URL/header parsers, allowlists, and stable translated error keys.
- Do not add arbitrary command execution, local MCP processes, filesystem Skill
  execution, runtime package installation, or runtime Git behavior without an
  explicit design and user approval.
- Keep Logfire process-global behavior explicit and HA-owned. Prompt/completion
  content capture must remain controlled by the entry setting.
- Classify external failures with typed exceptions and safe user-facing reasons.
  Avoid leaking provider response bodies or request metadata into errors.

## Documentation And Metadata

- Documentation must be generated from current source, not memory or planned
  spec text.
- Update README, translations, diagnostics, repairs, services/actions, sensors,
  binary sensors, `manifest.json`, and `hacs.json` when behavior changes.
- Keep release metadata aligned: `manifest.json` version, `pyproject.toml`
  version, `CHANGELOG.md`, and matching GitHub release tag when releasing.

## Progressive Reference Map

- Load the smallest relevant reference before changing a Home Assistant area;
  do not rely on generic Python or stale model knowledge for HA-specific code.
- For component layout, manifest fields, typed runtime data, setup/unload, and
  config-entry lifecycle, read
  `.agents/skills/ha-integration-dev/references/architecture.md`.
- For UI flows, selectors, validation, reauth/reconfigure, translations, and
  config-flow tests, read
  `.agents/skills/ha-integration-dev/references/config-flow.md`.
- For workspace/provider/conversation/AI task/MCP/Skill child resources, read
  `.agents/skills/ha-integration-dev/references/subentries.md`.
- For Assist, `ConversationEntity`, `ChatLog`, LLM API tools, and voice-agent
  behavior, read
  `.agents/skills/ha-integration-dev/references/conversation-agent.md`.
- For diagnostics, device diagnostics, system health, and redaction boundaries,
  read `.agents/skills/ha-integration-dev/references/diagnostics.md`.
- For unsandboxed custom-integration risk, credential handling, input validation,
  command-execution avoidance, dependency vetting, and safe errors, read
  `.agents/skills/ha-integration-dev/references/security.md`.
- External references: Home Assistant developer docs
  <https://developers.home-assistant.io/> and Integration Quality Scale
  <https://developers.home-assistant.io/docs/core/integration-quality-scale/>.
- HACS integration blueprint patterns are available at
  <https://github.com/jpawlowski/hacs.integration_blueprint>.
- OpenRouter core integration reference:
  <https://github.com/home-assistant/core/tree/dev/homeassistant/components/open_router>.

## Workflow Constraints

- Pre-commit runs Ruff check, Ruff format, JSON/YAML checks, yamllint,
  markdownlint, end-of-file fixing, and trailing-whitespace cleanup.
- CI also runs HACS validation and Hassfest. Treat manifest, translations,
  services/actions, and integration structure as validation surfaces, not just
  Python tests.
- Do not amend, squash, or rebase commits that have already been pushed to a PR
  branch; reviewers need incremental history.
