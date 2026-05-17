# Pydantic AI Agent Specification

## Status

This document is the functional and technical specification for the Home
Assistant custom integration in this repository. It also records implementation
direction, so current source, executable config, manifests, and tests remain the
authority when this document describes planned behavior.

Current implementation status: the repository contains the provider/configuration
foundation, parent config flow, conversation, AI task, and MCP server subentry
flows, conversation and AI task entity registration, diagnostics, setup-time
provider validation, repair issues for reconfigurable model validation failures,
Pydantic AI `Agent` runtime execution, Home Assistant LLM tool conversion,
remote Streamable HTTP MCP toolsets, per-agent/per-task WebFetch capability,
local `pydantic-ai-skills` capabilities, ChatLog history conversion, and
Pydantic AI message adaptation.

Known remaining gaps include runtime capability detection beyond validation
probes, translation coverage tests, cleanup/cancellation lifecycle tests, and
deeper edge-case coverage for MCP and skill runtime failures.

## Product Identity

| Field                       | Value                                                 |
| --------------------------- | ----------------------------------------------------- |
| Integration domain          | `pydantic_ai_agent`                                   |
| Display name                | `Pydantic AI Agent`                                   |
| Home Assistant package path | `custom_components/pydantic_ai_agent/`                |
| Primary platform            | `conversation`                                        |
| Distribution target         | HACS custom integration                               |
| Configuration model         | UI config flow with config subentry reconfigure flows |

Legacy names such as `hermes_agent_bridge` are obsolete and must not be used for
new source paths, entity names, constants, documentation, or tests.

## Purpose

`Pydantic AI Agent` provides Home Assistant Assist conversation agents and AI
task data-generation entities backed by Pydantic AI. It allows Home Assistant
users to create one or more provider connections, each with provider credentials
and mode on the parent config entry, and one or more independent Assist agents
with model, prompt, Home Assistant tool access, MCP toolsets, optional WebFetch,
selected skills, and model behavior settings on `conversation` config subentries.

The integration bridges three systems:

1. Home Assistant Assist and conversation entities.
2. Home Assistant `ChatLog` and LLM API tool execution.
3. Pydantic AI agents, providers, model settings, toolsets, and message events.

The implementation should follow the architecture of Home Assistant's official
OpenAI conversation integration where practical: one conversation entity handles
Assist input, asks `ChatLog` to provide LLM data and HA tools, sends the turn to
a Pydantic AI `Agent`, appends assistant/tool/reasoning content back into
`ChatLog`, and returns Home Assistant's conversation result.

## Implementation Research Requirement

Never start from scratch when implementing a Home Assistant behavior. Before
designing or coding each functional area, inspect Home Assistant core components
and established HACS custom integrations for real-world examples. Do not assume
API shape, lifecycle behavior, UX patterns, testing patterns, or edge-case
handling from memory. Learn from existing implementations first, then adapt the
smallest correct pattern for `pydantic_ai_agent`.

For each feature area, implementation work should record the examples inspected
in the relevant PR, issue, commit message, or implementation note. The record
should include the URLs reviewed and the specific pattern copied, adapted, or
rejected.

Research expectations:

- Check Home Assistant core first for official patterns.
- Prefer integrations with high Integration Quality Scale levels when looking for
  lifecycle, config flow, diagnostics, repairs, translations, and testing
  examples.
- Check popular HACS custom integrations for practical custom-integration
  packaging, HACS release, real-user diagnostics, and edge-case handling.
- Verify behavior against the target Home Assistant version used by this repo.
- Prefer source code, tests, config flows, translations, and diagnostics modules
  over blog posts or memory.
- When examples conflict, prefer Home Assistant core patterns unless a custom
  integration demonstrates a necessary HACS-specific constraint.

Suggested Home Assistant core URLs:

| Area                                     | URLs to inspect                                                                                |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Official OpenAI conversation integration | `https://github.com/home-assistant/core/tree/dev/homeassistant/components/openai_conversation` |
| Conversation platform source             | `https://github.com/home-assistant/core/tree/dev/homeassistant/components/conversation`        |
| Conversation entity developer docs       | `https://developers.home-assistant.io/docs/core/entity/conversation/`                          |
| Home Assistant LLM API docs              | `https://developers.home-assistant.io/docs/core/llm/`                                          |
| Home Assistant LLM helper source         | `https://github.com/home-assistant/core/blob/dev/homeassistant/helpers/llm.py`                 |
| Config flow developer docs               | `https://developers.home-assistant.io/docs/config_entries_config_flow_handler/`                |
| Options flow developer docs              | `https://developers.home-assistant.io/docs/config_entries_options_flow_handler/`               |
| Translations developer docs              | `https://developers.home-assistant.io/docs/internationalization/core/`                         |
| Diagnostics developer docs               | `https://developers.home-assistant.io/docs/core/integration-quality-scale/rules/diagnostics/`  |
| Repairs developer docs                   | `https://developers.home-assistant.io/docs/core/platform/repairs/`                             |
| Integration quality scale                | `https://developers.home-assistant.io/docs/core/integration-quality-scale/`                    |
| Home Assistant frontend Assist chat      | `https://github.com/home-assistant/frontend/blob/dev/src/components/ha-assist-chat.ts`         |
| Assist pipeline frontend data            | `https://github.com/home-assistant/frontend/blob/dev/src/data/assist_pipeline.ts`              |

Suggested Home Assistant core components to inspect by feature:

| Feature                          | Components to inspect                                                                            |
| -------------------------------- | ------------------------------------------------------------------------------------------------ |
| Conversation agent and streaming | `openai_conversation`, `conversation`                                                            |
| Config flow UX and validation    | `openai_conversation`, `homekit_controller`, `unifi`, `mqtt`                                     |
| Options flow organization        | `openai_conversation`, `mqtt`, `music_assistant` if present in the target HA version             |
| Diagnostics shape                | `unifi`, `homekit_controller`, `matter`, `mqtt`                                                  |
| Repairs and reauth               | `unifi`, `homekit_controller`, `google`, `nest`                                                  |
| Translation structure            | Any recently updated Platinum or Gold integration with `strings.json` and `translations/en.json` |
| Tests for config entries         | `tests/components/openai_conversation`, `tests/components/conversation`, `tests/components/mqtt` |

Suggested HACS/custom integration URLs for practical examples:

| Area                                                        | URLs to inspect                                               |
| ----------------------------------------------------------- | ------------------------------------------------------------- |
| HACS integration packaging and repository expectations      | `https://github.com/hacs/integration`                         |
| HACS custom integration examples                            | `https://github.com/hacs/default`                             |
| Large custom integration structure                          | `https://github.com/music-assistant/hass-music-assistant`     |
| Config flow and diagnostics in a popular custom integration | `https://github.com/alandtse/alexa_media_player`              |
| Camera/streaming-heavy custom integration patterns          | `https://github.com/blakeblackshear/frigate-hass-integration` |
| Lightweight custom integration patterns                     | `https://github.com/basnijholt/adaptive-lighting`             |
| Custom integration release and HACS metadata examples       | `https://github.com/custom-components/ble_monitor`            |

Suggested Pydantic AI URLs:

| Area                       | URLs to inspect                           |
| -------------------------- | ----------------------------------------- |
| Pydantic AI docs           | `https://ai.pydantic.dev/`                |
| Agent API                  | `https://ai.pydantic.dev/api/agent/`      |
| Agents and streaming       | `https://ai.pydantic.dev/agents/`         |
| Toolsets                   | `https://ai.pydantic.dev/toolsets/`       |
| Thinking/reasoning         | `https://ai.pydantic.dev/thinking/`       |
| OpenAI model/provider docs | `https://ai.pydantic.dev/models/openai/`  |
| Pydantic AI repository     | `https://github.com/pydantic/pydantic-ai` |

Implementation checklist before coding each feature:

1. Identify the closest Home Assistant core implementation.
2. Identify at least one HACS/custom integration example when the feature affects
   HACS packaging, custom integration lifecycle, diagnostics, or real-user UX.
3. Inspect tests for the selected examples.
4. Note the target pattern and any rejected alternatives.
5. Implement the smallest adaptation for this integration.
6. Add or update tests that prove the adapted behavior.

## Goals

- Expose Pydantic AI chat-only agents as selectable Home Assistant Assist agents.
- Support multiple independent integration instances.
- Allow each instance to use its own API key, provider mode, base URL, model,
  instructions, Home Assistant tool access, and advanced model options.
- Append assistant responses into Assist through Home Assistant `ChatLog`; the
  current conversation entity does not advertise streaming.
- Surface high-level tool calls, tool results, and displayable reasoning or
  thinking summaries through Home Assistant `ChatLog` so the Assist UI can show
  details.
- Provide a clear Home Assistant config flow and subentry reconfigure flow with helpful
  translated labels, descriptions, info text, and actionable validation errors.
- Use Home Assistant's LLM API for Home Assistant control tools instead of
  directly calling services/actions from provider tool calls.
- Keep Home Assistant `ChatLog` as the canonical conversation state for a turn.
- Isolate Pydantic AI event-shape handling in a narrow stream adapter.
- Use async-first implementation patterns for every provider call, Home Assistant
  interaction, tool execution, setup task, unload task, script, and test helper.
- Detect provider, model, Home Assistant, and Pydantic AI capabilities at runtime
  rather than relying on brittle hardcoded assumptions.
- Preserve future extension points for additional MCP transports, additional
  providers, diagnostics, and richer tracing.

## Non-Goals

- Do not implement STT or TTS in the MVP.
- Do not implement local/stdio/SSE MCP servers, remote skill registries, RAG,
  vector memory, or custom external tool catalogs in the MVP.
- Do not create a global singleton agent shared across all config entries.
- Do not share memory or conversation history across config entries by default.
- Do not bypass Home Assistant's LLM API for smart-home control.
- Do not expose raw chain-of-thought. Only surface provider-approved reasoning
  summaries or displayable Pydantic AI thinking content.
- Do not add compatibility shims for legacy names or deprecated behavior unless a
  concrete migration need is introduced later.

## User-Facing Behavior

Users can add `Pydantic AI Agent` from the Home Assistant Integrations UI as a
provider/service connection. Each service connection owns one or more
conversation subentries, and each conversation subentry creates one selectable
Assist conversation agent. The initial setup flow creates only the provider
service entry; users add conversation agents and AI task configurations from the
integration subentry actions.

Example instances:

| Instance title              | Provider mode     | Model           | HA tools |
| --------------------------- | ----------------- | --------------- | -------- |
| `Pydantic AI Agent - Home`  | OpenAI            | `gpt-5.1`       | Enabled  |
| `Pydantic AI Agent - Local` | OpenAI-compatible | `llama-3.3-70b` | Disabled |
| `Pydantic AI Agent - Admin` | OpenAI            | `gpt-5.1`       | Enabled  |

Each conversation subentry appears as a separate selectable conversation agent in
Home Assistant Assist configuration. Changing one conversation subentry must not
affect other subentries or provider/service entries.

## Multi-Instance Model

Multiple agents must follow the current Home Assistant core LLM integration
pattern: provider/service credentials live on the parent Home Assistant config
entry, and individual Assist agents live in `conversation` config subentries.

Rationale:

- Home Assistant 2026 core LLM integrations such as OpenAI, Anthropic, and
  Google Generative AI use parent entries plus conversation subentries.
- Credentials, base URL, provider mode, and provider client lifecycle belong to
  the provider/service connection.
- Model, prompt, Home Assistant LLM API access, and mutable Assist-agent behavior
  belong to each conversation subentry.
- Additional parent config entries are still allowed when the user has genuinely
  separate credentials, endpoints, or provider modes.

STT and TTS subentry types remain future extensions. The MVP supports
`conversation` subentries for Assist agents, `ai_task_data` subentries for Home
Assistant AI task data-generation entities, and `mcp_server` subentries for
remote Streamable HTTP MCP servers.

## Functional Requirements

### Configuration

- The integration must expose a UI config flow.
- The config flow must follow Home Assistant UX conventions: minimal required
  setup steps, clear field descriptions, preservation of user input after
  validation errors, and actionable feedback when validation fails.
- The integration must allow multiple parent config entries when provider mode,
  credentials, or endpoint differ.
- The integration should abort duplicate parent entries when the same provider
  mode, credential, and endpoint are already configured.
- Each parent config entry must have a user-visible service title.
- Each parent config entry must store its own credentials and provider settings.
- The parent config flow must collect provider credentials and endpoint settings.
  Model validation happens when creating or reconfiguring conversation and AI
  task subentries, then again during setup-time stored model validation.
- Invalid or expired credentials after setup must trigger reauthentication or a
  repair issue rather than only setting an entity attribute.
- User-facing config, options, abort, progress, warning, info, and error text
  must be provided by templated strings in `translations/en.json`.

### Conversation Entity

- Each `conversation` config subentry must create exactly one
  `ConversationEntity` in the MVP.
- The entity must be selectable as a Home Assistant Assist conversation agent.
- The entity must not advertise streaming support until the implementation can
  stream through Home Assistant's conversation APIs.
- The entity must advertise control capability only when a Home Assistant LLM API
  is configured for that instance.
- The entity unique ID must be stable across option changes.
- The entity display name must come from the conversation subentry agent name.

### Conversation Processing

- The entity must accept Home Assistant conversation input, language, context,
  device information, and conversation ID through Home Assistant's conversation
  platform APIs.
- The entity must request LLM data from `ChatLog` for the selected Home Assistant
  LLM API and prompt configuration.
- The entity must convert the current `ChatLog` messages into Pydantic AI message
  history for the run.
- The entity must call the shared Pydantic AI `Agent` runtime with the selected
  provider, model, model settings, ChatLog-derived message history, Home
  Assistant LLM API tools, selected MCP toolsets, and selected skills.
- The entity must append assistant output back into `ChatLog`.
- The entity must return Home Assistant's conversation result generated from the
  updated `ChatLog`.

### Streaming

- Text content from Pydantic AI must be mapped to Home Assistant assistant
  content.
- Displayable thinking or reasoning summary deltas must be mapped to Home
  Assistant `thinking_content` deltas.
- Tool-call lifecycle events must be mapped to Home Assistant-compatible tool
  call records.
- Tool results must be mapped to Home Assistant-compatible tool result content.
- Request cancellation must stop provider work promptly and avoid appending
  misleading final content.
- If a provider cannot return a specific detail type, the integration must
  degrade by returning what is available and producing a valid final response.

### Home Assistant Tool Use

- Home Assistant smart-home control must go through Home Assistant's LLM API.
- The integration must request the configured LLM API from `ChatLog` or Home
  Assistant's LLM helper APIs.
- The integration must expose Home Assistant LLM API tools to Pydantic AI as
  typed function tools or toolsets.
- Pydantic AI tool calls for Home Assistant control must execute through the
  selected Home Assistant `APIInstance.async_call_tool` path.
- Tool results and tool errors must be reported into the conversation turn.
- Tool execution must preserve Home Assistant's exposed-entity filtering,
  validation, tracing, context, and user permissions.

### Reasoning and Thinking Display

- Raw hidden reasoning or chain-of-thought must never be requested, stored, or
  displayed.
- Displayable reasoning summaries may be shown only when the provider contract
  explicitly exposes them as user-visible summaries.
- Pydantic AI thinking content may be shown only when it is safe and intended for
  display by the selected provider/model behavior.
- Reasoning visibility must be configurable and default to a conservative mode.

### Diagnostics

- The integration must provide diagnostics suitable for HACS troubleshooting.
- Diagnostics must redact sensitive data with Home Assistant's
  `async_redact_data()` helper before returning data to Home Assistant.
- Diagnostics may include provider mode, base URL when it is not secret, model,
  option values, feature flags, configured model setting keys, and safe last error
  details.
- Diagnostics must not expose API keys, auth headers, bearer tokens, cookies,
  passwords, secret/token fields, extra headers, raw prompts/instructions, or
  provider payloads that may contain private Home Assistant state.
- Redaction must be recursive so sensitive keys inside `model_settings`,
  `extra_headers`, `extra_body`, and provider metadata are masked as well.
- Diagnostics must still be structured and concise. Do not include tracebacks or
  noisy internal dumps in normal diagnostics output.

## Configuration Specification

### Config Flow Fields

The config flow should collect only values required to create a working,
independent instance.

| Field              | Required           | Stored in                        | Notes                                                         |
| ------------------ | ------------------ | -------------------------------- | ------------------------------------------------------------- |
| Service name       | Yes                | Config entry title/data          | User-facing name for the provider/service connection.         |
| Provider mode      | Yes                | Data                             | Example: `openai`, `openai_compatible`.                       |
| API key            | Yes                | Data                             | Credential used for provider validation and requests.         |
| Base URL           | Provider-dependent | Data                             | Required for OpenAI-compatible providers; optional otherwise. |
| Logfire token      | No                 | Data                             | Enables optional Logfire tracing for the provider entry.      |
| Skills folder      | No                 | Data                             | Must be `/config/skills` or a subfolder when configured.      |
| Initial agent name | No                 | Conversation subentry title/data | Collected by the conversation subentry flow.                  |
| Model              | No                 | Conversation/AI task subentry    | Entered and validated by subentry flows.                      |
| HA LLM API         | No                 | Conversation subentry data       | Tool access is enabled when this selector has values.         |

Provider-specific required fields belong in the config flow when the instance
cannot be validated or used without them.

Config flow UX requirements:

- Keep the initial step focused on identity and provider connectivity: instance
  name, provider mode, credentials, and base URL when required.
- Validate model access in the `conversation` and `ai_task_data` subentry flows
  by running a lightweight async Pydantic AI probe before saving each subentry.
- Preserve the user's entered values when validation fails so the user can fix a
  specific field without starting over.
- Store validated models on the relevant subentry, not on the parent provider
  entry.
- Use descriptions and placeholders to explain provider mode, base URL format,
  model ID expectations, and where the API key is used.
- Use `translations/en.json` for all form titles, descriptions, fields, errors,
  aborts, and progress/info messages.
- Do not rely on a generic message such as `error_connecting`, `unknown_error`,
  or `cannot_connect` unless the rendered message includes the concrete endpoint,
  provider, low-level error type, useful detail, and the next action.

### Conversation Subentry Reconfigure Fields

The `conversation` config subentry reconfigure flow should hold mutable Assist
agent behavior. Changing these values should update or reload the subentry
without recreating the parent provider/service entry.

| Option                     | Purpose                                                     |
| -------------------------- | ----------------------------------------------------------- |
| Agent name                 | Set the conversation entity display name.                   |
| Model                      | Change model after setup.                                   |
| System prompt/instructions | Customize assistant behavior.                               |
| HA LLM API selection       | Choose which Home Assistant LLM API tools to expose.        |
| MCP server selection       | Choose which configured MCP server toolsets to expose.      |
| WebFetch                   | Allow URL content fetching through Pydantic AI WebFetch.    |
| Skill selection            | Choose discovered local skills to expose as capabilities.   |
| Max iterations             | Bound Pydantic AI request/tool-loop iterations.             |
| Temperature                | Portable generation control where supported.                |
| Thinking                   | Provider-dependent thinking control.                        |
| Max tokens                 | Bound response size where supported.                        |
| Top-p                      | Provider/model-specific generation control where supported. |
| Timeout                    | Bound provider waits.                                       |
| Parallel tool calls        | Request parallel tool calls where supported.                |
| Seed                       | Provider-specific deterministic sampling hint.              |
| Presence penalty           | Provider-specific repetition control.                       |
| Frequency penalty          | Provider-specific frequency control.                        |
| Extra headers              | JSON object of string headers for provider requests.        |
| Extra body                 | JSON object merged into provider request bodies.            |

Subentry reconfigure UX requirements:

- Keep commonly changed behavior easy to find: agent name, model, instructions,
  HA LLM API tool access, and core model settings.
- Put provider-specific tuning behind advanced steps or sections.
- Validate option combinations before saving them when validation can be done
  without an expensive provider call.
- When validation fails, show an actionable translated message and keep the
  user's current inputs available for correction.

### Translation and Message Contract

All user-facing and operator-facing messages must be actionable templated strings
in `translations/en.json`.

This includes:

- Config flow titles, descriptions, field labels, placeholders, errors, aborts,
  and progress/info text.
- Conversation subentry flow titles, descriptions, field labels, errors, aborts,
  and info text.
- Repair issue titles and descriptions when repairs are added.
- User-visible runtime warning and error summaries where Home Assistant exposes
  them through the integration.
- Log messages intended for normal operator troubleshooting.

Translation keys should follow Home Assistant conventions, including keys such
as:

- `config.step.user.title`
- `config.step.user.description`
- `config.step.user.data.*`
- `config.error.*`
- `config.abort.*`
- `config_subentries.*.step.*`

Message templates must include the most useful concrete details available for
the problem. Useful placeholders include:

- `{provider}`
- `{model}`
- `{base_url}`
- `{host}`
- `{port}`
- `{status_code}`
- `{error_type}`
- `{error_message}`
- `{response_body}`
- `{retry_after}`
- `{tool_name}`
- `{tool_call_id}`

Examples:

| Bad message        | Better message                                                                                                                                         |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Error connecting` | `Connection to {base_url} for {provider} was refused: {error_message}. Check that the server is running and the URL is reachable from Home Assistant.` |
| `Invalid model`    | `Provider {provider} rejected model {model}: {error_message}. Choose a model available to this API key or endpoint.`                                   |
| `Tool failed`      | `Home Assistant tool {tool_name} failed for call {tool_call_id}: {error_message}. Check the exposed entity and tool arguments.`                        |

Rules:

- Surface the pertinent lower-level error detail. Do not replace useful provider,
  network, DNS, TLS, HTTP, or schema messages with vague summaries.
- Do not show tracebacks in config flow messages, subentry flow messages,
  diagnostics, or normal warning/error logs.
- Do not dump noisy internal state when one concrete cause is available.
- Prefer one concise user-facing sentence plus one concrete suggested action.
- Preserve useful low-level provider and network detail, but redact or omit
  credentials, prompts, auth headers, bearer tokens, cookies, and sensitive
  provider payload fields before they reach diagnostics, logs, repairs, or normal
  user-facing messages.

### Entity and Device Attributes

Entity attributes are for observability only. They are not the authoritative
configuration surface.

Allowed attributes:

- Provider mode.
- Current model.
- Current HA LLM API ID or tool-enabled status.
- Streaming enabled/supported status.
- Reasoning display enabled/supported status.
- Last safe error category and short message.
- Last request timestamp as an ISO 8601 string.
- Safe request, token, or tool-call counters if useful.

Forbidden attributes:

- API keys.
- Auth headers.
- Bearer tokens.
- Provider payloads containing credentials.
- Raw prompts or instructions by default.
- Full tool arguments or results that may contain private state.
- Any mutable option users are expected to edit as configuration.

## Technical Architecture

### Planned File Layout

```text
custom_components/pydantic_ai_agent/
├── __init__.py
├── manifest.json
├── const.py
├── config_flow.py
├── conversation.py
├── entity.py
├── provider.py
├── history.py
├── ha_toolset.py
├── stream_adapter.py
├── ai_task.py
├── mcp.py
├── skills.py
├── structured_output.py
├── logfire_support.py
├── diagnostics.py
├── repairs.py
└── translations/
    └── en.json
```

Optional future files:

```text
custom_components/pydantic_ai_agent/
├── system_health.py
└── providers.py
```

### Module Responsibilities

- `__init__.py`: Config entry setup/unload, runtime data creation, platform
  forwarding, update listeners, setup-time validation, repairs, and MCP response
  actions.
- `manifest.json`: Home Assistant metadata, HACS version, requirements, and
  config flow declaration.
- `const.py`: Domain, config keys, option keys, defaults, provider IDs, and
  structured-output constants.
- `config_flow.py`: Initial config flow, provider reauth/reconfigure, subentry
  flows, provider validation, MCP discovery, WebFetch selection, skill selection, and
  structured-output configuration.
- `conversation.py`: `ConversationEntity` implementation and Home Assistant
  conversation lifecycle.
- `entity.py`: Shared Pydantic AI `Agent` runtime, model settings handling, Home
  Assistant LLM API tools, MCP toolsets, WebFetch capability, skills
  capabilities, usage limits, and tool loop.
- `provider.py`: Build Pydantic AI `OpenAIChatModel` and `OpenAIProvider`
  instances from provider config.
- `history.py`: Convert Home Assistant `ChatLog` content and attachments into
  Pydantic AI model messages.
- `ha_toolset.py`: Convert Home Assistant LLM API tools into Pydantic AI tools.
- `stream_adapter.py`: Convert Pydantic AI message parts and tool calls into
  Home Assistant `ChatLog` deltas.
- `ai_task.py`: Home Assistant AI task entity implementation for data generation,
  attachment input, and structured validation.
- `mcp.py`: Remote Streamable HTTP MCP server validation, tool discovery/cache
  refresh, allowlist handling, secret redaction, and runtime toolset creation.
- `skills.py`: Local `pydantic-ai-skills` discovery from `/config/skills`,
  selected-skill capability construction, and script execution gating.
- `structured_output.py`: Pydantic AI structured output mode helpers for tool,
  native, and prompted output.
- `logfire_support.py`: Optional Logfire configuration, per-run instrumentation
  metadata, and Logfire conflict repair data.
- `diagnostics.py`: Redacted config entry diagnostics for provider settings,
  subentry summaries, model settings, feature flags, and runtime status.
- `repairs.py`: Model validation repair issue IDs, issue creation, issue
  deletion, and stale issue cleanup.
- `translations/en.json`: Config flow, subentry flow, actionable errors, aborts,
  progress/info text, and operator-facing templates.

### Runtime Data

Each config entry must own an independent runtime container stored on
`entry.runtime_data`.

The runtime container should include:

- The resolved provider configuration for this entry.
- Provider configuration used to create Pydantic AI model instances for each
  request.
- Optional Logfire instrumentation configuration.
- Cached immutable option defaults.
- Cleanup callbacks or context managers for runtime resources.

The implementation must not store shared mutable state in module globals.

### Async, Cleanup, and Error Containment

All implementation paths must be async-first and compatible with Home
Assistant's event loop.

Requirements:

- Do not use blocking network, file, subprocess, or sleep calls inside Home
  Assistant async code paths.
- Use async provider clients, async context managers, and Home Assistant-compatible
  async helpers for provider communication and test scripts.
- Bound all provider calls and tool executions with explicit timeouts or usage
  limits.
- Track background tasks created by a config entry and cancel/await them during
  unload.
- Close provider clients, MCP HTTP clients, Pydantic AI agent contexts, and tool
  resources during config entry unload and reload.
- Contain exceptions at integration boundaries: config flow validation, provider
  calls, message adapters, tool execution, diagnostics, setup, unload, and test
  scripts.
- Convert caught exceptions into typed integration error categories and
  actionable translated messages.
- Do not let large tracebacks, exception groups, provider dumps, or noisy retry
  loops reach normal Home Assistant logs.
- Preserve useful low-level error details without dumping unrelated stack frames
  or internal state.
- Cancellation must be handled explicitly. Do not convert Home Assistant unload or
  request cancellation into misleading provider errors.

Cleanup validation must be part of testing: unloading one config entry must close
that entry's clients/tasks without affecting other entries.

### Config Entry Lifecycle

Setup:

1. Read config entry data and options.
2. Build provider configuration and validate required fields.
3. Create runtime data.
4. Validate configured conversation and AI task subentry models.
5. Configure Logfire when enabled.
6. Forward setup to the `conversation` and `ai_task` platforms.
7. Register an update listener that reloads the entry when needed.

Unload:

1. Unload the `conversation` and `ai_task` platforms.
2. Close provider clients or agent contexts.
3. Remove update listeners.
4. Clear runtime data through Home Assistant's config entry lifecycle.

Reload:

1. Triggered by options changes or reauth completion.
2. Must not change entity unique IDs.
3. Must not affect other config entries.

### Conversation Entity Design

The conversation entity should subclass Home Assistant's conversation entity base
classes used by custom conversation agents for the target Home Assistant version.

Responsibilities:

- Register itself as a conversation agent when added to Home Assistant.
- Unregister itself when removed.
- Expose control features only when Home Assistant tool access is configured.
- Do not mark streaming support until streaming is implemented and enabled.
- Use the conversation subentry agent name as the entity name.
- Derive unique ID from the subentry ID to avoid entity recreation when the
  provider, model, or options change.

### Request Flow

```text
Assist UI / pipeline
        │
        ▼
Home Assistant ConversationEntity
        │
        ▼
ChatLog.async_provide_llm_data(...)
        │
        ├── system prompt
        ├── conversation history
        └── selected Home Assistant LLM API tools
        │
        ▼
Pydantic AI Agent runtime
        │
        ├── configured provider/model
        ├── model settings
        ├── ChatLog-derived message history
        ├── HA LLM API tools
        ├── selected MCP toolsets
        ├── WebFetch capability when enabled
        └── selected skills capabilities
        │
        ▼
Pydantic AI response/new messages
        │
        ▼
entity.py / stream_adapter.py
        │
        ▼
Home Assistant ChatLog deltas
        │
        ▼
conversation result returned to Assist
```

### Chat History Adapter

The integration needs an adapter from Home Assistant `ChatLog` content to
Pydantic AI message history.

Requirements:

- Preserve user messages.
- Preserve assistant messages.
- Preserve tool calls and tool results where Pydantic AI needs them for model
  continuity.
- Preserve provider-native objects only when they are safe and compatible.
- Avoid storing a duplicate long-lived history on the entity.
- Apply zero-cost history trimming only at the Pydantic AI request boundary.
  Home Assistant `ChatLog` remains canonical and unpruned.
- Windowing must preserve the active Pydantic AI run, the configured head
  window, and complete tool call/tool result pairs.
- Reserve LLM summarization for a future explicit memory/history component.

### Message Adapter

The message adapter is a boundary module. It must isolate Pydantic AI message and
event API details from the rest of the integration.

It must map:

| Pydantic AI concept        | Home Assistant output concept         |
| -------------------------- | ------------------------------------- |
| Text content               | Assistant content                     |
| Displayable thinking       | `thinking_content`                    |
| Tool call                  | Tool call placeholder/metadata        |
| Tool call arguments        | Home Assistant `ToolInput` equivalent |
| Tool result                | Tool result content                   |
| Final result               | Final assistant content state         |
| Usage data                 | Trace/diagnostic metadata where safe  |
| Provider error             | User-safe error response and logs     |

The adapter must handle missing provider IDs, provider-specific native events,
cancellation, and incomplete model responses.

### Pydantic AI Usage

The MVP uses the higher-level Pydantic AI `Agent` runtime. Conversation and AI
task requests build an `Agent` with the configured model, output type, model
settings, Home Assistant LLM API tools, selected remote MCP toolsets, selected
WebFetch capability, selected skills capabilities, automatic sliding-window
context management, `max_concurrency=1`, `tool_retries=0`, and
`output_retries=2`. Runtime bounds are enforced with Pydantic AI
`UsageLimits(request_limit=max_iterations)`.

Home Assistant `ChatLog` remains the canonical conversation history for each
conversation turn. The integration converts ChatLog content into Pydantic AI
message history, runs the agent, then appends the agent's new messages back into
ChatLog deltas before returning Home Assistant's conversation result.
Long histories are windowed only inside the Pydantic AI model request hook so
stored Assist history, diagnostics, and result appending continue to use the
full ChatLog-derived sequence.
The hidden default trigger is 100 prior-history messages. When triggered, the
request keeps the first prior-history message and the latest 50 prior-history
messages, expanding the kept range rather than splitting a tool call from its
tool result. Messages created during the active agent run are never trimmed.

AI task requests use the same shared agent runtime. Each AI task subentry selects
one of Pydantic AI's structured output modes: `tool` (default), `native`, or
`prompted`. The final provider response is parsed as JSON and validated against
the Home Assistant task schema before returning the `GenDataTaskResult`. AI task
entities advertise data generation and attachment input support; image generation
is not implemented.

### Provider Support

The MVP should implement explicit provider modes rather than an unbounded generic
provider registry.

Initial provider modes:

| Provider mode       | Purpose                                                             |
| ------------------- | ------------------------------------------------------------------- |
| `openai`            | OpenAI provider through Pydantic AI.                                |
| `openai_compatible` | OpenAI-compatible provider with user-provided base URL and API key. |

Each provider mode must define:

- Required credential fields.
- Whether `base_url` is allowed or required.
- Model ID validation behavior.
- Streaming support status.
- Tool-calling support status.
- Reasoning/thinking support status.
- Supported portable and provider-specific options.

Future provider modes may include Anthropic, Google, local providers, or other
Pydantic AI-supported model backends once their configuration and event semantics
are explicitly specified.

### Capability Detection and Shape Validation

The integration must prefer capability detection over hardcoded assumptions. Home
Assistant, Pydantic AI, model providers, model families, and internet-hosted APIs
change quickly; brittle static assumptions will break the integration.

Requirements:

- Detect provider and model capabilities before enabling streaming, tool calling,
  reasoning/thinking display, response summaries, JSON/schema output, and
  provider-specific options.
- Treat provider mode as a starting point, not proof that every feature is
  available for the selected model or endpoint.
- Validate every external response shape before reading nested fields.
- Validate Pydantic AI event types, part types, tool-call payloads, tool result
  payloads, and provider-native objects at the message adapter boundary.
- Validate Home Assistant `ChatLog`, LLM API, and tool schema assumptions against
  the target Home Assistant version during implementation.
- Fail closed for unsupported capabilities: disable the feature, show an
  actionable message, and keep plain chat working where possible.
- Log the detected capability set for each config entry in debug/test contexts so
  real-server exploration can explain why a feature was enabled or skipped.
- Add real-server tests that prove capability detection for the configured model
  and base URL.

Current implementation note: capability detection is not generally implemented.
Conversation entities do not advertise streaming support. Home Assistant control
is advertised when an LLM API is configured, and structured-output support is
validated for AI task models through the configured Pydantic AI output mode.

Examples:

- Do not assume an OpenAI-compatible endpoint supports tool calling just because
  it accepts OpenAI-shaped requests.
- Do not assume a model supports reasoning summaries because another model from
  the same provider does.
- Do not assume Pydantic AI message or event objects contain complete tool-call
  arguments without validation; validate the final object.
- Do not assume Home Assistant frontend details render every possible delta type;
  only emit deltas supported by the target Home Assistant `ChatLog` contract.

## Home Assistant LLM API Toolset

The Home Assistant toolset adapter must convert selected Home Assistant LLM API
tools into Pydantic AI function tools.

Requirements:

- Build an LLM context from the Home Assistant conversation input.
- Request only the configured Home Assistant LLM API.
- Expose tool names, descriptions, and schemas faithfully.
- Validate tool arguments according to Home Assistant's schema before execution.
- Execute tool calls through Home Assistant's LLM API instance.
- Return JSON-serializable tool results.
- Surface tool errors with the pertinent details and without stack traces.
- Preserve Home Assistant trace events by using the native LLM API execution path.

Security-sensitive behavior:

- Tool access must be explicit and visible in setup/options.
- The default should be conservative if there is uncertainty about provider
  behavior.
- Users must be able to disable Home Assistant tool access for any instance.

## WebFetch

The current implementation can expose Pydantic AI `WebFetch(local=True)` to
conversation agents and AI tasks.

Requirements:

- Store WebFetch enablement on conversation and AI task subentries.
- Default WebFetch to disabled for new and existing subentries.
- Keep WebFetch independent of MCP selection; WebFetch can be enabled without
  selecting any MCP servers.
- Use Pydantic AI's local WebFetch fallback from the `web-fetch` optional extra,
  which uses Pydantic AI's SSRF-protected download path.
- Do not pass Home Assistant credentials, MCP headers, or provider headers to
  WebFetch.
- Do not log fetched URL contents or fetched page content.

## Remote MCP Toolsets

The current implementation supports remote Streamable HTTP MCP server subentries
only. Stdio, SSE, and local command/process MCP servers are not implemented.

Requirements:

- Validate MCP URLs as HTTP or HTTPS URLs.
- Reject credentials embedded in MCP URLs.
- Accept optional JSON HTTP headers and redact sensitive header values in
  diagnostics.
- Discover server tools through FastMCP/Pydantic AI MCP tooling.
- Cache discovered tools and expose response actions for listing and refreshing
  them.
- Require explicit selected servers and at least one allowed tool before exposing
  MCP tools at runtime.
- Prefix runtime tool names so tools from different MCP servers do not collide.
- Close MCP HTTP clients after each agent run.

Registered response actions:

| Action                                | Purpose                                                   |
| ------------------------------------- | --------------------------------------------------------- |
| `pydantic_ai_agent.list_mcp_tools`    | Return cached discovered tools for a config entry/server. |
| `pydantic_ai_agent.refresh_mcp_tools` | Reconnect to configured MCP servers and refresh tools.    |

## Local Skills

The current implementation can expose local `pydantic-ai-skills` capabilities to
conversation agents and AI tasks.

Requirements:

- Discover skills only from `/config/skills` or subfolders of that directory.
- Store provider-level skills folder and script-execution settings on the parent
  config entry.
- Store selected skills on conversation and AI task subentries.
- Exclude `run_skill_script` unless script execution is explicitly enabled on the
  provider entry.
- Clear selected skills from existing subentries when the provider-level skills
  folder or script-execution setting changes.

## Error Handling

Errors must be normalized into typed integration error categories before they are
shown in the Home Assistant UI or normal logs. The category chooses the template;
the original useful details populate template placeholders.

Required error categories:

- Authentication failure.
- Network, DNS, or TLS failure.
- Connection refused.
- Timeout.
- Provider rate limit.
- Invalid or unavailable model.
- Malformed provider response.
- Provider API error.
- Pydantic AI runtime error.
- Tool-call validation failure.
- Home Assistant LLM API tool execution failure.
- Provider interruption or cancellation.

### Config Flow Errors

The config flow should distinguish:

- Invalid credentials.
- Unsupported provider mode.
- Invalid or unavailable model.
- Invalid base URL.
- Network timeout.
- Provider rate limit.
- Unknown provider error.

Every config flow error must map to a specific `translations/en.json` template
with placeholders for the relevant endpoint, provider, model, status code, and
low-level message where available.

### Runtime Errors

Runtime failures should be converted into Home Assistant-friendly conversation
responses and logs.

| Failure               | Behavior                                                                                  |
| --------------------- | ----------------------------------------------------------------------------------------- |
| Credential rejected   | Trigger reauth or repair; tell user configuration needs attention.                        |
| Provider timeout      | Return a concise failure response; log the pertinent timeout details.                     |
| Provider rate limit   | Return a rate-limit response; optionally expose safe attribute/diagnostic.                |
| Tool validation error | Add tool error result to conversation; allow model to recover if possible.                |
| Tool execution error  | Add an actionable tool error result; log the pertinent error details without a traceback. |
| Provider interrupted  | Stop appending content and return a safe partial/failure response.                        |
| Unsupported option    | Fail validation before runtime where possible.                                            |

Runtime errors should preserve the most pertinent lower-level detail in logs and
user-visible summaries without emitting tracebacks or noisy dumps. For example,
`Connection to 1.2.3.4:8000 refused` is useful; `error connecting` is not.

### Logging

Logs must never include:

- Tracebacks in normal operator-facing warnings and errors.
- Repeated noisy retries without a changed cause.
- Generic summaries that hide the actionable lower-level cause.

Debug logs may include config entry ID, provider mode, model name, event type,
exception class, and safe request or response details when useful. Logs must not
include credentials, auth headers, bearer tokens, cookies, raw prompts, or
provider payloads that may contain private Home Assistant state.

## Security and Privacy

The integration runs inside the Home Assistant process and must follow Home
Assistant custom integration security expectations.

Requirements:

- Use async HTTP clients provided by Pydantic AI/provider libraries or
  Home Assistant-compatible async clients.
- Do not use blocking network I/O in async Home Assistant paths.
- Store secrets only in config entry data.
- Respect Home Assistant user context and exposed entities through the LLM API.
- Do not send unnecessary entity or state data to providers.
- Make Home Assistant tool access explicit in user-facing options.
- Document that selected model providers may receive prompts, conversation text,
  exposed entity metadata, and tool results when tool access is enabled.
- Diagnostics, logs, repair issues, and test artifacts must redact or avoid
  credentials, auth headers, bearer tokens, cookies, raw prompts, and sensitive
  provider payload fields.

## Manifest and Packaging Requirements

The integration package must live at:

```text
custom_components/pydantic_ai_agent/
```

`manifest.json` must include:

- `domain`: `pydantic_ai_agent`.
- `name`: `Pydantic AI Agent`.
- `version`: required for HACS custom integrations.
- `config_flow`: `true`.
- `requirements`: pinned or constrained Pydantic AI dependencies compatible with
  the target Home Assistant dependency constraints. For Home Assistant 2026.5.1,
  executable dependencies include
  `pydantic-ai-slim[openai,mcp,web-fetch]==1.97.0`, `logfire==4.33.0`, and
  `pydantic-ai-skills==0.10.0` in both
  `pyproject.toml` and `manifest.json`. These supply the OpenAI-compatible
  provider path, remote MCP support, FastMCP client support through Pydantic AI's
  `mcp` extra, WebFetch support through Pydantic AI's `web-fetch` extra,
  optional Logfire tracing, and local skill capabilities.
- `documentation`: repository documentation URL once known.
- `issue_tracker`: repository issue URL once known.
- `codeowners`: repository owner(s) once known.

The repository-level `hacs.json` should use the display name
`Pydantic AI Agent`, the target Home Assistant version, and an explicit minimum
HACS version.

The integration must include `translations/en.json` with helpful user-facing
strings for the config flow, subentry flows, errors, warnings, aborts, and info
messages before the config flow is considered complete.

## Testing Requirements

Tests should be added under `tests/` and use Home Assistant custom component
test helpers.

Repository-owned scripts are the supported local and CI entrypoints:

```text
scripts/setup
scripts/lint-check
scripts/type-check
scripts/yaml-check
scripts/markdown-check
scripts/test
scripts/check
scripts/format
```

Do not document or add new validation paths that bypass these wrappers unless a
tool cannot reasonably run through them.

Required test coverage:

- Config flow success for each MVP provider mode.
- Config flow credential validation failures.
- Subentry reconfigure flow updates for model settings and HA tool access.
- Reauth flow for credential failure.
- Setup and unload of one config entry.
- Setup and unload of multiple independent config entries.
- Stable entity unique ID across option changes.
- Conversation entity registration and unregistration.
- ChatLog-to-Pydantic message history conversion.
- Pydantic message-to-ChatLog delta conversion.
- Non-streaming conversation response handling.
- Thinking/reasoning summary mapping when supported.
- Tool call and tool result mapping.
- Tool execution through Home Assistant LLM API mocks.
- Provider timeout and error handling.
- Translation coverage for config flow and subentry flow keys.
- Actionable error template rendering with concrete low-level details.
- Diagnostics redaction for credentials, prompts, extra headers, tokens, and
  nested sensitive provider/model settings.
- Repair issue creation and cleanup for reconfigurable stored model validation
  failures.
- Async cleanup and cancellation behavior for setup, unload, reload, provider
  calls, MCP clients, and provider client lifecycles.
- Capability detection and response-shape validation for provider/model features,
  Pydantic AI messages/events, and Home Assistant LLM API/tool data.

Current known coverage gaps:

- Real-server tests, `pytest.mark.real_server`, `.env.example`, and
  `scripts/test-real-server` exist, but they are outside the default test suite
  and require explicit credentials and endpoint configuration.
- Translation coverage and concrete error-template rendering need direct tests.
- Cleanup/cancellation behavior for provider calls, MCP clients, and provider
  lifecycle needs direct tests.
- General provider/model capability detection tests remain future work because
  general capability detection is not implemented.

Redaction tests are part of the current default suite. Tests must assert that
credentials, auth headers, raw prompts, tokens, and nested sensitive settings do
not leak through diagnostics, logs, repair issue data, or user-facing messages.

Repository test rules:

- Test function parameters must have type annotations.
- Prefer concrete Home Assistant test types over `Any`.
- Avoid branching in tests; use parametrization or separate tests.
- Use `pytest.mark.parametrize` when tests share the same body.

### Real-Server Integration Testing

Real-provider tests and scripts must be created early and used often during
development.

Requirements:

- Add scripts under `scripts/` that exercise real Pydantic AI provider
  connectivity, model calls, non-streaming conversation responses, and tool-call
  behavior.
- Add pytest integration tests that can connect to real OpenAI or
  OpenAI-compatible servers through Pydantic AI.
- Load credentials and provider details from a `.env` file only in test and
  script contexts.
- Do not load `.env` from Home Assistant integration runtime code.
- Keep `.env` uncommitted.
- Provide `.env.example` with variable names and placeholder values once scripts
  are added.
- Mark real-server tests explicitly, for example with `pytest -m real_server`.
- Skip real-server tests with an actionable message when required environment
  variables are missing.
- Do not run real-server tests in default CI unless secrets and endpoints are
  explicitly configured for that job.

Required `.env` keys for end-to-end tests and exploration:

```text
OPENAI_API_KEY=
OPENAI_MODEL=
OPENAI_BASE_URL=
```

Key usage rules:

- `OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_BASE_URL` drive both the hosted
  OpenAI path and OpenAI-compatible endpoint exploration.
- `OPENAI_BASE_URL` may point to the default hosted OpenAI endpoint or to an
  OpenAI-compatible server under test.
- `.env` values must be consumed only by scripts and tests. The Home Assistant
  integration runtime must not read `.env` directly.
- `.env` must remain uncommitted. `.env.example` should contain these keys with
  empty or placeholder values once scripts are added.

Real-server validation should cover:

- Credential validation success and failure.
- Base URL connection failures with actionable error details.
- Plain chat completion.
- Non-streaming conversation responses.
- Provider/model rejection details.
- Pydantic AI message/event shape used by `entity.py` and `stream_adapter.py`.
- Home Assistant LLM API tool adapter behavior with mocked HA tools and a real
  provider model call where feasible.
- Capability detection results for the configured provider, model, and base URL.
- Response-shape validation for the actual Pydantic AI messages/events emitted
  by the configured provider and model.

## Acceptance Criteria

The MVP is complete when all of the following are true:

- Home Assistant can load `custom_components/pydantic_ai_agent` as a HACS custom
  integration.
- A user can create two config entries with different provider settings.
- Each config entry can create one or more separate Assist conversation agents.
- Each agent uses its parent provider credentials plus its own model, prompt,
  Home Assistant tool access, MCP selections, WebFetch setting, skill
  selections, and model settings.
- Each agent can answer a plain chat request.
- AI task subentries create AI task entities for data generation and attachment
  input, with structured output validation when Home Assistant provides a schema.
- Remote Streamable HTTP MCP server subentries can discover, list, refresh, and
  expose explicitly allowlisted tools to selected agents/tasks.
- WebFetch can be enabled per conversation agent or AI task without requiring MCP
  selection.
- Conversation responses are surfaced in Assist; streaming text remains future
  work until the entity advertises streaming support.
- Home Assistant LLM API tools can be enabled per instance.
- Tool calls execute through Home Assistant's LLM API, not direct service calls.
- Tool calls and tool results are visible through Assist chat details when Home
  Assistant supports those deltas.
- Displayable reasoning or thinking summaries are surfaced only when explicitly
  supported and configured.
- Unloading or reloading one config entry does not affect another.
- Diagnostics, repair issues, logs, and debug/test artifacts avoid or redact
  credentials, auth headers, bearer tokens, cookies, raw prompts, and sensitive
  provider payload fields.
- Async setup, provider calls, tool execution, reload, unload, and provider
  cleanup do not block the Home Assistant event loop and do not leak large
  tracebacks into normal Home Assistant logs.
- Capability detection gates future streaming, tools, and reasoning features
  instead of hardcoded provider/model assumptions.
- Tests cover multi-instance isolation, message mapping, tool execution,
  config/subentry flows, actionable message templates, and real-server provider
  validation.
- Each implemented feature has a recorded source example review covering the
  Home Assistant core and HACS/custom integration examples that informed it.

## Future Extensions

Future work should be added behind narrow interfaces rather than expanding the
MVP surface area prematurely.

Potential extensions:

- Additional MCP transports beyond remote Streamable HTTP.
- Remote or user-defined skill/tool registries.
- Additional provider modes.
- Additional entity types or subentries under a shared provider account.
- Richer diagnostics and system health.
- Optional persistent memory with explicit privacy controls.
- Provider-specific built-in tools such as web search where Home Assistant UX and
  privacy implications are documented.
