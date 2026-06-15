# Pydantic AI Agent

[![HACS Custom](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/hacs/integration)

Home Assistant custom integration for Assist conversation agents and AI task data
generation backed by Pydantic AI.

## Status

Implemented capabilities include:

- OpenAI-compatible workspace providers using the in-repo
  `OpenAICompatibleChatModel` / `OpenAICompatibleProvider` and Home
  Assistant's shared async HTTP client. The OpenAI SDK is not required.
- Native Anthropic and Google Gemini workspace providers using Pydantic AI's
  provider/model classes with Home Assistant-managed credentials and HTTP
  clients. Google support is for the Gemini Developer API, not Vertex AI or
  Google Cloud IAM.
- Multiple configurable Assist conversation agents per workspace.
- Home Assistant control tools for conversation agents when an LLM API is
  selected for that agent.
- AI task entities for Home Assistant data generation, including attachment
  support and structured output validation.
- Configurable structured output modes for AI tasks: tool, native, and prompted,
  subject to the configured provider/model capabilities.
- Optional Web fetch capability for individual conversation agents and AI tasks,
  disabled by default.
- Native workspace Skill subentries that selected conversation agents and AI
  tasks can list and load as raw guidance. Skills do not run scripts or access
  files.
- Automatic zero-cost sliding-window context trimming before provider requests,
  while keeping Home Assistant `ChatLog` as the canonical history.
- Optional Logfire tracing with Home Assistant metadata.
- Provider reconfiguration, subentry reconfiguration, system health, diagnostics
  redaction, device diagnostics, and repair issues for provider authentication,
  model-validation, and Logfire-token conflicts.

## Installation

This integration requires Home Assistant 2026.5.1 or newer.

### HACS Custom Repository

1. Open HACS in Home Assistant.
2. Go to Integrations, then Custom repositories.
3. Add `https://github.com/bradsjm/hacs-pydantic-ai-agent` as an Integration.
4. Install `Pydantic AI Agent`.
5. Restart Home Assistant.

### Manual

1. Copy `custom_components/pydantic_ai_agent` into your Home Assistant
   `custom_components` directory.
2. Restart Home Assistant.

## Configuration

1. Go to Settings > Devices & services.
2. Add `Pydantic AI Agent`.
3. Create a workspace, then add a provider subentry with its mode and API key.
   Supported modes are OpenAI-compatible Chat Completions,
   OpenAI-compatible Responses, Anthropic, and Google Gemini.
4. Enter a custom base URL when needed. OpenAI-compatible providers default to
   `https://api.openai.com/v1`; Anthropic and Google Gemini use their hosted API
   endpoints when no custom base URL is configured.
5. Add provider, conversation, AI task, and Skill subentries as needed.
6. To expose shared external tools, configure MCP servers in Home Assistant
   Core, expose them through the Home Assistant LLM API you want this
   integration to use, then select that API via `Capabilities`.

Workspace entries own shared Logfire settings. Provider credentials live on
`provider` subentries, while conversation, AI task, and Skill settings remain
on their own subentries. Each provider subentry owns a stable
`model_profiles` map, and conversation/AI task subentries reference profiles
with workspace-local refs shaped like
`<provider_subentry_id>:<model_profile_id>`. Anthropic entries also accept
`anthropic:<model>` identifiers, and Google Gemini entries also accept
`google:<model>` or `google-gla:<model>` identifiers. A prefixed model ID must
match the selected provider mode. Provider credentials are always read from the
provider subentry, not from environment variables.

Model discovery is provider-specific. OpenAI-compatible modes use the
OpenAI-compatible `/models` shape, Anthropic uses Anthropic's model listing API,
and Google Gemini lists Gemini models that support `generateContent`. If model
listing fails or a provider omits a model, the provider flow still accepts
manual model entry and stores the selected provider-owned model profiles without
live model probing.

### Conversation Agents

Add a `Conversation agent` subentry for each Assist agent you want to expose.
Each subentry creates a distinct `conversation.*` entity, and that entity ID is
the Home Assistant conversation agent ID.

Conversation agents support:

- A per-agent name, model, and instruction prompt.
- Optional Home Assistant LLM API selection. Selecting an API enables Home
  Assistant control tools and makes the entity advertise conversation control
  support.
- Optional Web fetch URL content fetching. Web fetch is disabled by default and
  can be enabled independently of Home Assistant control tools.
- Optional workspace Skill selection. Selected Skills are exposed through
  `list_skills` and `load_skill` tools and are not auto-loaded into every
  request.
- Optional model settings including temperature, capability-backed thinking, max
  tokens, max iterations, top P, timeout, parallel tool calls, seed, penalties,
  and extra body fields.
- Optional provider HTTP headers configured on the provider entry and used for
  model discovery and model requests.
- Automatic hidden context trimming for very long conversations. Stored Assist
  history is not pruned; only the model request is windowed.
  When prior model history exceeds 100 messages, the request preserves the first
  message and the latest 50 prior-history messages. Messages from the active
  agent run are always preserved.
- Tool-call follow-up requests preserve provider reasoning metadata such as
  `reasoning` and `reasoning_content` for OpenAI-compatible endpoints that
  require it.
- Plain conversation entities stream Assist responses. Conversations that can
  call tools, including HA LLM APIs, Web fetch, or skills, return
  non-streamed responses so provider tool-result follow-up requests stay on the
  compatibility path.

### AI Tasks

Add an `AI task` subentry for each AI task configuration you want to expose.
Each subentry has its own task name and creates an `ai_task.*` entity that
supports Home Assistant data generation and attachments.

AI task entities can return plain text or validate structured results against
the schema requested by Home Assistant. When Home Assistant requests structured
data, runtime chooses the highest supported strategy in the order `tool`,
`native`, then `prompted` from the resolved model profile capabilities. Each AI
task can also enable Web fetch independently of Home Assistant control tool
selection.
AI task requests use the same automatic model-request context trimming as
conversation agents, including active-run preservation.
AI tasks default to 30 agent request iterations unless the selected language model
profile sets a max iterations override.

### Home Assistant LLM APIs and shared tools

This integration no longer manages its own MCP servers. Shared external tools
must be configured in Home Assistant Core and exposed through a Home Assistant
LLM API.

Recommended setup:

1. Configure any MCP servers in Home Assistant Core `mcp`.
2. Expose those servers through the Home Assistant LLM API you want the model
   to use.
3. In this integration, select that API in the conversation agent or AI task
   `Capabilities` section.

### Skills

Add a `Workspace Skill` subentry for reusable model guidance. The Skill content
field uses Home Assistant's template-style editor for a comfortable multiline
editing experience, but the integration stores and sends the text as raw content;
it is not rendered as a Home Assistant template.

Conversation agents and AI tasks can select Skill subentries from the same
workspace. At runtime, selected Skills are exposed through two Pydantic AI tools:
`list_skills` returns Skill names and descriptions, and `load_skill` returns the
raw content for a selected Skill ID. Skill content is user-managed guidance only;
it cannot override system, Home Assistant, developer, or safety instructions.

Skills do not run scripts, clone repositories, auto-update, or read filesystem
folders. Reference attachments are reserved for a later phase and are redacted in
diagnostics.

### Validation And Repairs

The integration validates provider connection settings and selected model
references during config flows, but it does not send live preflight model
requests during setup or AI task save. Authentication failures during
provider/profile flows are surfaced on those flows. Runtime provider credential
or permission failures create provider-scoped repair issues so the provider
connection can be reconfigured without removing the workspace.

Conversation and AI task entities stay loaded during degraded provider setup.
They are marked unavailable when required configured provider or model profile
references can no longer be resolved. Runtime conversation responses stream only
when the agent has no configured tool sources; tool-capable conversations use
non-streamed requests.

Diagnostics redact API keys, Logfire tokens, prompts, sensitive model settings,
provider headers, Skill content, and Skill references.
Runtime diagnostics expose safe workspace/provider state and latest bounded run
snapshots.
Device diagnostics are scoped to the matching agent or AI task subentry and
include safe runtime metrics for that subentry.

System health reports aggregate, non-secret counts for configured and loaded
entries, provider modes, model profiles, conversation agents, AI tasks,
Logfire-enabled entries, workspace Skills, and selected Skill references.

Entity unique IDs use the prefixed format
`pydantic_ai_agent_<entry_id>_<subentry_type>_<subentry_id>` for conversation and
AI task entities, and
`pydantic_ai_agent_<entry_id>_<subentry_type>_<subentry_id>_<metric_key>` for
sensor and binary sensor entities. Registry entries and empty devices for deleted
subentries are cleaned up during workspace setup and entry removal.

### Data Updates And Runtime Behavior

The integration does not poll devices or periodically fetch state. Conversation
and AI task entities make provider requests only when Home Assistant asks them to
handle a conversation turn or generate AI task data. Diagnostic sensors are
updated from in-memory run metrics after each successful or failed run.

### Known Limitations

- Provider credentials live on provider subentries, so credential repair is a
  provider reconfigure/repair workflow rather than a top-level workspace reauth
  flow.
- Google support targets the Gemini Developer API. Vertex AI and Google Cloud IAM
  are not supported.
- Native workspace Skills are static guidance. They do not execute code, read
  files, install packages, or clone repositories.
- Provider quality, model capabilities, tool-calling behavior, and structured
  output support vary by provider and selected model.

### Troubleshooting

- If a provider repair issue appears, reconfigure the provider connection and
  reload the workspace entry after saving.
- If an agent entity is unavailable, check that its configured provider and
  model profile references still exist and inspect runtime diagnostics for the
  latest run failure classification.
- If shared external tools are missing, confirm the selected Home Assistant LLM
  API exposes the expected Home Assistant Core MCP-backed tools.
- If provider requests fail intermittently, check Home Assistant logs for the
  safe failure category and configure fallback model profiles for retryable
  timeout, rate-limit, or provider-server failures.
- If Logfire tracing does not start for one workspace, check for a Logfire token
  conflict repair issue. Home Assistant can only use one active Logfire token in
  the current process.

### Logfire Tracing

Workspace setup includes optional Logfire tracing fields. Leave the Logfire token
blank to disable Logfire for that workspace. When a token is provided,
the integration adds Home Assistant metadata such as entry, subentry, entity,
model, and conversation IDs to Pydantic AI traces.

The `Include prompt and response content in Logfire` option is disabled by
default. Enable it only if you want Logfire to capture prompt, completion, and
tool payload content. Logfire is configured process-wide in Home Assistant: the
first loaded workspace entry with a token wins, later entries with a different
token are left loaded but get a repair warning and do not emit Logfire traces.

## Integration Quality Scale

This custom integration uses the [Home Assistant Integration Quality Scale][iqs]
as an aspirational checklist. The source code is the authority for implemented
behavior; this table documents current alignment and remaining gaps rather than
claiming official Home Assistant certification.

[iqs]: https://developers.home-assistant.io/docs/core/integration-quality-scale/

### Bronze

<!-- markdownlint-disable MD060 -->

| Criteria                                      | Status  | Proof                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| --------------------------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| UI-based setup via config flow                | Done    | [`config_flow.py`](custom_components/pydantic_ai_agent/config_flow.py) — full multi-step workspace, provider, subentry, and reconfigure flows                                                                                                                                                                                                                                                                                                      |
| Form fields have context help text            | Done    | [`translations/en.json`](custom_components/pydantic_ai_agent/translations/en.json) — per-field `data_description` for every form step                                                                                                                                                                                                                                                                                                              |
| Config entry data and options used correctly  | Done    | [`__init__.py`](custom_components/pydantic_ai_agent/__init__.py) — workspace data read from `entry.data`; per-agent settings in subentry data                                                                                                                                                                                                                                                                                                      |
| Full config flow test coverage                | Partial | [`test_workspace_config_flow_smoke.py`](tests/components/pydantic_ai_agent/test_workspace_config_flow_smoke.py), [`test_config_flow_helpers.py`](tests/components/pydantic_ai_agent/test_config_flow_helpers.py), and provider/config reconfigure tests under [`tests/components/pydantic_ai_agent/`](tests/components/pydantic_ai_agent/) — covers workspace/provider creation smoke paths, helper validation, and AI task/conversation form behavior; deeper provider reconfigure coverage remains pending |
| Connection tested before config entry created | Done    | [`config_flow.py`](custom_components/pydantic_ai_agent/config_flow.py) — validates provider connection settings and persists provider-owned model profiles without live model probing                                                                                                                                                                                                                                                               |
| Integration readiness checked during setup    | Done    | [`__init__.py`](custom_components/pydantic_ai_agent/__init__.py) and [`_setup_helpers.py`](custom_components/pydantic_ai_agent/_setup_helpers.py) — resolve configured providers and model profiles without blocking setup on live preflight requests                                                                                                                                                                                             |
| Duplicate config entries prevented            | Done    | [`config_flow.py`](custom_components/pydantic_ai_agent/config_flow.py) — workspace and provider flows prevent duplicate resources                                                                                                                                                                                                                                                                                                                  |
| Every entity has a unique ID                  | Done    | [`entity.py`](custom_components/pydantic_ai_agent/entity.py) [`sensor.py`](custom_components/pydantic_ai_agent/sensor.py) — every entity has a stable `unique_id`                                                                                                                                                                                                                                                                                  |
| Entities use `has_entity_name = True`         | Done    | [`conversation.py`](custom_components/pydantic_ai_agent/conversation.py), [`ai_task.py`](custom_components/pydantic_ai_agent/ai_task.py), [`sensor.py`](custom_components/pydantic_ai_agent/sensor.py), and [`binary_sensor.py`](custom_components/pydantic_ai_agent/binary_sensor.py) preserve `has_entity_name` on entity instances                                                                                                              |
| Runtime data stored on `entry.runtime_data`   | Done    | [`__init__.py`](custom_components/pydantic_ai_agent/__init__.py) — `WorkspaceRuntimeData` dataclass stored on `entry.runtime_data`                                                                                                                                                                                                                                                                                                                 |
| Service actions registered in `async_setup`   | Done    | [`__init__.py`](custom_components/pydantic_ai_agent/__init__.py) — response services are registered in `async_setup`                                                                                                                                                                                                                                                                                                                               |
| Shared logic in common modules                | Done    | [`entity.py`](custom_components/pydantic_ai_agent/entity.py) — shared `PydanticAIBaseLLMEntity` base; [`model_profiles.py`](custom_components/pydantic_ai_agent/model_profiles.py) — profile resolution helpers                                                                                                                                                                                                                                    |
| Dependencies fully declared                   | Done    | [`manifest.json`](custom_components/pydantic_ai_agent/manifest.json) — all runtime dependencies are explicit, no extras or transitive requires                                                                                                                                                                                                                                                                                                     |
| Entity events in correct lifecycle hooks      | Done    | [`sensor.py`](custom_components/pydantic_ai_agent/sensor.py) — `async_added_to_hass` subscribes to dispatcher signals                                                                                                                                                                                                                                                                                                                              |
| Branding assets present                       | Done    | [`brand/icon.png`](custom_components/pydantic_ai_agent/brand/icon.png)                                                                                                                                                                                                                                                                                                                                                                             |

### Silver

| Criteria                                                      | Status  | Proof                                                                                                                                                                                                         |
| ------------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Active integration code owner                                 | Done    | [`manifest.json`](custom_components/pydantic_ai_agent/manifest.json) — `codeowners: ["@bradsjm"]`                                                                                                             |
| Config entry unloading supported                              | Done    | [`__init__.py`](custom_components/pydantic_ai_agent/__init__.py) — `async_unload_entry` unloads platforms; `async_remove_entry` cleans up repair issues                                                       |
| Reauthentication flow available                               | Partial | Provider credentials are subentry-owned; provider auth failures create repair issues that direct the user to provider reconfiguration instead of a top-level workspace reauth flow                            |
| Reconfiguration flow available                                | Done    | [`config_flow.py`](custom_components/pydantic_ai_agent/config_flow.py) — reconfigure step plus per-subentry-type reconfigure flows (provider, conversation, AI task, Skill)                                   |
| Service actions raise exceptions on failure                   | Done    | [`__init__.py`](custom_components/pydantic_ai_agent/__init__.py) and [`debug_services.py`](custom_components/pydantic_ai_agent/debug_services.py) raise translated service errors for invalid service targets |
| Entity marked unavailable when appropriate                    | Done    | [`entity.py`](custom_components/pydantic_ai_agent/entity.py) marks agent entities unavailable when all configured model profiles failed setup validation                                                      |
| Log once when service becomes unavailable and again when back | Done    | [`__init__.py`](custom_components/pydantic_ai_agent/__init__.py) — provider validation failures are logged at warning                                                                                         |
| Parallel update concurrency specified                         | N/A     | No `DataUpdateCoordinator`; agent runs are serialized per entity with `max_concurrency=1`                                                                                                                     |
| Above 95% test coverage                                       | Partial | [`tests/components/pydantic_ai_agent/`](tests/components/pydantic_ai_agent/) — focused coverage exists, but the normal suite currently measures below the Silver target                                       |

### Gold (partial)

The integration meets several Gold-tier rules. Full Gold-style alignment is still
aspirational because coverage and top-level reauth semantics remain partial.

| Criteria                                    | Status  | Proof                                                                                                                                                                                                                                                                       |
| ------------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Devices created for entities                | Done    | [`entity.py`](custom_components/pydantic_ai_agent/entity.py) — `DeviceInfo` with `identifiers`, `manufacturer`, `model`, and `entry_type` for every agent subentry                                                                                                          |
| Entities assigned appropriate categories    | Done    | [`sensor.py`](custom_components/pydantic_ai_agent/sensor.py) — all metric and config sensors use `EntityCategory.DIAGNOSTIC`                                                                                                                                                |
| Entities use device classes where possible  | Done    | [`sensor.py`](custom_components/pydantic_ai_agent/sensor.py) — `SensorDeviceClass.DURATION` and `SensorStateClass` set where applicable                                                                                                                                     |
| Diagnostics implemented                     | Done    | [`diagnostics.py`](custom_components/pydantic_ai_agent/diagnostics.py) — config-entry and device-level diagnostics with comprehensive redaction                                                                                                                             |
| Repair issues for user-visible problems     | Done    | [`repair_issues.py`](custom_components/pydantic_ai_agent/repair_issues.py) — provider auth issues, model-validation issues, process-global Logfire-token-conflict warnings, and automatic issue lifecycle cleanup                                                           |
| Exception messages translatable             | Done    | [`translations/en.json`](custom_components/pydantic_ai_agent/translations/en.json) — per-reason error and abort translations with placeholders                                                                                                                              |
| Entity names translatable                   | Done    | [`translations/en.json`](custom_components/pydantic_ai_agent/translations/en.json), [`sensor.py`](custom_components/pydantic_ai_agent/sensor.py), and [`binary_sensor.py`](custom_components/pydantic_ai_agent/binary_sensor.py) — diagnostic entities use translation keys |
| Stale devices removed                       | Done    | [`__init__.py`](custom_components/pydantic_ai_agent/__init__.py) removes orphaned registry entities and empty subentry devices during setup and entry removal                                                                                                               |
| Devices added after integration setup       | N/A     | Devices are created at subentry load time, not discovered at runtime                                                                                                                                                                                                        |
| Device/service auto-discovery               | N/A     | LLM-based integration; no device or hardware discovery                                                                                                                                                                                                                      |
| Documentation describes supported functions | Done    | README §Configuration details provider, conversation, AI task, Skill, validation, diagnostics, and runtime behavior                                                                                                                                                         |
| Documentation includes troubleshooting      | Done    | README §Troubleshooting                                                                                                                                                                                                                                                     |
| Documentation includes automation examples  | Pending | Examples remain pending                                                                                                                                                                                                                                                     |
| Documentation describes known limitations   | Done    | README §Known Limitations                                                                                                                                                                                                                                                   |
| Documentation describes how data is updated | Done    | README §Data Updates And Runtime Behavior                                                                                                                                                                                                                                   |

<!-- markdownlint-enable MD060 -->

## Development

Use Python 3.14.2 or newer.

Runtime dependencies are declared in both `pyproject.toml` and
`custom_components/pydantic_ai_agent/manifest.json`. The integration uses
`pydantic-ai-slim`, explicit native provider SDK dependencies for Anthropic and
Google Gemini and the in-repo
OpenAI-compatible adapter instead of the OpenAI SDK.

The OpenAI-compatible adapter design is documented in
`docs/openai_compatible_provider_design.md`.

Install or update the development environment, then run the local checks:

```bash
scripts/setup
scripts/check
```

## Support

Issues: <https://github.com/bradsjm/hacs-pydantic-ai-agent/issues>

Code owner: `@bradsjm`
