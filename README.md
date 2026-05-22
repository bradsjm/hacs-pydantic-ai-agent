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
- Optional remote Streamable HTTP MCP server subentries, with tool discovery,
  explicit runtime allowlists, and response services for listing or refreshing
  discovered tools.
- Optional Web fetch capability for individual conversation agents and AI tasks,
  disabled by default.
- Optional local `pydantic-ai-skills` selection from `/config/skills` or a
  subfolder, with script execution disabled unless explicitly enabled on the
  conversation agent or AI task.
- Automatic zero-cost sliding-window context trimming before provider requests,
  while keeping Home Assistant `ChatLog` as the canonical history.
- Optional Logfire tracing with Home Assistant metadata.
- Config entry reauthentication, provider reconfiguration, subentry
  reconfiguration, system health, diagnostics redaction, device diagnostics, and
  repair issues for model-validation and Logfire-token conflicts.

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
5. Add provider, conversation, AI task, and MCP server subentries as needed.

Workspace entries own shared Logfire settings and default skills-folder policy.
Provider credentials now live on `provider` subentries, while conversation and
AI task settings remain on their own subentries. Each provider subentry owns a
stable `model_profiles` map, and conversation/AI task subentries reference
profiles with workspace-local refs shaped like
`<provider_subentry_id>:<model_profile_id>`. Anthropic entries also accept
`anthropic:<model>` identifiers, and Google Gemini entries also accept
`google:<model>` or `google-gla:<model>` identifiers. A prefixed model ID must
match the selected provider mode. Provider credentials are always read from the
provider subentry, not from environment variables.

Model discovery is provider-specific. OpenAI-compatible modes use the
OpenAI-compatible `/models` shape, Anthropic uses Anthropic's model listing API,
and Google Gemini lists Gemini models that support `generateContent`. If model
listing fails or a provider omits a model, the provider flow still accepts
manual model entry and validates the selected profiles with provider probes.

### Conversation Agents

Add a `Conversation agent` subentry for each Assist agent you want to expose.
Each subentry creates a distinct `conversation.*` entity, and that entity ID is
the Home Assistant conversation agent ID.

Conversation agents support:

- A per-agent name, model, and instruction prompt.
- Optional Home Assistant LLM API selection. Selecting an API enables Home
  Assistant control tools and makes the entity advertise conversation control
  support.
- Optional MCP server selection. Selected MCP servers must have at least one
  allowed tool configured before runtime use.
- Optional Web fetch URL content fetching. Web fetch is disabled by default and
  can be enabled without selecting any MCP servers.
- Optional local skill selection from the configured skills folder.
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
  call tools, including HA LLM APIs, MCP servers, Web fetch, or skills, return
  non-streamed responses so provider tool-result follow-up requests stay on the
  compatibility path.

### AI Tasks

Add an `AI task` subentry for each AI task configuration you want to expose.
Each subentry has its own task name and creates an `ai_task.*` entity that
supports Home Assistant data generation and attachments.

AI task entities can return plain text or validate structured results against
the schema requested by Home Assistant. Structured output defaults to tool output
and can be changed to native or prompted output in the advanced AI task settings.
Each AI task can also enable Web fetch independently of MCP server selection.
AI task requests use the same automatic model-request context trimming as
conversation agents, including active-run preservation.
AI tasks default to 30 agent request iterations unless the selected language model
profile sets a max iterations override.

### MCP Servers

Add an `MCP server` subentry to connect a remote Streamable HTTP MCP server.
Stdio and local command MCP servers are not supported by this integration.

MCP server configuration supports:

- HTTP or HTTPS MCP endpoint URLs.
- Optional JSON HTTP headers.
- Optional comma-separated allowed tool names.
- Optional tool return schema inclusion, enabled by default.

Use the `pydantic_ai_agent.list_mcp_tools` action to list cached discovered
tools, or `pydantic_ai_agent.refresh_mcp_tools` to reconnect and refresh tool
catalogs. Both actions require a Pydantic AI Agent config entry ID and can be
limited to one MCP server subentry ID.

MCP tools are available at runtime only when a conversation agent or AI task
selects the server and the server has an explicit allowed-tools list.

### Skills

Conversation agents and AI tasks can select discovered `pydantic-ai-skills`
from `/config/skills` or one of its subfolders.

Skill script execution is disabled by default. Enable `Allow skill script
execution` only for skills you trust; changing the skills folder or script
execution setting clears selected skills from existing agent and task subentries.

### Validation And Repairs

The integration validates configured models with provider test requests when
conversation-agent and AI-task subentries are created or reconfigured, and again
when the workspace entry loads. Authentication failures during provider/profile
flows are surfaced on those flows. Model, permission, provider-configuration, or
streaming-capability failures that can be fixed by reconfiguration are surfaced
as Home Assistant repair issues without preventing unrelated workspace resources
from loading.

Provider validation uses a short streamed Pydantic AI model probe. Runtime
conversation responses stream only when the agent has no configured tool sources;
tool-capable conversations use non-streamed requests.

Diagnostics redact API keys, Logfire tokens, prompts, sensitive model settings,
provider headers, MCP URLs, and MCP headers. Runtime diagnostics expose only safe
workspace/provider counts such as configured MCP server count, cached MCP server
count, and cached tool counts per server. Device diagnostics are scoped to the
matching agent or AI task subentry and include safe runtime metrics for that
subentry.

System health reports aggregate, non-secret counts for configured and loaded
entries, provider modes, model profiles, conversation agents, AI tasks, MCP
servers and caches, Logfire-enabled entries, and skill-script-execution entries.

Entity unique IDs use the breaking prefixed format
`pydantic_ai_agent_<subentry_type>_<subentry_id>` for conversation and AI task
entities, and `pydantic_ai_agent_<subentry_type>_<subentry_id>_<metric_key>` for
sensor and binary sensor entities. No migration shim is provided for older
development builds.

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

This integration targets the [Home Assistant Integration Quality Scale][iqs] at the
**Silver** tier. Several Gold-tier rules are also met. Rules marked N/A are
inapplicable to an LLM-based integration that does not poll devices or discover
hardware.

[iqs]: https://developers.home-assistant.io/docs/core/integration-quality-scale/

### Bronze

| Criteria                                      | Status | Proof                                                                                                                                                                                                           |
| --------------------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| UI-based setup via config flow                | Done   | [`config_flow.py`](custom_components/pydantic_ai_agent/config_flow.py) — full multi-step provider, subentry, reauth, and reconfigure flows                                                                      |
| Form fields have context help text            | Done   | [`translations/en.json`](custom_components/pydantic_ai_agent/translations/en.json) — per-field `data_description` for every form step                                                                           |
| Config entry data and options used correctly  | Done   | [`__init__.py`](custom_components/pydantic_ai_agent/__init__.py) — workspace data read from `entry.data`; per-agent settings in subentry data                                                                   |
| Full config flow test coverage                | Partial | [`test_workspace_config_flow_smoke.py`](tests/components/pydantic_ai_agent/test_workspace_config_flow_smoke.py), [`test_config_flow_helpers.py`](tests/components/pydantic_ai_agent/test_config_flow_helpers.py), and [`test_probe_model.py`](tests/components/pydantic_ai_agent/test_probe_model.py) — covers workspace/provider creation smoke paths, helper validation, and model probing; deeper provider reconfigure coverage remains pending |
| Connection tested before config entry created | Done   | [`config_flow.py`](custom_components/pydantic_ai_agent/config_flow.py) — `async_probe_model` stream-tests the provider before entry creation                                                                    |
| Integration readiness checked during setup    | Done   | [`__init__.py`](custom_components/pydantic_ai_agent/__init__.py) — `_async_validate_configured_models` probes every configured model at load                                                                    |
| Duplicate config entries prevented            | Done   | [`config_flow.py`](custom_components/pydantic_ai_agent/config_flow.py) — workspace and provider flows prevent duplicate resources                                                                               |
| Every entity has a unique ID                  | Done   | [`entity.py`](custom_components/pydantic_ai_agent/entity.py) [`sensor.py`](custom_components/pydantic_ai_agent/sensor.py) — every entity has a stable `unique_id`                                               |
| Entities use `has_entity_name = True`         | Done   | [`conversation.py`](custom_components/pydantic_ai_agent/conversation.py) [`sensor.py`](custom_components/pydantic_ai_agent/sensor.py) — `_attr_has_entity_name = True` on all entity classes                    |
| Runtime data stored on `entry.runtime_data`   | Done   | [`__init__.py`](custom_components/pydantic_ai_agent/__init__.py) — `WorkspaceRuntimeData` dataclass stored on `entry.runtime_data`                                                                              |
| Service actions registered in `async_setup`   | Done   | [`__init__.py`](custom_components/pydantic_ai_agent/__init__.py) — `list_mcp_tools` and `refresh_mcp_tools` registered in `async_setup`                                                                         |
| Shared logic in common modules                | Done   | [`entity.py`](custom_components/pydantic_ai_agent/entity.py) — shared `PydanticAIBaseLLMEntity` base; [`model_profiles.py`](custom_components/pydantic_ai_agent/model_profiles.py) — profile resolution helpers |
| Dependencies fully declared                   | Done   | [`manifest.json`](custom_components/pydantic_ai_agent/manifest.json) — all runtime dependencies are explicit, no extras or transitive requires                                                                  |
| Entity events in correct lifecycle hooks      | Done   | [`sensor.py`](custom_components/pydantic_ai_agent/sensor.py) — `async_added_to_hass` subscribes to dispatcher signals                                                                                           |
| Branding assets present                       | Done   | [`brand/icon.png`](custom_components/pydantic_ai_agent/brand/icon.png)                                                                                                                                          |

### Silver

| Criteria                                                      | Status  | Proof                                                                                                                                                                                                                                                         |
| ------------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Active integration code owner                                 | Done    | [`manifest.json`](custom_components/pydantic_ai_agent/manifest.json) — `codeowners: ["@bradsjm"]`                                                                                                                                                             |
| Config entry unloading supported                              | Done    | [`__init__.py`](custom_components/pydantic_ai_agent/__init__.py) — `async_unload_entry` unloads platforms; `async_remove_entry` cleans up repair issues                                                                                                       |
| Reauthentication flow available                               | Done    | [`config_flow.py`](custom_components/pydantic_ai_agent/config_flow.py) — reauth step (`async_step_reauth_confirm`) with provider re-validation                                                                                                                |
| Reconfiguration flow available                                | Done    | [`config_flow.py`](custom_components/pydantic_ai_agent/config_flow.py) — reconfigure step plus per-subentry-type reconfigure flows (model, conversation, AI task, MCP server)                                                                                 |
| Service actions raise exceptions on failure                   | Done    | [`__init__.py`](custom_components/pydantic_ai_agent/__init__.py) [`entity.py`](custom_components/pydantic_ai_agent/entity.py) — typed `MCPValidationError` and `_home_assistant_error` mapping                                                                |
| Entity marked unavailable when appropriate                    | N/A     | Entities are LLM-driven, not device status; diagnostic sensors expose `None` when no data is available                                                                                                                                                        |
| Log once when service becomes unavailable and again when back | Done    | [`__init__.py`](custom_components/pydantic_ai_agent/__init__.py) — MCP tool refresh failures logged at warning; provider validation failures logged at warning                                                                                                |
| Parallel update concurrency specified                         | N/A     | No `DataUpdateCoordinator`; agent runs are serialized per entity with `max_concurrency=1`                                                                                                                                                                     |
| Above 95% test coverage                                       | Partial | [`tests/components/pydantic_ai_agent/`](tests/components/pydantic_ai_agent/) — active coverage exists for helpers, model profiles, diagnostics, system health, runtime pieces, and real-server paths; legacy flow/entity modules still need workspace rewrite |

### Gold (partial)

The integration meets several Gold-tier rules. Full Gold alignment is pending documentation improvements.

| Criteria                                    | Status  | Proof                                                                                                                                                                                                                 |
| ------------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Devices created for entities                | Done    | [`entity.py`](custom_components/pydantic_ai_agent/entity.py) — `DeviceInfo` with `identifiers`, `manufacturer`, `model`, and `entry_type` for every agent subentry                                                    |
| Entities assigned appropriate categories    | Done    | [`sensor.py`](custom_components/pydantic_ai_agent/sensor.py) — all metric and config sensors use `EntityCategory.DIAGNOSTIC`                                                                                          |
| Entities use device classes where possible  | Done    | [`sensor.py`](custom_components/pydantic_ai_agent/sensor.py) — `SensorDeviceClass.DURATION` and `SensorStateClass` set where applicable                                                                               |
| Diagnostics implemented                     | Done    | [`diagnostics.py`](custom_components/pydantic_ai_agent/diagnostics.py) — config-entry and device-level diagnostics with comprehensive redaction                                                                       |
| Repair issues for user-actionable problems  | Done    | [`repairs.py`](custom_components/pydantic_ai_agent/repairs.py) — `is_fixable=True` model-validation issues; process-global Logfire-token-conflict warnings; automatic issue lifecycle (create, delete, stale cleanup) |
| Exception messages translatable             | Done    | [`translations/en.json`](custom_components/pydantic_ai_agent/translations/en.json) — per-reason error and abort translations with placeholders                                                                        |
| Entity names translatable                   | Done    | [`translations/en.json`](custom_components/pydantic_ai_agent/translations/en.json) — subentry-type labels (provider, conversation, AI task, MCP server) and form field translations                                   |
| Stale devices removed                       | N/A     | Devices are tied to subentry lifecycle; subentry removal triggers platform cleanup                                                                                                                                    |
| Devices added after integration setup       | N/A     | Devices are created at subentry load time, not discovered at runtime                                                                                                                                                  |
| Device/service auto-discovery               | N/A     | LLM-based integration; no device or hardware discovery                                                                                                                                                                |
| Documentation describes supported functions | Partial | README §Configuration details all entity types, MCP servers, skills, and validation; automation examples pending                                                                                                      |
| Documentation includes troubleshooting      | Pending | Not yet included                                                                                                                                                                                                      |
| Documentation includes automation examples  | Pending | Not yet included                                                                                                                                                                                                      |
| Documentation describes known limitations   | Pending | Not yet included                                                                                                                                                                                                      |
| Documentation describes how data is updated | Pending | Agent responses are per-request, not periodic; not yet documented                                                                                                                                                     |

## Development

Use Python 3.14.2 or newer.

Runtime dependencies are declared in both `pyproject.toml` and
`custom_components/pydantic_ai_agent/manifest.json`. The integration uses
`pydantic-ai-slim`, explicit native provider SDK dependencies for Anthropic and
Google Gemini, `fastmcp-slim[client,server]`, and the in-repo
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
