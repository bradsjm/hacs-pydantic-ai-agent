# Pydantic AI Agent

[![HACS Custom](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/hacs/integration)

Home Assistant custom integration for Assist conversation agents and AI task data
generation backed by Pydantic AI.

## Status

Implemented capabilities include:

- OpenAI-compatible provider connections using the in-repo
  `OpenAICompatibleChatModel` / `OpenAICompatibleProvider` and Home
  Assistant's shared async HTTP client. The OpenAI SDK is not required.
- Multiple configurable Assist conversation agents per provider connection.
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
  provider connection.
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
3. Add `https://github.com/bradsjm/hacs-pydantic-agent` as an Integration.
4. Install `Pydantic AI Agent`.
5. Restart Home Assistant.

### Manual

1. Copy `custom_components/pydantic_ai_agent` into your Home Assistant
   `custom_components` directory.
2. Restart Home Assistant.

## Configuration

1. Go to Settings > Devices & services.
2. Add `Pydantic AI Agent`.
3. Configure an OpenAI-compatible provider connection with an API key.
4. Enter a custom OpenAI-compatible base URL when needed, such as
   `http://localhost:11434/v1`. If no base URL is entered, the provider uses
   `https://api.openai.com/v1`.
5. Add subentries for the agents and tool sources you want to expose.

Provider-level settings are shared by all subentries under that provider
connection. Per-agent and per-task settings are stored on their own subentries.

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
- Conversation entities currently return non-streamed Assist responses and do
  not advertise Home Assistant streaming support.

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

Provider setup includes a skills folder field. The folder must be `/config/skills`
or one of its subfolders. Conversation agents and AI tasks can select discovered
`pydantic-ai-skills` from that folder.

Skill script execution is disabled by default. Enable `Allow skill script
execution` only for skills you trust; changing the skills folder or script
execution setting clears selected skills from existing agent and task subentries.

### Validation And Repairs

The integration validates configured models with provider test requests when
conversation-agent and AI-task subentries are created or reconfigured, and again
when the provider entry loads. Authentication failures trigger reauthentication.
Model, permission, provider-configuration, or streaming-capability failures that
can be fixed by reconfiguration are surfaced as Home Assistant repair issues
without preventing the provider entry from loading.

OpenAI-compatible provider validation uses a short streamed model probe, but
runtime conversation responses are still returned to Home Assistant as
non-streamed results.

Diagnostics redact API keys, Logfire tokens, prompts, sensitive model settings,
provider headers, MCP URLs, and MCP headers. Runtime diagnostics expose only safe
counts such as configured MCP server count, cached MCP server count, and cached
tool counts per server. Device diagnostics are scoped to the matching agent or AI
task subentry and include safe runtime metrics for that subentry.

System health reports aggregate, non-secret counts for configured and loaded
entries, provider modes, model profiles, conversation agents, AI tasks, MCP
servers and caches, Logfire-enabled entries, and skill-script-execution entries.

Entity unique IDs use the breaking prefixed format
`pydantic_ai_agent_<subentry_type>_<subentry_id>` for conversation and AI task
entities, and `pydantic_ai_agent_<subentry_type>_<subentry_id>_<metric_key>` for
sensor and binary sensor entities. No migration shim is provided for older
development builds.

### Logfire Tracing

Provider setup includes optional Logfire tracing fields. Leave the Logfire token
blank to disable Logfire for that provider connection. When a token is provided,
the integration adds Home Assistant metadata such as entry, subentry, entity,
model, and conversation IDs to Pydantic AI traces.

The `Include prompt and response content in Logfire` option is disabled by
default. Enable it only if you want Logfire to capture prompt, completion, and
tool payload content. Logfire is configured process-wide in Home Assistant: the
first loaded provider entry with a token wins, later entries with a different
token are left loaded but get a repair warning and do not emit Logfire traces.

## Development

Use Python 3.14.2 or newer.

Runtime dependencies are declared in both `pyproject.toml` and
`custom_components/pydantic_ai_agent/manifest.json`. The integration uses
`pydantic-ai-slim`, `fastmcp-slim[client,server]`, and the in-repo
OpenAI-compatible adapter instead of the OpenAI SDK.

The adapter design is documented in
`docs/openai_compatible_provider_design.md`.

Install or update the development environment, then run the local checks:

```bash
scripts/setup
scripts/check
```

## Support

Issues: <https://github.com/bradsjm/hacs-pydantic-agent/issues>

Code owner: `@bradsjm`
