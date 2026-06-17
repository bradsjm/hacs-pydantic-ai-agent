# Pydantic AI Agent

[![Pydantic AI Agent Logo](custom_components/pydantic_ai_agent/brand/logo@2x.png)](https://github.com/bradsjm/hacs-pydantic-ai-agent)

[![HACS Custom](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/hacs/integration)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Coding Harness](https://img.shields.io/badge/coding_agents-opencode/gpt--5.5-orange)](https://opencode.ai/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/bradsjm/hacs-pydantic-ai-agent)

Home Assistant custom integration providing Assist conversation agents and AI task data
generation backed by the [Pydantic AI library](https://pydantic.dev/pydantic-ai) and optional integrated observability via [Logfire](https://logfire.dev/).

## Key Features

- Wizard driven provider and model selection using [models.dev](https://models.dev/)
- Supports custom model providers in addition to models.dev
- Support for multiple workspaces each with its own provider configuration
- Built-in observability via Logfire
- Built-in support for remote MCP servers over Streamable HTTP only
  - Optional per-server tool allowlists and disabled/specified/all tool modes
  - Optional tool-result caching with TTL
  - Optional deferred tool disclosure so the model discovers tools on demand
- Support for simple skills (reusable capabilities) library
- Optional enablement of provider web search capability with local duckduckgo fallback
- Optional web fetch tool with SSRF protection to prevent server-side request forgery attacks
- Support for Home Assistant assist tools for controlling Home Assistant entities
- Optional per-run virtual workspace tools for temporary shell and file work

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
5. Add provider, conversation, AI task, MCP server, and Skill subentries as
   needed.
6. To expose Home Assistant control tools, select a Home Assistant LLM API via
   the conversation agent or AI task `Capabilities` section.
7. To expose remote MCP tools, add MCP server subentries and select them on the
   conversation agent or AI task external tools section.

Workspace entries own shared Logfire settings. Provider credentials live on
`provider` subentries, while conversation, AI task, MCP server, and Skill
settings remain on their own subentries. Each provider subentry owns a stable
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
- Optional remote MCP server selection. MCP tool names are prefixed per server,
  and servers can expose all tools, a selected allowlist, or no tools.
- Optional Web fetch URL content fetching and Web search. Both are disabled by
  default and can be enabled independently of Home Assistant control tools.
- Optional workspace Skill selection. Selected Skills are exposed through
  `list_skills` and `load_skill` tools and are not auto-loaded into every
  request.
- Optional per-run virtual workspace tools for temporary file and shell work.
- Optional model settings including temperature, capability-backed thinking, max
  tokens, max iterations, top P, timeout, parallel tool calls, seed, penalties,
  and extra body fields.
- Optional provider HTTP headers configured on the provider entry and used for
  model discovery and model requests.
- Automatic hidden context management for very long conversations. Stored Assist
  history is not pruned; only the model request is windowed. Conversation
  agents default to context-manager mode and AI tasks default to sliding-window
  mode, with an off switch available for either.
- Tool-call follow-up requests preserve provider reasoning metadata such as
  `reasoning` and `reasoning_content` for OpenAI-compatible endpoints that
  require it.
- Plain conversation entities stream Assist responses. Conversations that can
  call tools, including HA LLM APIs, MCP servers, Web fetch, Web search,
  skills, or the virtual workspace, return non-streamed responses so provider
  tool-result follow-up requests stay on the compatibility path.

### AI Tasks

Add an `AI task` subentry for each AI task configuration you want to expose.
Each subentry has its own task name and creates an `ai_task.*` entity that
supports Home Assistant data generation and attachments.

AI task entities can return plain text or validate structured results against
the schema requested by Home Assistant. When Home Assistant requests structured
data, runtime chooses the highest supported strategy in the order `tool`,
`native`, then `prompted` from the resolved model profile capabilities. Each AI
task can also enable Web fetch or Web search independently of Home Assistant
control tool selection. Web search uses provider-native search when available or
local DuckDuckGo search through Pydantic AI.
AI task requests use the same automatic model-request context management as
conversation agents.
AI tasks default to 30 agent request iterations unless the selected language model
profile sets a max iterations override.
AI tasks can also select MCP servers, Skills, Home Assistant LLM APIs, a
virtual workspace, and an optional Home Assistant todo entity for locked
per-run todo tool access.

### MCP Servers and Home Assistant LLM APIs

Remote MCP servers are configured as workspace subentries. Only Streamable HTTP
transport is supported. The integration validates MCP URLs, headers, and tool
allowlists, rejects redirects outside the validated origin, and supports
per-server call caching and deferred loading.

Recommended setup:

1. Add the MCP server subentry in this integration.
2. Select the MCP server on the conversation agent or AI task external tools
   section.
3. Select a Home Assistant LLM API only when you want HA control tools.

Registered MCP and debug services:

- `list_mcp_tools`
- `refresh_mcp_tools`
- `get_agent_run_diagnostics`
- `get_workspace_status`
- `list_model_profiles`
- `get_agent_metrics`
- `get_tool_source_status`

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
folders. Skill references are supported and surfaced in diagnostics.

### Virtual Workspace

Conversation agents and AI tasks can enable a per-run virtual workspace. The
workspace provides Bash, virtual file, directory, patch, copy, move, remove, and
metadata tools in an in-memory Bashkit filesystem rooted at `/workspace` with
no host mounts and no network access. Destructive actions require explicit
`confirm=true`, and overwrites also require `overwrite=true`.

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

### Diagnostics

The integration also exposes redacted diagnostics, system health data, metrics,
and repair handling through the runtime services listed above.

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
