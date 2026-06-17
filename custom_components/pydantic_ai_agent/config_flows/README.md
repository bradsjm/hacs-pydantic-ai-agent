# Config Flows

This package implements the Home Assistant configuration UI for the
`pydantic_ai_agent` integration. It owns the workspace config entry flow and the
provider, conversation, AI task, MCP server, and Skill config subentry flows.

## Runtime Shape

- `workspace_flow.py` creates and reconfigures workspace config entries.
- `ProviderSubentryFlowHandler` creates and manages provider subentries and
  provider-owned model profiles.
- `ConversationSubentryFlowHandler` creates conversation agent subentries.
- `AITaskDataSubentryFlowHandler` creates AI task data-generation subentries and
  validates selected model refs before saving.
- `MCPServerSubentryFlowHandler` creates remote Streamable HTTP MCP server
  subentries, validates connection details, discovers tool catalogs, and manages
  per-server tool exposure.
- `SkillSubentryFlowHandler` creates native text-only Skill subentries.

## Module Map

- `workspace_flow.py` - top-level `ConfigFlow` and subentry type registry.
- `workspace_helpers.py` - workspace name and Logfire form schema helpers.
- `provider_flow.py` - manual provider flow, guided wizard orchestration, model
  discovery, profile editing, profile enablement, pricing, and settings.
- `conversation_flow.py` - conversation form validation and persistence.
- `ai_task_flow.py` - AI task form validation, todo workspace selection, and
  persistence.
- `mcp_server_flow.py` - MCP server create/reconfigure, connection validation,
  tool discovery, and tool exposure management.
- `mcp_helpers.py` - MCP URL/header normalization, server selection, call-cache
  settings, allowlist parsing, and tool selector helpers.
- `skill_flow.py` - native Skill create and reconfigure flow.
- `common.py` - shared schemas, parsing, normalization, selector options, model
  profile refs, run settings, pricing, and provider validation helpers.
- `helpers.py` - generic section flattening and selector sorting helpers.
- `skill_helpers.py` - Skill validation and selected Skill reference helpers.
- `provider_wizard/` - models.dev catalog cache, normalization, filtering,
  selectors, and storage builders.

## Data Model

- Workspace data stores shared workspace settings, including Logfire settings.
- Provider subentry data stores credentials, provider mode, endpoint data,
  discovered model cache data, model profiles, model settings, and pricing.
- Conversation and AI task data store primary and fallback model profile refs,
  prompts, run settings, selected MCP server refs, Skill refs, and virtual
  workspace, web-fetch, or web-search flags.
- MCP server data stores the remote URL, optional headers and secret header
  metadata, return-schema disclosure, call-cache settings, deferred loading,
  tool exposure mode, and allowed tool names.
- Skill data stores a name, description, and content template.

## Provider Wizard

The guided provider wizard loads a compact models.dev catalog through
`ProviderWizardCatalogManager`. The cache uses an idle TTL and a hard TTL, and
shared inflight tasks are shielded so one flow cannot cancel another flow's
catalog request.

The wizard selects a catalog provider, provider driver, connection data, model
filters, and selected models. `build_provider_data()` and
`build_model_profiles()` turn those choices into the same provider subentry
shape used by the manual flow.

## Validation Boundaries

- Provider model names are discovered through `provider_validation.py` when a
  flow needs a live model list.
- Model profile refs are checked against loaded provider subentries.
- MCP server URLs, headers, duplicate URL identity, and tool allowlists are
  validated by `mcp_helpers.py` and `mcp_server_flow.py`.
- Todo workspace selection is checked against HA todo entities and features.
- Skill references are checked against Skill subentries before save.

## Testing

- `scripts/test -k config_flow`
- `scripts/test -k config_flow_helpers`
- `scripts/test -k provider_wizard`
- `scripts/test -k mcp_server_flow`
- `scripts/test -k mcp_server_manage_tools_flow`
- `scripts/test -k workspace_config_flow_smoke`

Related tests live under `tests/components/pydantic_ai_agent/`, including
`test_config_flow_*`, `test_mcp_server_*`, `test_workspace_entry_flow.py`, and
the `provider_wizard/` test package.
