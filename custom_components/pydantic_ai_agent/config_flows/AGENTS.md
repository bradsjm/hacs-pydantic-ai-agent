# Config Flows Agent Instructions

## Scope

These instructions apply to `custom_components/pydantic_ai_agent/config_flows`
and its `provider_wizard` subpackage.

## Agent Focus

- Treat this package as Home Assistant UI flow code, not runtime agent code.
- Preserve `ConfigFlow` and `ConfigSubentryFlow` lifecycle semantics.
- Keep flow output aligned with `strings.json`, tests, repairs, diagnostics, and
  runtime consumers when stored data changes.
- Prefer changing shared schema and normalization helpers in `common.py` only
  when more than one flow genuinely uses the behavior.
- Never add live provider model probes in config flows. Keep provider network
  validation limited to the existing HA-managed provider connection and
  model-list steps. MCP server flows may validate remote MCP URLs and discover
  tool catalogs through the existing MCP helpers.

## Read First

- `workspace_flow.py` - top-level workspace entry flow and supported subentry
  type registration.
- `provider_flow.py` - provider subentry creation, reconfiguration, model
  profile management, provider discovery, and guided setup wizard orchestration.
- `common.py` - shared selectors, section flattening consumers, provider data
  normalization, model profile references, run settings, pricing, and model
  setting parsing.
- `conversation_flow.py` - conversation subentry setup and validation.
- `ai_task_flow.py` - AI task subentry setup and local validation.
- `mcp_server_flow.py` and `mcp_helpers.py` - remote MCP server forms,
  Streamable HTTP URL/header validation, tool discovery, allowlists, deferred
  loading, and call-cache settings.
- `skill_flow.py` and `skill_helpers.py` - native Skill subentry forms and
  selection helpers.
- `provider_wizard/` - models.dev catalog loading, filtering, schemas, and
  wizard data builders.

## Invariants

- Workspace entries are created by `PydanticAIAgentConfigFlow`; child resources
  are config subentries returned by `async_get_supported_subentry_types()`.
- Conversation and AI task subentries reference provider-owned model profiles
  with `<provider_subentry_id>:<model_profile_id>` refs. Do not replace these
  refs with raw model names.
- Provider subentries own credentials, mode, base URL, headers, extra body,
  model profiles, model settings, and pricing.
- MCP server subentries own remote Streamable HTTP URL, headers, secret header
  metadata, return-schema disclosure, call-cache options, deferred loading, tool
  exposure mode, and allowlisted tools.
- Provider validation uses `async_list_provider_model_names()` for model
  discovery only; runtime model/tool/structured-output failures are handled at
  run time.
- AI task subentries validate selected model refs locally and save without live
  preflight requests.
- Conversation and AI task subentries can reference MCP server subentries by
  ID; keep this independent from Home Assistant LLM API selection.
- Use `_flatten_section_data()` when processing Home Assistant `section()` form
  input. Do not read nested form sections directly in new flow code.
- Keep selector options deterministic with `_sorted_select_options()` or sorted
  source data.
- Do not log API keys, provider headers, prompt text, or raw provider bodies.
- The provider wizard catalog cache is intentionally process-global through
  `hass.data[DOMAIN]` because it is shared across concurrent flows.

## High-Risk Changes

- Provider profile enable or delete logic can break existing conversation and
  AI task refs. Check `_referenced_provider_profile_ids()` and dependent errors.
- Blank form fields can mean either clear stored settings or keep untouched
  settings. Check `_merge_model_settings()` and `_merge_model_pricing()` before
  changing form sections.
- Base URL validation rejects endpoint-specific URLs such as
  `/v1/chat/completions`. Keep this guard unless the runtime provider builders
  change too.
- Provider `extra_body` cannot contain chat-template kwargs. Keep conflict
  validation in the normalization path.
- Catalog metadata and pricing are seeded from models.dev data but must remain
  editable and stable after reconfigure.
- MCP URL validation rejects unsafe schemes and duplicate normalized URL
  identities, and runtime MCP redirects must remain on the validated origin.

## Validation

- Run `scripts/test -k config_flow` for flow-level changes.
- Run `scripts/test -k config_flow_helpers` for shared helper changes.
- Run `scripts/test -k provider_flow` for provider creation, reconfigure, model,
  and profile-flow changes.
- Run `scripts/test -k provider_wizard` for guided provider setup changes.
- Run `scripts/test -k mcp_server_flow` for MCP server form changes.
- Run `scripts/test -k mcp_server_manage_tools_flow` for MCP tool exposure
  changes.
- Run `scripts/test -k "workspace_config_flow_smoke or config_flow_helpers or provider_wizard"` before finishing broad config-flow edits.
- Run `scripts/lint-check` when changing imports, schemas, or helper names.
