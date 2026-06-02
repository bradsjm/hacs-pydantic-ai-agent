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
- Never add provider probes or catalog fetches outside the existing HA-managed
  async validation steps.

## Read First

- `workspace_flow.py` - top-level workspace entry flow and supported subentry
  type registration.
- `provider_flow.py` - provider subentry creation, reconfiguration, model
  profile management, provider discovery, and guided setup wizard orchestration.
- `common.py` - shared selectors, section flattening consumers, provider data
  normalization, model profile references, run settings, pricing, and model
  setting parsing.
- `conversation_flow.py` - conversation subentry setup and validation.
- `ai_task_flow.py` - AI task subentry setup and model liveness probing.
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
- Provider validation must use `provider_validation.async_probe_model()` or
  `async_list_provider_model_names()` through the existing progress steps.
- AI task subentries probe the primary and fallback model refs before saving.
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

## Validation

- Run `scripts/test -k config_flow` for flow-level changes.
- Run `scripts/test -k config_flow_helpers` for shared helper changes.
- Run `scripts/test -k provider_wizard` for guided provider setup changes.
- Run `scripts/test -k "workspace_config_flow_smoke or config_flow_helpers or provider_wizard"` before finishing broad config-flow edits.
- Run `scripts/lint-check` when changing imports, schemas, or helper names.
