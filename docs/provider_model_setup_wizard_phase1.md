# Provider And Model Setup Wizard Phase 1

## Status

This document is the implementation reference and running checklist for phase 1
of the provider and model setup wizard for `custom_components/pydantic_ai_agent`.
Phase 1 implementation is complete in this branch. Current source, tests,
manifests, and translations remain authoritative for future changes.

## Phase 1 Purpose

Phase 1 adds a guided provider setup path that helps a user add a provider
account and immediately enable at least one usable language model profile. The
wizard reduces first-run friction without replacing the existing manual provider
form or reconfigure paths.

Primary goals:

- make the first selection a user-facing provider name, not SDK or driver type;
- use `https://models.dev/api.json` as a setup catalog of providers, model names,
  and model capabilities;
- fetch and normalize the catalog lazily when the wizard is opened;
- reuse the catalog during a short setup session so users can add multiple
  providers or workspaces without repeated network waits;
- create provider subentries whose selected model profiles are enabled
  immediately;
- keep advanced customization available through existing provider/profile
  reconfigure flows;
- keep wizard code isolated from runtime provider, entity, and adapter code.

Phase 1 non-goals:

- do not replace provider reconfiguration or per-profile customization;
- do not make runtime model construction depend on `models.dev` availability;
- do not persist the full `models.dev` catalog in config entries;
- do not infer broad Responses API support from `models.dev` unless a reliable
  catalog signal is introduced later;
- do not remove the custom/manual provider path.

## Current Source Baseline

Current provider setup is implemented by
`custom_components/pydantic_ai_agent/config_flows/provider_flow.py` and shared
helpers in `custom_components/pydantic_ai_agent/config_flows/common.py`.

Current provider creation behavior:

- `ProviderSubentryFlowHandler.async_step_user()` starts `async_step_init()`.
- `async_step_init()` delegates to `_async_provider_form_step("init", ...)`.
- The provider form asks for provider name, provider mode, API key, optional base
  URL, optional provider HTTP headers, optional provider extra body, and optional
  custom model names.
- `_normalise_provider_data()` flattens form sections, normalizes the base URL,
  parses headers and extra body, strips the API key, and normalizes custom model
  names.
- `_validate_provider_data()` checks the provider mode, rejects endpoint-shaped
  base URLs, and rejects provider extra body for modes that do not consume it.
- `_provider_already_configured()` prevents duplicate provider connection entries.
- `_normalise_provider_model_profiles()` creates new discovered or custom model
  profiles with `enabled` set to `False`.
- Reconfigure shows `edit_connection` and `customize_model_profile`.
- `conversation_flow.py` and `ai_task_flow.py` abort with `no_models_configured`
  when no enabled model profile exists.

Phase 1 must preserve the storage and runtime contracts above while improving the
initial provider setup path.

## UX Decisions

The wizard is a happy-path setup flow. It should get a typical user to a working
provider and enabled model profile with minimal choices.

UX rules:

- The first wizard choice is provider name.
- SDK, AI SDK package names, and internal provider modes are implementation
  details and must not be shown as the first-level choice.
- Driver or API-mode selection is shown only when the selected provider has more
  than one meaningful supported driver choice.
- Steps with zero or one meaningful choice should be skipped.
- Large gateway providers may require extra filtering steps.
- Advanced users can adjust provider fields and model-profile settings later
  through reconfigure.

Default visible model filters:

- hide non-chat and non-text-output models;
- hide deprecated models;
- hide models that do not support tool calling;
- hide models that do not support structured output according to catalog data.

The structured-output filter is a UX default, not a runtime guarantee. If this
filter hides too many otherwise useful models for a provider, the implementation
must expose clear advanced options to include additional models.

## Wizard Flow

Canonical conditional flow:

1. Choose guided setup or custom provider.
2. Load catalog if needed.
3. Pick provider by display name.
4. Pick driver or API mode only if needed.
5. Enter connection details.
6. Filter and select model profiles.
7. Save provider subentry with selected profiles enabled.

Catalog loading must use a Home Assistant progress step with
`progress_action="load_model_catalog"`. Loading must not run at Home Assistant
startup. Failure must offer retry and custom/manual provider setup without
creating a repair issue.

## models.dev Catalog Contract

The catalog source is `https://models.dev/api.json`. It is a provider-id keyed
object. Provider objects include `id`, `name`, `npm`, `env`, optional `api`,
`doc`, and `models`. Each provider `models` value is a model-id keyed object.

Relevant model fields include `id`, `name`, optional `family`, `attachment`,
`reasoning`, `tool_call`, optional `structured_output`, optional `temperature`,
`modalities.input`, `modalities.output`, `limit.context`, `limit.output`,
optional `cost`, `release_date`, `last_updated`, optional `status`, and optional
model-level `provider` override.

The wizard must treat catalog data as setup guidance, not as authoritative
runtime capability proof.

## Supported Provider Mapping

Phase 1 mapping:

| Catalog condition                                     | Supported mode                         | Notes                       |
| ----------------------------------------------------- | -------------------------------------- | --------------------------- |
| Provider ID `anthropic`                               | `anthropic`                            | Native Anthropic mode only  |
| Provider ID `google`                                  | `google_gemini`                        | Gemini Developer API only   |
| Provider ID `openai`                                  | `openai_compatible_completions`        | Default OpenAI choice       |
| Provider ID `openai`                                  | `openai_compatible_responses`          | Explicit choice only        |
| `npm == "@ai-sdk/openai-compatible"`                  | `openai_compatible_completions`        | Use valid provider `api`    |
| `npm == "@openrouter/ai-sdk-provider"`                | `openai_compatible_completions`        | Use valid provider `api`    |
| Other provider IDs or unsupported SDK/provider shapes | none                                   | Hide from guided list       |
| Custom provider                                       | user-selected supported provider mode  | Manual escape hatch         |

Do not expose `npm` or SDK names directly to the user. Do not assume Responses
support for every OpenAI-compatible provider.

## Catalog Cache And Efficiency

Cache requirements:

- lazy load only when the guided wizard is opened;
- no Home Assistant startup fetch;
- one shared in-flight load task across simultaneous flows;
- compact in-memory catalog reused across flows;
- minimum setup-session reuse window of 15 minutes after last use;
- hard freshness cap of 1 hour after load;
- discard the raw `api.json` object immediately after normalization;
- generate Home Assistant selector options lazily per current provider/filter;
- drop the compact in-memory catalog when the cache expires.

Recommended expiration formula:

```text
memory_retained_until = min(last_used_at + 15 minutes, loaded_at + 1 hour)
```

Store the catalog manager under `hass.data[DOMAIN]` because it is shared
process-wide integration state, not one config entry's runtime data. Do not store
catalog data in provider subentries, workspace entry data, or runtime provider
objects.

## Compact Catalog Shape

The compact catalog should keep only data needed for wizard display and profile
creation.

Suggested fields:

- provider: `id`, `name`, `doc_url`, `api_key_hints`, `default_base_url`,
  `supported_drivers`, `model_count`, and `families`;
- model: `id`, `name`, `provider_id`, `family`, `tool_call`,
  `structured_output`, `reasoning`, `attachment`, `text_output`,
  `context_limit`, `output_limit`, and `status`.

Avoid retaining raw provider dictionaries, raw model dictionaries, full nested
cost data, prebuilt selector options for every provider, unsupported provider
metadata, or unsupported model-level provider overrides.

## Persistence Contract

Wizard-created provider subentries must use the existing operational schema:

```python
{
    CONF_NAME: provider_name,
    CONF_PROVIDER_MODE: provider_mode,
    CONF_API_KEY: api_key,
    CONF_MODEL_PROFILES: model_profiles,
}
```

Optional fields are `CONF_BASE_URL`, `CONF_PROVIDER_HEADERS`, and
`CONF_PROVIDER_EXTRA_BODY` when configured.

Selected catalog models must create provider-owned profiles shaped as:

```python
{
    "id": profile_id,
    CONF_NAME: catalog_model_name,
    CONF_MODEL: catalog_model_id,
    CONF_ENABLED: True,
    CONF_DISCOVERED: True,
}
```

Phase 1 should not persist catalog identity fields. Existing provider and
model-profile data remains the source of truth for model construction.

## Package Boundary

Wizard code must live in:

```text
custom_components/pydantic_ai_agent/config_flows/provider_wizard/
```

Suggested module layout:

```text
provider_wizard/
├── __init__.py
├── catalog_cache.py
├── const.py
├── filters.py
├── flow.py
├── mapping.py
├── models_dev.py
├── normalize.py
├── schemas.py
└── types.py
```

Runtime modules such as `provider.py`, `entity.py`, `model_profiles.py`, and
`openai_compatible_adapter/` must not import wizard modules. `provider_flow.py`
may delegate guided setup to the wizard package.

## Validation And Failure Behavior

Catalog loading failures:

- do not abort provider setup;
- show a controlled retry/manual fallback;
- do not create repair issues.

Provider connection validation:

- keep existing `_validate_provider_data()` behavior;
- keep duplicate-provider detection;
- keep base URL endpoint rejection;
- keep provider extra body compatibility validation.

Model validation:

- phase 1 avoids provider probes during provider wizard creation;
- AI task and setup-time model validation remain the stronger runtime checks.

## Home Assistant Progress Pattern

Catalog loading should follow Home Assistant's current data-entry-flow progress
pattern and the existing in-repo `ai_task_flow.py` and `mcp_server_flow.py`
implementations:

- create the long-running load with `hass.async_create_task(...)`;
- pass the task to `async_show_progress(progress_task=task, ...)`;
- when the frontend refreshes the flow while the task is unfinished, return
  `async_show_progress(...)` with the same progress action and task;
- when the task is complete, store its result on flow state and return
  `async_show_progress_done(next_step_id="...")`;
- handle the result or controlled error in the next step;
- do not call `async_show_progress()` without `progress_task`.

References checked before implementation:

- Home Assistant developer docs for data entry flow progress;
- Home Assistant `data_entry_flow.py` `async_show_progress()` and
  `async_show_progress_done()` behavior;
- in-repo `config_flows/ai_task_flow.py` model probe progress;
- in-repo `config_flows/mcp_server_flow.py` MCP validation progress.

## Execution Checklist

- [x] Restore this phase 1 implementation reference in the worktree.
- [x] Inspect Home Assistant progress-flow references and record the selected
  pattern before coding catalog loading.
- [x] Add provider wizard package skeleton, compact dataclasses, mapping,
  filtering, normalization, and unit tests.
- [x] Add lazy catalog fetch/cache lifecycle and unit tests.
- [x] Add wizard schemas, selector builders, provider-data builders, and tests.
- [x] Integrate guided/manual setup entry point into `ProviderSubentryFlowHandler`
  while preserving reconfigure behavior.
- [x] Complete guided end-to-end provider creation with enabled selected profiles.
- [x] Add translations, diagnostics coverage, and smoke tests for rendered flow
  steps and progress actions.
- [x] Run focused tests after each section and `scripts/check` at completion.
- [x] Commit each successful tested section before moving to the next one.

## Testing Plan

Required test areas:

- provider mapping from catalog provider data to supported integration modes;
- default model filtering hides non-text, deprecated, non-tool, and
  non-structured-output models;
- advanced filter flags include additional models;
- provider selection hides unsupported providers but includes custom provider;
- driver step is skipped for providers with one supported mode;
- driver step is shown for providers with multiple supported modes;
- model filter step is shown for large provider catalogs;
- small provider catalogs go directly to model selection;
- selected models create enabled provider-owned profiles;
- wizard-created provider data validates through existing provider validation;
- custom/manual path remains available when catalog loading fails;
- catalog cache reuses one in-flight task for concurrent flows;
- catalog cache reuses compact memory state within the 15-minute idle window;
- catalog cache expires no later than the 1-hour hard cap;
- raw catalog data is not persisted in config entries;
- no unit test hits `https://models.dev/api.json` directly.

## Risks And Open Questions

Risks:

- `models.dev` may be unavailable, slow, or return a schema that changed;
- AI SDK provider metadata is not proof of Pydantic AI or provider wire
  compatibility;
- Responses API support is ambiguous for most OpenAI-compatible providers;
- some catalog `api` values may be endpoint-shaped and fail current base URL
  validation;
- large gateway model catalogs can produce unusable Home Assistant forms if model
  selectors are not filtered before rendering;
- capability and cost metadata can be stale;
- strict structured-output filtering may hide providers or models that still work
  with the integration's tool-output mode.

Resolved phase 1 defaults:

- no bundled fallback catalog;
- no catalog identity fields persisted;
- 100 eligible models forces a family/filter step;
- OpenAI defaults to Chat Completions but offers Responses as a user-visible
  API-mode choice;
- catalog fetch failure shows retry/manual fallback;
- default structured-output filter requires `structured_output is True`.
