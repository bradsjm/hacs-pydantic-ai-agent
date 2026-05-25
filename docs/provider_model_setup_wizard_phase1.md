# Provider And Model Setup Wizard Phase 1

## Status

This document is the implementation reference for phase 1 of the provider and
model setup wizard for `custom_components/pydantic_ai_agent`. It records the
target UX, source boundaries, catalog handling, data contracts, and test plan
that future implementation plans should follow.

Current source, tests, manifests, and translations remain authoritative. When
this document describes planned behavior that is not yet implemented, the
implementation must still be verified against the current Home Assistant and
Pydantic AI APIs before coding.

## Phase 1 Purpose

Phase 1 adds a guided provider setup path that helps a user add a provider
account and immediately enable at least one usable language model profile. The
wizard should reduce first-run friction without replacing the existing manual
and reconfigure paths.

Primary goals:

- make the first selection a user-facing provider name, not SDK or driver type;
- use `https://models.dev/api.json` as a catalog of providers, model names, and
  model capabilities;
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
- `_provider_already_configured()` prevents duplicate provider connection
  entries by comparing provider mode, API key, base URL, headers, and extra
  body.
- `_async_validate_provider_form()` builds the stored provider data and model
  profiles.
- `_finish_provider_form()` creates or updates the provider subentry.

Current model-profile behavior:

- Provider subentries own a `model_profiles` map.
- Profiles are referenced by conversation and AI task subentries with
  workspace-local refs shaped as `<provider_subentry_id>:<profile_id>`.
- `_normalise_provider_model_profiles()` creates new profiles for discovered or
  custom model names with `enabled` set to `False`.
- When a provider is in discovery mode, profile editing triggers provider model
  discovery through `async_list_provider_model_names()`.
- Reconfigure shows a menu with `edit_connection` and
  `customize_model_profile`.
- `customize_model_profile` lets the user pick and edit one model profile at a
  time.

Current consumer-flow behavior:

- `conversation_flow.py` aborts with `no_models_configured` when no enabled
  model profiles exist.
- `ai_task_flow.py` also aborts with `no_models_configured` when no enabled
  model profiles exist.
- AI task creation probes selected model profiles with `async_probe_model()`.
- Provider creation does not probe a model.

Phase 1 should preserve the storage and runtime contracts above while improving
the initial provider setup path.

## User Experience Decisions

The wizard is a happy-path setup flow. It should get a typical user to a working
provider and enabled model profile with minimal choices.

UX rules:

- The first wizard choice is provider name.
- SDK, AI SDK package names, and internal provider modes are implementation
  details and should not be shown as the first-level choice.
- Driver or API-mode selection is shown only when the selected provider has more
  than one meaningful supported driver choice.
- Steps with zero or one meaningful choice should be skipped.
- Simple providers should be fast to configure.
- Large gateway providers, especially providers with hundreds or thousands of
  models, may require extra filtering steps.
- The default model list should favor models that are likely to work well for
  Home Assistant control and AI task usage.
- Advanced users can adjust provider fields and model-profile settings later
  through reconfigure.

Default visible model filters:

- hide non-chat and non-text-output models;
- hide deprecated models;
- hide models that do not support tool calling;
- hide models that do not support structured output according to catalog data.

The structured-output filter is a UX default, not a runtime guarantee. If this
filter hides too many otherwise useful models for a provider, the implementation
should expose a clear advanced option to include additional models.

Advanced model-list options may include:

- include models without tool calling;
- include models without structured output;
- include deprecated models;
- include beta or alpha models;
- include non-text input models when they still output text.

## Wizard Flow

The phase 1 flow should be conditional rather than a fixed sequence of screens.

Canonical flow:

1. Load catalog if needed.
2. Pick provider.
3. Pick driver or API mode only if needed.
4. Enter connection details.
5. Filter and select model profiles.
6. Save provider subentry with selected profiles enabled.

### Catalog Loading

When the wizard starts and no fresh compact catalog is available, the flow should
show a Home Assistant progress screen while loading the catalog.

Expected progress behavior:

- start one shared catalog load task;
- show progress action `load_model_catalog` while the task runs;
- when loading succeeds, continue to provider selection;
- when loading fails, continue with a manual/custom provider path and a
  controlled error or warning.

Catalog loading must not run at Home Assistant startup.

### Provider Selection

Provider selection should show recognizable provider names from the normalized
catalog, plus `Custom provider`.

Examples:

| Provider option | Expected user-facing meaning                  |
| --------------- | --------------------------------------------- |
| `Anthropic`     | Native Anthropic provider mode                |
| `Google`        | Gemini Developer API provider mode            |
| `OpenAI`        | OpenAI through in-repo OpenAI-compatible mode |
| `OpenRouter`    | OpenRouter gateway through OpenAI-compatible  |
| `DeepSeek`      | DeepSeek OpenAI-compatible endpoint           |
| `Custom`        | Manual provider mode, URL, and model names    |

Provider list display should be bounded. If the supported provider list becomes
large, provider selection should include search or simple grouping rather than a
long unfiltered dropdown.

### Driver Or API Mode Selection

Driver selection is conditional.

Skip the step when:

- the selected provider maps to exactly one supported integration mode;
- the selected provider is `Custom` and another form already requires explicit
  mode selection.

Show the step when:

- more than one supported integration mode is valid for the provider;
- choosing the mode materially changes available models or runtime behavior.

Labels should be user-facing:

| Internal mode                         | Suggested label                         |
| ------------------------------------- | --------------------------------------- |
| `openai_compatible_completions`       | Chat Completions                        |
| `openai_compatible_responses`         | Responses                               |
| `anthropic`                           | Anthropic                               |
| `google_gemini`                       | Google Gemini                           |

For OpenAI, phase 1 may show both Chat Completions and Responses if the
implementation can produce a clear recommendation. For other OpenAI-compatible
providers, default to Chat Completions unless future catalog metadata provides a
reliable Responses signal.

### Connection Details

Connection details should be prefilled from catalog data when available.

| Field                    | Catalog source or default                                      |
| ------------------------ | -------------------------------------------------------------- |
| Provider name            | Provider `name`                                                |
| Provider mode            | Mapping result from provider and driver selection              |
| API key                  | Empty password field with hint from provider `env`             |
| Base URL                 | Provider `api` when compatible and not endpoint-shaped         |
| Documentation/help text  | Provider `doc`                                                 |
| Provider headers         | Advanced field, empty by default                               |
| Provider extra body      | Advanced field, empty by default                               |

When a catalog provider does not need a base URL because the runtime has a native
default, leave the field empty.

Runtime defaults that already exist in source:

- OpenAI-compatible provider omitted base URL defaults to
  `https://api.openai.com/v1` in the in-repo provider adapter.
- Anthropic model listing defaults to `https://api.anthropic.com` and runtime
  SDK construction strips a trailing `/v1` from a configured base URL.
- Google Gemini model listing defaults to
  `https://generativelanguage.googleapis.com` and runtime SDK construction strips
  trailing `/v1beta` or `/v1` from a configured base URL.

Base URL validation must continue to reject endpoint-shaped values such as URLs
ending in `/models`, `/responses`, `/chat/completions`, or Google
`:generateContent` endpoint suffixes.

### Model Filtering And Selection

For small provider catalogs, show model selection directly after connection
details. For large provider catalogs, show filter controls before rendering a
model selector.

Suggested threshold behavior:

- if eligible model count is small enough for a Home Assistant multi-select,
  show the selector directly;
- if eligible model count is large, show a family/search/filter step first;
- generate selector options only for the current provider, driver, and filter
  state.

Useful filters:

- model family;
- model ID prefix for gateway providers;
- tool calling;
- structured output;
- reasoning;
- attachment or multimodal input;
- status;
- context window range.

For OpenRouter-style providers, the family or upstream provider prefix should be
available before showing individual model options. Example families can be
derived from model IDs such as `anthropic/*`, `openai/*`, `google/*`,
`deepseek/*`, `qwen/*`, and `meta-llama/*` when catalog family metadata is not
enough.

### Save Behavior

The wizard should create the same operational provider subentry shape that the
runtime already consumes.

Selected catalog models should become enabled provider-owned model profiles.
Unselected catalog models should not need to be persisted in phase 1.

For each selected model:

- use the catalog model ID as `model`;
- use the catalog model display name as `name`;
- set `enabled` to `True`;
- set `discovered` to `True` if the profile came from catalog or live discovery;
- store only model settings that the user explicitly chose in the wizard.

The wizard should not silently set advanced generation settings such as
temperature, max tokens, top P, seed, or penalties from catalog data. Catalog
limits can be displayed as information or suggested bounds, but user settings
should remain explicit.

## models.dev Catalog Contract

The catalog source is `https://models.dev/api.json`.

The upstream project documents that `api.json` is a provider-id keyed object and
that model IDs are the identifiers used by AI SDK. The upstream schema defines a
provider object with:

- `id`;
- `name`;
- `npm`;
- `env`;
- optional `api`;
- `doc`;
- `models`.

Each provider `models` value is a model-id keyed object. Relevant model fields
include:

- `id`;
- `name`;
- optional `family`;
- `attachment`;
- `reasoning`;
- `tool_call`;
- optional `structured_output`;
- optional `temperature`;
- optional `interleaved`;
- `modalities.input`;
- `modalities.output`;
- `limit.context`;
- optional `limit.input`;
- `limit.output`;
- optional `cost`;
- `release_date`;
- `last_updated`;
- optional `status`;
- optional model-level `provider` override.

The wizard should treat catalog data as setup guidance, not as an authoritative
runtime capability source. Provider probes and runtime requests remain the final
validation.

## Supported Provider Mapping

The implementation should normalize catalog providers into provider options with
supported integration drivers.

Phase 1 mapping:

| Catalog condition                                      | Supported mode                         | Notes                                      |
| ------------------------------------------------------ | -------------------------------------- | ------------------------------------------ |
| Provider ID `anthropic`                                | `anthropic`                            | Native Anthropic mode only                 |
| Provider ID `google`                                   | `google_gemini`                        | Gemini Developer API only                  |
| Provider ID `openai`                                   | `openai_compatible_completions`        | Base URL omitted by default                |
| Provider ID `openai`                                   | `openai_compatible_responses`          | Explicit choice only                       |
| `npm == "@ai-sdk/openai-compatible"`                   | `openai_compatible_completions`        | Use provider `api` when valid              |
| `npm == "@openrouter/ai-sdk-provider"`                 | `openai_compatible_completions`        | Use provider `api` when valid              |
| Other provider IDs or unsupported SDK/provider shapes  | none                                   | Hide from guided list                      |
| Custom provider                                        | user-selected supported provider mode  | Manual escape hatch                        |

Do not expose `npm` or SDK names directly to the user.

Do not assume `openai_compatible_responses` for every OpenAI-compatible provider.
The live catalog has model-level `provider.shape` support in the schema, but
phase 1 should only rely on it if a reliable `responses` signal exists for the
specific provider/model. Otherwise, Responses remains an explicit OpenAI or
custom-provider choice.

## Catalog Cache And Efficiency

Provider setup is expected to happen in short bursts: a user may add multiple
provider accounts, models, or workspaces in one setup session, then rarely touch
account configuration again. The catalog cache should optimize that setup
session without keeping large structures in memory indefinitely.

Cache requirements:

- lazy load only when the wizard is opened;
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

Cache ownership:

- store the catalog manager under `hass.data[DOMAIN]` because it is shared
  process-wide integration state, not one config entry's runtime data;
- do not store catalog data in provider subentries, workspace entry data, or
  runtime provider objects;
- do not expose the catalog manager to runtime entity/model construction.

Suggested cache manager state:

```python
@dataclass(slots=True)
class ProviderWizardCatalogManager:
    catalog: CompactCatalog | None
    loaded_at: datetime | None
    last_used_at: datetime | None
    inflight_task: asyncio.Task[CompactCatalog] | None
    cleanup_unsub: CALLBACK_TYPE | None
```

Implementation notes:

- fetch with Home Assistant's managed async HTTP client;
- use a short timeout;
- parse and normalize large JSON off the event loop if profiling shows event-loop
  blocking;
- cancel and reschedule cleanup when the catalog is touched;
- never log raw catalog payloads;
- never include API keys in catalog cache keys or logs.

Disk cache is optional and should not be part of phase 1 unless implementation
testing shows repeated network fetches are still a UX problem. If added later,
disk cache should store only compact normalized catalog data, contain no secrets,
and expire no later than the 1-hour hard cap.

## Compact Catalog Shape

The compact catalog should keep only data needed for wizard display and profile
creation.

Suggested types:

```python
@dataclass(frozen=True, slots=True)
class CatalogProviderOption:
    id: str
    name: str
    doc_url: str
    api_key_hints: tuple[str, ...]
    default_base_url: str | None
    supported_drivers: tuple[str, ...]
    model_count: int
    families: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CatalogModelOption:
    id: str
    name: str
    provider_id: str
    family: str | None
    tool_call: bool
    structured_output: bool | None
    reasoning: bool
    attachment: bool
    text_output: bool
    context_limit: int
    output_limit: int
    status: str | None
```

Avoid retaining:

- raw provider dictionaries;
- raw model dictionaries;
- full nested cost data unless the UI explicitly displays it;
- prebuilt selector options for every provider;
- unsupported provider metadata;
- unsupported model-level provider overrides.

## Persistence Contract

Wizard-created provider subentries should use the existing operational schema.

Required provider data:

```python
{
    CONF_NAME: provider_name,
    CONF_PROVIDER_MODE: provider_mode,
    CONF_API_KEY: api_key,
    CONF_MODEL_PROFILES: model_profiles,
}
```

Optional provider data when configured:

```python
{
    CONF_BASE_URL: base_url,
    CONF_PROVIDER_HEADERS: headers,
    CONF_PROVIDER_EXTRA_BODY: extra_body,
}
```

Model profile data for selected models:

```python
{
    "id": profile_id,
    CONF_NAME: catalog_model_name,
    CONF_MODEL: catalog_model_id,
    CONF_ENABLED: True,
    CONF_DISCOVERED: True,
}
```

Optional lightweight catalog identity can be added only if useful for future
refresh or diagnostics:

```python
{
    "catalog_source": "models.dev",
    "catalog_provider_id": provider_id,
    "catalog_model_id": model_id,
}
```

If catalog identity is added, runtime code must not require it. Existing provider
and model-profile data remains the source of truth for model construction.

## Package Boundary

Wizard code should live in its own package so it does not mix with runtime
provider and entity behavior.

Recommended package:

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

Responsibilities:

| Module             | Responsibility                                                   |
| ------------------ | ---------------------------------------------------------------- |
| `catalog_cache.py` | Shared lazy catalog manager, TTL, cleanup, in-flight task reuse  |
| `models_dev.py`    | HTTP fetch and raw payload parsing boundary                      |
| `normalize.py`     | Convert raw catalog data into compact provider/model options     |
| `mapping.py`       | Map catalog providers to supported integration provider modes    |
| `filters.py`       | Apply default and advanced model filters                         |
| `schemas.py`       | Home Assistant form schemas and selector-option construction     |
| `flow.py`          | Wizard step orchestration and final provider data construction   |
| `types.py`         | Wizard-only dataclasses and typed structures                     |
| `const.py`         | Wizard step IDs, cache constants, thresholds, and labels         |

Boundary rules:

- runtime modules such as `provider.py`, `entity.py`, `model_profiles.py`, and
  `openai_compatible_adapter/` must not import wizard modules;
- wizard modules may import shared constants and validation helpers;
- `provider_flow.py` should delegate guided setup to the wizard package rather
  than absorbing catalog and filtering logic;
- the wizard package should produce ordinary provider subentry data, not a
  separate runtime model system.

## Integration With ProviderSubentryFlowHandler

Phase 1 can add a guided setup entry point without removing the existing manual
form.

Recommended user entry shape:

- provider subentry `user` step shows a menu or initial selector with guided setup
  and custom/manual setup;
- guided setup delegates to `provider_wizard.flow`;
- custom/manual setup uses the existing provider form path;
- reconfigure keeps existing `edit_connection` and `customize_model_profile`
  paths, with optional future bulk model-management improvements.

The exact Home Assistant config-subentry step names should be chosen during
implementation and covered by translations and smoke tests.

## Validation And Failure Behavior

Catalog loading failures:

- do not abort the entire provider setup;
- show a controlled error or warning;
- allow the custom/manual provider path;
- do not create repair issues for a transient catalog load failure.

Provider connection validation:

- keep existing `_validate_provider_data()` behavior;
- keep duplicate-provider detection;
- keep base URL endpoint rejection;
- keep provider extra body compatibility validation.

Model validation:

- phase 1 may avoid provider probes during provider wizard creation to keep setup
  fast and avoid extra provider calls;
- if a probe is added, it should be explicit, bounded, and use the existing
  `async_probe_model()` error mapping;
- AI task and setup-time model validation remain the stronger runtime checks.

Capability metadata:

- use catalog capabilities for filtering and display;
- do not treat catalog capabilities as proof that Pydantic AI runtime requests
  will work;
- stale or missing catalog metadata must not block custom/manual configuration.

## Testing Plan

Add tests under the existing integration test tree. A package-specific test
folder is preferred if it matches the implementation structure:

```text
tests/components/pydantic_ai_agent/provider_wizard/
```

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

Translation/smoke coverage should verify all wizard steps, sections, errors,
progress actions, and selector translation keys that are rendered by the new
schemas.

## Implementation Sequence

Recommended phase 1 implementation order:

1. Add wizard package skeleton and compact dataclasses.
2. Implement mapping and filtering with unit tests using small fixture catalogs.
3. Implement catalog normalization with unit tests.
4. Implement lazy cache manager with mocked HTTP client and time control.
5. Add wizard schemas and selector option builders.
6. Integrate guided setup entry point into `ProviderSubentryFlowHandler`.
7. Create provider subentry data with enabled selected profiles.
8. Add translations and smoke tests for rendered wizard steps.
9. Run focused config-flow tests, then full local validation.

## Risks And Open Questions

Risks:

- `models.dev` may be unavailable, slow, or return a schema that changed.
- AI SDK provider metadata is not proof of Pydantic AI or provider wire
  compatibility.
- Responses API support is ambiguous for most OpenAI-compatible providers.
- Some catalog `api` values may be endpoint-shaped and fail current base URL
  validation.
- Large gateway model catalogs can produce unusable Home Assistant forms if model
  selectors are not filtered before rendering.
- Capability and cost metadata can be stale.
- Strict structured-output filtering may hide providers or models that still work
  with the integration's tool-output mode.

Open questions for implementation planning:

- Should phase 1 include a bundled fallback catalog for OpenAI, Anthropic, Google,
  and a small set of OpenAI-compatible providers?
- Should selected catalog identity fields be persisted for future refresh, or
  should phase 1 persist only existing operational fields?
- What model-count threshold should force a family/search step before model
  selection?
- Should OpenAI default to Chat Completions, Responses, or a user-visible choice?
- Should catalog fetch failure show a warning before custom setup or silently open
  custom setup with a description placeholder?
- Should the default structured-output filter require `structured_output is True`
  or treat missing structured-output metadata as unknown and hide it only when the
  user keeps strict defaults enabled?
