# Home Semantic Agent Instructions

## Scope

These instructions apply to `custom_components/pydantic_ai_agent/home_semantic`.

## Agent Focus

- Treat this package as the local semantic model of a Home Assistant instance.
- Keep it deterministic, local, vector-less, and free of provider/model calls.
- Preserve Home Assistant exposure checks for every user-facing context or
  control response.
- Keep response service outputs and tool outputs JSON-serializable.
- Do not add broad home-control capabilities without updating schemas, tests,
  service strings, and safety checks.

## Read First

- `builder.py` - snapshots HA registries and state, then builds the index off
  the event loop.
- `models.py` - frozen source and document dataclasses used by the index.
- `index.py` - token index, deterministic search, and aggregate diagnostics.
- `ranker.py` - scoring rules and penalties.
- `query.py` - exposed summary, target resolution, context lookup, supported
  domain checks, and service mapping.
- `llm_api.py` - Home Assistant `llm.API` tools, including `control_home`.
- `manager.py` - entry-scoped lifecycle, delayed initial build, debounce, and
  periodic refresh.
- `services.py` - read-only response service actions.
- `diagnostics.py` - aggregate manager and index diagnostics.

## Invariants

- The builder must copy HA-owned registry and state data on the event loop before
  executor work. Executor-side code must not touch `hass` directly.
- Search is symbolic token overlap plus deterministic ranking. Do not introduce
  embeddings, network calls, or model calls here.
- `async_should_expose()` is the authority for entity visibility in summaries,
  target resolution, context responses, and control actions.
- `control_home` is an LLM API tool only. The registered Home Assistant response
  services are read-only summary, resolve, and context services.
- Supported control domains are constrained in `query.py`. Add domains only with
  matching service mapping and tests.
- Group control must expand only exposed and supported members.
- `HomeSemanticIndexManager` is entry-scoped and belongs on
  `entry.runtime_data.home_semantic`.
- State change handling intentionally ignores raw sensor and binary sensor state
  value churn unless indexed attributes changed.

## High-Risk Changes

- Changing ranking can alter which entity receives a control action. Update
  ambiguity and specific-target tests with any ranking change.
- Changing `phrase_matches_specific_target()` can make generic phrases control
  a target. Keep generic action words from being enough by themselves.
- Changing manager delays, debounce, or periodic refresh behavior can increase
  HA startup load or miss registry updates.
- Adding fields to service responses or diagnostics must preserve primitive JSON
  values only.
- Do not expose hidden or unexposed entities through diagnostics, services, or
  LLM tools unless the surrounding product policy explicitly changes.

## Validation

- Run `scripts/test -k home_semantic` for package changes.
- Run `scripts/test -k home_semantic_llm_api` for service or tool behavior.
- Run `scripts/lint-check` when changing imports or dataclass fields.
