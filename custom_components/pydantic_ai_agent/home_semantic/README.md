# Home Semantic

`home_semantic` builds and serves a local symbolic view of the Home Assistant
installation. It indexes floors, areas, devices, entities, groups, and
capability summaries, then exposes deterministic search and scoped home context
to Assist LLM tools and read-only response services.

No external model, embedding service, or vector database is used.

## Runtime Flow

- `HomeSemanticIndexManager.async_start()` schedules a delayed initial build and
  subscribes to HA state and registry events.
- `async_build_home_semantic_index()` snapshots HA registries and states on the
  event loop.
- `_build_home_semantic_index_from_snapshot()` runs in the executor and produces
  a `HomeSemanticIndex`.
- The manager atomically swaps in the new index and records generation, status,
  timing, and error type.
- `HomeSemanticAPI` exposes four Assist tools for the owning workspace entry.
- `services.py` registers three response services for external read-only access.

## Modules

- `models.py` - typed source data, searchable documents, graph edges, rank
  features, and capability summaries.
- `builder.py` - HA snapshot collection and pure index construction.
- `index.py` - in-memory token index, search, and aggregate diagnostics.
- `ranker.py` - deterministic score contributions and penalties.
- `query.py` - summary, target resolution, scoped context, exposure checks, and
  action-to-service mapping.
- `llm_api.py` - entry-scoped `llm.API` implementation and semantic tools.
- `manager.py` - lifecycle, listeners, debounce, refresh tasks, and diagnostics.
- `services.py` - `SupportsResponse.ONLY` Home Assistant actions.
- `diagnostics.py` - aggregate diagnostics helpers.

## LLM Tools

- `get_home_summary` returns exposed areas, capabilities, preferred targets, and
  control domain counts.
- `resolve_home_target` resolves a phrase and optional action to one exposed
  supported entity target.
- `get_home_context` returns compact live state for explicit entity, phrase,
  domain, or area scopes.
- `control_home` executes constrained actions for exposed light, switch, scene,
  script, or group targets.

## Response Services

- `pydantic_ai_agent.get_home_semantic_summary`
- `pydantic_ai_agent.resolve_home_semantic_target`
- `pydantic_ai_agent.get_home_semantic_context`

Each service requires `config_entry_id`, accepts an optional `assistant_id`, and
returns a JSON-serializable response with `success`, `ready`, `status`,
`generation`, `config_entry_id`, `assistant_id`, and `errors` fields.

## Safety Boundaries

- Entity exposure is checked with Home Assistant's Assist exposure helper.
- Target resolution requires specific target tokens, not only generic action
  words such as `turn`, `on`, or `light`.
- Group control expands to exposed supported members and calls HA services with
  the LLM context.
- Registered response services do not call HA control services.
- Sensor and binary sensor value churn is excluded from refresh relevance checks
  unless indexed attributes change.

## Testing

- `scripts/test -k home_semantic`
- `scripts/test -k home_semantic_llm_api`

Related tests live in `tests/components/pydantic_ai_agent/test_home_semantic.py`
and `tests/components/pydantic_ai_agent/test_home_semantic_llm_api.py`.
