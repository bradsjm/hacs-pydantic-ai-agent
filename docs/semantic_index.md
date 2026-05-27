# Home Semantic Index + Home Assistant execution boundary

## Recommendation

Build a **local Home Semantic Indexer** inside the integration, then expose it through a small, custom, in-process Home Assistant `llm.API`.

The core principle should be:

> Use semantic memory and retrieval to understand the home.  
> Use Home Assistant LLM APIs, intents, services, scripts, scenes, and exposure policy to act.

This avoids three bad defaults:

1. dumping thousands of entities into prompts;
2. using broad search as the main control path;
3. letting an LLM call arbitrary HA services directly.

## Target architecture

```text
Home Assistant registries, states, groups, history summaries
        ↓
Home Semantic Indexer
        ↓
device / area / group / capability documents
        ↓
semantic retrieval + usage-aware ranking
        ↓
Pydantic AI conversation or AI task
        ↓
custom entry-scoped HA llm.API tools
        ↓
HA intents / services / scripts / scenes
```

## Product goal

For a command like:

> “Turn off bedroom lights”
> The ideal path is:

```text
utterance
  → semantic index resolves bedroom + lights
  → preferred target = light.bedroom_lights
  → one HA action/service call
  → concise confirmation
```

Not:

```text
utterance
  → model call
  → dump/search all entities
  → model call
  → multiple per-light service calls
  → verification waits
  → model call
```

## Why this direction

The real scale target matters:

- ~2,892 entities;
- ~239 physical devices;
- 20 areas;
- 3 floors;
- noisy `sensor`, `switch`, and `binary_sensor` domains;
- rooms with hundreds of entities;
- useful room/group entities;
- high-churn logbook/state activity from sensors and thermostats.
  At that scale, the primary abstraction should not be “entity search.” It should be:
- areas;
- floors;
- devices;
- capabilities;
- groups;
- aliases;
- routines;
- user corrections.
  Entities remain necessary, but mostly as implementation details behind those higher-level concepts.

## Implementation recommendation

### 1. Add a semantic subsystem

Suggested package:

```text
custom_components/pydantic_ai_agent/home_semantic/
├── models.py          # typed document and edge models
├── builder.py         # registry/state extraction
├── index.py           # in-memory searchable index
├── ranker.py          # scoring and usage-aware ranking
├── store.py           # HA Store persistence for corrections/usage
├── llm_api.py         # custom HA llm.API
├── tools.py           # llm.Tool implementations
├── control.py         # validated HA action execution
└── diagnostics.py     # redacted diagnostics helpers
```

Store runtime objects on typed `entry.runtime_data`, not module globals or `hass.data`.

### 2. Build semantic documents

Inputs:

- entity registry;
- device registry;
- area registry;
- floor registry;
- labels;
- aliases;
- custom names;
- original names;
- current states;
- selected attributes;
- device class;
- entity category;
- supported features;
- unit of measurement;
- group entities and helper groups;
- scripts and scenes;
- safe summaries from recorder/logbook/statistics/traces, if available.
  Outputs:
- floor documents;
- area documents;
- device documents;
- group documents;
- entity documents;
- capability documents;
- graph edges;
- searchable text;
- ranking features;
- optional embedding jobs.
  Example area document:

```json
{
  "type": "area",
  "area_id": "bedroom",
  "name": "Bedroom",
  "floor": "Upstairs",
  "aliases": ["main bedroom", "our room"],
  "capabilities": {
    "lights": {
      "preferred_target": "light.bedroom_lights",
      "entity_count": 6
    },
    "climate": {
      "preferred_target": "climate.bedroom",
      "entity_count": 1
    },
    "covers": {
      "entity_count": 2
    }
  }
}
```

The model should see this, not hundreds of raw bedroom entities.

### 3. Start symbolic/local, add embeddings later

Start with:

- normalized token search;
- alias matching;
- area/floor/device graph traversal;
- capability matching;
- fuzzy matching;
- domain/action compatibility;
- group preference;
- usage-aware ranking.

Add embeddings only later if proven effective at increasing result quality.

### 4. Register a custom HA `llm.API`

Create an entry-scoped API, for example:

```text
pydantic_ai_agent_home_<entry_id>
```

This prevents cross-entry leakage and lets each workspace/conversation choose whether to use the semantic API.

The API should expose a compact prompt plus a small tool surface.

Recommended tools:

#### `get_home_summary`

Returns compact floor/area/capability overview.

#### `resolve_home_target`

Resolves a phrase into a preferred HA target.
Example input:

```json
{
  "phrase": "bedroom lights",
  "action": "turn_off"
}
```

Example output:

```json
{
  "confidence": 0.96,
  "target_type": "entity",
  "entity_id": "light.bedroom_lights",
  "reason": "Preferred grouped light entity for Bedroom",
  "alternatives": []
}
```

#### `get_home_context`

Returns scoped live state.
Inputs should require a scope:

- area;
- floor;
- domain;
- entity IDs;
- device;
- capability.
  Never default to all exposed entities.

#### `control_home`

Executes validated actions.
Example input:

```json
{
  "action": "turn_off",
  "target": {
    "area": "Bedroom",
    "domain": "light"
  },
  "verification": "none"
}
```

The tool should internally prefer:

1. scripts/scenes/routines when the user requests one;
2. explicit group entities;
3. area + domain targets;
4. HA intents;
5. individual entity calls only as fallback.

#### `remember_home_correction`

Optional and explicit.
Example:

> “When I say bedroom lamp, I mean Jane’s nightstand.”
> This should update the semantic overlay, not mutate HA registry data.

### 5. Add opt-in semantic context to conversations and AI tasks

Each conversation and AI task subentry should be able to choose:

- disabled;
- compact home summary;
- retrieval context (w/corrections memory);
- retrieval + control tools.

This keeps the feature transparent and avoids surprising users.

### 6. Use usage-aware ranking carefully

Boost:

- successful target resolutions;
- explicit corrections;
- preferred groups;
- scripts/scenes used successfully;
- area/device/capability matches;
- exposed, user-named physical controls.
  Penalize:
- high-churn telemetry;
- diagnostic entities;
- hidden/disabled entities;
- noisy binary sensors;
- raw sensors unless directly queried;
- previously ambiguous matches.
  Use capped, typed, time-decayed signals. Do not let logbook volume dominate ranking.

### 7. Keep execution safe

Use HA exposure settings as the default allowlist.
Avoid exposing a generic model-facing `ha_call_service(domain, service, data)` tool. That is too broad for end-user control.
Prefer a constrained tool like `control_home` that validates:

- action;
- domain;
- target;
- confidence;
- exposure;
- domain safety policy;
- ambiguity.
  Recommended safety rules:
  | Domain/action type | Default behavior |
  |---|---|
  | lights, media, benign scenes | execute if confidence is high |
  | switches | execute only if exposed and target is unambiguous |
  | covers, climate | allow, but support verification/clarification |
  | locks, alarms, garage doors, security | require stricter confirmation |
  | config/admin/registry/files/add-ons | out of scope |

## Latency strategy

1. No full entity dumps.
2. No unfiltered live context.
3. Prefer group/area/domain control.
4. Keep tool results compact.
5. Cache semantic index snapshots.
6. Rebuild index in background on registry/state changes.
7. Use optional verification, not default verification.
8. Add a deterministic fast path later for high-confidence common commands.
   The fastest future path is:

```text
local semantic resolver → high-confidence control → HA action
```

with the LLM used when ambiguity, explanation, or multi-step reasoning is needed.

## Suggested implementation phases

### Phase 1 — symbolic semantic index

- Add semantic document models.
- Build floor/area/device/group/entity/capability docs.
- Build graph edges.
- Add compact search and ranking.
- Store index on `entry.runtime_data`.
- Add redacted diagnostics.

### Phase 2 — custom HA LLM API

- Register entry-scoped custom `llm.API`.
- Add `get_home_summary`.
- Add `resolve_home_target`.
- Add `get_home_context`.
- Add `control_home` for lights, groups, scenes, scripts, and simple switches.
- Respect exposed entities.

### Phase 3 — opt-in semantic context

- Add conversation subentry option.
- Add AI task subentry option.
- Inject compact retrieved context before model calls.
- Keep context size bounded.

### Phase 4 — correction and usage memory

- Persist explicit corrections with HA `Store`.
- Record successful resolutions.
- Record ambiguity penalties.
- Add time decay.
- Add user-inspectable diagnostics.

### Phase 5 — advanced optimization

- Optional embeddings.
- Optional recorder/logbook/statistics summaries.
- Optional deterministic fast path.
- Optional streaming with tools.

## Final recommendation

Implement this as:

> **a local, inspectable, device/capability-first Home Semantic Index feeding a constrained Home Assistant execution API.**
> The foundation should be:

```text
local semantic index
+ usage-aware ranking
+ explicit correction memory
+ compact retrieval
+ constrained HA control tools
```

That architecture best matches large Home Assistant installs, protects privacy, minimizes latency, and aligns with how users actually think about their homes.
