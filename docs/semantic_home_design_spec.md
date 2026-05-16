# Semantic Home Design Specification

## Status

This document is the functional and technical design specification for the
proposed `semantic_home` subsystem in the `pydantic_ai_agent` Home Assistant
custom integration.

Current implementation status: not implemented. This document describes planned
behavior and architecture. Current source, manifests, executable config, and
tests remain the authority for behavior that already exists.

The subsystem should be developed inside this repository first, under
`custom_components/pydantic_ai_agent/semantic_home/`, while remaining isolated
enough that it can later become its own Home Assistant custom component.

## Purpose

`semantic_home` provides a scalable Home Assistant LLM API for physical home
control and state queries. It replaces raw entity search and large static entity
prompts with a compact, cached semantic map of the home.

The subsystem should optimize for low-latency common interactions such as:

- `Turn off the bedroom lights.`
- `Is the garage door open?`
- `Set the office fan to medium.`
- `Close the living room curtains.`
- `What is happening in the kitchen?`

The primary design goal is not to expose more tools. The goal is to resolve
natural-language home intents into safe, deterministic Home Assistant actions
with minimal model/tool round trips.

## Source-Grounded Context

The current integration already has the right provider and agent shape for this
subsystem:

| Area                          | Current source                                        |
| ----------------------------- | ----------------------------------------------------- |
| Integration domain            | `custom_components/pydantic_ai_agent/const.py`        |
| Parent config entry runtime   | `custom_components/pydantic_ai_agent/__init__.py`     |
| Conversation platform         | `custom_components/pydantic_ai_agent/conversation.py` |
| Pydantic AI runtime           | `custom_components/pydantic_ai_agent/entity.py`       |
| HA LLM API tool adapter       | `custom_components/pydantic_ai_agent/ha_toolset.py`   |
| HA LLM API selection          | `custom_components/pydantic_ai_agent/config_flow.py`  |
| Diagnostics redaction         | `custom_components/pydantic_ai_agent/diagnostics.py`  |
| Product architecture baseline | `docs/pydantic_ai_agent_spec.md`                      |

Verified current behavior:

- The integration domain is `pydantic_ai_agent`.
- Parent provider credentials and connection data are stored on the parent
  config entry and copied into `entry.runtime_data`.
- Conversation agents are represented by `conversation` config subentries.
- New conversation subentries default to Home Assistant Assist LLM API access
  through `CONF_LLM_HASS_API: [llm.LLM_API_ASSIST]`.
- Conversation entities set `ConversationEntityFeature.CONTROL` only when
  `CONF_LLM_HASS_API` is configured.
- Runtime execution uses `pydantic_ai.Agent` and converts HA `llm.APIInstance`
  tools into Pydantic AI executable tools through `ha_toolset.py`.
- Current package dependencies include `fastmcp==3.3.1` and
  `pydantic-ai-slim[openai,mcp]==1.97.0`.

## Real Home Data Used For Design

The following sampled Home Assistant environment data was collected through
read-only Home Assistant tools during design discussion. It should be treated as
a design benchmark, not as a fixture.

| Metric                                  |    Observed value |
| --------------------------------------- | ----------------: |
| Entities                                |             2,892 |
| Domains                                 |                37 |
| Devices                                 | approximately 242 |
| Areas                                   |                20 |
| Floors                                  |                 3 |
| `sensor` entities                       |               724 |
| `switch` entities                       |               405 |
| `binary_sensor` entities                |               200 |
| `light` entities                        |                99 |
| `master_bedroom` entities               |               179 |
| `master_bedroom` lights                 |                19 |
| Office presence sensor entities         |                78 |
| 24 hour logbook entries sampled         |            11,511 |
| Explicit conversation exposure settings |        4 entities |

Important observed patterns:

- Raw entity lists are too large for prompt context or direct user selection.
- Many devices expose large diagnostic/configuration surfaces.
- Presence and thermostat activity can dominate recent activity signals.
- Room-level light groups exist and are better command targets than individual
  bulbs for broad room commands.
- Hidden entities, unavailable entities, duplicate-like entities, IP address
  sensors, firmware entities, and display indicator entities are common noise.
- Conversation exposure metadata is sparse in the sampled home, so exposure
  policy must be explicit and visible.

## Design Goals

- Build a cached semantic map of the home from Home Assistant registries,
  states, and selected recorder/statistics signals.
- Represent physical home targets as devices, areas, capabilities, and canonical
  groups instead of raw entity rows.
- Resolve common physical-control commands deterministically without requiring a
  preliminary search tool call.
- Keep model-visible tool results bounded, compact, paginated, and stable.
- Use ranking signals from Home Assistant and internal usage telemetry to
  improve result ordering over time.
- Respect Home Assistant exposure, hidden/disabled state, and safety boundaries.
- Keep embeddings optional and out of the v1 hot path.
- Keep the subsystem isolated enough to extract into a separate component later.

## Non-Goals

- Do not replace Home Assistant as the source of truth for state, registries, or
  service/action execution.
- Do not expose raw unrestricted service-call tools by default.
- Do not require an external database, external MCP server, LangChain,
  LangGraph, Postgres, or pgvector.
- Do not require embeddings for common control commands.
- Do not index or prompt with raw recorder history, raw logbook streams, or all
  entity attributes.
- Do not silently expose entities that Home Assistant or the integration policy
  has excluded.
- Do not build long-term conversation memory in this subsystem.

## High-Level Architecture

`semantic_home` should be implemented as an internal package with three layers:

| Layer                  | Modules                                                                                    | Responsibility                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| Core                   | `models.py`, `capabilities.py`, `index.py`, `ranker.py`, `retrieval.py`, `result_cache.py` | Pure semantic map, indexing, ranking, matching, pagination                                     |
| Home Assistant adapter | `ha_adapter.py`, `stats.py`, `storage.py`                                                  | Registry/state reads, event listeners, recorder/statistics enrichment, persisted learned stats |
| LLM API edge           | `llm_api.py`, `tools.py`                                                                   | HA `llm.API` registration and bounded tool implementations                                     |

Initial package layout:

```text
custom_components/pydantic_ai_agent/
  semantic_home/
    __init__.py
    capabilities.py
    ha_adapter.py
    index.py
    llm_api.py
    models.py
    ranker.py
    result_cache.py
    retrieval.py
    stats.py
    storage.py
    tools.py
```

Dependency direction:

```text
conversation.py/config_flow.py/__init__.py
  -> semantic_home.llm_api / semantic_home.ha_adapter

semantic_home.llm_api
  -> semantic_home.tools
  -> semantic_home.retrieval

semantic_home.retrieval
  -> semantic_home.index
  -> semantic_home.ranker

semantic_home.ha_adapter
  -> Home Assistant registries, states, services, recorder APIs
```

Forbidden dependency direction:

```text
semantic_home core -> pydantic_ai.Agent
semantic_home core -> conversation.py
semantic_home core -> config_flow.py
semantic_home core -> provider credentials
semantic_home core -> raw ConfigEntry/Subentry objects
```

The core should operate on typed snapshots and cards. Home Assistant objects
should be converted at the adapter edge.

## Functional Requirements

| ID     | Requirement                                                                                                                              |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| FR-001 | Build an in-memory semantic map from HA areas, floors, devices, entities, states, labels, aliases, and supported features.               |
| FR-002 | Collapse noisy entity surfaces into device and capability cards.                                                                         |
| FR-003 | Prefer canonical room/group entities for broad commands, such as room light groups over individual bulbs.                                |
| FR-004 | Exclude diagnostic, configuration, disabled, hidden, and unavailable entities from default control candidates unless policy allows them. |
| FR-005 | Resolve common natural-language physical-control commands without requiring a search tool call.                                          |
| FR-006 | Provide bounded search for ambiguous or exploratory queries.                                                                             |
| FR-007 | Cache result sets and support pagination through stable cursors or result set IDs.                                                       |
| FR-008 | Track internal usage and success statistics to improve ranking.                                                                          |
| FR-009 | Use Home Assistant recorder/logbook/statistics signals as ranking enrichment, not as prompt context.                                     |
| FR-010 | Require deterministic safety validation before any action is executed.                                                                   |
| FR-011 | Register as a Home Assistant `llm.API` so it appears in existing `CONF_LLM_HASS_API` selection.                                          |
| FR-012 | Expose diagnostics for index health, ranking inputs, cache size, excluded counts, and last rebuild time.                                 |
| FR-013 | Support future optional embedding and SQLite backends behind stable interfaces.                                                          |

## Technical Requirements

| ID     | Requirement                                                                                                  |
| ------ | ------------------------------------------------------------------------------------------------------------ |
| TR-001 | Do not block the Home Assistant event loop during index builds, stats refreshes, or embedding work.          |
| TR-002 | Store runtime objects on typed `entry.runtime_data`, not `hass.data`.                                        |
| TR-003 | Register any integration-wide services in `async_setup()`, not `async_setup_entry()`.                        |
| TR-004 | Keep persisted storage versioned and rebuildable.                                                            |
| TR-005 | Redact prompts, provider credentials, headers, tokens, and any learned free-text corrections in diagnostics. |
| TR-006 | Use JSON-serializable diagnostics attributes only.                                                           |
| TR-007 | Unit tests must not call real LLM providers.                                                                 |
| TR-008 | Embedding providers, if added later, must be optional and must not be required for setup or basic control.   |

## Runtime Ownership

The existing `PydanticAIAgentRuntimeData` currently stores provider data and MCP
server data in a frozen dataclass. `semantic_home` should add a contained runtime
object at construction time rather than scattering state across modules or
mutating frozen runtime data after setup.

Proposed runtime shape:

```python
@dataclass(frozen=True, kw_only=True)
class PydanticAIAgentRuntimeData:
    provider_mode: str
    name: str
    api_key: str
    base_url: str | None
    mcp_servers: list[dict[str, Any]]
    semantic_home: SemanticHomeRuntime | None = None
```

Proposed semantic runtime shape:

```python
@dataclass(kw_only=True)
class SemanticHomeRuntime:
    index: SemanticHomeIndex
    ranker: SemanticHomeRanker
    retriever: SemanticHomeRetriever
    stats: SemanticHomeStats
    result_cache: ResultSetCache
    unregister_api: Callable[[], None]
```

The runtime object should own in-memory state and unload callbacks. It should not
own provider credentials. Because the parent runtime dataclass is frozen today,
`async_setup_entry()` must build the semantic runtime before assigning
`entry.runtime_data`, or the design must introduce another explicit immutable
container that is assigned as part of setup.

## Future Extraction Boundary

If extracted later, the new component should own:

- semantic map construction
- registry and state listeners
- stats and ranking storage
- Home Assistant `llm.API` registration
- diagnostics for semantic indexing and ranking

`pydantic_ai_agent` would then consume it through normal HA LLM API selection:

```text
llm.async_get_apis(hass)
CONF_LLM_HASS_API
chat_log.async_provide_llm_data(...)
```

This means the in-tree version should avoid assumptions that only make sense for
Pydantic AI. The `llm.API` surface is the stable integration boundary.

## Data Model

### AreaCard

Represents a Home Assistant area and its ranking context.

Required fields:

- `area_id`
- `name`
- `aliases`
- `floor_id`
- `labels`
- `device_count`
- `entity_count`
- `capabilities`
- `recent_activity_score`

### DeviceCard

Represents a physical or logical device after noisy entities are collapsed.

Required fields:

- `device_id`
- `name`
- `aliases`
- `area_id`
- `floor_id`
- `manufacturer`
- `model`
- `labels`
- `integration_domain`
- `capability_ids`
- `primary_entity_ids`
- `support_entity_ids`
- `excluded_entity_count`
- `availability_state`
- `usage_score`
- `activity_score`
- `safety_class`

### CapabilityCard

Represents what can be read or controlled.

Required fields:

- `capability_id`
- `device_id`
- `area_id`
- `type`
- `actions`
- `read_entities`
- `control_entities`
- `canonical_entity_id`
- `confidence`
- `requires_confirmation`
- `is_group`
- `is_diagnostic`
- `is_configuration`

Example capability types:

- `light.on_off`
- `light.brightness`
- `light.color`
- `cover.position`
- `lock.lock_state`
- `climate.temperature`
- `fan.speed`
- `presence.occupancy`
- `camera.observation`
- `sensor.physical_measurement`

### EntityCard

Represents one HA entity after normalization.

Required fields:

- `entity_id`
- `domain`
- `name`
- `aliases`
- `device_id`
- `area_id`
- `device_class`
- `entity_category`
- `hidden`
- `disabled`
- `available`
- `state_summary`
- `supported_features`
- `labels`
- `conversation_exposure`
- `include_policy`
- `noise_tags`

### ResultSet

Represents a stable cached retrieval result.

Required fields:

- `result_set_id`
- `created_at`
- `expires_at`
- `query_fingerprint`
- `ranked_ids`
- `total_estimate`
- `sort_version`

The result set should store references and compact ranked IDs, not full prompt
payloads.

## Capability Mapping

Capability mapping should convert raw HA domains, device classes, and supported
features into physical capabilities.

Initial control capability mapping:

| HA domain      | Capability types                                  |                                         Default priority |
| -------------- | ------------------------------------------------- | -------------------------------------------------------: |
| `light`        | `light.on_off`, `light.brightness`, `light.color` |                                                     High |
| `cover`        | `cover.open_close`, `cover.position`              |                                                     High |
| `climate`      | `climate.temperature`, `climate.mode`             |                                                     High |
| `fan`          | `fan.on_off`, `fan.speed`                         |                                                     High |
| `lock`         | `lock.lock_state`                                 |                                   High with confirmation |
| `media_player` | `media.playback`, `media.volume`                  |                                                   Medium |
| `switch`       | `switch.on_off`                                   | Medium by default, high only with strong physical naming |
| `scene`        | `scene.activate`                                  |                        Medium with explicit scene intent |
| `script`       | `script.run`                                      |                       Medium with explicit script intent |

Initial read capability mapping:

| HA domain        | Capability types                                                                            |                             Default priority |
| ---------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------: |
| `binary_sensor`  | `presence.occupancy`, `door.opening`, `motion.activity`, `moisture.leak`                    |               Medium to high by device class |
| `sensor`         | `sensor.temperature`, `sensor.humidity`, `sensor.power`, `sensor.energy`, `sensor.activity` | Low to high by device class and query intent |
| `camera`         | `camera.observation`                                                                        |                 High for observation intents |
| `person`         | `person.location`                                                                           |                      Context only by default |
| `device_tracker` | `tracker.location`                                                                          |                      Context only by default |

Noise tags should be assigned during mapping.

Initial noise tags:

- `diagnostic`
- `configuration`
- `firmware`
- `ip_address`
- `mac_address`
- `rssi`
- `battery_telemetry`
- `update_status`
- `engineering_mode`
- `calibration`
- `display_indicator`
- `duplicate_like`
- `unavailable`
- `hidden`

## Inclusion Policy

The subsystem needs an explicit policy because sampled conversation exposure
settings were sparse.

Recommended default policy for development:

```text
eligible = enabled and not hidden and available and not diagnostic/config
control_allowed = eligible and passes integration exposure policy
read_allowed = eligible and passes read exposure policy
```

Supported policy modes:

| Mode                  | Behavior                                                                                                                                     |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `assist_exposed_only` | Only entities exposed to HA conversation are indexed for read/control. Safest, but sparse in the sampled home.                               |
| `physical_default`    | Index physical non-diagnostic entities, still requiring explicit safety checks for control. Recommended development default if user opts in. |
| `areas_and_domains`   | Index only configured areas/domains. Useful for staged rollout.                                                                              |
| `debug_all_visible`   | Index visible non-disabled entities for diagnostics only. Must not be the default.                                                           |

The UI must make the selected policy visible. Diagnostics must report counts by
included and excluded reason.

## Ranking Design

Ranking should combine static registry quality, dynamic usage, activity, query
match, and safety confidence.

High-level score:

```text
final_score =
  query_match_score
  + location_score
  + capability_score
  + canonical_score
  + usage_score
  + activity_score
  + freshness_score
  + availability_score
  - noise_penalty
  - ambiguity_penalty
  - safety_penalty
```

Scores should be capped and normalized. No single noisy signal should dominate.

### Static Ranking Signals

| Signal                                 | Effect                             |
| -------------------------------------- | ---------------------------------- |
| Exact area match                       | Strong positive                    |
| Current conversation device area match | Strong positive                    |
| Floor match                            | Medium positive                    |
| Exact friendly name or alias match     | Strong positive                    |
| Domain/capability matches intent       | Strong positive                    |
| Room-level group entity                | Strong positive for broad commands |
| Device has physical area               | Positive                           |
| Label indicates canonical group        | Positive                           |
| Hidden or disabled                     | Strong negative or excluded        |
| Diagnostic/config category             | Strong negative or excluded        |
| Unavailable                            | Negative or excluded for control   |
| Duplicate-like suffix                  | Negative                           |
| Display indicator or telemetry name    | Negative                           |

### Dynamic Ranking Signals

| Signal                         | Source                | Effect                                                   |
| ------------------------------ | --------------------- | -------------------------------------------------------- |
| Successful tool use count      | Internal stats        | Strong positive with time decay                          |
| Recent successful control      | Internal stats        | Strong positive                                          |
| Manual state changes           | Recorder/logbook      | Positive if meaningful                                   |
| Recent meaningful state change | State history/logbook | Positive for read relevance                              |
| High unavailable rate          | State history         | Negative                                                 |
| Ambiguity frequency            | Internal stats        | Negative until disambiguated                             |
| User correction                | Internal stats        | Positive for corrected target, negative for wrong target |
| High churn telemetry           | Logbook/history       | Negative unless query intent asks for activity           |

### Churn-Normalized Activity

The sampled logbook showed heavy churn from presence and thermostat updates. Raw
event count must not be treated as importance.

Use meaningful transition scoring:

```text
meaningful_activity = unique_state_transitions * domain_weight
churn_penalty = max(0, raw_update_count - meaningful_transition_count * ratio)
activity_score = cap(meaningful_activity) - cap(churn_penalty)
```

Examples:

- `binary_sensor.office_presence_sensor_occupant_presence` is useful for office
  occupancy context.
- `sensor.office_presence_sensor_moving_energy_g8` should not become a top
  result because it changes often.
- `climate.thermostat` can rank highly because thermostat state changes are
  user-relevant and there is only one climate entity.

### Usage Learning

Internal usage stats should be stored separately from HA recorder data.

Track per target:

- `successful_control_count_1d`
- `successful_control_count_7d`
- `successful_control_count_30d`
- `last_successful_control`
- `last_read_result`
- `failed_control_count`
- `ambiguity_count`
- `correction_count`
- `last_corrected_alias`

Use exponential decay so current habits matter more than old habits.

Example:

```text
usage_score =
  4.0 * controls_1d
  + 2.0 * controls_7d
  + 0.5 * controls_30d
  + recent_success_bonus
```

The constants should be implementation defaults, not user-facing configuration.

## Retrieval Strategy

Retrieval should run in stages. Each stage reduces the candidate set.

```text
1. Parse intent hints from tool arguments.
2. Apply metadata filters: area, floor, domain, capability, exposure, safety.
3. Apply lexical match over names, aliases, area names, and capability tokens.
4. Apply ranking using static and dynamic scores.
5. Optionally apply semantic reranking when enabled and confidence is low.
6. Return bounded compact cards or execute deterministic control.
```

The default path for common control commands should not call `search_home` first.
The control tool should perform internal resolution.

## LLM Vocabulary And Prompting Contract

`ControlHome` is not designed around assumed model knowledge. The model must be
grounded at runtime with the HA `llm.API` prompt, tool schemas, tool and parameter
descriptions, and curated dynamic examples from the current Home Assistant
instance. These prompting surfaces help the model choose valid values, but all
selected values remain untrusted input and must be validated server-side before
execution.

The LLM does not need to know canonical Home Assistant area IDs, capability IDs,
entity IDs, service names, or action names before the request. It should provide
natural-language intent hints or opaque references it has already received from
`SearchHome` or prior tool results. The deterministic resolver owns
canonicalization.

Resolution priority:

```text
explicit user location hint
  > target_ref from prior semantic_home result
  > conversation origin area from llm_context.device_id
  > learned usage/default ranking
```

The conversation origin area is only a default. It must help with commands like
`turn off the lights`, but it must not override explicit requests like
`turn off the kitchen lights`.

### Prompting Surfaces

The subsystem should document and control every prompt surface it provides to the
model.

| Surface                | Source                            | Purpose                                                                                       | Constraints                                              |
| ---------------------- | --------------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| API prompt             | `llm.APIInstance.api_prompt`      | Explain resolver behavior, safety rules, when to call each tool, and how origin area is used. | Compact, no raw entity dump, no secrets.                 |
| Tool descriptions      | `llm.Tool.description`            | Tell the model which tool to use for control, state, search, or diagnostics.                  | May include bounded real examples from the current home. |
| Parameter descriptions | Tool parameter schema metadata    | Explain accepted hint fields, opaque refs, limits, and when a value is optional.              | Descriptions should discourage invented canonical IDs.   |
| Dynamic examples       | Generated from the semantic index | Improve selection for real areas, devices, groups, and capabilities.                          | Representative examples only; never all entities.        |
| Injected context       | Generated per `llm_context`       | Provide origin area/floor, policy mode, and a few relevant examples.                          | Must be minimal and regenerated when stale.              |
| Tool results           | Runtime tool responses            | Return validated refs, compact cards, ambiguity reasons, and execution summaries.             | Bounded, paginated, and redacted.                        |

### API Prompt Content

The Semantic Home API prompt should include these instructions:

```text
You can control and query the home through semantic_home tools.
Do not assume Home Assistant entity IDs, area IDs, capabilities, or service names.
Use ControlHome for simple physical control requests when the user intent is clear.
Use GetHomeState for current state questions.
Use SearchHome when the target is ambiguous or exploratory.
Use target_ref values from SearchHome results when available.
The user's explicit location words override the origin area.
If no location is specified, the origin area may be used as a default hint.
Do not invent target_ref values.
If a request is ambiguous or unsafe, ask for clarification or let the tool return
clarification candidates.
```

When `llm_context.device_id` maps to an area, the prompt may include compact
origin context:

```text
Origin context: the request was heard in Master Bedroom on Top Floor.
Use this only when the user did not name another location.
```

### Dynamic Tool And Parameter Examples

Tool and parameter descriptions may include real examples from the current home
when they improve model success. These examples should be generated from the
semantic index and ranking data.

Example `ControlHome` description fragment:

```text
Use this for physical control such as lights, fans, covers, climate, locks, and
media players. Examples in this home: "turn off the master bedroom lights" ->
location_hint="master bedroom", target_hint="lights", action_hint="turn off";
"close the living room curtains" -> location_hint="living room",
target_hint="curtains", action_hint="close".
```

Example `location_hint` parameter description fragment:

```text
Natural-language location from the user's request, such as "kitchen", "office",
"master bedroom", or "garage". Leave empty when the user did not specify a
location; the resolver can use the origin area.
```

Example `target_hint` parameter description fragment:

```text
Natural-language target from the user's request, such as "lights", "bedside
lamp", "curtains", "garage door", "thermostat", or "fan". Do not provide an
entity ID unless a prior tool result explicitly returned it.
```

Dynamic examples should be selected by relevance:

- Prefer the origin area and nearby/frequently used areas.
- Prefer canonical groups and high-confidence capabilities.
- Prefer common physical control targets over diagnostic entities.
- Include no more examples than needed for tool-use accuracy.
- Regenerate examples after index generation changes.

Examples are hints, not permissions. A target shown in a tool description still
must pass policy, safety, and availability checks before execution.

### Prompting Boundaries

- Treat all model-provided values as untrusted.
- Prompting may improve selection but must never replace resolver validation.
- Tool handlers must validate areas, target refs, capabilities, actions, and
  parameters against the current semantic index.
- Invalid, stale, excluded, unsafe, or ambiguous values must fail closed or return
  clarification candidates.
- Do not expose secrets, tokens, provider credentials, precise private locations,
  raw history, or unnecessary sensitive states in prompts.
- Do not inject the full semantic map. Inject only compact, relevant examples and
  use tools for retrieval.

## LLM API Tool Contract

`semantic_home` should register one HA `llm.API` with a small, stable tool set.

Recommended API name:

```text
Semantic Home
```

Recommended API ID:

```text
pydantic_ai_agent_semantic_home
```

If extracted later, the ID can move to the new domain. A migration plan should
preserve user selection or document the breaking change.

### `ControlHome`

Primary control tool. This should be the hot path.

`ControlHome` accepts intent hints, not authoritative canonical values. It may
also accept an opaque `target_ref` returned by `SearchHome` or a previous
`semantic_home` tool result. The resolver must canonicalize all hints before it
chooses any Home Assistant service/action.

Arguments:

```json
{
  "request": "turn off the bedroom lights",
  "target_ref": null,
  "action_hint": "turn off",
  "target_hint": "lights",
  "location_hint": "bedroom",
  "parameters": {}
}
```

Follow-up arguments using a validated opaque reference:

```json
{
  "request": "turn those off",
  "target_ref": "capability:master_bedroom:light.group",
  "action_hint": "turn off",
  "parameters": {}
}
```

Responsibilities:

- Treat tool arguments as untrusted hints.
- Canonicalize action, target, location, and parameters internally.
- Use conversation origin area only when the user did not specify a location.
- Resolve target internally or return clarification candidates.
- Prefer canonical group targets when appropriate.
- Validate action against capability.
- Validate safety policy.
- Execute one or more Home Assistant service/action calls.
- Record success/failure stats.
- Return a compact execution summary.

Return shape:

```json
{
  "status": "success",
  "resolved_target": "Master Bedroom Lights Group",
  "area": "master_bedroom",
  "action": "turn_off",
  "controlled_entities": ["light.master_bedroom_lights_group"],
  "resolution_reasons": ["explicit_location", "canonical_group"],
  "requires_follow_up": false
}
```

Ambiguity return shape:

```json
{
  "status": "needs_clarification",
  "message": "Which bedroom lights should I control?",
  "candidates": [
    {
      "target_ref": "capability:master_bedroom:light.group",
      "name": "Master Bedroom Lights Group",
      "area": "Master Bedroom"
    },
    {
      "target_ref": "capability:guest_room:light.group",
      "name": "Guest Room Lights Group",
      "area": "Guest Room"
    }
  ]
}
```

### `GetHomeState`

Primary read tool.

Arguments:

```json
{
  "request": "is the garage door open",
  "target_ref": null,
  "target_hint": "garage door",
  "location_hint": "garage",
  "include_related": true
}
```

Responsibilities:

- Resolve relevant state targets.
- Include compact related context only when useful.
- Avoid raw diagnostic dumps.
- Return current state and freshness.

### `SearchHome`

Fallback discovery tool.

Arguments:

```json
{
  "query": "bedroom lamps",
  "filters": {
    "location_hint": "bedroom",
    "capability_hint": "light"
  },
  "limit": 5,
  "cursor": null
}
```

Responsibilities:

- Return compact ranked cards.
- Return a stable cursor for more results.
- Do not execute actions.
- Explain why top results ranked highly when helpful.

Return shape:

```json
{
  "result_set_id": "rs_abc123",
  "matches": [
    {
      "target_ref": "capability:master_bedroom:light.group",
      "name": "Master Bedroom Lights Group",
      "area": "Master Bedroom",
      "capabilities": ["light.on_off", "light.brightness"],
      "score": 0.94,
      "reasons": ["area_match", "canonical_group", "usage"]
    }
  ],
  "next_cursor": null,
  "total_estimate": 1
}
```

### `ExplainHomeResolution`

Optional diagnostics/debug tool. It should be disabled by default for normal
users unless needed for troubleshooting.

Responsibilities:

- Explain candidate filtering and scoring.
- Show excluded reason counts.
- Avoid exposing sensitive or hidden raw metadata.

## Fast Path Control Flow

Example prompt:

```text
Turn off the lights.
```

Conversation context:

```text
device_id -> area master_bedroom
```

Expected flow:

```text
1. API prompt tells the model the origin area is Master Bedroom.
2. Model calls ControlHome(request="Turn off the lights", action_hint="turn off", target_hint="lights").
3. Semantic resolver uses origin area because the user did not specify another location.
4. Semantic resolver picks light.master_bedroom_lights_group.
5. Tool validates policy and executes light.turn_off.
6. Tool returns compact success summary.
7. Assistant returns final response.
```

Explicit-location example:

```text
Prompt: Turn off the kitchen lights.
Origin area: Master Bedroom.
Tool call: ControlHome(request="Turn off the kitchen lights", action_hint="turn off", target_hint="lights", location_hint="kitchen").
Resolver behavior: kitchen overrides origin area, then resolves kitchen lights.
```

Target behavior:

- No `SearchHome` call.
- No raw list of 19 bedroom lights returned to the model.
- No control of AWTRIX indicators or hidden display lights.
- No duplicate service calls for both group and member lights.

The latency target is best-effort, not a hard guarantee. The architecture should
aim for one model tool call and one Home Assistant service/action execution for
high-confidence commands.

## Search and Pagination

Search must be bounded by design.

Default limits:

| Setting                                |   Default |
| -------------------------------------- | --------: |
| Search result limit                    |         5 |
| Maximum search result limit            |        20 |
| Result set TTL                         | 5 minutes |
| Maximum cached result sets per runtime |       100 |

Result sets should be keyed by:

- query text
- normalized filters
- index generation
- policy mode
- conversation subentry ID when relevant

Pagination should read from cached result IDs rather than rerunning retrieval.

## Indexing Lifecycle

Initial setup:

```text
1. Read area and floor registries.
2. Read device registry.
3. Read entity registry.
4. Read current states.
5. Normalize entity cards.
6. Build device and capability cards.
7. Compute static scores.
8. Load learned stats.
9. Mark index ready.
```

Invalidation triggers:

- entity registry create/update/remove
- device registry create/update/remove
- area registry create/update/remove
- floor registry create/update/remove
- config entry or subentry update
- exposure policy update
- Home Assistant start
- relevant service availability change

State update handling:

- Current state changes should update lightweight state summaries.
- State changes should not rebuild the full index unless metadata changed.
- High-volume updates should be debounced and summarized.

Stats refresh handling:

- Internal usage stats should update synchronously after tool calls.
- Recorder/logbook/statistics enrichment should run in background refresh jobs.
- Background refresh should be cancellable on unload.

## Storage Design

V1 should keep the index in memory and persist only learned metadata.

Persisted storage contents:

- schema version
- target usage counters
- learned aliases and corrections
- ambiguity and failure counts
- optional last scoring snapshots for diagnostics

Do not persist:

- provider credentials
- raw prompts
- raw conversation history
- raw recorder history
- full state history
- embeddings in v1

Future optional SQLite backend may persist:

- embedding vectors
- card text hashes
- vector model metadata
- rebuild generation
- search cache metadata

SQLite must remain optional and rebuildable.

## Embeddings Design

Embeddings are a future enhancement, not a v1 requirement.

Embedding scope should be compact semantic cards, not raw entities.

Good embedding text:

```text
Master Bedroom Lights Group. Area: Master Bedroom. Type: room light group.
Capabilities: turn on, turn off, set brightness. Common aliases: bedroom lights.
```

Bad embedding text:

```text
Every entity ID, every attribute, all state history, all diagnostic sensors.
```

Embedding retrieval should always be metadata-filtered first:

```text
area/domain/capability/exposure/safety filters -> vector rerank
```

Embedding provider configuration must be optional and explicit because cloud
embedding providers may receive home metadata.

## Safety Model

Safety checks must happen after retrieval and before actuation.

Safety classes:

| Safety class          | Examples                                     | Default behavior                          |
| --------------------- | -------------------------------------------- | ----------------------------------------- |
| `read_only`           | occupancy, temperature, door state           | Allow if readable by policy               |
| `low_risk_control`    | lights, fans, media playback                 | Allow if high confidence                  |
| `medium_risk_control` | covers, climate changes, scenes              | Allow with stronger confidence and bounds |
| `high_risk_control`   | locks, garage doors, alarms, unknown scripts | Require confirmation                      |

Safety requirements:

- Never execute an action against an excluded target.
- Never execute a low-confidence action silently.
- Never use unbounded raw service calls as the default control path.
- Require confirmation for locks, garage doors, alarm controls, and unknown
  scripts.
- Avoid double-control when both a group and its members match.
- Return ambiguity instead of guessing when top candidates are too close.

## Configuration Design

V1 should require minimal configuration.

Recommended user-facing settings:

| Setting                    | Scope                                     | Default                                                        |
| -------------------------- | ----------------------------------------- | -------------------------------------------------------------- |
| Enable Semantic Home API   | Parent entry or global integration option | Enabled only after user opts in during development             |
| Exposure policy            | Parent entry                              | `physical_default` or `assist_exposed_only`, final default TBD |
| Diagnostic/config entities | Parent entry                              | Excluded                                                       |
| Result verbosity           | Conversation subentry                     | Compact                                                        |
| Control enabled            | Conversation subentry                     | Mirrors HA LLM API selection                                   |
| Confirmation strictness    | Conversation subentry                     | Standard                                                       |

Advanced settings should be hidden initially:

- result limit
- ranking debug mode
- embedding mode
- embedding model
- SQLite backend
- inclusion overrides

Provider credentials remain on the parent provider config entry. Semantic Home
settings must not require duplicating provider credentials.

## Diagnostics and Observability

Diagnostics should expose enough data to tune ranking without leaking secrets or
large histories.

Recommended diagnostics fields:

- `semantic_home.enabled`
- `semantic_home.index_ready`
- `semantic_home.index_generation`
- `semantic_home.last_rebuild_time`
- `semantic_home.entity_count_seen`
- `semantic_home.entity_count_indexed`
- `semantic_home.device_count_indexed`
- `semantic_home.capability_count_indexed`
- `semantic_home.excluded_by_reason`
- `semantic_home.result_cache_size`
- `semantic_home.stats_store_version`
- `semantic_home.embedding_enabled`
- `semantic_home.embedding_backend`
- `semantic_home.last_error_reason`

Optional debug diagnostics should be separately gated:

- top ranked targets by area/capability
- sample ranking reasons
- ambiguous target examples
- slowest resolution timings

Debug output must redact learned aliases or free-text corrections if they might
contain sensitive user text.

## Performance Design

The sampled home size should be treated as a baseline performance target.

Targets:

| Operation                                  |                                 Target |
| ------------------------------------------ | -------------------------------------: |
| Build in-memory index for 2,892 entities   |   Non-blocking; complete in background |
| Resolve high-confidence area light command | Less than 100 ms inside tool execution |
| Return search page from cache              |                        Less than 50 ms |
| Default search results                     |                        5 compact cards |
| Hot path model tool calls                  |          1 for high-confidence control |
| Tool calls before final answer             |     No more than 3 for normal commands |

Performance tactics:

- Precompute normalized tokens and aliases.
- Precompute area and capability indexes.
- Precompute canonical group preferences.
- Keep result payloads compact.
- Use debounced incremental updates for registry changes.
- Use background refresh for recorder/statistics enrichment.
- Do not compute embeddings during user requests.

## Testing Strategy

New tests should live under:

```text
tests/components/pydantic_ai_agent/test_semantic_home_*.py
```

Recommended test modules:

| Test module                         | Coverage                                                    |
| ----------------------------------- | ----------------------------------------------------------- |
| `test_semantic_home_index.py`       | Card construction, grouping, exclusion reasons              |
| `test_semantic_home_ranker.py`      | Static and dynamic ranking, group preference, churn penalty |
| `test_semantic_home_retrieval.py`   | Query filters, pagination, result cache stability           |
| `test_semantic_home_tools.py`       | Tool schemas, target resolution, safety rejection           |
| `test_semantic_home_llm_api.py`     | HA `llm.API` registration and API instance contents         |
| `test_semantic_home_diagnostics.py` | Redaction and index health diagnostics                      |

Required test scenarios:

- Master bedroom light command prefers `light.master_bedroom_lights_group` over
  individual lights and AWTRIX indicators.
- Presence sensor with many entities collapses into one device card with a small
  number of useful capabilities.
- Diagnostic/config entities are excluded by default.
- Unavailable and hidden entities do not appear as control candidates.
- Large search results return a bounded first page and stable cursor.
- High-risk controls require confirmation.
- Usage stats improve ranking without overriding explicit user location or target
  hints.
- High-churn telemetry is penalized for generic queries.
- Recorder/statistics failures do not prevent basic control.
- No test calls a real LLM provider.

Run validation through existing scripts:

```text
scripts/test
scripts/lint-check
scripts/type-check
scripts/check
```

## Phased Delivery Plan

### Phase 1: In-Memory Semantic Map

Deliverables:

- `semantic_home` package skeleton.
- Typed card models.
- HA adapter for areas, devices, entities, and states.
- In-memory index.
- Default exclusion policy.
- Diagnostics counts.
- Unit tests for indexing and filtering.

Exit criteria:

- Sample-sized home can be indexed without event-loop blocking.
- No prompts or tools expose raw 2,892-entity lists.

### Phase 2: Deterministic Resolver and LLM API

Deliverables:

- HA `llm.API` implementation.
- `ControlHome`, `GetHomeState`, and `SearchHome` tools.
- Capability mapping for lights, covers, climate, fans, locks, switches, sensors,
  and binary sensors.
- Safety validation.
- Result cache and pagination.

Exit criteria:

- `turn off master bedroom lights` resolves to the canonical room light group.
- Common natural-language control commands do not require a search call when
  confidence is high.
- Ambiguous commands return a bounded clarification result.

### Phase 3: Usage and Activity Ranking

Deliverables:

- Internal usage stats store.
- Ranking updates after successful and failed tool calls.
- Recorder/logbook/statistics enrichment hooks.
- Churn-normalized activity scoring.
- Diagnostics for ranking signals.

Exit criteria:

- Frequently used targets rise in rankings.
- Noisy telemetry does not dominate generic search results.
- Stale or unavailable entities are demoted.

### Phase 4: Optional Embeddings and SQLite

Deliverables:

- Embedding backend interface.
- Optional embedding model configuration.
- Optional SQLite vector/cache storage.
- Metadata-first vector reranking.
- Rebuild and privacy diagnostics.

Exit criteria:

- Lexical mode remains fully functional without embeddings.
- Embeddings improve fuzzy-match queries without changing safety behavior.
- SQLite data is rebuildable and versioned.

### Phase 5: Extraction Readiness

Deliverables:

- Stable internal interface for consumers.
- No provider-specific dependencies in semantic core.
- Clear migration note for separate component ownership.
- Optional service/API boundary for external consumers.

Exit criteria:

- `semantic_home` can be moved to another custom component with minimal changes to
  core models, ranker, retrieval, and storage.

## Acceptance Criteria

- The subsystem indexes the sampled scale of 2,892 entities without blocking Home
  Assistant startup or the event loop.
- The subsystem collapses noisy devices, including a presence sensor with dozens
  of entities, into compact device and capability cards.
- The subsystem excludes diagnostic/config/hidden/unavailable entities from
  default control candidates.
- The subsystem prefers room-level light groups for broad room light commands.
- The subsystem supports deterministic control resolution for common commands.
- The subsystem treats LLM-provided tool arguments as hints and validates all
  canonical areas, target refs, capabilities, actions, and parameters before
  execution.
- The subsystem documents and tests the API prompt, tool descriptions, parameter
  descriptions, dynamic examples, and injected context used to ground the model.
- The subsystem returns paginated bounded search results with stable cursors.
- The subsystem records internal usage stats and applies time-decayed ranking.
- The subsystem uses recorder/logbook/statistics data only as bounded ranking
  signals.
- The subsystem handles recorder/statistics unavailability gracefully.
- The subsystem registers a Home Assistant `llm.API` that can be selected through
  the existing `CONF_LLM_HASS_API` flow.
- The subsystem provides diagnostics with index health and exclusion counts.
- Tests cover filtering, grouping, ranking, pagination, safety, diagnostics, and
  API registration.

## Open Questions

| Question                                                                                       | Decision needed before |
| ---------------------------------------------------------------------------------------------- | ---------------------- |
| Should the development default exposure policy be `assist_exposed_only` or `physical_default`? | Phase 2                |
| Should Semantic Home be enabled automatically for new agents or require explicit opt-in?       | Phase 2                |
| Should learned aliases be global per provider entry or scoped per conversation subentry?       | Phase 3                |
| Which recorder/statistics APIs are reliable enough across HA installs for enrichment?          | Phase 3                |
| Which embedding providers should be supported first, if any?                                   | Phase 4                |
| What component domain should be used if extracted later?                                       | Phase 5                |

## Recommended Initial Decisions

- Build in-tree under `custom_components/pydantic_ai_agent/semantic_home/`.
- Keep the semantic core independent of Pydantic AI.
- Register a normal HA `llm.API` and let the existing conversation flow select it.
- Start with lexical, metadata, and ranking retrieval only.
- Use embeddings only after measuring miss cases.
- Store only learned usage and alias metadata in v1.
- Treat raw service calls as out of scope for the default API.
- Make safety and exposure policy visible in config and diagnostics.
