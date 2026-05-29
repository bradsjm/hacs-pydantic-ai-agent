# Entities And Runtime Execution

## Contents

- Purpose
- Authoritative Sources
- Platform Setup
- Coordinator-Backed Entities
- Entity Descriptions
- Unique IDs Device Info And Ownership
- State Attributes And JSON Safety
- Diagnostic Entities And Metrics
- Conversation Entities
- AI Task Entities
- Shared Runtime Execution
- Events And Dispatchers
- Anti-Patterns
- Tests To Expect

## Purpose

Use this reference when creating or reviewing entity platforms, shared runtime execution, diagnostic entities, metrics, conversation entities, AI task entities, state attributes, device info, and dispatcher updates.

## Authoritative Sources

- Entity platform docs: https://developers.home-assistant.io/docs/core/entity/
- Entity properties: https://developers.home-assistant.io/docs/core/entity/#generic-properties
- Fetching data and coordinator entities: https://developers.home-assistant.io/docs/integration_fetching_data/
- Device registry: https://developers.home-assistant.io/docs/device_registry_index/
- Conversation integration: https://developers.home-assistant.io/docs/core/entity/conversation/
- AI Task entity: https://developers.home-assistant.io/docs/core/entity/ai-task
- Integration Quality Scale rules: https://developers.home-assistant.io/docs/core/integration-quality-scale/rules/

## Platform Setup

Each platform implements `async_setup_entry(hass, entry, async_add_entities)` and creates entities from current config-entry runtime data.

Checklist:

- Use platform setup only after `entry.runtime_data` is assigned.
- Filter invalid child resources through a central iterator/resolver.
- Pass `config_subentry_id` when adding entities for config subentries.
- Use `AddConfigEntryEntitiesCallback` where available.
- Avoid network calls in entity constructors.
- Do not create entities for disabled/invalid child resources unless HA UX requires diagnostic visibility.

Entity constructors should receive stable runtime references, descriptions, and identity values. Keep expensive setup in coordinator/runtime managers.

## Coordinator-Backed Entities

When entities use `CoordinatorEntity`, read the coordinator from typed `entry.runtime_data` and pass the coordinator to the entity constructor. Do not retrieve it from `hass.data[DOMAIN][entry_id]`.

Coordinator-backed entity properties should be pure reads from coordinator data or precomputed fields. After a mutating action, request a coordinator refresh when the API/device state should be confirmed before the next scheduled poll.

Use `context` values for coordinator entities when the coordinator can use `async_contexts()` to limit fetched data to enabled/listening entities.

## Entity Descriptions

Use entity descriptions when a platform has repeated entity definitions that differ by key, device class, unit, state class, icon, or value extraction.

Rules:

- Use domain-specific description classes such as `SensorEntityDescription` or `BinarySensorEntityDescription`.
- Use `@dataclass(frozen=True, kw_only=True)` for custom description subclasses unless the base/domain pattern requires otherwise.
- Keep description keys stable and unique within the platform.
- Prefer `translation_key` and `icons.json` over hard-coded English names or dynamic icon properties.
- Use `exists_fn`-style predicates only when device capabilities truly vary.
- Use value functions only for cheap reads from already-fetched data.

## Unique IDs Device Info And Ownership

Build unique IDs from stable persisted identity, not display names.

HA scopes entity unique IDs by integration domain and platform, so unique IDs should not include the domain or platform name just to make them global.

Common patterns within a platform:

- Root-entry entity: `<entry_id>_<description_key>`.
- Subentry entity: `<subentry_id>_<description_key>`.
- External device entity: stable external serial/account/device ID plus entity description key.

Use `DeviceInfo` identifiers that match the same ownership model. For service-like entities, use a config/service device identity rather than pretending the integration is physical hardware.

Set `_attr_has_entity_name = True` and let HA compose names when possible.

## State Attributes And JSON Safety

Entity state and attributes must be cheap, deterministic, and JSON-safe.

Allowed output shapes:

- `None`, bool, int, float, str.
- Lists/tuples containing JSON-safe values.
- Dicts with string keys and JSON-safe values.

Convert before exposing:

- Dataclasses to dicts or selected fields.
- `datetime` to ISO strings using HA time utilities.
- Exceptions to safe string/category structures.
- SDK objects to primitive summaries.
- Long traces to bounded head/tail summaries.

Avoid raw prompts, credentials, request bodies, huge histories, or unbounded lists in attributes. Put large debug data behind diagnostics or response-only debug actions with explicit bounds.

## Diagnostic Entities And Metrics

Diagnostic entities are useful when users need ongoing health visibility, not when a debug response action is enough.

Patterns:

- Use `EntityCategory.DIAGNOSTIC`.
- Disable noisy or advanced diagnostic entities by default.
- Store metrics per entry in runtime data.
- Separate cumulative metrics from last-run metrics.
- Use dispatcher signals to update diagnostic entities after runtime changes.
- Return `None` for derived metrics when inputs are incomplete.

Good diagnostic entities include provider health, last-run success, cumulative request count, last error class, or estimated cost when pricing is configured. Bad diagnostic attributes include full traces, raw request content, or per-tool arguments.

## Conversation Entities

Use HA `ConversationEntity` and `ChatLog` when implementing Assist-facing agents.

Rules:

- Treat `ChatLog` as canonical conversation history.
- Do not add global conversation memory unless the user explicitly configures a separate memory feature and privacy model.
- Call `chat_log.async_provide_llm_data()` when HA LLM context/tools are enabled.
- Convert HA chat messages to SDK messages at a narrow boundary.
- Read attachments off the event loop with `hass.async_add_executor_job()`.
- Keep tool exposure tied to HA Assist permissions and configured LLM APIs.

Conversation tests should cover entity creation, supported languages/features, chat-log conversion, tool availability, streaming or partial responses if implemented, and error handling.

## AI Task Entities

Use AI task entities for structured data generation or non-conversational model tasks.

Rules:

- Validate returned data against the schema HA provides.
- Make attachments, tools, or workspace access explicit capabilities.
- Lock shared mutable user resources such as todo lists or workspaces.
- Avoid hidden conversation history unless the entity contract says it is stateful.
- Map provider/runtime failures to `HomeAssistantError` or the platform-specific failure contract.

AI task tests should cover schema validation, attachment handling, capability flags, tool composition, and cleanup after failure.

## Shared Runtime Execution

When multiple entity platforms use the same execution pipeline, keep it behind a shared runtime helper or base class.

Runtime execution should produce separate outputs:

- User-facing result or safe error.
- Developer log detail.
- Stable error category.
- Retry/fallback decision.
- Bounded diagnostics record.
- Metrics update.

For fallback chains, retry only on transient errors. Do not fallback on auth failure, permission denied, invalid model/input, unsupported feature, or bad request unless there is a deliberate product reason.

## Events And Dispatchers

Use dispatcher signals for internal entity updates. Use HA events only for user-observable outcomes that automations or debugging tools may reasonably consume.

Event payloads must be JSON-safe and secret-safe. Include stable IDs, categories, and counts rather than raw content.

Register dispatcher cleanup with entity `async_on_remove()` or entry unload callbacks.

## Anti-Patterns

- Building entities from display names instead of stable IDs.
- Doing network I/O in entity constructors or property methods.
- Looking up per-entry coordinators from `hass.data` instead of typed runtime data.
- Hard-coding English entity names where translations or device-class naming should be used.
- Returning dataclasses, datetimes, exceptions, SDK objects, or large traces in attributes.
- Storing metrics in module globals.
- Creating one global conversation history across config entries.
- Using entity state attributes as a debug dump.
- Fallback on auth/permission/bad-request errors.

## Tests To Expect

Entity and runtime tests should assert:

- Entity count and `config_subentry_id` ownership.
- Unique IDs and device info.
- Entity category and disabled-by-default registry behavior.
- State and attribute primitive shapes.
- Dispatcher update behavior.
- Entity description keys, translation keys, device classes, units, and existence predicates.
- Coordinator-backed refresh and context behavior when used.
- Conversation/AI task capabilities and failure paths.
- Metrics mutation on success and failure.
- Fallback classification matrix.
- Event payload shape when events are emitted.
