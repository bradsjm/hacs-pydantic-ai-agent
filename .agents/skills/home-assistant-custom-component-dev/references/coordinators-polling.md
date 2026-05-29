# Coordinators And Polling

## Contents

- Purpose
- Authoritative Sources
- When To Use A Coordinator
- Setup And Runtime Ownership
- Coordinator Design
- Error Semantics
- Coordinator-Backed Entities
- Push APIs And Coordinator Updates
- Multi-Coordinator Design
- Polling Without A Coordinator
- Anti-Patterns
- Tests To Expect

## Purpose

Use this reference when an integration polls an API, receives push updates that feed shared entity state, uses `DataUpdateCoordinator`, uses `CoordinatorEntity`, or needs multiple refresh managers with different cadences or failure modes.

## Authoritative Sources

- Fetching data and `DataUpdateCoordinator`: https://developers.home-assistant.io/docs/integration_fetching_data/
- Config entry setup failures: https://developers.home-assistant.io/docs/integration_setup_failures/
- Entity update strategies: https://developers.home-assistant.io/docs/core/entity/#updating-the-entity
- Entity registry behavior: https://developers.home-assistant.io/docs/entity_registry_index/
- Home Assistant Core examples: https://github.com/home-assistant/core/tree/dev/homeassistant/components

## When To Use A Coordinator

Use `DataUpdateCoordinator` when one refresh can produce data for multiple entities, devices, or platforms. It centralizes polling, error handling, availability, listener updates, and optional request de-duplication.

Do not claim coordinators are mandatory for every integration. Simpler entity-level polling can be acceptable when exactly one entity maps to one endpoint and there is no shared cache, no fan-out, and no meaningful setup data to pre-process.

Use a push/subscription model instead of polling when the API can notify HA about changes. A push integration can still use a coordinator if shared state and `CoordinatorEntity` semantics are useful.

## Setup And Runtime Ownership

Store coordinators on typed `entry.runtime_data`, not `hass.data[DOMAIN][entry_id]`.

Common setup sequence:

- Create HA-managed client or SDK wrapper.
- Create coordinator or runtime dataclass containing multiple coordinators.
- Call `async_config_entry_first_refresh()` before forwarding platforms when entities require initial data.
- Assign `entry.runtime_data` before platform setup reads it.
- Register update listeners, subscription cancel callbacks, and close callbacks with `entry.async_on_unload()`.

Use `async_config_entry_first_refresh()` when setup should retry on initial refresh failure. Use `async_refresh()` only when failure should not make the config entry unavailable during setup.

If the coordinator has one-time expensive setup, put it in `_async_setup()` so it runs as part of `async_config_entry_first_refresh()` and shares the same setup-failure semantics.

## Coordinator Design

Good coordinator design:

- Pass `config_entry=entry` to the coordinator constructor.
- Choose `update_interval` based on API limits and product needs, not arbitrary freshness.
- Use `always_update=False` when returned data supports meaningful equality and duplicate listener callbacks would cause noisy state writes.
- Pre-process fetched data into lookup maps so entity properties stay cheap.
- Use `async_contexts()` when the API can fetch only data for enabled/listening entities.
- Keep `_async_update_data()` focused on one refresh and avoid side effects unrelated to the data snapshot.
- Keep request timeouts bounded.
- Convert API objects to integration-owned models or primitive data before exposing to entities.

Avoid holding raw SDK response objects in coordinator data when those objects are not JSON-safe, stable, or easy to test.

## Error Semantics

Map errors deliberately:

- Raise `ConfigEntryAuthFailed` for invalid, expired, revoked, or rejected credentials that need reauth.
- Raise `UpdateFailed` for transient refresh failures such as timeout, connection failure, rate limit, or temporary server errors.
- Honor provider/device backoff signals when exposed by the API.
- Let setup-time failures become `ConfigEntryNotReady` through `async_config_entry_first_refresh()` when retry is appropriate.
- Do not leak raw response bodies, credentials, or user data through `UpdateFailed` messages.

Keep raw details in safe logs or bounded diagnostics only when they do not expose secrets.

## Coordinator-Backed Entities

Use `CoordinatorEntity[MyCoordinator]` for entities backed by coordinator data.

Entity rules:

- Pass a stable `context` value to `CoordinatorEntity` when the coordinator can use `async_contexts()` to limit fetches.
- Keep entity properties as pure reads from coordinator data or precomputed attributes.
- Use stable unique IDs from entry/device identity plus entity description key.
- Use `available` from `CoordinatorEntity` unless the entity has narrower availability semantics.
- Call `await coordinator.async_request_refresh()` after a mutating service/action when the device/API state should be refreshed immediately.

Use entity descriptions for repeated entity definitions across devices or platforms. The description `key` should be unique within the platform and usually participates in the unique ID.

## Push APIs And Coordinator Updates

For push APIs, entities should set `should_poll` to `False` unless the coordinator or entity still needs HA polling.

When new push data arrives:

- Update integration-owned state on the event loop.
- Call `coordinator.async_set_updated_data(data)` if using a coordinator.
- Otherwise call `entity.async_write_ha_state()` or schedule an entity update from the entity owner.
- Unsubscribe in `async_will_remove_from_hass()` or entry unload callbacks.

Do not leave websocket subscriptions, callbacks, or library listeners orphaned after unload.

## Multi-Coordinator Design

Use multiple coordinators when data has truly different refresh semantics:

- Different API endpoints have different rate limits.
- Some data is fast-changing and other data is slow/static.
- Failure of one endpoint should not make unrelated entities unavailable.
- Push and poll data need different ownership.
- One coordinator serves device inventory and another serves measurements.

Store them in a runtime dataclass:

```python
@dataclass(frozen=True, kw_only=True)
class MyRuntimeData:
    status: StatusCoordinator
    history: HistoryCoordinator
    client: MyApiClient
```

Avoid creating one coordinator per entity unless the external API truly requires per-entity polling. Per-entity coordinators can hide rate-limit problems and make unload/test behavior harder.

## Polling Without A Coordinator

Entity-level `async_update()` can be acceptable when:

- One entity maps to one API endpoint.
- There is no shared data fan-out.
- No setup-time shared fetch is needed.
- The API can tolerate HA platform polling.

Keep `SCAN_INTERVAL` conservative. HA's minimum interval is not a recommendation; cloud APIs and constrained local devices often need slower polling.

## Anti-Patterns

- Storing coordinators in `hass.data[DOMAIN][entry_id]` for per-entry runtime.
- Calling network I/O from entity property methods.
- Fetching the same endpoint separately for each entity when one shared refresh would work.
- Marking all coordinator errors as auth failures.
- Returning raw provider response text in update exceptions.
- Using a single coordinator for unrelated endpoints with different availability or rate limits.
- Forgetting unsubscribe/close cleanup for push subscriptions.

## Tests To Expect

Coordinator tests should assert:

- Initial refresh success assigns runtime data before platform forwarding.
- Initial refresh failure maps to retry/auth behavior correctly.
- `_async_setup()` behavior when one-time setup exists.
- `UpdateFailed` and `ConfigEntryAuthFailed` mapping.
- `always_update=False` or context filtering behavior when used.
- Entity unique IDs, availability, and state reads from coordinator data.
- Mutating actions call `async_request_refresh()` when required.
- Multiple coordinators refresh independently and unload cleanly.
- Push subscriptions call `async_set_updated_data()` and unsubscribe on unload.
