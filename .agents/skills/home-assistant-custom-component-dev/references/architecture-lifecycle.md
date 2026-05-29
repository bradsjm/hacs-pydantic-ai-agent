# Architecture And Lifecycle

## Contents

- Purpose
- Authoritative Sources
- Repository Shape
- Manifest And Distribution Metadata
- Setup Entry Runtime Model
- Unload Remove And Migration
- Runtime Data Versus Process Globals
- Data Fetching Runtime Ownership
- Services Registered At Domain Setup
- Patterns To Copy
- Anti-Patterns
- Tests To Expect

## Purpose

Use this reference when designing or changing the root structure of a Home Assistant custom component: `manifest.json`, HACS metadata, `__init__.py`, typed runtime data, setup/unload/remove/migrate, domain-level services/actions, and process-global coordination.

## Authoritative Sources

- Integration file structure: https://developers.home-assistant.io/docs/creating_integration_file_structure/
- Manifest fields: https://developers.home-assistant.io/docs/creating_integration_manifest/
- Config entries: https://developers.home-assistant.io/docs/config_entries_index/
- Integration Quality Scale: https://developers.home-assistant.io/docs/core/integration-quality-scale/
- HACS publishing docs: https://hacs.xyz/docs/publish/start/
- Home Assistant developer blog: https://developers.home-assistant.io/blog/

## Repository Shape

Use the normal custom component layout and treat every HA surface as part of the product:

```text
custom_components/<domain>/
  __init__.py
  manifest.json
  const.py
  config_flow.py
  translations/en.json
  services.yaml
  icons.json
  diagnostics.py
  system_health.py
  repairs.py
  <platform>.py
```

For HACS distribution, keep root-level project metadata aligned:

```text
hacs.json
README.md
CHANGELOG.md
LICENSE
.github/workflows/validate.yml
pyproject.toml
scripts/check
scripts/test
```

Do not treat these as optional cleanup files. HACS, Hassfest, users, and future maintainers consume them.

## Manifest And Distribution Metadata

Keep `manifest.json` complete and current:

- `domain`, `name`, `version`, `config_flow`, `integration_type`, and `iot_class` describe the integration contract.
- `requirements` controls what HA installs at runtime; keep it aligned with development dependencies and lockfiles.
- `codeowners`, `documentation`, and `issue_tracker` are user support surfaces.
- `dependencies` and `after_dependencies` express HA setup ordering, not import convenience.
- `loggers` should include third-party libraries whose logs matter for debugging.

Keep `hacs.json`, README, changelog, and release tags in sync with the manifest version when publishing. If behavior changes user-visible setup, diagnostics, services, or entities, update docs and translations in the same change.

## Setup Entry Runtime Model

Root setup should establish typed runtime state before platform setup needs it.

Checklist:

- Define `PLATFORMS` once as a tuple.
- Define a typed config-entry alias such as `type MyConfigEntry = ConfigEntry[MyRuntimeData]`.
- Build a frozen or otherwise explicit runtime dataclass for per-entry state.
- Assign `entry.runtime_data` before calling `async_forward_entry_setups`.
- Validate required child resources or external configuration before exposing entities.
- Create repair issues for user-actionable setup validation failures.
- Register update listeners with `entry.async_on_unload()`.
- Register background tasks with `entry.async_create_background_task()` or `hass.async_create_task()` plus cleanup.

Prefer this shape:

```python
type MyConfigEntry = ConfigEntry[MyRuntimeData]

async def async_setup_entry(hass: HomeAssistant, entry: MyConfigEntry) -> bool:
    entry.runtime_data = MyRuntimeData(...)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True
```

## Unload Remove And Migration

Unload is a first-class behavior, not an afterthought.

- Unload all forwarded platforms.
- Cancel listeners, timers, manager refreshes, and background tasks.
- Close entry-owned clients if HA does not own their lifecycle.
- Release process-global ownership when an entry unloads.
- Delete stale repair issues on successful validation, unload, or remove when they no longer apply.
- Clean persistent per-entry artifacts on remove.
- Fail migrations explicitly if a stored version cannot be upgraded safely.

Tests should prove unload and remove behavior. A custom component that cannot unload cleanly is painful to develop, test, and reload in production.

## Runtime Data Versus Process Globals

Use `entry.runtime_data` for per-entry runtime state: clients, managers, caches, metrics, last-run diagnostics, child resource maps, and dispatcher handles.

Use `hass.data[DOMAIN]` only for intentional process-global coordination that cannot belong to a single entry, such as a singleton instrumentation owner, a shared catalog cache, or a cross-entry lock registry.

When process-global state is required:

- Document why it cannot be entry-scoped.
- Protect it with `asyncio.Lock` when ownership can change concurrently.
- Use deterministic ownership rules.
- Create repair issues or warnings on conflicts instead of silently changing global behavior.
- Release ownership on unload and promote another waiting entry only when safe.

## Data Fetching Runtime Ownership

Polling coordinators, websocket clients, Bluetooth subscriptions, local device clients, and external SDK wrappers are per-entry runtime objects unless they intentionally coordinate across entries.

Store these objects directly or inside a runtime dataclass on `entry.runtime_data`. Platform setup should receive them from the typed entry, not by looking up `hass.data[DOMAIN][entry_id]`.

If setup performs an initial data refresh, assign runtime data before platform forwarding and use the correct setup-failure semantics. For coordinator details, read `coordinators-polling.md`; for OAuth, websocket, Bluetooth, and discovery details, read `specialized-integrations.md`.

## Services Registered At Domain Setup

Register domain-level response actions in `async_setup()`, not `async_setup_entry()`, when the action describes the integration domain rather than one loaded entity.

Use `SupportsResponse.ONLY` for read/query developer actions. Keep schemas strict and response envelopes stable:

```python
hass.services.async_register(
    DOMAIN,
    "get_status",
    _handle_get_status,
    schema=vol.Schema({...}),
    supports_response=SupportsResponse.ONLY,
)
```

Domain actions should resolve entries at call time so unloaded or missing entries return structured errors instead of stale runtime access.

## Patterns To Copy

- Source and tests are authoritative; design docs are intent unless backed by implementation.
- Constants live in `const.py`, including config keys, subentry types, defaults, and supported modes.
- Mutable defaults come from factory functions so config entries do not share state.
- Setup-time validation deduplicates identical checks and turns current failures into repair issues.
- Validation success cleans stale issues so users are not left with obsolete repairs.
- Release validation is one command for local use, with live external-service tests split into a separate command.

## Anti-Patterns

- Storing per-entry runtime data in `hass.data[DOMAIN][entry_id]` by habit.
- Copying examples or templates that use `hass.data[DOMAIN][entry_id]` for per-entry clients or coordinators.
- Creating unmanaged module-level client/session globals.
- Registering the same domain service once per config entry.
- Spawning background tasks without `entry.async_on_unload()` cleanup.
- Treating HACS metadata, translations, services, icons, and README as separate from code behavior.
- Editing `.storage` directly to work around config-flow limitations.
- Hiding migration failures by keeping compatibility shims without a real persisted-data need.

## Tests To Expect

Write tests that assert HA behavior rather than only internal helpers:

- Setup assigns `entry.runtime_data` and forwards platforms.
- Setup failures produce `ConfigEntryNotReady`, `ConfigEntryAuthFailed`, or repair issues as appropriate.
- Domain actions are registered once and return stable response shapes.
- Update listener reloads the entry or updates runtime state correctly.
- Unload cancels listeners/tasks and unloads platforms.
- Remove deletes repair issues and persistent entry artifacts.
- Process-global ownership conflict and promotion behavior are covered when such state exists.
