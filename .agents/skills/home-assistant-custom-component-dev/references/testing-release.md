# Testing Validation And Release

## Contents

- Purpose
- Authoritative Sources
- Test Philosophy
- Config Flow Tests
- Setup Lifecycle Tests
- Entity Runtime Tests
- Coordinator And Polling Tests
- External Boundary Tests
- Specialized Integration Tests
- Observability Tests
- Security Tests
- Live Tests
- Validation Scripts
- Release Readiness
- Anti-Patterns

## Purpose

Use this reference when planning tests, updating test coverage, validating a custom component locally, preparing HACS/Core-style checks, or releasing behavior changes.

## Authoritative Sources

- Home Assistant testing docs: https://developers.home-assistant.io/docs/development_testing/
- Integration Quality Scale: https://developers.home-assistant.io/docs/core/integration-quality-scale/
- Hassfest custom-component validation: https://developers.home-assistant.io/blog/2020/04/16/hassfest/
- Hassfest GitHub action: https://github.com/home-assistant/actions#hassfest
- HACS publish docs: https://hacs.xyz/docs/publish/start/
- Pytest Home Assistant custom component plugin: https://github.com/MatthewFlamm/pytest-homeassistant-custom-component
- Home Assistant Core tests: https://github.com/home-assistant/core/tree/dev/tests/components

## Test Philosophy

Tests should prove HA concepts and user-visible behavior.

Prioritize tests for:

- Config-flow UX and validation.
- Setup/unload/remove lifecycle.
- Entity registry and entity state behavior.
- Coordinator refresh, availability, and push/poll behavior.
- External client boundaries and error mapping.
- Diagnostics/system health/repairs/debug surfaces.
- Security validation and allowlists.
- Release metadata drift.

Avoid tests that only lock implementation details unless the helper is reusable infrastructure with meaningful edge cases.

## Config Flow Tests

Assert:

- Initial form type, step ID, and schema defaults.
- Successful entry creation and stored normalized data.
- Duplicate aborts.
- Cannot connect/auth/permission/timeout/server error keys.
- Reconfigure and options flows preserve unrelated values.
- Reauth updates credentials correctly.
- Subentry create/reconfigure behavior.
- Progress steps for slow probes/discovery.
- Translation keys and placeholders, not English strings.
- Selector options are deterministic.

Mock external probes unless the test is explicitly a live integration test.

## Setup Lifecycle Tests

Assert:

- Runtime data is assigned and typed.
- Platforms are forwarded.
- Domain services/actions are registered once.
- Setup validation creates and clears repair issues.
- Config entry update listener reloads or updates runtime state.
- Unload unloads platforms and releases listeners/tasks/resources.
- Remove cleans persistent artifacts and repair issues.
- Process-global singleton ownership conflict and unload behavior when relevant.

Use HA fixtures and registry helpers rather than manually constructing internal state when HA behavior matters.

## Entity Runtime Tests

Assert:

- Entity creation count and ownership.
- Unique IDs and device info.
- `config_subentry_id` when subentries own entities.
- Diagnostic category and disabled-by-default behavior.
- State and attributes are JSON-safe expected shapes.
- Dispatcher updates after metrics/runtime changes.
- Conversation or AI task result behavior.
- Fallback/retry classification.
- Event payload shape if events are emitted.

Do not overmock the entity if the registry/platform behavior is the important contract.

## Coordinator And Polling Tests

Assert:

- Initial coordinator refresh succeeds before platform forwarding when entities require initial data.
- Initial refresh failures map to retry or reauth behavior.
- Coordinator data is stored on typed `entry.runtime_data`.
- `UpdateFailed`, `ConfigEntryAuthFailed`, timeout, and rate-limit paths are classified correctly.
- Coordinator entities expose availability and state from coordinator data.
- Mutating entity actions request refresh when required.
- Push integrations update entities through `async_set_updated_data()` or equivalent state scheduling.
- Subscriptions and multiple coordinators unload cleanly.

## External Boundary Tests

Assert:

- HA-managed client/session construction.
- Request serialization and supported settings only.
- URL/header/object normalization and rejection.
- Probe path matches runtime path when possible.
- HTTP status and transport errors map to stable reasons.
- Streaming event parsing, terminal events, malformed events, timeout, and cancellation.
- Adapter mapping between HA/framework types and protocol types.
- Forbidden SDK/import guardrails when avoiding an unsafe dependency.

Keep networked tests out of the default suite.

## Specialized Integration Tests

Assert:

- OAuth2 flows update the existing entry during reauth and reject mismatched accounts.
- Websocket/push clients subscribe, update state, reconnect or fail safely, and unsubscribe on unload.
- Bluetooth integrations use HA scanner/coordinator APIs and handle connectable versus non-connectable devices.
- Discovery flows set stable unique IDs, abort duplicates, update changed host/port safely, and confirm user setup when required.
- Device registry identifiers, connections, and `via_device` match the real device model.
- Manifest dependencies and requirements match OAuth, Bluetooth, discovery, or websocket behavior.

## Observability Tests

Assert:

- Diagnostics include intended fields and exclude unintended fields.
- Large data is bounded with clear markers/counts.
- Device diagnostics filter correctly.
- System health is aggregate and secret-safe.
- Debug response actions are read-only, redacted, filtered, and bounded.
- Repair issues have stable IDs, translations, placeholders, and cleanup.
- Metrics update on success/failure.

Tests should encode the privacy decision for each surface. If owner diagnostics intentionally include raw non-sensitive fields, assert that intentionally and keep credentials/secrets redacted across diagnostics and support-safe surfaces.

## Security Tests

Assert:

- URL userinfo and invalid schemes are rejected.
- Header names/values reject unsafe characters.
- Path traversal and protected roots are rejected.
- Destructive operations require confirmation.
- Service schemas reject unsafe inputs.
- Allowlists are enforced at call time.
- Error messages do not include secret-bearing raw response bodies.
- Debug/system health surfaces summarize secrets as booleans/counts.

## Live Tests

Use live tests sparingly and mark them separately.

Rules:

- Require credentials through environment variables.
- Skip by default when credentials are absent.
- Serialize tests when external resources have rate limits or shared state.
- Cover only high-value end-to-end behavior that mocks cannot prove.
- Never make normal unit tests depend on network access.

## Validation Scripts

Prefer one aggregate local command and focused subcommands.

Typical commands:

- `scripts/check` for lint, format check, YAML/JSON/markdown, type check, and tests.
- `scripts/lint-check` for Ruff or project linter.
- `scripts/type-check` for mypy/pyright based on repo choice.
- `scripts/yaml-check` and `scripts/markdown-check` for metadata/docs.
- `scripts/test` for normal unit tests.
- `scripts/test-live` or equivalent for networked tests.

Use the repo's actual commands, not generic commands, when working in an existing project.

## Release Readiness

Before release or handoff, verify:

- `manifest.json` version, requirements, docs, issue tracker, codeowners, and loggers.
- `hacs.json` minimum HA/HACS versions.
- README and docs match source behavior.
- `CHANGELOG.md` describes user-visible changes.
- `services.yaml`, `icons.json`, and translations match registered services/actions.
- Diagnostics, repairs, and system health strings are translated.
- New entity names, device info, attributes, and categories are tested.
- Migration is implemented for persisted data changes.
- Hassfest/HACS validation passes.
- Release tag matches manifest/project version when publishing.

## Anti-Patterns

- Adding tests only for the helper you wrote while missing HA lifecycle behavior.
- Branching inside tests instead of using parametrization or separate cases.
- Live provider tests in the default suite.
- Tests that assert English UI text instead of translation keys/placeholders.
- Diagnostics tests that do not check bounding or privacy boundaries.
- Updating code without matching translations/services/icons/README metadata.
- Declaring completion without unload/remove validation for lifecycle-heavy changes.
