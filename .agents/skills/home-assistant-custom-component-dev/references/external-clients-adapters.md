# External Clients And Adapter Boundaries

## Contents

- Purpose
- Authoritative Sources
- Specialized Integration Selection
- HA Managed Client Ownership
- SDK Wrapper Strategy
- Validation Probes
- Adapter Layer Design
- Streaming Protocols
- Error Mapping
- Settings Boundaries
- Dependency Hygiene
- Anti-Patterns
- Tests To Expect

## Purpose

Use this reference when integrating cloud APIs, local HTTP devices, websocket clients, SDKs, LLM/model providers, MCP-like tool servers, or protocol adapters.

For OAuth2, websocket/push APIs, Bluetooth, discovery protocols, and device-registry-specific integration patterns, also read `specialized-integrations.md`.

## Authoritative Sources

- Fetching data: https://developers.home-assistant.io/docs/integration_fetching_data/
- Network helpers: https://developers.home-assistant.io/docs/network_discovery/
- Config entry exceptions: https://developers.home-assistant.io/docs/integration_setup_failures/
- Integration Quality Scale rules: https://developers.home-assistant.io/docs/core/integration-quality-scale/rules/
- Home Assistant Core source examples: https://github.com/home-assistant/core/tree/dev/homeassistant/components
- Developer blog: https://developers.home-assistant.io/blog/

## Specialized Integration Selection

Choose the HA-native integration pattern before writing a custom client wrapper:

- OAuth2 account linking should use HA config-flow OAuth helpers and Application Credentials when applicable.
- Websocket or push APIs need explicit subscription ownership and unload cleanup.
- Bluetooth integrations should use HA's shared Bluetooth scanner/coordinator APIs rather than private scanners.
- Zeroconf, SSDP, DHCP, USB, and Bluetooth discovery should use HA discovery steps and stable unique IDs.
- Simple polling APIs should usually start with `DataUpdateCoordinator`; read `coordinators-polling.md` before designing polling fan-out.

## HA Managed Client Ownership

Home Assistant should own event-loop, SSL, proxy, and connection-pooling concerns.

Use HA-managed clients:

- `async_get_clientsession(hass)` for aiohttp integrations.
- `get_async_client(hass)` from HA httpx helpers when using httpx-compatible clients.
- HA SSL context and network helpers when integrations expose configurable hosts.
- `hass.async_add_executor_job()` for blocking libraries or file I/O that cannot be made async.

Avoid creating ad-hoc sessions without a cleanup owner. If a third-party SDK requires owning a client/session, wrap it in the smallest adapter and register close/cleanup through entry unload.

## SDK Wrapper Strategy

Third-party SDKs should not leak through the integration.

Create a narrow boundary:

- Config data -> normalized runtime config.
- Runtime config -> HA-managed client and SDK/client wrapper.
- SDK responses -> integration dataclasses or primitive dicts.
- SDK exceptions -> typed integration exceptions or HA exceptions.

Keep SDK-specific data structures out of entity attributes, diagnostics, services, and tests unless the integration explicitly exposes a protocol surface.

## Validation Probes

Config-flow probes should use the same request path as runtime when possible.

Good probe behavior:

- Validates auth, base URL, model/device/resource existence, and feature support.
- Uses the same adapter/client construction as runtime.
- Times out quickly and maps failures to stable translated reasons.
- Does not mutate external resources unless the user explicitly confirms.
- Does not log secrets or provider response bodies.

Avoid a separate lightweight probe implementation that can pass while the real runtime path fails.

## Adapter Layer Design

For complex protocols, use a two-layer design:

- Protocol client: independent of HA and framework logic, accepts caller-owned async HTTP client, serializes requests, parses responses, raises typed protocol exceptions.
- Framework/HA adapter: maps HA config/runtime types to protocol requests and maps protocol responses/errors to entity/platform results.

This separation makes tests clearer:

- Protocol client tests verify wire payloads, response parsing, streaming, and status errors.
- Adapter tests verify message/content/tool/usage mapping and HA-facing error conversion.

Use sentinels to distinguish omitted fields from explicit `None` when external APIs care about the difference.

Use permissive typed models for vendor-specific response fields when round-tripping metadata matters.

## Streaming Protocols

Streaming should have explicit ownership and cleanup.

Patterns:

- Use an async context manager that opens and closes the stream.
- Convert wire events to stable internal events at the boundary.
- Handle keepalive/empty events and terminal markers.
- Propagate cancellation cleanly.
- Preserve partial output separately from final success/failure.
- Bound stream diagnostics and do not expose raw unbounded event logs in entity attributes.

Tests should simulate normal events, malformed events, terminal markers, cancellation, timeout, and status failures.

## Error Mapping

Map low-level errors into categories that HA UX and tests can rely on:

- Auth failed.
- Permission denied.
- Not found.
- Timeout.
- DNS/connect/TLS failure.
- Rate limited.
- Provider/server error.
- Bad request or invalid configuration.
- Unsupported feature.

Use typed exceptions, HTTP status codes, `TimeoutError`, `OSError.errno`, SSL errors, and HTTP client exception types. Avoid localized string matching.

Keep user-facing messages safe and actionable. Put additional safe detail in logs or diagnostics when needed.

## Settings Boundaries

Separate settings by boundary:

- Persisted config entry data: user choices and credentials.
- Runtime config: normalized settings used to build clients/managers.
- Request settings: only fields supported by the external API/framework request.
- Integration-only settings: retries, max iterations, fallback policy, diagnostics flags.
- Probe settings: minimal request needed to validate setup.

Strip integration-only settings before sending provider/API requests. Tests should fail if unsupported settings leak to a provider payload.

## Dependency Hygiene

Every runtime dependency affects HA startup, install size, and compatibility.

Rules:

- Pin runtime requirements in `manifest.json` and development dependencies consistently.
- Prefer lightweight protocol clients when a large SDK fights HA lifecycle.
- Do not import private modules from third-party libraries unless there is no public API and the risk is accepted.
- Add regression tests when intentionally avoiding a forbidden SDK or dependency path.
- Keep optional/live-provider tests separated from normal unit tests.

## Anti-Patterns

- `requests` or other blocking I/O in async code.
- Module-level unmanaged clients/sessions.
- Validation probes that use a different code path than runtime.
- Passing raw config dicts directly into external SDK requests.
- Branching on exception class-name strings or localized messages.
- Leaking response bodies or credentials into flow errors.
- Streaming implementations without close/cancel behavior.

## Tests To Expect

External boundary tests should assert:

- HA-managed client construction and cleanup.
- Request serialization and omission/null handling.
- URL/header/body validation and rejection cases.
- Probe success and failure reason mapping.
- HTTP status and transport error mapping.
- Streaming event parsing and cancellation.
- Adapter message/content/tool/usage mapping.
- Dependency guardrails when a specific SDK must not be used.
