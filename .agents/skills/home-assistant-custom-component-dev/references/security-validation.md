# Security And Validation

## Contents

- Purpose
- Authoritative Sources
- Threat Model
- Secrets And Credentials
- URL Header And Object Validation
- Path And File Safety
- Service Action Safety
- Unsafe Template And Generator Patterns
- Exposure Permission And Allowlists
- Logging Errors And Diagnostics
- Dependency And Supply Chain Safety
- Anti-Patterns
- Tests To Expect

## Purpose

Use this reference when handling credentials, user input, URLs, headers, paths, templates, tools, service actions, diagnostics, logging, dependencies, or any feature that can mutate HA state or external resources.

## Authoritative Sources

- Integration Quality Scale security-related rules: https://developers.home-assistant.io/docs/core/integration-quality-scale/rules/
- Config flows and validation: https://developers.home-assistant.io/docs/config_entries_config_flow_handler/
- Service actions: https://developers.home-assistant.io/docs/dev_101_services/
- Diagnostics rule: https://developers.home-assistant.io/docs/core/integration-quality-scale/rules/diagnostics/
- Python package requirements: https://developers.home-assistant.io/docs/creating_integration_manifest/#requirements
- Home Assistant architecture docs: https://developers.home-assistant.io/docs/architecture_index/

## Threat Model

Home Assistant custom components are not sandboxed. They run in the HA process and can access the filesystem, network, memory, service registry, event bus, and user data.

Assume bugs can:

- Block the HA event loop.
- Leak credentials through logs, diagnostics, services, or errors.
- Execute unintended HA services.
- Persist unsafe config.
- Overwrite user data.
- Exhaust memory with large diagnostics or traces.
- Break setup/unload for all entries.

Design every boundary as if the integration is part of HA Core.

## Secrets And Credentials

Rules:

- Collect secrets with password selectors.
- Store secrets only in config entry data/options when needed.
- Do not duplicate secrets into runtime diagnostics, metrics, logs, events, or entity attributes.
- Do not put credentials in URLs; reject URL userinfo.
- Redact or summarize secrets in debug services and system health.
- Redact credentials, tokens, cookies, precise locations, and other sensitive values from diagnostics; raw diagnostic fields are acceptable only when they are explicitly non-sensitive and intentionally useful.
- Never log API keys, bearer tokens, cookies, private URLs, or raw headers.

Use summary fields such as `has_api_key`, `header_count`, `token_configured`, or `content_length` for support surfaces.

## URL Header And Object Validation

Validate before storage and before use.

URL checklist:

- Require allowed schemes, usually `https` for cloud APIs.
- Normalize host, default ports, paths, and query ordering when using URLs for duplicate detection.
- Reject userinfo.
- Reject empty host.
- Consider local-network SSRF risk before allowing arbitrary URLs.
- Keep base URL and credentials separate.

Header checklist:

- Reject empty names.
- Reject names/values with control characters.
- Normalize or preserve case according to protocol needs.
- Treat authorization/cookie-like headers as secrets.

Object/JSON checklist:

- Parse to typed objects before storage.
- Reject unexpected top-level keys when strictness matters.
- Bound nested size/depth for user-provided objects.
- Strip integration-only fields before external requests.

## Path And File Safety

Avoid filesystem access in HA integrations unless it is a core feature.

If paths are needed:

- Use HA config path helpers where appropriate.
- For real filesystem access, resolve/canonicalize the final path and verify it remains under the allowed base before read, write, move, or delete operations.
- For purely virtual path spaces, normalize lexically and reject root escape.
- Reject NUL bytes, empty paths, and protected roots.
- Require explicit confirmation for overwrite/delete/move.
- Run blocking file I/O in `hass.async_add_executor_job()`.
- Bound file sizes and read lengths.
- Use snapshot/rollback for multi-step mutations.

Never add arbitrary command execution without explicit design approval.

## Unsafe Template And Generator Patterns

Treat old generated integration templates as untrusted examples until source-verified.

Reject or rewrite templates that include:

- Per-entry clients, hubs, or coordinators stored in `hass.data[DOMAIN][entry_id]` instead of typed `entry.runtime_data`.
- Generated attribution headers or third-party branding in every source file.
- Blocking HTTP clients such as `requests` in async setup, config flows, entity methods, or service handlers.
- Raw response-body logging or user-facing errors.
- Broad service schemas that accept arbitrary commands, actions, URLs, paths, or JSON objects.
- Command execution helpers unless the feature has explicit approval and a threat model.

Generated boilerplate is useful only after it has been reduced to current HA patterns and validated against current docs/source.

## Service Action Safety

Service/action schemas are security boundaries.

Rules:

- Use strict voluptuous schemas.
- Prefer selectors in `services.yaml` so UI users choose valid targets.
- Separate `target` from service `data` for entity-targeting actions.
- Use `SupportsResponse.ONLY` for read-only query services.
- Name mutating actions clearly.
- Require confirmation fields for destructive operations.
- Return stable structured errors instead of raw exceptions.
- Do not perform network probes or mutations in actions documented as status queries.

## Exposure Permission And Allowlists

For LLM, Assist, automation, or external tool surfaces:

- Gate entity visibility through HA exposure/permission APIs where available.
- Enforce allowlists at execution time.
- Treat stale allowlist entries as denial, not silent broad access.
- Keep live-control tools narrow.
- Provide dry-run plan services for validation.
- Use sequential execution when tools mutate shared state.

Do not rely on UI filtering alone; configuration can become stale or be edited by other paths.

## Logging Errors And Diagnostics

Separate outputs:

- User-facing error: safe, actionable, translated when shown in UI.
- Log message: safe detail for developers/operators.
- Diagnostics: chosen privacy policy plus bounding.
- Metrics: category/count/duration, no raw sensitive content.
- Repair issue: translated problem and fix path.

Avoid raw provider response bodies in user errors. They may include prompts, secrets, account metadata, or policy text.

## Dependency And Supply Chain Safety

Runtime dependencies are loaded inside the HA process.

Rules:

- Keep dependencies minimal.
- Pin requirements in `manifest.json`.
- Keep dev/test dependencies separate.
- Prefer public APIs over private module imports.
- Vet SDK behavior for blocking I/O, hidden threads, global clients, telemetry, and event-loop ownership.
- Add tests or import guards when intentionally avoiding unsafe/heavy dependencies.

## Anti-Patterns

- Accepting arbitrary `http://` cloud URLs with credentials.
- Logging headers for debug convenience.
- Free-form JSON stored without shape validation.
- Diagnostic dumps that include secrets by accident.
- Service actions with permissive `dict` schemas.
- Tools that execute HA services without allowlist checks.
- Blocking file/network I/O in config flows or entity methods.
- Adding fallback paths for invalid input instead of rejecting it.

## Tests To Expect

Security and validation tests should cover:

- URL normalization and rejection cases.
- Credential userinfo rejection.
- Header validation and redaction summaries.
- JSON/object parser strictness and bounds.
- Path escape and protected-root rejection.
- Destructive confirmation requirements.
- Service schema accept/reject behavior.
- Exposure and allowlist enforcement at call time.
- Secret absence in system health, debug services, logs when practical, and redacted diagnostics.
- Error message safety for provider/API failures.
