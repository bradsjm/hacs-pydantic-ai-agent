# Observability And Reliability

## Contents

- Purpose
- Authoritative Sources
- Observability Surface Matrix
- Diagnostics
- Bounded Diagnostics
- System Health
- Debug Response Actions
- Repair Issues
- Metrics
- Error Classification
- Process Global Reliability
- Anti-Patterns
- Tests To Expect

## Purpose

Use this reference when adding or reviewing diagnostics, device diagnostics, system health, developer/debug response services, repair issues, runtime metrics, error classification, and reliability-oriented cleanup.

## Authoritative Sources

- Diagnostics: https://developers.home-assistant.io/docs/core/integration-quality-scale/rules/diagnostics/
- System health: https://developers.home-assistant.io/docs/core/integration/system_health/
- Repairs: https://developers.home-assistant.io/docs/core/integration-quality-scale/rules/repair-issues/
- Entity categories: https://developers.home-assistant.io/docs/core/entity/#generic-properties
- Service actions: https://developers.home-assistant.io/docs/dev_101_services/
- Integration Quality Scale: https://developers.home-assistant.io/docs/core/integration-quality-scale/

## Observability Surface Matrix

Choose the audience before choosing fields.

| Surface | Audience | Recommended content |
| --- | --- | --- |
| Logs | Developer/operator | Safe categories, entry IDs, status codes, non-secret detail |
| Config-entry diagnostics | Owner exporting diagnostics | Deliberately chosen config/runtime data, redacted or raw by explicit policy, always bounded |
| Device diagnostics | Owner debugging one device/subentry | Targeted subset only, no unrelated child resources |
| System health | Support and UI summaries | Aggregate counts and loaded/error states, no secrets |
| Debug response actions | Developer or agent investigation | Redacted targeted summaries and bounded slices |
| Diagnostic entities | User dashboard/history | Small stable health/metric values |
| Repair issues | User action path | Translated problem and fix flow |

The important design decision is not "redact everything" or "dump everything". It is a deliberate privacy boundary per surface.

## Diagnostics

Diagnostics must be intentional.

Checklist:

- Register config-entry diagnostics when the integration has meaningful support data.
- Add device diagnostics when child devices/subentries need targeted support bundles.
- Redact credentials, tokens, cookies, precise locations, and other sensitive values from diagnostics.
- Include raw values only for explicitly non-sensitive fields that are intentionally useful for support.
- Bound data size/depth even when data is not secret.
- Convert to JSON-safe values.
- Exclude live stream traces, huge histories, and raw runtime objects unless deliberately summarized.
- Keep runtime diagnostics narrower than stored config if they contain URLs, tool names, prompts, or headers.

If owner diagnostics intentionally include raw non-sensitive values, document that decision in tests and keep system health/debug services safer for sharing.

## Bounded Diagnostics

A diagnostics bounding helper should:

- Recurse through dicts, lists, tuples, dataclasses, Pydantic models, exceptions, and unknown objects.
- Cap depth.
- Cap mapping entries.
- Cap sequence entries.
- Cap string length with head/tail preservation.
- Include omitted counts or markers.
- Never raise because a diagnostic value is unusual.

Bounding is separate from redaction. A non-secret 5 MB prompt, device list, or trace can still break support tooling.

## System Health

System health should answer "is this integration broadly healthy?" without becoming a diagnostics dump.

Good fields:

- Configured entry count.
- Loaded entry count.
- Child resource counts by type.
- Cache loaded/count state.
- Enabled feature counts.
- Last known aggregate health category.

Avoid:

- URLs.
- Tokens or headers.
- Prompts, tool names, user text.
- Raw entity IDs when counts are enough.
- External probes during health collection.

Register system health callbacks with HA's system health integration and keep helpers pure for easy testing.

## Debug Response Actions

Debug actions are powerful for development and ha-dev/MCP investigation, but must stay safe by default.

Rules:

- Register with `SupportsResponse.ONLY`.
- Make read-only actions truly read-only: no provider probes, no refreshes, no external calls, no mutation.
- If an action refreshes or mutates, name it accordingly and require explicit parameters.
- Use strict voluptuous schemas for filters and limits.
- Return stable envelopes: `success`, `count`, `items`, `errors` when needed.
- Provide filters such as `config_entry_id`, `subentry_id`, `enabled_only`, `limit`, and `offset`.
- Return redacted summaries such as `has_api_key`, `header_count`, `content_length`, `tool_count`, and timestamps.

Debug actions should make source inspection and live state validation faster without becoming a secret exfiltration API.

## Repair Issues

Repair issues turn detected failures into user action.

Checklist:

- Use deterministic issue IDs.
- Hash canonical non-secret identity when issue IDs depend on settings.
- Use `translation_key` and `translation_placeholders`.
- Choose fixable only when HA can route the user to a fix.
- Clean stale issues when config changes or validation succeeds.
- Clean entry-scoped issues on remove.
- Do not create duplicate issues for the same root cause.

Use repair issues for persistent setup/config problems, not every transient runtime failure.

## Metrics

Runtime metrics should be explicit and lightweight.

Patterns:

- Store per-entry metrics in `entry.runtime_data`.
- Use dataclasses or typed dicts internally.
- Separate last-run values from cumulative counters.
- Track success, failure category, duration, request count, token/byte counts, and cost only when inputs are reliable.
- Use dispatcher signals to update diagnostic entities.
- Return `None` for unknown derived metrics.

Avoid raw prompts, response text, headers, or unbounded traces in metrics.

## Error Classification

Classify failures once and reuse the result for flow errors, runtime errors, fallback decisions, metrics, repairs, and diagnostics.

Implementation guidance:

- Walk exception chains with cycle protection and max depth.
- Classify by typed exception, HTTP status code, timeout type, DNS/connect/TLS error, and `errno`.
- Keep user message, log detail, stable category, retry/fallback decision, and diagnostic detail separate.
- Wrap HA-facing runtime failures in `HomeAssistantError` or platform-specific errors.
- Do not leak provider response bodies to users unless explicitly safe.

## Process Global Reliability

When process-global state is unavoidable:

- Keep one owner model.
- Protect updates with `asyncio.Lock`.
- Surface conflicts with repair issues or explicit debug state.
- Run blocking setup in executor jobs.
- Release ownership on unload.
- Test competing entries and unload promotion.

## Anti-Patterns

- Treating diagnostics as a raw recursive dump by accident.
- Redacting secrets but leaving unbounded huge values.
- System health that includes private identifiers or performs network probes.
- Debug actions that mutate state despite sounding read-only.
- Repair issue IDs that change on every validation run.
- Metrics stored globally across entries.
- Error handling that mixes user message, log detail, and fallback policy in one string.

## Tests To Expect

Observability tests should assert:

- Diagnostics include exactly the intended fields for the chosen privacy boundary.
- Large strings/lists/maps are bounded.
- Device diagnostics filter to the requested device/subentry.
- System health returns aggregate counts and excludes secrets.
- Debug actions return stable redacted envelopes and enforce limits.
- Repair issues create, update, and clean stale issues.
- Metrics update on success/failure and diagnostic entities receive dispatcher updates.
- Error classifications cover HTTP status, transport failures, chained exceptions, and fallback decisions.
