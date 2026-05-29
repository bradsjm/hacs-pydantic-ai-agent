# Derived State Tools And Sandboxed Resources

## Contents

- Purpose
- Authoritative Sources
- Derived Indexes From HA State
- Dry Run And Trace Services
- HA LLM Or Tool APIs
- Service Call Tool Boundaries
- External Tool Catalogs
- Sandboxed Runtime Resources
- Documentation Versus Implementation
- Anti-Patterns
- Tests To Expect

## Purpose

Use this reference when a custom component derives searchable knowledge from HA registries/state, exposes LLM/tool APIs, wraps HA services as tools, discovers external tool catalogs, or provides a sandboxed virtual resource such as a per-run workspace.

## Authoritative Sources

- State and entity registry concepts: https://developers.home-assistant.io/docs/entity_registry_index/
- Device registry: https://developers.home-assistant.io/docs/device_registry_index/
- Service actions: https://developers.home-assistant.io/docs/dev_101_services/
- Conversation entity: https://developers.home-assistant.io/docs/core/entity/conversation/
- Integration Quality Scale: https://developers.home-assistant.io/docs/core/integration-quality-scale/
- Home Assistant Core source: https://github.com/home-assistant/core/tree/dev/homeassistant/components

## Derived Indexes From HA State

Some integrations need a derived view of HA registries and states, such as semantic search, planning, inventory, or routing.

Design rules:

- Treat HA registries and state machine as source of truth.
- Snapshot HA-owned data on the event loop into primitive structures.
- Build heavy indexes off the event loop with `hass.async_add_executor_job()`.
- Represent indexed documents with frozen `kw_only` dataclasses or typed dicts.
- Include source metadata: entity ID, area, device, labels, exposure, domain, hidden/disabled flags, and update timestamp.
- Penalize hidden, disabled, diagnostic, high-churn, or noisy entities for user-facing search.
- Gate user/LLM-visible output through HA exposure/permission checks.
- Keep query functions as pure as possible and return JSON-safe dicts.

Refresh rules:

- Listen to registry changes that alter the index.
- Debounce rebuilds.
- Add periodic refresh when source events may be missed.
- Avoid rebuilding on high-churn state changes unless state is actually indexed.
- Expose loaded/stale/rebuilding counts through diagnostics or debug actions.

## Dry Run And Trace Services

Any derived planner or control surface should include read-only validation services.

Good services:

- Refresh index, if explicitly named and safe.
- Search/trace resolution for a phrase or filter.
- Plan a control action without executing it.
- Return a bounded document or explanation for a specific result.
- Benchmark or validate deterministic behavior with limits.

Keep live control separate from trace/dry-run services. A debug service should not turn on lights, delete records, or call external tools unless the name and schema make the mutation explicit.

## HA LLM Or Tool APIs

When exposing tools to HA Assist, an LLM framework, or another tool runner:

- Use the HA abstraction for the surface when one exists.
- Make IDs entry-scoped to avoid cross-entry leakage.
- Define tool schemas with voluptuous or the framework's public schema API.
- Resolve runtime managers from `entry.runtime_data` at call time.
- Check exposure, permissions, and allowlists at call time.
- Return stable structured errors rather than raw exceptions.
- Keep tools narrow and predictable.

Do not use private library modules for tool conversion when a public API exists.

## Service Call Tool Boundaries

When converting HA services/actions into tool calls:

- Preserve schema and call ID information.
- Use `hass.services.async_call(..., blocking=True)` when later logic depends on the effect.
- Use `target={ATTR_ENTITY_ID: entity_id}` for entity-targeting services.
- Keep data and target separate.
- Validate entity IDs before execution when possible.
- Reject tools not selected or allowlisted at call time, even if configuration previously filtered them.
- Use sequential toolsets when tools mutate shared HA or integration state.

## External Tool Catalogs

For MCP-like or remote tool catalogs:

- Validate URLs and headers before storing.
- Reject credentials in URL userinfo.
- Discover tools with timeouts and safe error classification.
- Cache catalogs per entry or subentry with clear refresh semantics.
- Store allowlists by stable tool name/ID.
- Enforce allowlists during execution, not just during discovery.
- Summarize catalog diagnostics with counts, not full secret-bearing headers or raw URLs unless the diagnostics policy allows it.

## Sandboxed Runtime Resources

If an integration exposes a virtual workspace, temporary file tree, patch system, or similar resource:

- Create it per run/session.
- Never use a global mutable workspace for independent user runs.
- Enforce limits on total size, file count, read length, write length, patch size, and command count.
- For virtual paths, normalize lexically; for real filesystem-backed paths, resolve/canonicalize the final path and verify it remains under the allowed base.
- Reject root escape, empty paths, NUL bytes, absolute paths when not allowed, and protected roots.
- Use snapshot/rollback around multi-step mutations.
- Require explicit confirmation for destructive operations: overwrite, delete, move, patch replace.
- Return typed JSON-safe result dicts.
- Keep mutating operations sequential.

Sandboxing inside a Home Assistant custom component is a design discipline, not a security sandbox. HA integrations run with process access, so do not add arbitrary command execution or uncontrolled filesystem access without explicit approval and threat modeling.

## Documentation Versus Implementation

For complex derived systems, design documents can drift.

Rules:

- Treat source, tests, manifests, and lockfiles as implementation truth.
- Mark future phases clearly.
- Keep external research separate from implemented features.
- When updating docs, cite current source behavior and avoid presenting design intent as shipped behavior.

## Anti-Patterns

- Building indexes directly from registry objects in executor threads.
- Returning hidden or unexposed HA entities to LLM/user-facing tools.
- Exposing a debug service that performs live control.
- Assuming discovery allowlists are enough and skipping execution-time checks.
- Shared mutable workspaces across runs.
- Path normalization that uses string prefix checks without resolving escapes.
- Tool errors that leak raw headers, URLs, prompts, or stack traces.

## Tests To Expect

Tests should assert:

- Registry/state snapshot behavior.
- Executor use for heavy index building.
- Debounce, cancellation, and periodic refresh behavior.
- Ranking penalties and deterministic ordering.
- Exposure and permission gating.
- Dry-run planning does not mutate state.
- Tool schemas and service targets are correct.
- Allowlist rejection at call time.
- Remote catalog validation and cache refresh behavior.
- Virtual resource path safety, size limits, destructive confirmation, rollback, and JSON result shapes.
