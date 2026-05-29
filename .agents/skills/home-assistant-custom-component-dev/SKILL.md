---
name: home-assistant-custom-component-dev
description: "Design, implement, review, or harden Home Assistant custom components and HACS integrations with production-quality lifecycle, config entries, subentries, DataUpdateCoordinator, entity platforms, OAuth2, Bluetooth, discovery, external clients, diagnostics, repairs, system health, debug services, security validation, tests, and release metadata. Use when Codex is creating or changing files under custom_components/, config_flow.py, manifest.json, translations, services.yaml, diagnostics.py, system_health.py, repairs.py, entity platforms, Home Assistant conversation or AI task entities, integration adapters, or integration tests; also use for architecture reviews of HA custom integrations."
---

# Home Assistant Custom Component Dev

## Overview

Use this skill as an architecture and quality playbook for Home Assistant custom components. Prioritize HA ownership of lifecycle, config, runtime state, registries, entities, diagnostics, repair issues, tests, and release metadata over generic Python app patterns.

## Operating Rules

- Inspect current source before changing behavior; docs and plans are not implementation proof.
- Use current Home Assistant developer docs and source examples when APIs or conventions may have changed.
- Prefer small HA-native changes: config entries, selectors, lifecycle helpers, repair issues, diagnostics, and typed runtime data.
- Avoid `.storage` edits, YAML-only workarounds, blocking I/O on the event loop, unmanaged tasks, global clients, and singleton runtime state unless explicitly justified.
- Keep user-facing surfaces aligned: `manifest.json`, `hacs.json`, translations, `services.yaml`, `icons.json`, diagnostics, system health, repairs, README, tests, and release metadata.

Authoritative starting points:

- Home Assistant developer docs: https://developers.home-assistant.io/
- Integration Quality Scale: https://developers.home-assistant.io/docs/core/integration-quality-scale/
- Config entries: https://developers.home-assistant.io/docs/config_entries_index/
- Entity platform development: https://developers.home-assistant.io/docs/core/entity/
- Integration manifest: https://developers.home-assistant.io/docs/creating_integration_manifest/
- Developer blog for breaking/API changes: https://developers.home-assistant.io/blog/

## Workflow

1. Identify the integration shape: cloud/local, polling/push, auth model, config-entry model, child resources, entities, services/actions, diagnostics, and tests.
2. Read only the reference files needed for the task from the map below.
3. Inspect comparable Home Assistant Core integrations before designing unfamiliar HA surfaces.
4. Implement through HA lifecycle helpers and typed boundaries.
5. Update every product surface affected by the behavior change.
6. Verify with focused tests first, then broader validation scripts available in the repo.

## Reference Map

- Read `references/architecture-lifecycle.md` for repository shape, manifest/HACS metadata, setup, unload, remove, migration, runtime data, and process-global coordination.
- Read `references/config-flows-subentries.md` for config flows, options flows, reconfigure, subentries, selectors, progress steps, normalization, duplicate detection, and translations.
- Read `references/coordinators-polling.md` for DataUpdateCoordinator, push versus poll, multi-coordinator design, coordinator-backed entities, and polling tests.
- Read `references/entities-runtime.md` for entity platforms, unique IDs, device info, diagnostic entities, JSON-safe state, conversation entities, AI task entities, metrics, and runtime execution.
- Read `references/external-clients-adapters.md` for HA-managed HTTP clients, SDK wrappers, protocol adapters, streaming, validation probes, typed exception mapping, and dependency boundaries.
- Read `references/specialized-integrations.md` for OAuth2, websocket/push APIs, Bluetooth, discovery protocols, and specialized manifest/lifecycle requirements.
- Read `references/observability-reliability.md` for diagnostics, system health, debug response actions, repair issues, run diagnostics, metrics, error classification, and privacy boundaries.
- Read `references/derived-state-tools.md` for derived indexes from HA registries, LLM/API tool surfaces, sandboxed virtual resources, allowlists, service-call tools, and dry-run/debug services.
- Read `references/security-validation.md` for threat modeling, secret handling, URL/header/path validation, destructive operations, exposure checks, logging, and safe error messages.
- Read `references/testing-release.md` for test layering, pytest-homeassistant patterns, live-test separation, Hassfest/HACS validation, metadata drift checks, documentation, and release readiness.

## First Pass Checklist

- Use `ConfigEntry[T]` with typed `entry.runtime_data` for per-entry runtime state.
- Register domain-level response services/actions in `async_setup()`, not per-entry setup.
- Use `entry.async_on_unload()` for listeners, update callbacks, background tasks, and cleanup.
- Use HA-managed async clients/sessions for network I/O and executor jobs for blocking file/library work.
- Keep entity attributes, diagnostics, service responses, and events JSON-safe and size-bounded.
- Decide explicit privacy boundaries for diagnostics, system health, debug services, logs, and errors.
- Add or update tests for HA UX/lifecycle behavior, not only helper functions.
- Reconcile source, tests, translations, service schemas, icons, README, manifest, HACS metadata, and changelog before completion.
