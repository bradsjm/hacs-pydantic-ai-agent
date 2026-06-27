# Pydantic AI Agent Test Instructions

## Test Strategy

- Prefer tests that verify Home Assistant-visible behavior: service responses, config entry lifecycle, diagnostics payloads, system health data, emitted graph facts, source coverage, and stable reason or warning keys.
- Keep tests deterministic. Build compact snapshots or HA registry fixtures with explicit entities, areas, labels, source references, and exposure boundaries instead of relying on incidental ordering.
- For service tests, call the registered Home Assistant services and assert the returned response shape. Do not bypass service validation when the behavior being tested is user-visible through services.
- Do not add tests that pass only because a mock is configured to return the expected value. Mock only external or Home Assistant-owned boundaries that are slow, unavailable, or outside the component's control.
- Keep each test file under the repository file-size guard. Split by behavior area rather than adding exceptions.

## Validation

- Run focused tests while iterating, for example `scripts/test tests/components/semantic_home/test_query.py` or the specific file being changed.
- Run `scripts/format` before final validation.
- Run `scripts/check` before considering a test or behavior change complete.
