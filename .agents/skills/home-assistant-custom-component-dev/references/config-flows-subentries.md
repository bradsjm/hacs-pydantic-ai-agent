# Config Flows And Subentries

## Contents

- Purpose
- Authoritative Sources
- Config Entry Model
- Options Reconfigure And Reauth
- Config Subentries
- Selectors And Schemas
- Parsing Normalization And Storage
- Validation And Progress Steps
- Translations And Errors
- Anti-Patterns
- Tests To Expect

## Purpose

Use this reference when implementing or reviewing `config_flow.py`, split config-flow modules, options flows, reconfigure flows, reauth, config subentries, selectors, validation, and translated errors.

## Authoritative Sources

- Config entries overview: https://developers.home-assistant.io/docs/config_entries_index/
- Config flow handler: https://developers.home-assistant.io/docs/config_entries_config_flow_handler/
- Options flow: https://developers.home-assistant.io/docs/config_entries_options_flow_handler/
- Selectors: https://developers.home-assistant.io/docs/data_entry_flow_index/#selectors
- Translations: https://developers.home-assistant.io/docs/internationalization/core/
- Developer blog: https://developers.home-assistant.io/blog/

## Config Entry Model

Use a root config entry for the account, hub, workspace, or other top-level installation. Keep the root entry stable and avoid creating multiple entries when child resources are better represented as subentries or options.

Root flow checklist:

- Use `ConfigFlow` with a stable `VERSION` and `MINOR_VERSION` when migrations matter.
- Set a unique ID when the external service/device has a stable identity.
- Abort duplicate entries before creating a second root entry.
- Validate connection/auth before creating the entry unless the integration is intentionally offline-capable.
- Store normalized, typed data rather than raw form strings.
- Keep titles deterministic and user-friendly.

If flow logic grows, keep `config_flow.py` as a small public entrypoint and split implementation into focused modules. Do not make tests import private split modules unless they are stable helpers worth testing directly.

## Options Reconfigure And Reauth

Use the right flow for the user intent:

- Options flow changes non-identity runtime preferences.
- Reconfigure changes setup data without removing and re-adding the integration.
- Reauth fixes expired or invalid credentials.
- Repairs should point users to the correct flow when a validation failure is fixable.

Preserve existing data unless the user explicitly changes or clears a field. Treat absent, blank, and explicit values as different states:

- Absent means preserve current value.
- Blank may mean clear optional value.
- Explicit value means validate and store normalized value.

Avoid compatibility fallbacks unless existing persisted entries require them.

## Config Subentries

Use `ConfigSubentryFlow` when one root entry owns independently managed child resources, such as model profiles, devices, agents, dashboards, tool servers, or data sources.

Subentry checklist:

- Expose supported subentry types from the root flow.
- Keep each child flow focused in its own module when logic is non-trivial.
- Create and reconfigure paths should share schema builders and normalizers.
- Abort child flows if the parent entry is not loaded and runtime data is required.
- Store subentry type constants in `const.py`.
- Add entities with `config_subentry_id` when the entity belongs to a subentry.
- Use deterministic titles and collision avoidance for generated child names.
- Validate references between subentries before storing them.

Subentries are not a dumping ground for arbitrary options. Use them when a child resource needs independent add/edit/delete UI and lifecycle ownership.

## Selectors And Schemas

Prefer HA selectors over free-form input:

- `TextSelector` for plain text and password secrets.
- `NumberSelector` for numeric settings.
- `SelectSelector` for finite choices.
- `BooleanSelector` for toggles.
- `EntitySelector` for HA entity references.
- `ObjectSelector` for JSON-like advanced configuration when unavoidable.
- `TemplateSelector` only when templates are truly the user-facing contract.

Use `section()` for advanced settings when it improves UI clarity. Always flatten sectioned user input before reading it.

Build deterministic select options sorted by label then value. This makes UI behavior predictable and tests stable.

## Parsing Normalization And Storage

Never store raw complex form strings directly when they represent typed data.

Normalize before storage:

- Numbers to `int` or `float` with range validation.
- JSON/object text to dict/list with shape validation.
- Comma-separated lists to stripped deduplicated tuples/lists.
- URLs to canonical form with scheme/host/default port/query normalization.
- Headers to validated key/value pairs with secret handling.
- Entity IDs through HA selectors or explicit validation.
- References to other entries/subentries through current registry/runtime lookup.

Reject unsafe URL userinfo such as `https://user:pass@example.com`. Credentials belong in separate secret fields, not URLs.

Preserve user customization when refreshing discovered data. A catalog sync should not silently wipe custom display names, enabled flags, pricing, selected references, or user-edited settings.

## Validation And Progress Steps

Validation should use the same runtime path that real operation uses when feasible. Separate probe-only implementations often drift.

Use progress steps for slow validation or discovery:

- Start an HA task for the probe or discovery.
- Show progress while it runs.
- Collect the result in a finish step.
- Map failures to stable translated reason keys.

Classify validation failures into actionable categories: cannot connect, auth failed, permission denied, not found, timeout, rate limited, invalid input, unsupported feature, provider/server error.

Do not surface raw exception bodies into flow errors. Logs and diagnostics can contain more detail when safe.

## Translations And Errors

Every user-visible flow state needs translation coverage:

- Step titles and descriptions.
- Field labels and descriptions.
- Selector option labels when labels are not self-explanatory.
- `config.error` keys.
- `config.abort` keys.
- Reconfigure, reauth, and subentry flow strings.
- Repair issue strings when flow failures create repairs.

Use placeholders for dynamic details such as status code, host, child resource title, or safe reason text.

Tests should assert translation keys and placeholders, not English text.

## Anti-Patterns

- Free-form strings where HA selectors can constrain input.
- Reconfigure flows that wipe unrelated stored data.
- Duplicate detection based on raw URL strings instead of normalized identity.
- Parent/child references stored without checking current validity.
- Exception message matching for validation reason mapping.
- Blocking network probes on the event loop.
- Overloading options flow for credential repair when reauth is the right UX.

## Tests To Expect

Cover both success and UX failure paths:

- Initial form schema and suggested values.
- Successful entry creation and stored normalized data.
- Duplicate entry abort.
- Cannot connect/auth/permission/timeout/error mappings.
- Options and reconfigure preserve unrelated values.
- Reauth updates credentials and reloads or completes correctly.
- Subentry create/reconfigure/delete if supported.
- Parent-not-loaded abort for subentry flows that need runtime data.
- Selector option ordering and stale reference display.
- Section flattening and blank-versus-absent semantics.
