# Specialized Integration Patterns

## Contents

- Purpose
- Authoritative Sources
- OAuth2 And Reauth
- Websocket And Push APIs
- Bluetooth Integrations
- Discovery Protocols
- Device And Entity Registry Details
- Publishing Metadata For Specialized Integrations
- Anti-Patterns
- Tests To Expect

## Purpose

Use this reference when a custom component needs OAuth2, websocket/push updates, Bluetooth, Zeroconf, SSDP, USB, DHCP, device registry details, or specialized manifest metadata beyond a simple cloud/local polling integration.

## Authoritative Sources

- Config flow OAuth2, Application Credentials, reauth, reconfigure, and discovery steps: https://developers.home-assistant.io/docs/config_entries_config_flow_handler/#configuration-via-oauth2
- Network discovery: https://developers.home-assistant.io/docs/network_discovery/
- Bluetooth best practices: https://developers.home-assistant.io/docs/bluetooth/
- Bluetooth API: https://developers.home-assistant.io/docs/core/bluetooth/api/
- Bluetooth data coordinators: https://developers.home-assistant.io/docs/core/bluetooth/bluetooth_fetching_data/
- Device registry: https://developers.home-assistant.io/docs/device_registry_index/
- Manifest fields: https://developers.home-assistant.io/docs/creating_integration_manifest/

## OAuth2 And Reauth

Use HA's OAuth2/config-flow helpers instead of hand-rolling token storage and refresh behavior.

Design rules:

- Use Application Credentials for user-provided client ID/secret or HA Cloud account linking when applicable.
- Structure the API library so HA can own token refresh.
- Store token data in config entry data using HA's OAuth helpers.
- Raise `ConfigEntryAuthFailed` from setup/runtime paths when credentials are expired or revoked.
- Implement `async_step_reauth` and update the existing entry; do not create a second entry during reauth.
- Use `async_update_reload_and_abort()` for successful reauth/reconfigure paths when reload is needed.
- Verify unique ID mismatch during reauth so users cannot accidentally attach the entry to a different account.

Tests should cover initial OAuth completion, reauth confirmation, successful token replacement, mismatch abort, reload behavior, and no duplicate config entry creation.

## Websocket And Push APIs

Push integrations should subscribe when HA is ready and unsubscribe cleanly.

Patterns:

- Represent the subscription/client on `entry.runtime_data`.
- Register close/unsubscribe callbacks with `entry.async_on_unload()`.
- Use `should_poll = False` for entities driven entirely by push data.
- Use `coordinator.async_set_updated_data()` when push events update shared coordinator state.
- Reconnect with bounded backoff and clear setup/runtime error categories.
- Keep incoming event handling cheap; offload parsing or blocking library work if needed.
- Bound any retained event history.

Avoid long-lived library tasks that HA cannot cancel during unload.

## Bluetooth Integrations

Bluetooth integrations need HA's shared Bluetooth stack, not private scanners.

Rules:

- Add `bluetooth_adapters` to `manifest.json` dependencies when the integration needs to use a Bluetooth adapter.
- Use `bluetooth.async_get_scanner(hass)` when a library needs a scanner.
- Avoid starting additional scanners.
- Avoid reusing a `BleakClient` between connections.
- Use connection timeouts of at least 10 seconds for active connections.
- Use `connectable=False` for advertisement-only devices so non-connectable remote controllers can contribute data.
- Check `service_info.connectable` or `async_ble_device_from_address()` before flows/devices that need active connections.
- Prefer HA Bluetooth processor coordinators for sensor/binary sensor/event-style BLE devices when they fit the data model.
- Register discovery callbacks with unload cleanup.

Bluetooth discovery flows need stable unique IDs, usually from address or manufacturer/service data that remains stable for the physical device. Confirm with the user before completing discovered flows when required by HA UX.

## Discovery Protocols

Discovery should identify devices, avoid duplicates, and preserve user confirmation.

Rules:

- Add appropriate manifest discovery keys for `zeroconf`, `ssdp`, `dhcp`, `usb`, `bluetooth`, or other supported mechanisms.
- Add integration dependencies such as `zeroconf` or `ssdp` when using their helpers after setup.
- Use discovery dataclass properties, not old dict-style access.
- Set a stable unique ID and abort if already configured.
- Use `_abort_if_unique_id_configured(updates={...})` when discovery should update host/port for an already configured local device.
- Implement `is_matching()` when multiple discovery sources can report the same device with different identifiers.
- Never finish a discovered config flow without appropriate confirmation when HA docs require confirmation.

For USB, use HA USB helpers to check whether expected hardware is present and register scan callbacks through unload cleanup.

## Device And Entity Registry Details

Device registry data should model real ownership:

- Use stable identifiers from the external device/account, not display names or mutable IP addresses.
- Use `connections` only for stable hardware identifiers such as MAC when legitimately obtained from the device/discovery data.
- Use `via_device` for child devices behind a hub/bridge.
- Set manufacturer, model, serial, hardware version, software version, and configuration URL only when accurate and safe.
- Use service-level device info for cloud service/account entities when there is no physical device.

Do not create fake devices just to group unrelated entities.

## Publishing Metadata For Specialized Integrations

Specialized integrations often require extra metadata:

- OAuth2 integrations need documentation for application credentials or account linking.
- Bluetooth/discovery integrations need correct manifest matchers and dependencies.
- Cloud integrations need accurate `iot_class` and support URLs.
- Local push integrations should distinguish local push from local polling.
- Runtime requirements in `manifest.json` must include Bluetooth, API, websocket, or OAuth client packages actually imported at runtime.

Keep README, translations, config flow strings, repairs, and diagnostics aligned with these capabilities.

## Anti-Patterns

- Hand-rolled OAuth token refresh when HA OAuth helpers apply.
- Reauth flows that create new entries.
- Websocket subscriptions without unload cleanup.
- Starting private Bluetooth scanners.
- Treating all Bluetooth devices as connectable.
- Using IP address, URL, or mutable hostname as a unique ID.
- Completing discovery flows without duplicate checks or user confirmation.
- Copying old examples that store device clients in `hass.data`.

## Tests To Expect

Specialized integration tests should assert:

- OAuth2 create-entry, token refresh/reauth, mismatch abort, and reload behavior.
- Push/websocket subscription start, event update, reconnect classification, and unload cleanup.
- Bluetooth discovery matchers, connectable handling, callback cleanup, and scanner usage boundaries.
- Discovery duplicate aborts, host update behavior, and `is_matching()` for multi-source discovery.
- Device registry identifiers, `via_device`, and entity registry unique IDs.
- Manifest dependencies and requirements match the specialized integration behavior.
