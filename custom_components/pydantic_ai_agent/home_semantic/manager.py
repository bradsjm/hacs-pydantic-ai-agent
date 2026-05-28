"""Entry-scoped lifecycle manager for the Home Semantic Index."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
import logging
from time import monotonic
from typing import Any, Literal

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_DEVICE_CLASS,
    ATTR_FRIENDLY_NAME,
    ATTR_SUPPORTED_FEATURES,
    ATTR_UNIT_OF_MEASUREMENT,
    EVENT_STATE_CHANGED,
)
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant, State, callback
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    floor_registry as fr,
    label_registry as lr,
)
from homeassistant.helpers.event import async_call_later
from homeassistant.util import dt as dt_util

from .builder import async_build_home_semantic_index
from .index import HomeSemanticIndex

_LOGGER = logging.getLogger(__name__)

BuildIndexCallable = Callable[[], Awaitable[HomeSemanticIndex]]
RefreshStatus = Literal["loading", "ready", "refreshing", "failed", "stopped"]

_INDEXED_STATE_ATTRIBUTES = (
    ATTR_FRIENDLY_NAME,
    ATTR_DEVICE_CLASS,
    ATTR_UNIT_OF_MEASUREMENT,
    ATTR_SUPPORTED_FEATURES,
)
_HIGH_CHURN_STATE_DOMAINS = {"binary_sensor", "sensor"}


class HomeSemanticIndexManager:
    """Maintain an entry-scoped semantic index without blocking HA setup."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry[Any],
        *,
        build_index: BuildIndexCallable | None = None,
        initial_delay_seconds: float = 45,
        initial_jitter_seconds: float = 15,
        debounce_seconds: float = 60,
        periodic_seconds: float = 600,
    ) -> None:
        """Initialize the manager."""
        self._hass = hass
        self._entry = entry
        self._build_index = build_index or (
            lambda: async_build_home_semantic_index(hass)
        )
        self._initial_delay_seconds = initial_delay_seconds
        self._initial_jitter_seconds = initial_jitter_seconds
        self._debounce_seconds = debounce_seconds
        self._periodic_seconds = periodic_seconds
        self._unsubscribers: list[CALLBACK_TYPE] = []
        self._refresh_handle: CALLBACK_TYPE | None = None
        self._periodic_handle: CALLBACK_TYPE | None = None
        self._refresh_task: asyncio.Task[None] | None = None
        self._scheduled_reason: str | None = None
        self._next_refresh_at: str | None = None
        self._next_refresh_monotonic: float | None = None
        self._next_periodic_refresh_at: str | None = None
        self._structural_dirty = False
        self._dirty_entity_ids: set[str] = set()
        self._stopped = False

        self.index: HomeSemanticIndex | None = None
        self.status: RefreshStatus = "loading"
        self.generation = 0
        self.last_refresh_reason: str | None = None
        self.last_success_at: str | None = None
        self.last_duration_ms: int | None = None
        self.last_error_type: str | None = None

    @callback
    def async_start(self) -> None:
        """Start listeners and schedule the initial delayed build."""
        if self._stopped:
            return
        self._unsubscribers.extend(
            (
                self._hass.bus.async_listen(
                    EVENT_STATE_CHANGED, self._async_state_changed
                ),
                self._hass.bus.async_listen(
                    er.EVENT_ENTITY_REGISTRY_UPDATED,
                    self._async_registry_updated,
                ),
                self._hass.bus.async_listen(
                    dr.EVENT_DEVICE_REGISTRY_UPDATED,
                    self._async_registry_updated,
                ),
                self._hass.bus.async_listen(
                    ar.EVENT_AREA_REGISTRY_UPDATED,
                    self._async_registry_updated,
                ),
                self._hass.bus.async_listen(
                    fr.EVENT_FLOOR_REGISTRY_UPDATED,
                    self._async_registry_updated,
                ),
                self._hass.bus.async_listen(
                    lr.EVENT_LABEL_REGISTRY_UPDATED,
                    self._async_registry_updated,
                ),
            )
        )
        self._schedule_refresh(self._initial_delay(), "initial")
        self._schedule_periodic_refresh()

    @callback
    def async_stop(self) -> None:
        """Stop listeners, timers, and any active refresh task."""
        self._stopped = True
        self.status = "stopped"
        self._cancel_refresh_timer()
        self._cancel_periodic_timer()
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()
        if self._refresh_task is not None and not self._refresh_task.done():
            self._refresh_task.cancel()

    @callback
    def async_request_refresh(
        self,
        *,
        reason: str,
        structural: bool = False,
        entity_id: str | None = None,
        delay: float | None = None,
    ) -> None:
        """Mark the index dirty and schedule a coalesced refresh."""
        if self._stopped:
            return
        if structural:
            self._structural_dirty = True
        if entity_id is not None:
            self._dirty_entity_ids.add(entity_id)
        if self._refresh_task is not None and not self._refresh_task.done():
            return
        self._schedule_refresh(
            self._debounce_seconds if delay is None else delay, reason
        )

    def diagnostics(self) -> dict[str, object]:
        """Return aggregate, redacted manager diagnostics."""
        data: dict[str, object] = {
            "loaded": self.index is not None,
            "ready": self.index is not None,
            "status": self.status,
            "generation": self.generation,
            "scheduled": self._refresh_handle is not None,
            "refreshing": self._refresh_task is not None
            and not self._refresh_task.done(),
            "structural_dirty": self._structural_dirty,
            "dirty_entity_count": len(self._dirty_entity_ids),
            "next_refresh_at": self._next_refresh_at,
            "next_periodic_refresh_at": self._next_periodic_refresh_at,
            "last_refresh_reason": self.last_refresh_reason,
            "last_success_at": self.last_success_at,
            "last_duration_ms": self.last_duration_ms,
            "last_error_type": self.last_error_type,
        }
        if self.index is not None:
            data.update(self.index.diagnostics_summary())
        return data

    @callback
    def _async_state_changed(self, event: Event[Any]) -> None:
        """Handle state changes using cheap relevance checks."""
        entity_id = event.data.get("entity_id")
        if not isinstance(entity_id, str):
            return
        old_state = event.data.get("old_state")
        new_state = event.data.get("new_state")
        if not isinstance(old_state, State) and not isinstance(new_state, State):
            return
        if _state_index_fields(old_state) == _state_index_fields(new_state):
            return
        known = (
            self.index is not None and entity_id in self.index.documents_by_entity_id
        )
        structural = old_state is None or new_state is None
        if self.index is not None and not known and not structural:
            return
        self.async_request_refresh(
            reason="state_changed",
            structural=structural,
            entity_id=entity_id,
        )

    @callback
    def _async_registry_updated(self, event: Event[Any]) -> None:
        """Handle structural registry changes."""
        self.async_request_refresh(reason=str(event.event_type), structural=True)

    @callback
    def _schedule_refresh(self, delay: float, reason: str) -> None:
        """Schedule one refresh, replacing a later pending refresh if needed."""
        if self._stopped:
            return
        delay = max(0, delay)
        target_time = monotonic() + delay
        if (
            self._refresh_handle is not None
            and self._next_refresh_monotonic is not None
            and self._next_refresh_monotonic <= target_time
        ):
            return
        self._cancel_refresh_timer()
        self._scheduled_reason = reason
        self._next_refresh_monotonic = target_time
        self._next_refresh_at = (
            dt_util.utcnow() + timedelta(seconds=delay)
        ).isoformat()
        self._refresh_handle = async_call_later(
            self._hass, delay, self._async_refresh_timer_fired
        )

    @callback
    def _schedule_periodic_refresh(self) -> None:
        """Schedule the next periodic full rescan."""
        if self._stopped or self._periodic_handle is not None:
            return
        self._next_periodic_refresh_at = (
            dt_util.utcnow() + timedelta(seconds=self._periodic_seconds)
        ).isoformat()
        self._periodic_handle = async_call_later(
            self._hass, self._periodic_seconds, self._async_periodic_timer_fired
        )

    @callback
    def _async_refresh_timer_fired(self, now: datetime) -> None:
        """Start a background refresh when the debounce timer fires."""
        self._refresh_handle = None
        self._next_refresh_at = None
        self._next_refresh_monotonic = None
        reason = self._scheduled_reason or "scheduled"
        self._scheduled_reason = None
        if self._stopped:
            return
        if self._refresh_task is not None and not self._refresh_task.done():
            return
        self._start_refresh_task(reason)

    @callback
    def _async_periodic_timer_fired(self, now: datetime) -> None:
        """Request a coalesced periodic full rescan."""
        self._periodic_handle = None
        self._next_periodic_refresh_at = None
        if self._stopped:
            return
        self.async_request_refresh(
            reason="periodic_rescan",
            structural=True,
            delay=0,
        )
        self._schedule_periodic_refresh()

    @callback
    def _start_refresh_task(self, reason: str) -> None:
        """Create an HA-managed background refresh task."""
        self.status = "loading" if self.index is None else "refreshing"
        task = self._entry.async_create_background_task(
            self._hass,
            self._async_refresh(reason),
            name=f"{self._entry.title} Home Semantic Index refresh",
        )
        self._refresh_task = task
        task.add_done_callback(self._async_refresh_done)

    async def _async_refresh(self, reason: str) -> None:
        """Build a new index and atomically swap it into place."""
        self.last_refresh_reason = reason
        self._structural_dirty = False
        self._dirty_entity_ids.clear()
        started = monotonic()
        try:
            index = await self._build_index()
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001 - safe aggregate diagnostics only.
            self.last_error_type = type(err).__name__
            self.status = "failed" if self.index is None else "ready"
            _LOGGER.warning(
                "Home Semantic Index refresh failed: %s",
                self.last_error_type,
                exc_info=True,
            )
        else:
            self.index = index
            self.generation += 1
            self.status = "ready"
            self.last_success_at = dt_util.utcnow().isoformat()
            self.last_error_type = None
        finally:
            self.last_duration_ms = round((monotonic() - started) * 1000)

    @callback
    def _async_refresh_done(self, task: asyncio.Task[None]) -> None:
        """Clear task state and run one follow-up pass if changes arrived."""
        if self._refresh_task is task:
            self._refresh_task = None
        if self._stopped:
            return
        if self._structural_dirty or self._dirty_entity_ids:
            self._schedule_refresh(self._debounce_seconds, "dirty_during_refresh")

    @callback
    def _cancel_refresh_timer(self) -> None:
        """Cancel the pending one-shot refresh timer."""
        if self._refresh_handle is not None:
            self._refresh_handle()
            self._refresh_handle = None
        self._next_refresh_at = None
        self._next_refresh_monotonic = None
        self._scheduled_reason = None

    @callback
    def _cancel_periodic_timer(self) -> None:
        """Cancel the periodic refresh timer."""
        if self._periodic_handle is not None:
            self._periodic_handle()
            self._periodic_handle = None
        self._next_periodic_refresh_at = None

    def _initial_delay(self) -> float:
        """Return initial build delay with deterministic per-entry jitter."""
        if self._initial_jitter_seconds <= 0:
            return self._initial_delay_seconds
        jitter = sum(ord(char) for char in self._entry.entry_id) % int(
            self._initial_jitter_seconds
        )
        return self._initial_delay_seconds + jitter


def _state_index_fields(state: State | None) -> tuple[object, ...] | None:
    """Return the state fields that affect semantic documents."""
    if state is None:
        return None
    state_value = None if state.domain in _HIGH_CHURN_STATE_DOMAINS else state.state
    return (
        state_value,
        *((key, state.attributes.get(key)) for key in _INDEXED_STATE_ATTRIBUTES),
    )
