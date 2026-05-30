"""Shared provider wizard catalog cache."""

import asyncio
from datetime import datetime

from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers.event import async_call_later
from homeassistant.util import dt as dt_util

from ...const import DOMAIN
from .const import (
    DATA_CATALOG_MANAGER,
    MODEL_CATALOG_HARD_TTL,
    MODEL_CATALOG_IDLE_TTL,
)
from .models_dev import async_fetch_catalog
from .types import CompactCatalog


class ProviderWizardCatalogManager:
    """Lazy compact catalog manager shared across provider wizard flows."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the catalog manager."""
        self.hass = hass
        self.catalog: CompactCatalog | None = None
        self.loaded_at: datetime | None = None
        self.last_used_at: datetime | None = None
        self.inflight_task: asyncio.Task[CompactCatalog] | None = None
        self.cleanup_unsub: CALLBACK_TYPE | None = None

    @callback
    def cached_catalog(self) -> CompactCatalog | None:
        """Return a fresh cached catalog, if available."""
        if self.catalog is None or not self.cache_is_fresh():
            return None
        self.touch()
        return self.catalog

    @callback
    def cache_is_fresh(self) -> bool:
        """Return if cached catalog data is still usable."""
        return self.catalog is not None and self.expires_at() > dt_util.utcnow()

    @callback
    def expires_at(self) -> datetime:
        """Return the current cache expiration time."""
        if self.loaded_at is None or self.last_used_at is None:
            return dt_util.utcnow()
        return min(
            self.last_used_at + MODEL_CATALOG_IDLE_TTL,
            self.loaded_at + MODEL_CATALOG_HARD_TTL,
        )

    @callback
    def touch(self) -> None:
        """Refresh idle cache lifetime after use."""
        if self.catalog is None or self.loaded_at is None:
            return
        self.last_used_at = dt_util.utcnow()
        self._schedule_cleanup()

    @callback
    def load_task(self) -> asyncio.Task[CompactCatalog]:
        """Return a per-flow catalog load task, creating the shared fetch if needed."""
        cached_catalog = self.cached_catalog()
        if cached_catalog is not None:
            return self.hass.async_create_task(_return_catalog(cached_catalog))
        if self.catalog is not None:
            self._clear_cache_state(cancel_inflight=False)
        if self.inflight_task is None or self.inflight_task.done():
            self.inflight_task = self.hass.async_create_task(self._async_load_catalog())
        return self.hass.async_create_task(_await_shared_catalog(self.inflight_task))

    async def async_get_catalog(self) -> CompactCatalog:
        """Return a fresh catalog, loading it if needed."""
        return await self.load_task()

    async def _async_load_catalog(self) -> CompactCatalog:
        """Load and store the compact provider catalog."""
        catalog = await async_fetch_catalog(self.hass)
        now = dt_util.utcnow()
        self.catalog = catalog
        self.loaded_at = now
        self.last_used_at = now
        self._schedule_cleanup()
        return catalog

    @callback
    def clear(self) -> None:
        """Drop cached catalog data and cleanup handles."""
        self._clear_cache_state(cancel_inflight=True)

    @callback
    def _clear_cache_state(self, *, cancel_inflight: bool) -> None:
        """Drop cached catalog state, optionally cancelling the shared fetch."""
        self.catalog = None
        self.loaded_at = None
        self.last_used_at = None
        if cancel_inflight and self.inflight_task is not None:
            if not self.inflight_task.done():
                self.inflight_task.cancel()
            self.inflight_task = None
        if self.cleanup_unsub is not None:
            self.cleanup_unsub()
            self.cleanup_unsub = None

    @callback
    def clear_expired(self) -> None:
        """Drop the cached catalog if it is expired."""
        if self.catalog is None:
            return
        if self.cache_is_fresh():
            self._schedule_cleanup()
            return
        self._clear_cache_state(cancel_inflight=False)

    @callback
    def _schedule_cleanup(self) -> None:
        """Schedule cache cleanup for the current expiration time."""
        if self.catalog is None:
            return
        if self.cleanup_unsub is not None:
            self.cleanup_unsub()
        delay = max((self.expires_at() - dt_util.utcnow()).total_seconds(), 0)
        self.cleanup_unsub = async_call_later(
            self.hass, delay, lambda _now: self.clear_expired()
        )


def catalog_manager(hass: HomeAssistant) -> ProviderWizardCatalogManager:
    """Return the shared provider wizard catalog manager."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    manager = domain_data.get(DATA_CATALOG_MANAGER)
    if isinstance(manager, ProviderWizardCatalogManager):
        return manager
    manager = ProviderWizardCatalogManager(hass)
    domain_data[DATA_CATALOG_MANAGER] = manager
    return manager


async def _return_catalog(catalog: CompactCatalog) -> CompactCatalog:
    """Return a cached catalog through a task interface."""
    return catalog


async def _await_shared_catalog(
    task: asyncio.Task[CompactCatalog],
) -> CompactCatalog:
    """Await the shared fetch without allowing one flow to cancel it."""
    return await asyncio.shield(task)
