"""Tests for provider wizard catalog loading and cache lifecycle."""

import asyncio

import pytest
from custom_components.pydantic_ai_agent.config_flows.provider_wizard import (
    catalog_cache,
    const,
    models_dev,
    types,
)
from custom_components.pydantic_ai_agent.const import DOMAIN
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util


async def test_catalog_manager_reuses_single_inflight_fetch(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test concurrent catalog callers share one fetch through per-flow tasks."""
    calls = 0
    release = asyncio.Event()
    catalog = _catalog()

    async def fake_fetch_catalog(hass: HomeAssistant) -> types.CompactCatalog:
        nonlocal calls
        calls += 1
        await release.wait()
        return catalog

    monkeypatch.setattr(catalog_cache, "async_fetch_catalog", fake_fetch_catalog)
    manager = catalog_cache.ProviderWizardCatalogManager(hass)

    first_task = manager.load_task()
    second_task = manager.load_task()

    assert first_task is not second_task
    assert manager.inflight_task is not None
    release.set()
    assert await first_task is catalog
    assert await second_task is catalog
    assert calls == 1


async def test_catalog_manager_flow_task_cancel_does_not_cancel_shared_fetch(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test cancelling one flow task does not cancel the shared catalog fetch."""
    calls = 0
    release = asyncio.Event()
    catalog = _catalog()

    async def fake_fetch_catalog(hass: HomeAssistant) -> types.CompactCatalog:
        nonlocal calls
        calls += 1
        await release.wait()
        return catalog

    monkeypatch.setattr(catalog_cache, "async_fetch_catalog", fake_fetch_catalog)
    manager = catalog_cache.ProviderWizardCatalogManager(hass)

    cancelled_task = manager.load_task()
    surviving_task = manager.load_task()
    cancelled_task.cancel()
    await asyncio.gather(cancelled_task, return_exceptions=True)

    assert manager.inflight_task is not None
    assert not manager.inflight_task.cancelled()
    release.set()
    assert await surviving_task is catalog
    assert calls == 1


async def test_catalog_manager_clear_cancels_inflight_fetch(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test clearing the cache cancels an active shared catalog fetch."""
    started = asyncio.Event()
    catalog = _catalog()

    async def fake_fetch_catalog(hass: HomeAssistant) -> types.CompactCatalog:
        started.set()
        await asyncio.Event().wait()
        return catalog

    monkeypatch.setattr(catalog_cache, "async_fetch_catalog", fake_fetch_catalog)
    manager = catalog_cache.ProviderWizardCatalogManager(hass)

    flow_task = manager.load_task()
    await started.wait()
    shared_task = manager.inflight_task
    assert shared_task is not None

    manager.clear()

    assert manager.inflight_task is None
    await asyncio.gather(flow_task, shared_task, return_exceptions=True)
    assert shared_task.cancelled()


async def test_catalog_manager_clear_allows_new_fetch(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test loading after clear starts a new shared catalog fetch."""
    calls = 0
    first_started = asyncio.Event()
    catalog = _catalog()

    async def fake_fetch_catalog(hass: HomeAssistant) -> types.CompactCatalog:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            await asyncio.Event().wait()
        return catalog

    monkeypatch.setattr(catalog_cache, "async_fetch_catalog", fake_fetch_catalog)
    manager = catalog_cache.ProviderWizardCatalogManager(hass)

    first_task = manager.load_task()
    await first_started.wait()
    manager.clear()
    await asyncio.gather(first_task, return_exceptions=True)

    assert await manager.async_get_catalog() is catalog
    assert calls == 2


async def test_catalog_manager_expired_cleanup_does_not_cancel_new_fetch(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test stale cache cleanup does not cancel a fresh shared fetch."""
    release = asyncio.Event()
    catalog = _catalog()

    async def fake_fetch_catalog(hass: HomeAssistant) -> types.CompactCatalog:
        await release.wait()
        return catalog

    monkeypatch.setattr(catalog_cache, "async_fetch_catalog", fake_fetch_catalog)
    manager = catalog_cache.ProviderWizardCatalogManager(hass)
    manager.catalog = _catalog()
    manager.loaded_at = (
        dt_util.utcnow() - const.MODEL_CATALOG_HARD_TTL - const.MODEL_CATALOG_IDLE_TTL
    )
    manager.last_used_at = dt_util.utcnow() - const.MODEL_CATALOG_IDLE_TTL

    flow_task = manager.load_task()
    shared_task = manager.inflight_task
    assert shared_task is not None

    manager.clear_expired()

    assert not shared_task.cancelled()
    release.set()
    assert await flow_task is catalog


async def test_catalog_manager_reuses_fresh_memory_catalog(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test a fresh compact catalog is reused without another fetch."""
    calls = 0
    catalog = _catalog()

    async def fake_fetch_catalog(hass: HomeAssistant) -> types.CompactCatalog:
        nonlocal calls
        calls += 1
        return catalog

    monkeypatch.setattr(catalog_cache, "async_fetch_catalog", fake_fetch_catalog)
    manager = catalog_cache.ProviderWizardCatalogManager(hass)

    assert await manager.async_get_catalog() is catalog
    assert await manager.async_get_catalog() is catalog
    assert calls == 1


def test_catalog_manager_expiration_uses_idle_and_hard_cap(
    hass: HomeAssistant,
) -> None:
    """Test cache expiration uses the earlier idle or hard-cap expiry."""
    manager = catalog_cache.ProviderWizardCatalogManager(hass)
    now = dt_util.utcnow()
    manager.catalog = _catalog()
    manager.loaded_at = now
    manager.last_used_at = now + const.MODEL_CATALOG_HARD_TTL

    assert manager.expires_at() == now + const.MODEL_CATALOG_HARD_TTL

    manager.last_used_at = now
    assert manager.expires_at() == now + const.MODEL_CATALOG_IDLE_TTL


def test_catalog_manager_clear_expired_drops_stale_catalog(
    hass: HomeAssistant,
) -> None:
    """Test stale compact catalog data is dropped."""
    manager = catalog_cache.ProviderWizardCatalogManager(hass)
    manager.catalog = _catalog()
    manager.loaded_at = (
        dt_util.utcnow() - const.MODEL_CATALOG_HARD_TTL - const.MODEL_CATALOG_IDLE_TTL
    )
    manager.last_used_at = dt_util.utcnow() - const.MODEL_CATALOG_IDLE_TTL

    manager.clear_expired()

    assert manager.catalog is None
    assert manager.loaded_at is None
    assert manager.last_used_at is None


async def test_catalog_manager_fetch_failure_is_retriable(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test a failed fetch is not cached and can be retried."""
    calls = 0
    catalog = _catalog()

    async def fake_fetch_catalog(hass: HomeAssistant) -> types.CompactCatalog:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise models_dev.CatalogLoadError("failed")
        return catalog

    monkeypatch.setattr(catalog_cache, "async_fetch_catalog", fake_fetch_catalog)
    manager = catalog_cache.ProviderWizardCatalogManager(hass)

    with pytest.raises(models_dev.CatalogLoadError):
        await manager.async_get_catalog()

    assert manager.catalog is None
    assert await manager.async_get_catalog() is catalog
    assert calls == 2


def test_catalog_manager_is_stored_in_domain_data(hass: HomeAssistant) -> None:
    """Test catalog manager is shared through domain process-global data."""
    manager = catalog_cache.catalog_manager(hass)

    assert catalog_cache.catalog_manager(hass) is manager
    assert hass.data[DOMAIN][const.DATA_CATALOG_MANAGER] is manager


def test_decode_catalog_rejects_invalid_shape() -> None:
    """Test raw catalog parser rejects unexpected JSON shapes."""
    with pytest.raises(models_dev.CatalogLoadError):
        models_dev._decode_catalog("[]")


def _catalog() -> types.CompactCatalog:
    """Return a compact catalog fixture."""
    provider = types.CatalogProviderOption(
        id="openai",
        name="OpenAI",
        doc_url="https://models.dev/providers/openai",
        api_key_hints=("OPENAI_API_KEY",),
        default_base_url=None,
        supported_drivers=("openai_compatible_completions",),
        model_count=0,
        families=(),
    )
    return types.CompactCatalog(providers={"openai": provider}, models_by_provider={})
