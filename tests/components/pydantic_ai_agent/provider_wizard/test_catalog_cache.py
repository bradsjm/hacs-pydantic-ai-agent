"""Tests for provider wizard catalog loading and cache lifecycle."""

import asyncio

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.pydantic_ai_agent.config_flows.provider_wizard import (
    catalog_cache,
)
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.catalog_cache import (
    ProviderWizardCatalogManager,
    catalog_manager,
)
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.const import (
    DATA_CATALOG_MANAGER,
    MODEL_CATALOG_HARD_TTL,
    MODEL_CATALOG_IDLE_TTL,
)
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.models_dev import (
    CatalogLoadError,
    _decode_catalog,
)
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.types import (
    CatalogProviderOption,
    CompactCatalog,
)
from custom_components.pydantic_ai_agent.const import DOMAIN


async def test_catalog_manager_reuses_single_inflight_fetch(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test concurrent catalog callers share one fetch through per-flow tasks."""
    calls = 0
    release = asyncio.Event()
    catalog = _catalog()

    async def fake_fetch_catalog(hass: HomeAssistant) -> CompactCatalog:
        nonlocal calls
        calls += 1
        await release.wait()
        return catalog

    monkeypatch.setattr(catalog_cache, "async_fetch_catalog", fake_fetch_catalog)
    manager = ProviderWizardCatalogManager(hass)

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

    async def fake_fetch_catalog(hass: HomeAssistant) -> CompactCatalog:
        nonlocal calls
        calls += 1
        await release.wait()
        return catalog

    monkeypatch.setattr(catalog_cache, "async_fetch_catalog", fake_fetch_catalog)
    manager = ProviderWizardCatalogManager(hass)

    cancelled_task = manager.load_task()
    surviving_task = manager.load_task()
    cancelled_task.cancel()
    await asyncio.gather(cancelled_task, return_exceptions=True)

    assert manager.inflight_task is not None
    assert not manager.inflight_task.cancelled()
    release.set()
    assert await surviving_task is catalog
    assert calls == 1


async def test_catalog_manager_reuses_fresh_memory_catalog(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test a fresh compact catalog is reused without another fetch."""
    calls = 0
    catalog = _catalog()

    async def fake_fetch_catalog(hass: HomeAssistant) -> CompactCatalog:
        nonlocal calls
        calls += 1
        return catalog

    monkeypatch.setattr(catalog_cache, "async_fetch_catalog", fake_fetch_catalog)
    manager = ProviderWizardCatalogManager(hass)

    assert await manager.async_get_catalog() is catalog
    assert await manager.async_get_catalog() is catalog
    assert calls == 1


def test_catalog_manager_expiration_uses_idle_and_hard_cap(
    hass: HomeAssistant,
) -> None:
    """Test cache expiration uses the earlier idle or hard-cap expiry."""
    manager = ProviderWizardCatalogManager(hass)
    now = dt_util.utcnow()
    manager.catalog = _catalog()
    manager.loaded_at = now
    manager.last_used_at = now + MODEL_CATALOG_HARD_TTL

    assert manager.expires_at() == now + MODEL_CATALOG_HARD_TTL

    manager.last_used_at = now
    assert manager.expires_at() == now + MODEL_CATALOG_IDLE_TTL


def test_catalog_manager_clear_expired_drops_stale_catalog(
    hass: HomeAssistant,
) -> None:
    """Test stale compact catalog data is dropped."""
    manager = ProviderWizardCatalogManager(hass)
    manager.catalog = _catalog()
    manager.loaded_at = dt_util.utcnow() - MODEL_CATALOG_HARD_TTL - MODEL_CATALOG_IDLE_TTL
    manager.last_used_at = dt_util.utcnow() - MODEL_CATALOG_IDLE_TTL

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

    async def fake_fetch_catalog(hass: HomeAssistant) -> CompactCatalog:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise CatalogLoadError("failed")
        return catalog

    monkeypatch.setattr(catalog_cache, "async_fetch_catalog", fake_fetch_catalog)
    manager = ProviderWizardCatalogManager(hass)

    with pytest.raises(CatalogLoadError):
        await manager.async_get_catalog()

    assert manager.catalog is None
    assert await manager.async_get_catalog() is catalog
    assert calls == 2


def test_catalog_manager_is_stored_in_domain_data(hass: HomeAssistant) -> None:
    """Test catalog manager is shared through domain process-global data."""
    manager = catalog_manager(hass)

    assert catalog_manager(hass) is manager
    assert hass.data[DOMAIN][DATA_CATALOG_MANAGER] is manager


def test_decode_catalog_rejects_invalid_shape() -> None:
    """Test raw catalog parser rejects unexpected JSON shapes."""
    with pytest.raises(CatalogLoadError):
        _decode_catalog("[]")


def _catalog() -> CompactCatalog:
    """Return a compact catalog fixture."""
    provider = CatalogProviderOption(
        id="openai",
        name="OpenAI",
        doc_url="https://models.dev/providers/openai",
        api_key_hints=("OPENAI_API_KEY",),
        default_base_url=None,
        supported_drivers=("openai_compatible_completions",),
        model_count=0,
        families=(),
    )
    return CompactCatalog(providers={"openai": provider}, models_by_provider={})
