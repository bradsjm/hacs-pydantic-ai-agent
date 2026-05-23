"""models.dev catalog client for the provider setup wizard."""

import json
from collections.abc import Mapping

import httpx
from homeassistant.core import HomeAssistant
from homeassistant.helpers.httpx_client import get_async_client

from .const import CATALOG_SOURCE_URL, MODEL_CATALOG_TIMEOUT
from .normalize import normalize_catalog
from .types import CompactCatalog


class CatalogLoadError(Exception):
    """Raised when the provider catalog cannot be loaded."""


async def async_fetch_catalog(hass: HomeAssistant) -> CompactCatalog:
    """Fetch and normalize the models.dev catalog."""
    client = get_async_client(hass)
    try:
        response = await client.get(CATALOG_SOURCE_URL, timeout=MODEL_CATALOG_TIMEOUT)
        response.raise_for_status()
    except httpx.HTTPError as err:
        raise CatalogLoadError("Unable to load provider model catalog.") from err
    return await hass.async_add_executor_job(_decode_catalog, response.text)


def _decode_catalog(raw_json: str) -> CompactCatalog:
    """Decode and normalize raw catalog JSON off the event loop."""
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as err:
        raise CatalogLoadError("Provider model catalog returned invalid JSON.") from err
    if not isinstance(payload, Mapping):
        raise CatalogLoadError("Provider model catalog has an unexpected shape.")
    return normalize_catalog(payload)
