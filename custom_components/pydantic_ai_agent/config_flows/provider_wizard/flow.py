"""Provider setup wizard data builders."""

from collections.abc import Callable, Mapping
from uuid import uuid4

from homeassistant.const import CONF_API_KEY, CONF_NAME

from ...const import (
    CONF_BASE_URL,
    CONF_DISCOVERED,
    CONF_ENABLED,
    CONF_MODEL,
    CONF_MODEL_PRICING,
    CONF_MODEL_PROFILES,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_METADATA,
    CONF_PROVIDER_MODE,
    CONF_PROVIDER_SECRET_HEADER_KEYS,
)
from .const import CONF_CATALOG_PROVIDER_ID
from .types import CatalogModelOption, CatalogProviderOption

type ProfileIdFactory = Callable[[], str]


def build_provider_data(
    provider: CatalogProviderOption,
    *,
    provider_mode: str,
    api_key: str,
    selected_models: tuple[CatalogModelOption, ...],
    provider_name: str | None = None,
    base_url: str | None = None,
    provider_headers: Mapping[str, str] | None = None,
    provider_secret_header_keys: object = (),
    provider_extra_body: Mapping[str, object] | None = None,
    profile_id_factory: ProfileIdFactory | None = None,
) -> dict[str, object]:
    """Return provider subentry data from guided wizard selections."""
    profile_id_factory = profile_id_factory or _profile_id
    data: dict[str, object] = {
        CONF_NAME: provider_name or provider.name,
        CONF_PROVIDER_MODE: provider_mode,
        CONF_API_KEY: api_key.strip(),
        CONF_PROVIDER_METADATA: {CONF_CATALOG_PROVIDER_ID: provider.id},
        CONF_MODEL_PROFILES: build_model_profiles(
            selected_models, profile_id_factory=profile_id_factory
        ),
    }
    effective_base_url = base_url or provider.default_base_url
    if effective_base_url:
        data[CONF_BASE_URL] = effective_base_url.rstrip("/")
    if provider_headers:
        data[CONF_PROVIDER_HEADERS] = dict(provider_headers)
        if (
            isinstance(provider_secret_header_keys, list)
            and provider_secret_header_keys
        ):
            data[CONF_PROVIDER_SECRET_HEADER_KEYS] = list(provider_secret_header_keys)
    if provider_extra_body:
        data[CONF_PROVIDER_EXTRA_BODY] = dict(provider_extra_body)
    return data


def build_model_profiles(
    selected_models: tuple[CatalogModelOption, ...],
    *,
    profile_id_factory: ProfileIdFactory | None = None,
) -> dict[str, dict[str, object]]:
    """Return enabled provider-owned model profiles for selected catalog models."""
    profile_id_factory = profile_id_factory or _profile_id
    profiles: dict[str, dict[str, object]] = {}
    for model in selected_models:
        profile_id = profile_id_factory()
        profiles[profile_id] = {
            "id": profile_id,
            CONF_NAME: model.name,
            CONF_MODEL: model.id,
            CONF_ENABLED: True,
            CONF_DISCOVERED: True,
        }
        pricing = _model_pricing(model)
        if pricing:
            profiles[profile_id][CONF_MODEL_PRICING] = pricing
    return profiles


def selected_models_by_id(
    models: tuple[CatalogModelOption, ...], selected_model_ids: object
) -> tuple[CatalogModelOption, ...]:
    """Return selected catalog models preserving catalog order."""
    if isinstance(selected_model_ids, str) or not isinstance(selected_model_ids, list):
        return ()
    selected_ids = {
        model_id for model_id in selected_model_ids if isinstance(model_id, str)
    }
    return tuple(model for model in models if model.id in selected_ids)


def _profile_id() -> str:
    """Return a new model profile ID."""
    return uuid4().hex


def _model_pricing(model: CatalogModelOption) -> dict[str, float]:
    """Return configured catalog pricing for one model profile."""
    pricing: dict[str, float] = {}
    if model.input_price is not None:
        pricing["input"] = model.input_price
    if model.output_price is not None:
        pricing["output"] = model.output_price
    if model.cache_read_price is not None:
        pricing["cache_read"] = model.cache_read_price
    return pricing
