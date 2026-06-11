"""Pure helper functions extracted from provider_flow.py."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..const import (
    CONF_BASE_URL,
    CONF_PROVIDER_MODE,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE_GEMINI,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
)
from ._provider_data import _normalise_base_url
from .provider_wizard.const import CUSTOM_PROVIDER_ID
from .provider_wizard.types import CatalogModelOption

_DISCOVERED_PROVIDER_ID = "__provider_discovery__"


def _custom_model_options(model_names: list[str]) -> tuple[CatalogModelOption, ...]:
    """Return custom model names as availability-management options."""
    return tuple(
        CatalogModelOption(
            id=model_name,
            name=model_name,
            provider_id=CUSTOM_PROVIDER_ID,
            family=None,
            tool_call=True,
            structured_output=None,
            reasoning=False,
            attachment=False,
            text_output=True,
            context_limit=0,
            output_limit=0,
            status=None,
        )
        for model_name in model_names
    )


def _catalog_model_pricing(model: CatalogModelOption | None) -> dict[str, float]:
    """Return stored pricing seeded from one catalog model option."""
    if model is None:
        return {}
    pricing: dict[str, float] = {}
    if model.input_price is not None:
        pricing["input"] = model.input_price
    if model.output_price is not None:
        pricing["output"] = model.output_price
    if model.cache_read_price is not None:
        pricing["cache_read"] = model.cache_read_price
    return pricing


def _catalog_provider_metadata_still_valid(
    existing_data: Mapping[str, Any], storage_data: Mapping[str, Any]
) -> bool:
    """Return if stored catalog metadata still matches the edited endpoint."""
    return _effective_catalog_base_url(existing_data) == _effective_catalog_base_url(
        storage_data
    )


def _effective_catalog_base_url(data: Mapping[str, Any]) -> str | None:
    """Return the effective endpoint used by model discovery/catalog metadata."""
    base_url = _normalise_base_url(data.get(CONF_BASE_URL))
    provider_mode = data.get(CONF_PROVIDER_MODE)
    if provider_mode == PROVIDER_ANTHROPIC:
        if base_url and base_url.endswith("/v1"):
            return base_url.removesuffix("/v1")
        return base_url or "https://api.anthropic.com"
    if provider_mode == PROVIDER_GOOGLE_GEMINI:
        if base_url and base_url.endswith(("/v1beta", "/v1")):
            return base_url.rsplit("/", 1)[0]
        return base_url or "https://generativelanguage.googleapis.com"
    if provider_mode in {
        PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
    }:
        return base_url or "https://api.openai.com/v1"
    return base_url


def _discovered_model_options(
    model_names: list[str],
) -> tuple[CatalogModelOption, ...]:
    """Return discovered model names as CatalogModelOption tuples."""
    return tuple(
        CatalogModelOption(
            id=model_name,
            name=model_name,
            provider_id=_DISCOVERED_PROVIDER_ID,
            family=None,
            tool_call=True,
            structured_output=None,
            reasoning=False,
            attachment=False,
            text_output=True,
            context_limit=0,
            output_limit=0,
            status=None,
        )
        for model_name in model_names
    )
