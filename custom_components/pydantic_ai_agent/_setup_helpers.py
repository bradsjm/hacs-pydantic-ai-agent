"""Entry setup helper functions."""

import logging
from collections.abc import Mapping
from typing import Any

from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_API_KEY

from ._types import ProviderRuntimeData, PydanticAIAgentConfigEntry
from .const import (
    CONF_BASE_URL,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_MODE,
)
from .model_profiles import (
    enabled_model_profile_refs,
    parse_model_profile_ref,
    resolve_model_profile,
)

_LOGGER = logging.getLogger(__name__)


def _provider_runtimes(
    entry: PydanticAIAgentConfigEntry,
) -> dict[str, ProviderRuntimeData]:
    """Return runtime provider data for structurally valid provider subentries."""
    runtimes: dict[str, ProviderRuntimeData] = {}
    for subentry in _provider_subentries(entry):
        api_key = subentry.data.get(CONF_API_KEY)
        provider_mode = subentry.data.get(CONF_PROVIDER_MODE)
        if not isinstance(api_key, str) or not api_key:
            _LOGGER.warning(
                "Skipping provider subentry %s without an API key",
                subentry.subentry_id,
            )
            continue
        if not isinstance(provider_mode, str) or not provider_mode:
            _LOGGER.warning(
                "Skipping provider subentry %s without a provider mode",
                subentry.subentry_id,
            )
            continue
        headers = subentry.data.get(CONF_PROVIDER_HEADERS)
        provider_extra_body = subentry.data.get(CONF_PROVIDER_EXTRA_BODY)
        runtimes[subentry.subentry_id] = ProviderRuntimeData(
            provider_subentry_id=subentry.subentry_id,
            name=subentry.title,
            api_key=api_key,
            provider_mode=provider_mode,
            base_url=subentry.data.get(CONF_BASE_URL),
            provider_headers=dict(headers) if isinstance(headers, Mapping) else {},
            provider_extra_body=dict(provider_extra_body)
            if isinstance(provider_extra_body, Mapping)
            else {},
        )
    return runtimes


def _resolved_model_profiles(
    entry: PydanticAIAgentConfigEntry,
    provider_runtimes: Mapping[str, ProviderRuntimeData],
) -> dict[str, Any]:
    """Return enabled model profiles for providers that loaded successfully."""
    profiles: dict[str, Any] = {}
    for ref in enabled_model_profile_refs(entry):
        provider_subentry_id, _profile_id = parse_model_profile_ref(ref)
        if provider_subentry_id not in provider_runtimes:
            continue
        profiles[ref] = resolve_model_profile(entry, ref)
    return profiles


def _provider_subentries(
    entry: PydanticAIAgentConfigEntry,
) -> list[ConfigSubentry]:
    """Return all provider subentries."""
    from .const import SUBENTRY_TYPE_PROVIDER

    return [
        subentry
        for subentry in entry.subentries.values()
        if subentry.subentry_type == SUBENTRY_TYPE_PROVIDER
    ]
