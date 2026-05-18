"""Provider-owned model profile helpers."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from pydantic_ai.settings import ModelSettings

from .const import (
    CONF_FALLBACK_MODEL_SUBENTRY_IDS,
    CONF_MAX_ITERATIONS,
    CONF_MODEL,
    CONF_MODEL_SETTINGS,
    CONF_MODEL_SUBENTRY_ID,
    DEFAULT_TIMEOUT,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
    SUBENTRY_TYPE_MODEL,
)
from .provider import (
    openai_compatible_completions_model,
    openai_compatible_responses_model,
)

if TYPE_CHECKING:
    from . import PydanticAIAgentConfigEntry


@dataclass(frozen=True, kw_only=True)
class ModelProfile:
    """Resolved model profile runtime data."""

    subentry_id: str
    title: str
    model_name: str
    model_settings: dict[str, Any]


def model_profile_subentries(entry: PydanticAIAgentConfigEntry) -> list[ConfigSubentry]:
    """Return configured model profile subentries."""
    return [
        subentry
        for subentry in entry.subentries.values()
        if subentry.subentry_type == SUBENTRY_TYPE_MODEL
    ]


def model_profile(entry: PydanticAIAgentConfigEntry, subentry_id: str) -> ModelProfile:
    """Resolve one model profile subentry by ID."""
    subentry = entry.subentries.get(subentry_id)
    if subentry is None or subentry.subentry_type != SUBENTRY_TYPE_MODEL:
        raise HomeAssistantError("Configured model profile was not found")
    model_name = subentry.data.get(CONF_MODEL)
    if not isinstance(model_name, str) or not model_name.strip():
        raise HomeAssistantError("Configured model profile is missing a model name")
    raw_settings = subentry.data.get(CONF_MODEL_SETTINGS)
    model_settings = dict(raw_settings) if isinstance(raw_settings, Mapping) else {}
    return ModelProfile(
        subentry_id=subentry.subentry_id,
        title=subentry.title,
        model_name=model_name,
        model_settings=model_settings,
    )


def model_profile_chain(
    entry: PydanticAIAgentConfigEntry, owner_subentry: ConfigSubentry
) -> list[ModelProfile]:
    """Return primary profile followed by ordered fallback profiles."""
    primary_id = owner_subentry.data.get(CONF_MODEL_SUBENTRY_ID)
    if not isinstance(primary_id, str) or not primary_id:
        raise HomeAssistantError("Subentry is missing a model profile")
    fallback_ids = owner_subentry.data.get(CONF_FALLBACK_MODEL_SUBENTRY_IDS, [])
    if isinstance(fallback_ids, str) or not isinstance(fallback_ids, list):
        fallback_ids = []
    chain_ids = [primary_id, *fallback_ids]
    if primary_id in fallback_ids:
        raise HomeAssistantError("Primary model profile cannot also be a fallback")
    return [model_profile(entry, profile_id) for profile_id in chain_ids]


def model_settings(profile: ModelProfile) -> ModelSettings:
    """Return Pydantic AI model settings for one profile."""
    settings = dict(profile.model_settings)
    settings.pop(CONF_MAX_ITERATIONS, None)
    settings.setdefault("timeout", DEFAULT_TIMEOUT)
    return ModelSettings(**settings)


def max_iterations(profile: ModelProfile, default: int) -> int:
    """Return the configured agent iteration limit for one profile."""
    value = profile.model_settings.get(CONF_MAX_ITERATIONS)
    if type(value) is int and value > 0:
        return value
    return default


def model_display_names(profiles: list[ModelProfile]) -> list[str]:
    """Return display labels for a resolved model chain."""
    return [profile.title for profile in profiles]


def chat_model_for_profile(
    hass: HomeAssistant,
    entry: PydanticAIAgentConfigEntry,
    profile: ModelProfile,
) -> Any:
    """Build the configured OpenAI-compatible Pydantic AI model for one profile."""
    runtime_data = entry.runtime_data
    kwargs = {
        "api_key": runtime_data.api_key,
        "base_url": runtime_data.base_url,
        "headers": runtime_data.provider_headers,
        "model_name": profile.model_name,
    }
    if runtime_data.provider_mode == PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS:
        return openai_compatible_completions_model(hass, **kwargs)
    if runtime_data.provider_mode == PROVIDER_OPENAI_COMPATIBLE_RESPONSES:
        return openai_compatible_responses_model(hass, **kwargs)
    raise HomeAssistantError(
        f"Unsupported provider mode: {runtime_data.provider_mode!r}"
    )
