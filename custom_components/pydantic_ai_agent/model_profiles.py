"""Provider-owned model profile helpers."""

from collections.abc import Mapping
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any, cast

from homeassistant.config_entries import ConfigEntryState, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from pydantic_ai.capabilities import Thinking
from pydantic_ai.settings import ModelSettings

from .const import (
    CONF_CHAT_TEMPLATE_KWARGS,
    CONF_FALLBACK_MODEL_SUBENTRY_IDS,
    CONF_MAX_ITERATIONS,
    CONF_MODEL,
    CONF_MODEL_SETTINGS,
    CONF_MODEL_SUBENTRY_ID,
    DEFAULT_TIMEOUT,
    DOMAIN,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE_GEMINI,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
    SUBENTRY_TYPE_MODEL,
)
from .chat_template_kwargs import reject_chat_template_kwargs_in_extra_body
from .provider import (
    anthropic_model,
    google_gemini_model,
    openai_compatible_completions_model,
    openai_compatible_responses_model,
)

if TYPE_CHECKING:
    from . import PydanticAIAgentConfigEntry

_MODEL_SETTING_THINKING = "thinking"
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ModelProfile:
    """Resolved model profile runtime data."""

    subentry_id: str
    owner_entry_id: str
    title: str
    provider_title: str
    provider_mode: str
    model_name: str
    model_settings: dict[str, Any]


def model_profile_subentries(entry: PydanticAIAgentConfigEntry) -> list[ConfigSubentry]:
    """Return configured model profile subentries."""
    return [
        subentry
        for subentry in entry.subentries.values()
        if subentry.subentry_type == SUBENTRY_TYPE_MODEL
    ]


def model_profile_ref(entry_id: str, subentry_id: str) -> str:
    """Return the canonical cross-entry model profile reference."""
    return f"{entry_id}:{subentry_id}"


def parse_model_profile_ref(
    current_entry: PydanticAIAgentConfigEntry, raw_ref: str
) -> tuple[str, str]:
    """Parse a model profile reference, treating legacy bare IDs as local."""
    if ":" in raw_ref:
        entry_id, subentry_id = raw_ref.split(":", 1)
        return entry_id, subentry_id
    return current_entry.entry_id, raw_ref


def resolve_model_profile_ref(
    hass: HomeAssistant, current_entry: PydanticAIAgentConfigEntry, raw_ref: str
) -> tuple[PydanticAIAgentConfigEntry, ConfigSubentry]:
    """Resolve a possibly cross-entry model profile reference."""
    entry_id, subentry_id = parse_model_profile_ref(current_entry, raw_ref)
    entry = hass.config_entries.async_get_entry(entry_id)
    if (
        entry is None
        or entry.domain != DOMAIN
        or entry.state != ConfigEntryState.LOADED
    ):
        raise HomeAssistantError("Configured model profile was not found")
    owner_entry = cast("PydanticAIAgentConfigEntry", entry)
    subentry = owner_entry.subentries.get(subentry_id)
    if subentry is None or subentry.subentry_type != SUBENTRY_TYPE_MODEL:
        raise HomeAssistantError("Configured model profile was not found")
    return owner_entry, subentry


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
        owner_entry_id=entry.entry_id,
        title=subentry.title,
        provider_title=entry.title,
        provider_mode=entry.runtime_data.provider_mode,
        model_name=model_name,
        model_settings=model_settings,
    )


def primary_model_profile(
    entry: PydanticAIAgentConfigEntry, owner_subentry: ConfigSubentry
) -> ModelProfile:
    """Return the primary model profile for one consumer subentry."""
    primary_id = owner_subentry.data.get(CONF_MODEL_SUBENTRY_ID)
    if not isinstance(primary_id, str) or not primary_id:
        raise HomeAssistantError("Subentry is missing a model profile")
    return model_profile(entry, primary_id)


def model_profile_chain(
    hass: HomeAssistant,
    entry: PydanticAIAgentConfigEntry,
    owner_subentry: ConfigSubentry,
) -> list[ModelProfile]:
    """Return primary profile followed by ordered fallback profiles."""
    primary = primary_model_profile(entry, owner_subentry)
    fallback_ids = owner_subentry.data.get(CONF_FALLBACK_MODEL_SUBENTRY_IDS, [])
    if isinstance(fallback_ids, str) or not isinstance(fallback_ids, list):
        fallback_ids = []
    primary_ref = model_profile_ref(entry.entry_id, primary.subentry_id)
    profiles = [primary]
    for fallback_id in fallback_ids:
        if not isinstance(fallback_id, str):
            continue
        entry_id, subentry_id = parse_model_profile_ref(entry, fallback_id)
        if model_profile_ref(entry_id, subentry_id) == primary_ref:
            raise HomeAssistantError("Primary model profile cannot also be a fallback")
        try:
            owner_entry, subentry = resolve_model_profile_ref(hass, entry, fallback_id)
            profiles.append(model_profile(owner_entry, subentry.subentry_id))
        except HomeAssistantError:
            _LOGGER.warning(
                "Skipping unavailable fallback model profile %s for subentry %s",
                fallback_id,
                owner_subentry.subentry_id,
            )
    return profiles


def model_settings(profile: ModelProfile) -> ModelSettings:
    """Return Pydantic AI model settings for one profile."""
    settings = dict(profile.model_settings)
    settings.pop(CONF_MAX_ITERATIONS, None)
    settings.pop(CONF_CHAT_TEMPLATE_KWARGS, None)
    settings.pop(_MODEL_SETTING_THINKING, None)
    reject_chat_template_kwargs_in_extra_body(settings.get("extra_body"))
    settings.setdefault("timeout", DEFAULT_TIMEOUT)
    return ModelSettings(**settings)


def thinking_capability(profile: ModelProfile) -> Thinking | None:
    """Return the configured Thinking capability for one profile."""
    if _MODEL_SETTING_THINKING not in profile.model_settings:
        return None
    return Thinking(effort=profile.model_settings[_MODEL_SETTING_THINKING])


def max_iterations(profile: ModelProfile, default: int) -> int:
    """Return the configured agent iteration limit for one profile."""
    value = profile.model_settings.get(CONF_MAX_ITERATIONS)
    if type(value) is int and value > 0:
        return value
    return default


def model_display_names(profiles: list[ModelProfile]) -> list[str]:
    """Return display labels for a resolved model chain."""
    return [f"{profile.provider_title} / {profile.title}" for profile in profiles]


def chat_model_for_profile(
    hass: HomeAssistant,
    profile: ModelProfile,
) -> Any:
    """Build the configured Pydantic AI model for one profile."""
    entry = hass.config_entries.async_get_entry(profile.owner_entry_id)
    if (
        entry is None
        or entry.domain != DOMAIN
        or entry.state != ConfigEntryState.LOADED
    ):
        raise HomeAssistantError("Configured model profile provider was not found")
    runtime_data = entry.runtime_data
    kwargs = {
        "api_key": runtime_data.api_key,
        "base_url": runtime_data.base_url,
        "headers": runtime_data.provider_headers,
        "model_name": profile.model_name,
    }
    try:
        if runtime_data.provider_mode == PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS:
            return openai_compatible_completions_model(hass, **kwargs)
        if runtime_data.provider_mode == PROVIDER_OPENAI_COMPATIBLE_RESPONSES:
            return openai_compatible_responses_model(hass, **kwargs)
        if runtime_data.provider_mode == PROVIDER_ANTHROPIC:
            return anthropic_model(hass, **kwargs)
        if runtime_data.provider_mode == PROVIDER_GOOGLE_GEMINI:
            return google_gemini_model(hass, **kwargs)
    except ValueError as err:
        raise HomeAssistantError(str(err)) from err
    raise HomeAssistantError(
        f"Unsupported provider mode: {runtime_data.provider_mode!r}"
    )
