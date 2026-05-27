"""Workspace-local provider-owned model profile helpers."""

from collections.abc import Mapping
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_NAME
from homeassistant.exceptions import HomeAssistantError
from pydantic_ai.capabilities import Thinking
from pydantic_ai.settings import ModelSettings

from .chat_template_kwargs import reject_chat_template_kwargs_in_extra_body
from .const import (
    CONF_CHAT_TEMPLATE_KWARGS,
    CONF_ENABLED,
    CONF_FALLBACK_MODEL_REFS,
    CONF_MAX_ITERATIONS,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROVIDER_MODE,
    DEFAULT_TIMEOUT,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE_GEMINI,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
    SUBENTRY_TYPE_PROVIDER,
)
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
class ResolvedModelProfile:
    """Resolved provider-owned model profile runtime data."""

    ref: str
    provider_subentry_id: str
    profile_id: str
    title: str
    provider_title: str
    provider_mode: str
    model_name: str
    model_settings: dict[str, Any]


ModelProfile = ResolvedModelProfile


def provider_subentries(entry: PydanticAIAgentConfigEntry) -> list[ConfigSubentry]:
    """Return configured provider subentries."""
    return [
        subentry
        for subentry in entry.subentries.values()
        if subentry.subentry_type == SUBENTRY_TYPE_PROVIDER
    ]


def provider_model_profiles(subentry: ConfigSubentry) -> dict[str, dict[str, Any]]:
    """Return persisted model profiles for one provider subentry."""
    raw_profiles = subentry.data.get(CONF_MODEL_PROFILES)
    if not isinstance(raw_profiles, Mapping):
        return {}
    profiles: dict[str, dict[str, Any]] = {}
    for profile_id, profile in raw_profiles.items():
        if not isinstance(profile_id, str) or not isinstance(profile, Mapping):
            continue
        profiles[profile_id] = dict(profile)
    return profiles


def enabled_model_profile_refs(entry: PydanticAIAgentConfigEntry) -> list[str]:
    """Return enabled workspace-local model profile refs."""
    refs: list[str] = []
    for provider_subentry in provider_subentries(entry):
        for profile_id, profile in provider_model_profiles(provider_subentry).items():
            if not _profile_enabled(profile):
                continue
            refs.append(model_profile_ref(provider_subentry.subentry_id, profile_id))
    return refs


def model_profile_ref(provider_subentry_id: str, profile_id: str) -> str:
    """Return the canonical workspace-local model profile reference."""
    return f"{provider_subentry_id}:{profile_id}"


def parse_model_profile_ref(raw_ref: str) -> tuple[str, str]:
    """Parse a workspace-local provider/profile reference."""
    provider_subentry_id, separator, profile_id = raw_ref.partition(":")
    if not separator or not provider_subentry_id or not profile_id:
        raise HomeAssistantError("Configured model profile was not found")
    return provider_subentry_id, profile_id


def model_profile_exists(entry: PydanticAIAgentConfigEntry, raw_ref: str) -> bool:
    """Return if a workspace-local provider/profile ref exists and is enabled."""
    try:
        resolve_model_profile(entry, raw_ref)
    except HomeAssistantError:
        return False
    return True


def configured_model_profile_exists(
    entry: PydanticAIAgentConfigEntry, raw_ref: str
) -> bool:
    """Return if a persisted workspace-local provider/profile ref is usable."""
    try:
        provider_subentry_id, profile_id = parse_model_profile_ref(raw_ref)
    except HomeAssistantError:
        return False
    provider_subentry = entry.subentries.get(provider_subentry_id)
    if (
        provider_subentry is None
        or provider_subentry.subentry_type != SUBENTRY_TYPE_PROVIDER
    ):
        return False
    profile = provider_model_profiles(provider_subentry).get(profile_id)
    if profile is None or not _profile_enabled(profile):
        return False
    model_name = profile.get(CONF_MODEL)
    return isinstance(model_name, str) and bool(model_name.strip())


def resolve_model_profile(
    entry: PydanticAIAgentConfigEntry, raw_ref: str
) -> ResolvedModelProfile:
    """Resolve one workspace-local provider-owned model profile."""
    provider_subentry_id, profile_id = parse_model_profile_ref(raw_ref)
    provider_subentry = entry.subentries.get(provider_subentry_id)
    if (
        provider_subentry is None
        or provider_subentry.subentry_type != SUBENTRY_TYPE_PROVIDER
    ):
        raise HomeAssistantError("Configured model profile was not found")
    runtime_data = getattr(entry, "runtime_data", None)
    if runtime_data is not None and provider_subentry_id not in runtime_data.providers:
        raise HomeAssistantError("Configured model profile provider was not found")
    profile = provider_model_profiles(provider_subentry).get(profile_id)
    if profile is None or not _profile_enabled(profile):
        raise HomeAssistantError("Configured model profile was not found")
    model_name = profile.get(CONF_MODEL)
    if not isinstance(model_name, str) or not model_name.strip():
        raise HomeAssistantError("Configured model profile is missing a model name")
    raw_settings = profile.get(CONF_MODEL_SETTINGS)
    model_settings = dict(raw_settings) if isinstance(raw_settings, Mapping) else {}
    return ResolvedModelProfile(
        ref=raw_ref,
        provider_subentry_id=provider_subentry_id,
        profile_id=profile_id,
        title=_profile_title(profile, model_name),
        provider_title=provider_subentry.title,
        provider_mode=str(provider_subentry.data.get(CONF_PROVIDER_MODE, "")),
        model_name=model_name,
        model_settings=model_settings,
    )


def primary_model_profile(
    entry: PydanticAIAgentConfigEntry, owner_subentry: ConfigSubentry
) -> ResolvedModelProfile:
    """Return the primary model profile for one consumer subentry."""
    primary_ref = owner_subentry.data.get(CONF_PRIMARY_MODEL_REF)
    if not isinstance(primary_ref, str) or not primary_ref:
        raise HomeAssistantError("Subentry is missing a model profile")
    return resolve_model_profile(entry, primary_ref)


def model_profile_chain(
    entry: PydanticAIAgentConfigEntry,
    owner_subentry: ConfigSubentry,
) -> list[ResolvedModelProfile]:
    """Return primary profile followed by ordered fallback profiles."""
    primary = primary_model_profile(entry, owner_subentry)
    fallback_refs = owner_subentry.data.get(CONF_FALLBACK_MODEL_REFS, [])
    if isinstance(fallback_refs, str) or not isinstance(fallback_refs, list):
        fallback_refs = []
    profiles = [primary]
    for fallback_ref in fallback_refs:
        if not isinstance(fallback_ref, str):
            continue
        if fallback_ref == primary.ref:
            raise HomeAssistantError("Primary model profile cannot also be a fallback")
        try:
            profiles.append(resolve_model_profile(entry, fallback_ref))
        except HomeAssistantError:
            _LOGGER.warning(
                "Skipping unavailable fallback model profile %s for subentry %s",
                fallback_ref,
                owner_subentry.subentry_id,
            )
    return profiles


def model_settings(profile: ResolvedModelProfile) -> ModelSettings:
    """Return Pydantic AI model settings for one profile."""
    settings = dict(profile.model_settings)
    settings.pop(CONF_MAX_ITERATIONS, None)
    settings.pop(CONF_CHAT_TEMPLATE_KWARGS, None)
    settings.pop(_MODEL_SETTING_THINKING, None)
    settings.pop("extra_body", None)
    reject_chat_template_kwargs_in_extra_body(settings.get("extra_body"))
    settings.setdefault("timeout", DEFAULT_TIMEOUT)
    return ModelSettings(**settings)


def provider_extra_body(
    entry: PydanticAIAgentConfigEntry, profile: ResolvedModelProfile
) -> dict[str, Any]:
    """Return provider-level extra request body fields for one profile."""
    provider_subentry = entry.subentries.get(profile.provider_subentry_id)
    if provider_subentry is None:
        return {}
    extra_body = provider_subentry.data.get(CONF_PROVIDER_EXTRA_BODY)
    if not isinstance(extra_body, Mapping) or not extra_body:
        return {}
    return dict(extra_body)


def thinking_capability(profile: ResolvedModelProfile) -> Thinking | None:
    """Return the configured Thinking capability for one profile."""
    if _MODEL_SETTING_THINKING not in profile.model_settings:
        return None
    return Thinking(effort=profile.model_settings[_MODEL_SETTING_THINKING])


def max_iterations(profile: ResolvedModelProfile, default: int) -> int:
    """Return the configured agent iteration limit for one profile."""
    value = profile.model_settings.get(CONF_MAX_ITERATIONS)
    if type(value) is int and value > 0:
        return value
    return default


def model_display_names(profiles: list[ResolvedModelProfile]) -> list[str]:
    """Return display labels for a resolved model chain."""
    return [f"{profile.provider_title} / {profile.title}" for profile in profiles]


def model_profile_display_name(profile: Mapping[str, Any]) -> str:
    """Return the user-facing display name for one persisted model profile."""
    name = profile.get(CONF_NAME)
    if isinstance(name, str) and name.strip():
        return name
    model_name = profile.get(CONF_MODEL)
    if isinstance(model_name, str):
        return model_name
    return ""


def chat_model_for_profile(
    hass: Any,
    entry: PydanticAIAgentConfigEntry,
    profile: ResolvedModelProfile,
) -> Any:
    """Build the configured Pydantic AI model for one profile."""
    provider_runtime = entry.runtime_data.providers.get(profile.provider_subentry_id)
    if provider_runtime is None:
        raise HomeAssistantError("Configured model profile provider was not found")
    kwargs = {
        "api_key": provider_runtime.api_key,
        "base_url": provider_runtime.base_url,
        "headers": provider_runtime.provider_headers,
        "model_name": profile.model_name,
    }
    try:
        if provider_runtime.provider_mode == PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS:
            return openai_compatible_completions_model(hass, **kwargs)
        if provider_runtime.provider_mode == PROVIDER_OPENAI_COMPATIBLE_RESPONSES:
            return openai_compatible_responses_model(hass, **kwargs)
        if provider_runtime.provider_mode == PROVIDER_ANTHROPIC:
            return anthropic_model(hass, **kwargs)
        if provider_runtime.provider_mode == PROVIDER_GOOGLE_GEMINI:
            return google_gemini_model(hass, **kwargs)
    except ValueError as err:
        raise HomeAssistantError(str(err)) from err
    raise HomeAssistantError(
        f"Unsupported provider mode: {provider_runtime.provider_mode!r}"
    )


def _profile_enabled(profile: Mapping[str, Any]) -> bool:
    """Return if one persisted model profile is enabled."""
    return bool(profile.get(CONF_ENABLED, False))


def _profile_title(profile: Mapping[str, Any], model_name: str) -> str:
    """Return a human-readable profile title."""
    return model_profile_display_name(profile) or model_name
