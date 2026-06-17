"""Workspace-local provider-owned model profile helpers."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from pydantic_ai.capabilities import Thinking
from pydantic_ai.models import Model
from pydantic_ai.profiles import ModelProfile as PydanticAIProfile
from pydantic_ai.settings import (
    ModelSettings as PydanticAIModelSettings,
)
from pydantic_ai.settings import ThinkingLevel

from ..const import (
    CONF_CONTEXT_WINDOW_SOURCE,
    CONF_CONTEXT_WINDOW_TOKENS,
    CONF_ENABLED,
    CONF_FALLBACK_MODEL_REFS,
    CONF_MAX_ITERATIONS,
    CONF_MODEL,
    CONF_MODEL_PRICING,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_MODE,
    CONF_STRUCTURED_OUTPUT_SUPPORT,
    CONF_SUPPORTS_TOOLS,
    CONF_THINKING,
    CONF_THINKING_SUPPORT,
    CONF_TIMEOUT,
    CONTEXT_WINDOW_SOURCE_DEFAULT,
    CONTEXT_WINDOW_SOURCES,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    DEFAULT_TIMEOUT,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE_GEMINI,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
    SUBENTRY_TYPE_PROVIDER,
)
from .model_settings import profile_model_settings, runtime_model_settings_data
from .openai_compatible_profile import (
    PersistedOpenAICompatibleProfile,
    is_openai_compatible_provider_mode,
)
from .provider import (
    anthropic_model,
    effective_thinking_setting,
    google_gemini_model,
    model_thinking_support,
    openai_compatible_completions_model,
    openai_compatible_effective_thinking_setting,
    openai_compatible_model_profile,
    openai_compatible_responses_model,
    openai_compatible_thinking_support,
)

if TYPE_CHECKING:
    from ..runtime.types import PydanticAIAgentConfigEntry

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
    model_pricing: dict[str, float] = field(default_factory=dict)
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS
    context_window_source: str = CONTEXT_WINDOW_SOURCE_DEFAULT
    thinking_support: str | None = None
    structured_output_support: str | None = None
    supports_tools: bool | None = None


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
    provider_mode = provider_subentry.data.get(CONF_PROVIDER_MODE)
    if not isinstance(provider_mode, str):
        return False
    model_name = profile.get(CONF_MODEL)
    if not isinstance(model_name, str) or not model_name.strip():
        return False
    if is_openai_compatible_provider_mode(provider_mode):
        try:
            PersistedOpenAICompatibleProfile.from_mapping(profile)
        except KeyError, ValueError:
            return False
    return True


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
    provider_mode = provider_subentry.data.get(CONF_PROVIDER_MODE)
    if not isinstance(provider_mode, str):
        raise HomeAssistantError("Configured model profile provider was not found")
    model_name = profile.get(CONF_MODEL)
    if not isinstance(model_name, str) or not model_name.strip():
        raise HomeAssistantError("Configured model profile is missing a model name")
    raw_settings = profile.get(CONF_MODEL_SETTINGS)
    model_settings = profile_model_settings(
        raw_settings if isinstance(raw_settings, Mapping) else None
    )
    raw_pricing = profile.get(CONF_MODEL_PRICING)
    model_pricing = _profile_pricing(raw_pricing)
    context_window_tokens = _profile_context_window_tokens(profile)
    context_window_source = _profile_context_window_source(profile)
    openai_profile = _resolved_openai_compatible_profile(provider_mode, profile)
    return ResolvedModelProfile(
        ref=raw_ref,
        provider_subentry_id=provider_subentry_id,
        profile_id=profile_id,
        title=_profile_title(profile, model_name),
        provider_title=provider_subentry.title,
        provider_mode=provider_mode,
        model_name=model_name,
        model_pricing=model_pricing,
        model_settings=model_settings,
        context_window_tokens=context_window_tokens,
        context_window_source=context_window_source,
        thinking_support=(
            openai_profile.thinking_support if openai_profile is not None else None
        ),
        structured_output_support=(
            openai_profile.structured_output_support
            if openai_profile is not None
            else None
        ),
        supports_tools=(
            openai_profile.supports_tools if openai_profile is not None else None
        ),
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


def model_settings(
    profile: ResolvedModelProfile, run_settings: Mapping[str, Any] | None = None
) -> PydanticAIModelSettings:
    """Return Pydantic AI model settings for one profile."""
    settings = runtime_model_settings_data(profile.model_settings, run_settings)
    settings.setdefault(CONF_TIMEOUT, DEFAULT_TIMEOUT)
    return PydanticAIModelSettings(**settings)


def _profile_pricing(raw_pricing: object) -> dict[str, float]:
    """Return sanitized USD-per-million-token pricing for one profile."""
    if not isinstance(raw_pricing, Mapping):
        return {}
    pricing: dict[str, float] = {}
    for key in ("input", "output", "cache_read"):
        value = raw_pricing.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        price = float(value)
        if price >= 0:
            pricing[key] = price
    return pricing


def _profile_context_window_tokens(profile: Mapping[str, Any]) -> int:
    """Return a valid context window token budget for one profile."""
    value = profile.get(CONF_CONTEXT_WINDOW_TOKENS)
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return DEFAULT_CONTEXT_WINDOW_TOKENS
    try:
        parsed = int(value)
    except ValueError:
        return DEFAULT_CONTEXT_WINDOW_TOKENS
    return parsed if parsed > 0 else DEFAULT_CONTEXT_WINDOW_TOKENS


def _profile_context_window_source(profile: Mapping[str, Any]) -> str:
    """Return a valid context window source for one profile."""
    source = profile.get(CONF_CONTEXT_WINDOW_SOURCE)
    if isinstance(source, str) and source in CONTEXT_WINDOW_SOURCES:
        return source
    return CONTEXT_WINDOW_SOURCE_DEFAULT


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


def thinking_capability(
    run_settings: Mapping[str, Any], profile: ResolvedModelProfile | None = None
) -> Thinking | None:
    """Return the configured Thinking capability for one agent/task run."""
    if CONF_THINKING not in run_settings:
        return None
    if profile is not None:
        requested_thinking = cast(ThinkingLevel, run_settings[CONF_THINKING])
        if is_openai_compatible_provider_mode(profile.provider_mode):
            thinking = _resolved_openai_effective_thinking_setting(
                profile, requested_thinking
            )
        else:
            thinking = effective_thinking_setting(
                profile.provider_mode,
                profile.model_name,
                requested_thinking,
            )
        if thinking is None:
            return None
        return Thinking(effort=thinking)
    return Thinking(effort=cast(ThinkingLevel, run_settings[CONF_THINKING]))


def max_iterations(run_settings: Mapping[str, Any], default: int) -> int:
    """Return the configured agent iteration limit for one agent/task run."""
    value = run_settings.get(CONF_MAX_ITERATIONS)
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
    hass: HomeAssistant,
    entry: PydanticAIAgentConfigEntry,
    profile: ResolvedModelProfile,
) -> Model:
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
            return openai_compatible_completions_model(
                hass,
                **kwargs,
                profile=_resolved_openai_model_profile(profile),
            )
        if provider_runtime.provider_mode == PROVIDER_OPENAI_COMPATIBLE_RESPONSES:
            return openai_compatible_responses_model(
                hass,
                **kwargs,
                profile=_resolved_openai_model_profile(profile),
            )
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


def provider_profile_thinking_support(
    provider_mode: str, profile: Mapping[str, Any]
) -> tuple[bool, bool]:
    """Return thinking support for one persisted provider-owned profile."""
    if is_openai_compatible_provider_mode(provider_mode):
        support = openai_compatible_thinking_support(profile)
        return support.supported, support.can_disable
    model_name = profile.get(CONF_MODEL)
    if not isinstance(model_name, str) or not model_name.strip():
        return False, False
    support = model_thinking_support(provider_mode, model_name)
    return support.supported, support.can_disable


def effective_profile_thinking_setting(
    provider_mode: str,
    profile: Mapping[str, Any],
    thinking: ThinkingLevel | None,
) -> ThinkingLevel | None:
    """Return thinking validated against one persisted provider-owned profile."""
    if is_openai_compatible_provider_mode(provider_mode):
        return openai_compatible_effective_thinking_setting(profile, thinking)
    model_name = profile.get(CONF_MODEL)
    if not isinstance(model_name, str) or not model_name.strip():
        return None
    return effective_thinking_setting(provider_mode, model_name, thinking)


def _resolved_openai_compatible_profile(
    provider_mode: str, profile: Mapping[str, Any]
) -> PersistedOpenAICompatibleProfile | None:
    """Return parsed persisted OpenAI-compatible capabilities for one profile."""
    if not is_openai_compatible_provider_mode(provider_mode):
        return None
    try:
        return PersistedOpenAICompatibleProfile.from_mapping(profile)
    except (KeyError, ValueError) as err:
        raise HomeAssistantError(
            "Configured OpenAI-compatible model profile is incomplete"
        ) from err


def _resolved_openai_profile_data(profile: ResolvedModelProfile) -> dict[str, Any]:
    """Return persisted OpenAI-compatible capability fields for one profile."""
    if (
        profile.thinking_support is None
        or profile.structured_output_support is None
        or profile.supports_tools is None
    ):
        raise HomeAssistantError(
            "Configured OpenAI-compatible model profile is incomplete"
        )
    return {
        CONF_THINKING_SUPPORT: profile.thinking_support,
        CONF_STRUCTURED_OUTPUT_SUPPORT: profile.structured_output_support,
        CONF_SUPPORTS_TOOLS: profile.supports_tools,
    }


def _resolved_openai_model_profile(
    profile: ResolvedModelProfile,
) -> PydanticAIProfile:
    """Return an OpenAIModelProfile synthesized from one resolved profile."""
    return openai_compatible_model_profile(_resolved_openai_profile_data(profile))


def _resolved_openai_effective_thinking_setting(
    profile: ResolvedModelProfile, thinking: ThinkingLevel | None
) -> ThinkingLevel | None:
    """Return persisted-profile thinking for one resolved OpenAI-compatible model."""
    return openai_compatible_effective_thinking_setting(
        _resolved_openai_profile_data(profile), thinking
    )
