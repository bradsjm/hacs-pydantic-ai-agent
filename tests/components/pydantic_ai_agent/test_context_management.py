"""Test Pydantic AI context management helpers."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast

from custom_components.pydantic_ai_agent.agent.context_management import (
    context_management_capability,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_CONTEXT_MANAGEMENT_MODE,
    CONF_CONTEXT_SUMMARIZATION_MODEL_REF,
    CONTEXT_MANAGEMENT_CONTEXT_MANAGER,
    CONTEXT_MANAGEMENT_OFF,
    CONTEXT_MANAGEMENT_SLIDING_WINDOW,
    CONTEXT_WINDOW_SOURCE_MODELS_DEV,
)
from custom_components.pydantic_ai_agent.models.model_profiles import (
    ResolvedModelProfile,
    resolve_model_profile,
)
from custom_components.pydantic_ai_agent.runtime.types import PydanticAIAgentConfigEntry
from homeassistant.core import HomeAssistant
from pydantic_ai_summarization import ContextManagerCapability, SlidingWindowCapability

from .support.builders import (
    loaded_workspace_entry,
    model_profile_data,
    provider_runtime_data,
    provider_subentry_data,
)


def _entry(
    hass: HomeAssistant, profile_ids: Sequence[str] = ("profile-1",)
) -> PydanticAIAgentConfigEntry:
    """Return a loaded workspace entry with configured model profiles."""
    model_profiles = {
        profile_id: model_profile_data(
            profile_id=profile_id,
            name=profile_id,
            model=f"{profile_id}-model",
            context_window_tokens=1000,
            context_window_source=CONTEXT_WINDOW_SOURCE_MODELS_DEV,
        )
        for profile_id in profile_ids
    }
    provider = provider_subentry_data(model_profiles=model_profiles)
    entry = loaded_workspace_entry(
        (provider,),
        providers={"provider-1": provider_runtime_data()},
    )
    entry.add_to_hass(hass)
    return cast(PydanticAIAgentConfigEntry, entry)


def _profile(
    hass: HomeAssistant,
) -> tuple[PydanticAIAgentConfigEntry, ResolvedModelProfile]:
    """Return a loaded entry and its default resolved profile."""
    entry = _entry(hass)
    return entry, resolve_model_profile(entry, "provider-1:profile-1")


def _recording_model_factory(resolved_refs: list[str]):
    """Return a model factory that records the resolved model profile refs."""

    def model_factory(
        _hass: HomeAssistant,
        _entry: PydanticAIAgentConfigEntry,
        model_profile: ResolvedModelProfile,
    ) -> Any:
        resolved_refs.append(model_profile.ref)
        return object()

    return model_factory


def test_off_mode_returns_no_capability(hass: HomeAssistant) -> None:
    """Test off mode disables context management."""
    entry, profile = _profile(hass)

    capability = context_management_capability(
        hass,
        entry,
        {CONF_CONTEXT_MANAGEMENT_MODE: CONTEXT_MANAGEMENT_OFF},
        profile,
        default_mode=CONTEXT_MANAGEMENT_CONTEXT_MANAGER,
        model_factory=lambda *args: cast(Any, object()),
    )

    assert capability is None


def test_default_conversation_mode_builds_context_manager(
    hass: HomeAssistant,
) -> None:
    """Test conversation default mode builds the summarizing context manager."""
    entry, profile = _profile(hass)
    summarization_model = object()

    capability = context_management_capability(
        hass,
        entry,
        {},
        profile,
        default_mode=CONTEXT_MANAGEMENT_CONTEXT_MANAGER,
        model_factory=lambda *args: cast(Any, summarization_model),
    )

    assert isinstance(capability, ContextManagerCapability)
    assert capability.max_tokens == 1000
    assert capability.compress_threshold == 0.9
    assert capability.keep == ("messages", 10)
    assert capability.include_compact_tool is False
    assert capability.summarization_model is summarization_model


def test_default_ai_task_mode_builds_sliding_window(hass: HomeAssistant) -> None:
    """Test AI task default mode builds a token-budgeted sliding window."""
    entry, profile = _profile(hass)

    capability = context_management_capability(
        hass,
        entry,
        {},
        profile,
        default_mode=CONTEXT_MANAGEMENT_SLIDING_WINDOW,
        model_factory=lambda *args: cast(Any, object()),
    )

    assert isinstance(capability, SlidingWindowCapability)
    assert capability.trigger == ("tokens", 900)
    assert capability.keep == ("tokens", 500)
    assert capability.keep_head == ("messages", 1)


def test_explicit_mode_overrides_default(hass: HomeAssistant) -> None:
    """Test stored context mode overrides the entity default mode."""
    entry, profile = _profile(hass)

    capability = context_management_capability(
        hass,
        entry,
        {CONF_CONTEXT_MANAGEMENT_MODE: CONTEXT_MANAGEMENT_SLIDING_WINDOW},
        profile,
        default_mode=CONTEXT_MANAGEMENT_CONTEXT_MANAGER,
        model_factory=lambda *args: cast(Any, object()),
    )

    assert isinstance(capability, SlidingWindowCapability)


def test_sliding_window_ratios_floor_at_one(hass: HomeAssistant) -> None:
    """Test tiny context windows still produce valid token sizes."""
    entry, profile = _profile(hass)
    profile = replace(profile, context_window_tokens=1)

    capability = context_management_capability(
        hass,
        entry,
        {CONF_CONTEXT_MANAGEMENT_MODE: CONTEXT_MANAGEMENT_SLIDING_WINDOW},
        profile,
        default_mode=CONTEXT_MANAGEMENT_CONTEXT_MANAGER,
        model_factory=lambda *args: cast(Any, object()),
    )

    assert isinstance(capability, SlidingWindowCapability)
    assert capability.trigger == ("tokens", 1)
    assert capability.keep == ("tokens", 1)


def test_summarization_model_uses_active_profile_when_ref_unset(
    hass: HomeAssistant,
) -> None:
    """Test unset summarization model ref uses the active model profile."""
    entry, profile = _profile(hass)
    resolved_refs: list[str] = []

    context_management_capability(
        hass,
        entry,
        {CONF_CONTEXT_MANAGEMENT_MODE: CONTEXT_MANAGEMENT_CONTEXT_MANAGER},
        profile,
        default_mode=CONTEXT_MANAGEMENT_SLIDING_WINDOW,
        model_factory=_recording_model_factory(resolved_refs),
    )

    assert resolved_refs == [profile.ref]


def test_summarization_model_uses_configured_ref(hass: HomeAssistant) -> None:
    """Test configured summarization model ref is resolved for context manager."""
    entry = _entry(hass, ("profile-1", "summary-profile"))
    profile = resolve_model_profile(entry, "provider-1:profile-1")
    resolved_refs: list[str] = []

    context_management_capability(
        hass,
        entry,
        {
            CONF_CONTEXT_MANAGEMENT_MODE: CONTEXT_MANAGEMENT_CONTEXT_MANAGER,
            CONF_CONTEXT_SUMMARIZATION_MODEL_REF: "provider-1:summary-profile",
        },
        profile,
        default_mode=CONTEXT_MANAGEMENT_CONTEXT_MANAGER,
        model_factory=_recording_model_factory(resolved_refs),
    )

    assert resolved_refs == ["provider-1:summary-profile"]


def test_invalid_summarization_ref_falls_back_to_active_profile(
    hass: HomeAssistant,
) -> None:
    """Test stale summarization model refs do not break runtime execution."""
    entry, profile = _profile(hass)
    resolved_refs: list[str] = []

    context_management_capability(
        hass,
        entry,
        {
            CONF_CONTEXT_MANAGEMENT_MODE: CONTEXT_MANAGEMENT_CONTEXT_MANAGER,
            CONF_CONTEXT_SUMMARIZATION_MODEL_REF: "missing:profile",
        },
        profile,
        default_mode=CONTEXT_MANAGEMENT_CONTEXT_MANAGER,
        model_factory=_recording_model_factory(resolved_refs),
    )

    assert resolved_refs == [profile.ref]
