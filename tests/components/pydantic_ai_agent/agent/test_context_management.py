"""Tests for context-management capability selection."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from custom_components.pydantic_ai_agent.agent.context_management import (
    context_management_capability,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_CONTEXT_MANAGEMENT_MODE,
    CONF_CONTEXT_SUMMARIZATION_MODEL_REF,
    CONF_CONTEXT_WINDOW_TOKENS,
    CONF_ENABLED,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_PROVIDER_MODE,
    CONTEXT_MANAGEMENT_CONTEXT_MANAGER,
    CONTEXT_MANAGEMENT_OFF,
    CONTEXT_MANAGEMENT_SLIDING_WINDOW,
    PROVIDER_ANTHROPIC,
    SUBENTRY_TYPE_PROVIDER,
)
from custom_components.pydantic_ai_agent.models.model_profiles import (
    ResolvedModelProfile,
)
from homeassistant.core import HomeAssistant
from pydantic_ai_summarization import ContextManagerCapability, SlidingWindowCapability


def test_context_management_off_returns_none(
    hass: HomeAssistant,
    make_profile: Callable[..., ResolvedModelProfile],
) -> None:
    """The explicit off mode disables context-management capability creation."""
    capability = context_management_capability(
        hass,
        SimpleNamespace(subentries={}),
        {CONF_CONTEXT_MANAGEMENT_MODE: CONTEXT_MANAGEMENT_OFF},
        make_profile(),
        default_mode=CONTEXT_MANAGEMENT_SLIDING_WINDOW,
        model_factory=lambda *_args: object(),
    )

    assert capability is None


def test_invalid_mode_falls_back_to_default_sliding_window(
    hass: HomeAssistant,
    make_profile: Callable[..., ResolvedModelProfile],
) -> None:
    """An invalid stored mode uses the caller-provided default mode."""
    capability = context_management_capability(
        hass,
        SimpleNamespace(subentries={}),
        {CONF_CONTEXT_MANAGEMENT_MODE: "invalid"},
        make_profile(context_window_tokens=100),
        default_mode=CONTEXT_MANAGEMENT_SLIDING_WINDOW,
        model_factory=lambda *_args: object(),
    )

    assert isinstance(capability, SlidingWindowCapability)
    assert capability.trigger == ("tokens", 90)
    assert capability.keep == ("tokens", 50)
    assert capability.keep_head == ("messages", 1)


def test_sliding_window_clamps_tiny_context_counts(
    hass: HomeAssistant,
    make_profile: Callable[..., ResolvedModelProfile],
) -> None:
    """Tiny model context windows still produce positive token thresholds."""
    capability = context_management_capability(
        hass,
        SimpleNamespace(subentries={}),
        {CONF_CONTEXT_MANAGEMENT_MODE: CONTEXT_MANAGEMENT_SLIDING_WINDOW},
        make_profile(context_window_tokens=1),
        default_mode=CONTEXT_MANAGEMENT_CONTEXT_MANAGER,
        model_factory=lambda *_args: object(),
    )

    assert isinstance(capability, SlidingWindowCapability)
    assert capability.trigger == ("tokens", 1)
    assert capability.keep == ("tokens", 1)


def test_context_manager_uses_active_profile_without_summarization_ref(
    hass: HomeAssistant,
    make_profile: Callable[..., ResolvedModelProfile],
) -> None:
    """Context manager mode builds the summarizer from the active profile by default."""

    def model_factory(
        _hass: HomeAssistant, _entry: Any, profile: ResolvedModelProfile
    ) -> object:
        return f"built:{profile.ref}"

    active_profile = make_profile(context_window_tokens=200)
    capability = context_management_capability(
        hass,
        SimpleNamespace(subentries={}),
        {CONF_CONTEXT_MANAGEMENT_MODE: CONTEXT_MANAGEMENT_CONTEXT_MANAGER},
        active_profile,
        default_mode=CONTEXT_MANAGEMENT_SLIDING_WINDOW,
        model_factory=model_factory,
    )

    assert isinstance(capability, ContextManagerCapability)
    assert capability.max_tokens == 200
    assert capability.keep == ("messages", 10)
    assert capability.summarization_model == "built:provider-1:default"


def test_context_manager_uses_configured_summarization_profile(
    hass: HomeAssistant,
    make_profile: Callable[..., ResolvedModelProfile],
    make_subentry: Callable[..., Any],
) -> None:
    """A valid summarization ref is resolved and passed to the model factory."""
    provider = make_subentry(
        subentry_id="provider-2",
        title="Summary Provider",
        subentry_type=SUBENTRY_TYPE_PROVIDER,
        data={
            CONF_PROVIDER_MODE: PROVIDER_ANTHROPIC,
            CONF_MODEL_PROFILES: {
                "summary": {
                    CONF_ENABLED: True,
                    CONF_MODEL: "summary-model",
                    CONF_CONTEXT_WINDOW_TOKENS: 123,
                }
            },
        },
    )

    def model_factory(
        _hass: HomeAssistant, _entry: Any, profile: ResolvedModelProfile
    ) -> object:
        return f"built:{profile.ref}:{profile.model_name}"

    capability = context_management_capability(
        hass,
        SimpleNamespace(subentries={"provider-2": provider}),
        {
            CONF_CONTEXT_MANAGEMENT_MODE: CONTEXT_MANAGEMENT_CONTEXT_MANAGER,
            CONF_CONTEXT_SUMMARIZATION_MODEL_REF: "provider-2:summary",
        },
        make_profile(ref="provider-1:active", context_window_tokens=300),
        default_mode=CONTEXT_MANAGEMENT_SLIDING_WINDOW,
        model_factory=model_factory,
    )

    assert isinstance(capability, ContextManagerCapability)
    assert capability.summarization_model == (
        "built:provider-2:summary:summary-model"
    )
