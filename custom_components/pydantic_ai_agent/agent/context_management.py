"""Pydantic AI context management helpers."""

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai_summarization import ContextManagerCapability, SlidingWindowCapability

from ..const import (
    CONF_CONTEXT_MANAGEMENT_MODE,
    CONF_CONTEXT_SUMMARIZATION_MODEL_REF,
    CONTEXT_MANAGEMENT_CONTEXT_MANAGER,
    CONTEXT_MANAGEMENT_MODES,
    CONTEXT_MANAGEMENT_OFF,
)
from ..models.model_profiles import ResolvedModelProfile, resolve_model_profile

if TYPE_CHECKING:
    from pydantic_ai.models import Model

    from ..runtime.types import PydanticAIAgentConfigEntry

type ModelFactory = Callable[[HomeAssistant, PydanticAIAgentConfigEntry, ResolvedModelProfile], Model]

_CONTEXT_MANAGER_COMPRESS_THRESHOLD = 0.9
_SLIDING_WINDOW_TRIGGER_RATIO = 0.9
_SLIDING_WINDOW_KEEP_RATIO = 0.5


def context_management_capability(
    hass: HomeAssistant,
    entry: PydanticAIAgentConfigEntry,
    data: Mapping[str, Any],
    active_profile: ResolvedModelProfile,
    *,
    default_mode: str,
    model_factory: ModelFactory,
) -> AbstractCapability | None:
    """Return the context-management capability for one active profile."""
    mode = data.get(CONF_CONTEXT_MANAGEMENT_MODE, default_mode)
    if mode not in CONTEXT_MANAGEMENT_MODES:
        mode = default_mode
    if mode == CONTEXT_MANAGEMENT_OFF:
        return None
    if mode == CONTEXT_MANAGEMENT_CONTEXT_MANAGER:
        return ContextManagerCapability(
            max_tokens=active_profile.context_window_tokens,
            compress_threshold=_CONTEXT_MANAGER_COMPRESS_THRESHOLD,
            keep=("messages", 10),
            summarization_model=model_factory(
                hass,
                entry,
                _summarization_profile(entry, data, active_profile),
            ),
            include_compact_tool=False,
        )
    return SlidingWindowCapability(
        trigger=(
            "tokens",
            max(
                1,
                int(active_profile.context_window_tokens * _SLIDING_WINDOW_TRIGGER_RATIO),
            ),
        ),
        keep=(
            "tokens",
            max(
                1,
                int(active_profile.context_window_tokens * _SLIDING_WINDOW_KEEP_RATIO),
            ),
        ),
        keep_head=("messages", 1),
    )


def _summarization_profile(
    entry: PydanticAIAgentConfigEntry,
    data: Mapping[str, Any],
    active_profile: ResolvedModelProfile,
) -> ResolvedModelProfile:
    """Return the configured summarization profile or the active profile."""
    raw_ref = data.get(CONF_CONTEXT_SUMMARIZATION_MODEL_REF)
    if not isinstance(raw_ref, str) or not raw_ref:
        return active_profile
    try:
        return resolve_model_profile(entry, raw_ref)
    except HomeAssistantError:
        return active_profile
