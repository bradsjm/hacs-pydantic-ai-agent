"""Tests for config entry migration helpers."""

from custom_components.pydantic_ai_agent.const import (
    CONF_MODEL_PROFILES,
    CONF_THINKING_SUPPORT,
)
from custom_components.pydantic_ai_agent.runtime.migration import (
    _migrated_provider_thinking_support,
    _migrated_runtime_thinking,
)


def test_migrated_provider_thinking_support_maps_old_modes_to_boolean() -> None:
    """Legacy profile thinking support modes migrate to model-level booleans."""
    data, changed = _migrated_provider_thinking_support(
        {
            CONF_MODEL_PROFILES: {
                "disabled": {CONF_THINKING_SUPPORT: "none"},
                "optional": {CONF_THINKING_SUPPORT: "supported"},
                "always": {CONF_THINKING_SUPPORT: "always"},
            }
        }
    )

    assert changed is True
    assert data[CONF_MODEL_PROFILES] == {
        "disabled": {CONF_THINKING_SUPPORT: False},
        "optional": {CONF_THINKING_SUPPORT: True},
        "always": {CONF_THINKING_SUPPORT: True},
    }


def test_migrated_runtime_thinking_maps_legacy_values() -> None:
    """Legacy runtime thinking values migrate to explicit effort selections."""
    assert _migrated_runtime_thinking(False) == "none"
    assert _migrated_runtime_thinking(True) == "medium"
    assert _migrated_runtime_thinking("minimal") == "low"
    assert _migrated_runtime_thinking("low") is None
