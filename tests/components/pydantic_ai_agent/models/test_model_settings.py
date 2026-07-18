"""Tests for model setting normalization helpers."""

from custom_components.pydantic_ai_agent.const import (
    CONF_MAX_ITERATIONS,
    CONF_MAX_TOKENS,
    CONF_TEMPLATED_EXTRA_BODY,
    CONF_THINKING,
    CONF_TIMEOUT,
    CONF_TOOL_RETRIES,
)
from custom_components.pydantic_ai_agent.models.model_settings import (
    MODEL_SETTING_EXTRA_BODY,
    normalise_applied_model_settings,
    normalise_persisted_model_settings,
    profile_model_settings,
    runtime_model_settings_data,
    strip_model_settings,
)


def test_strip_model_settings_removes_only_requested_keys() -> None:
    """strip_model_settings returns a copy without mutating caller data."""
    settings = {
        "temperature": 0.2,
        CONF_MAX_TOKENS: 100,
        CONF_TIMEOUT: 30,
    }

    stripped = strip_model_settings(settings, {CONF_MAX_TOKENS})

    assert stripped == {"temperature": 0.2, CONF_TIMEOUT: 30}
    assert settings == {
        "temperature": 0.2,
        CONF_MAX_TOKENS: 100,
        CONF_TIMEOUT: 30,
    }


def test_normalise_persisted_model_settings_strips_profile_owned_noise() -> None:
    """Persisted comparison JSON excludes run-owned and API-only keys."""
    settings = {
        "z": 2,
        "a": {"b": 1},
        CONF_MAX_TOKENS: 10,
        CONF_MAX_ITERATIONS: 4,
        CONF_TIMEOUT: 12,
        CONF_THINKING: "low",
        CONF_TOOL_RETRIES: 2,
        MODEL_SETTING_EXTRA_BODY: {"provider": True},
    }

    normalised = normalise_persisted_model_settings(settings)

    assert normalised == '{"a":{"b":1},"z":2}'


def test_normalise_applied_model_settings_returns_sorted_compact_json() -> None:
    """Applied settings comparison preserves all keys but is stable."""
    assert (
        normalise_applied_model_settings({"z": 2, "a": {"b": 1}})
        == '{"a":{"b":1},"z":2}'
    )


def test_profile_model_settings_keeps_extra_body_but_removes_run_keys() -> None:
    """Profile settings persist provider API keys but not run controls."""
    settings = {
        "temperature": 0.1,
        MODEL_SETTING_EXTRA_BODY: {"provider": True},
        CONF_MAX_TOKENS: 20,
        CONF_TIMEOUT: 15,
    }

    assert profile_model_settings(settings) == {
        "temperature": 0.1,
        MODEL_SETTING_EXTRA_BODY: {"provider": True},
    }


def test_runtime_model_settings_data_applies_only_run_overrides() -> None:
    """Runtime data strips API-only keys and applies max_tokens/timeout only."""
    profile_settings = {
        "temperature": 0.5,
        CONF_MAX_TOKENS: 10,
        CONF_TIMEOUT: 30,
        CONF_THINKING: "high",
        CONF_TOOL_RETRIES: 5,
        CONF_TEMPLATED_EXTRA_BODY: [{"key": "a", "value_template": "1"}],
        MODEL_SETTING_EXTRA_BODY: {"provider": True},
    }
    run_settings = {
        CONF_MAX_TOKENS: 50,
        CONF_TIMEOUT: 8,
        CONF_THINKING: "low",
        CONF_MAX_ITERATIONS: 3,
    }

    data = runtime_model_settings_data(profile_settings, run_settings)

    assert data == {"temperature": 0.5, CONF_MAX_TOKENS: 50, CONF_TIMEOUT: 8}
    assert profile_settings[CONF_MAX_TOKENS] == 10
    assert run_settings[CONF_TIMEOUT] == 8
