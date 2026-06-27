"""Tests for provider-owned model profile resolution."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from custom_components.pydantic_ai_agent.const import (
    CONF_CONTEXT_WINDOW_TOKENS,
    CONF_ENABLED,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_PROVIDER_MODE,
    CONF_STRUCTURED_OUTPUT_SUPPORT,
    CONF_SUPPORTS_TOOLS,
    CONF_THINKING_SUPPORT,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_PROVIDER,
)
from custom_components.pydantic_ai_agent.models.model_profiles import (
    resolve_model_profile,
)
from homeassistant.const import CONF_NAME


def test_resolve_model_profile_reads_provider_owned_profile(
    make_subentry: Callable[..., Any],
) -> None:
    """Enabled provider-owned profiles resolve persisted runtime settings."""
    provider = make_subentry(
        subentry_id="provider-1",
        title="OpenAI Compatible",
        subentry_type=SUBENTRY_TYPE_PROVIDER,
        data={
            CONF_MODEL_PROFILES: {
                "default": {
                    CONF_ENABLED: True,
                    CONF_NAME: "Fast Model",
                    CONF_MODEL: "fast-model",
                    CONF_MODEL_SETTINGS: {"temperature": 0.2},
                    CONF_CONTEXT_WINDOW_TOKENS: 12345,
                    CONF_THINKING_SUPPORT: "supported",
                    CONF_STRUCTURED_OUTPUT_SUPPORT: "json_schema",
                    CONF_SUPPORTS_TOOLS: False,
                }
            },
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        },
    )

    profile = resolve_model_profile(
        SimpleNamespace(subentries={"provider-1": provider}),
        "provider-1:default",
    )

    assert profile.ref == "provider-1:default"
    assert profile.provider_title == "OpenAI Compatible"
    assert profile.title == "Fast Model"
    assert profile.provider_mode == PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS
    assert profile.model_name == "fast-model"
    assert profile.model_settings == {"temperature": 0.2}
    assert profile.context_window_tokens == 12345
    assert profile.thinking_support == "supported"
    assert profile.structured_output_support == "json_schema"
    assert profile.supports_tools is False
