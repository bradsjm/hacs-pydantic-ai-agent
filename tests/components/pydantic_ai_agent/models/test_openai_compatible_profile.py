"""Tests for persisted OpenAI-compatible profile capabilities."""

from custom_components.pydantic_ai_agent.const import (
    CONF_STRUCTURED_OUTPUT_SUPPORT,
    CONF_SUPPORTS_TOOLS,
    CONF_THINKING_SUPPORT,
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
)
from custom_components.pydantic_ai_agent.models.openai_compatible_profile import (
    PersistedOpenAICompatibleProfile,
    default_openai_compatible_profile_data,
    is_openai_compatible_provider_mode,
)
import pytest


def test_default_openai_compatible_profile_data_is_conservative() -> None:
    """Default discovered/custom profiles support tools but not special output."""
    assert default_openai_compatible_profile_data() == {
        CONF_THINKING_SUPPORT: "none",
        CONF_STRUCTURED_OUTPUT_SUPPORT: "none",
        CONF_SUPPORTS_TOOLS: True,
    }


def test_persisted_profile_from_mapping_parses_capabilities() -> None:
    """Valid persisted mappings become typed capability profiles."""
    profile = PersistedOpenAICompatibleProfile.from_mapping(
        {
            CONF_THINKING_SUPPORT: "supported",
            CONF_STRUCTURED_OUTPUT_SUPPORT: "json_schema",
            CONF_SUPPORTS_TOOLS: False,
        }
    )

    assert profile.thinking_support == "supported"
    assert profile.structured_output_support == "json_schema"
    assert profile.supports_tools is False
    assert profile.supports_thinking() is True
    assert profile.can_disable_thinking() is True


@pytest.mark.parametrize(
    "profile_data",
    [
        {
            CONF_THINKING_SUPPORT: "invalid",
            CONF_STRUCTURED_OUTPUT_SUPPORT: "none",
            CONF_SUPPORTS_TOOLS: True,
        },
        {
            CONF_THINKING_SUPPORT: "none",
            CONF_STRUCTURED_OUTPUT_SUPPORT: "yaml",
            CONF_SUPPORTS_TOOLS: True,
        },
        {
            CONF_THINKING_SUPPORT: "none",
            CONF_STRUCTURED_OUTPUT_SUPPORT: "none",
            CONF_SUPPORTS_TOOLS: "yes",
        },
    ],
)
def test_persisted_profile_from_mapping_rejects_invalid_values(
    profile_data: dict[str, object],
) -> None:
    """Invalid persisted capability values fail with a stable exception class."""
    with pytest.raises(ValueError, match="Invalid"):
        PersistedOpenAICompatibleProfile.from_mapping(profile_data)


@pytest.mark.parametrize(
    ("thinking_support", "requested", "expected"),
    [("none", "low", None), ("supported", "low", "low"), ("always", False, None)],
)
def test_effective_thinking_setting_respects_support_mode(
    thinking_support: str, requested: object, expected: object
) -> None:
    """Thinking is only passed through when the persisted profile supports it."""
    profile = PersistedOpenAICompatibleProfile(
        thinking_support=thinking_support,
        structured_output_support="none",
        supports_tools=True,
    )

    assert profile.effective_thinking_setting(requested) == expected


@pytest.mark.parametrize(
    ("structured_support", "json_schema", "json_object"),
    [("none", False, False), ("json_object", False, True), ("json_schema", True, True)],
)
def test_as_model_profile_reflects_structured_output_flags(
    structured_support: str, json_schema: bool, json_object: bool
) -> None:
    """Synthesized Pydantic AI model profiles mirror persisted capabilities."""
    model_profile = PersistedOpenAICompatibleProfile(
        thinking_support="always",
        structured_output_support=structured_support,
        supports_tools=False,
    ).as_model_profile()

    assert model_profile["supports_thinking"] is True
    assert model_profile["thinking_always_enabled"] is True
    assert model_profile["supports_tools"] is False
    assert model_profile["supports_json_schema_output"] is json_schema
    assert model_profile["supports_json_object_output"] is json_object


def test_is_openai_compatible_provider_mode_matches_supported_modes() -> None:
    """Only the in-repo OpenAI-compatible modes are classified as compatible."""
    assert is_openai_compatible_provider_mode(PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS)
    assert is_openai_compatible_provider_mode(PROVIDER_OPENAI_COMPATIBLE_RESPONSES)
    assert not is_openai_compatible_provider_mode(PROVIDER_ANTHROPIC)
