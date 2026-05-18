"""Test model profile helpers."""

from custom_components.pydantic_ai_agent.model_profiles import (
    ModelProfile,
    model_settings,
    thinking_capability,
)


def test_model_settings_excludes_capability_backed_thinking() -> None:
    """Test thinking is exposed through capabilities, not ModelSettings."""
    profile = ModelProfile(
        subentry_id="model_profile_1",
        title="Fast GPT",
        model_name="gpt-test",
        model_settings={"temperature": 0.7, "thinking": "high"},
    )

    settings = model_settings(profile)
    thinking = thinking_capability(profile)

    assert settings["temperature"] == 0.7
    assert settings.get("thinking") is None
    assert thinking is not None
    assert thinking.effort == "high"


def test_thinking_capability_keeps_explicit_false() -> None:
    """Test explicit thinking=False creates a capability."""
    profile = ModelProfile(
        subentry_id="model_profile_1",
        title="Fast GPT",
        model_name="gpt-test",
        model_settings={"thinking": False},
    )

    thinking = thinking_capability(profile)

    assert thinking is not None
    assert thinking.effort is False


def test_thinking_capability_absent_when_unconfigured() -> None:
    """Test absent thinking means no Thinking capability."""
    profile = ModelProfile(
        subentry_id="model_profile_1",
        title="Fast GPT",
        model_name="gpt-test",
        model_settings={},
    )

    assert thinking_capability(profile) is None
