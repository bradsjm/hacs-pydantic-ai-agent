"""Tests for OpenAI-compatible reasoning request helpers."""

from typing import cast

from custom_components.pydantic_ai_agent.openai_compatible_adapter import (
    _chat_model,
    _responses_model,
)
from custom_components.pydantic_ai_agent.openai_compatible_client import omit
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
import pytest


@pytest.mark.parametrize(
    ("thinking", "expected"),
    [
        (False, omit),
        ("none", omit),
        (True, "medium"),
        ("low", "low"),
    ],
)
def test_chat_reasoning_effort_maps_thinking(thinking: bool | str, expected: object) -> None:
    """Chat reasoning effort maps enabled values and omits disabled values."""
    result = _chat_model._reasoning_effort(ModelRequestParameters(thinking=thinking))

    if expected is omit:
        assert result is omit
    else:
        assert result == expected


@pytest.mark.parametrize(
    ("model_settings", "thinking", "expected"),
    [
        (cast(ModelSettings, {"openai_reasoning_effort": "none"}), None, omit),
        ({}, False, omit),
        ({}, "none", omit),
        (
            cast(ModelSettings, {"openai_reasoning_effort": "high"}),
            None,
            {"effort": "high"},
        ),
        (
            cast(ModelSettings, {"openai_reasoning_effort": "high"}),
            "low",
            {"effort": "low"},
        ),
    ],
)
def test_responses_reasoning_maps_effort_and_thinking_precedence(
    model_settings: ModelSettings,
    thinking: bool | str | None,
    expected: object,
) -> None:
    """Request thinking overrides model effort and disabled values are omitted."""
    result = _responses_model._reasoning(
        model_settings,
        ModelRequestParameters(thinking=thinking),
    )

    if expected is omit:
        assert result is omit
    else:
        assert result == expected


def test_responses_reasoning_keeps_summary_when_effort_omitted() -> None:
    """Reasoning summary settings remain independent from effort omission."""
    assert _responses_model._reasoning(
        cast(
            ModelSettings,
            {"openai_reasoning_effort": "none", "openai_reasoning_summary": "auto"},
        ),
        ModelRequestParameters(thinking=False),
    ) == {"summary": "auto"}
