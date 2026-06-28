"""Tests for OpenAI-compatible reasoning request helpers."""

from typing import cast

from custom_components.pydantic_ai_agent.openai_compatible_adapter import (
    _chat_model,
    _responses_model,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings


def test_chat_reasoning_effort_omits_disabled_or_none() -> None:
    """Disabled thinking values never emit a provider reasoning effort."""
    assert (
        _chat_model._reasoning_effort(ModelRequestParameters(thinking=False))
        is _chat_model.omit
    )
    assert (
        _chat_model._reasoning_effort(ModelRequestParameters(thinking="none"))
        is _chat_model.omit
    )


def test_responses_reasoning_omits_disabled_or_none_effort() -> None:
    """Responses payload omits reasoning effort for disabled thinking values."""
    assert (
        _responses_model._reasoning(
            cast(ModelSettings, {"openai_reasoning_effort": "none"}),
            ModelRequestParameters(),
        )
        is _responses_model.omit
    )
    assert (
        _responses_model._reasoning({}, ModelRequestParameters(thinking=False))
        is _responses_model.omit
    )
    assert (
        _responses_model._reasoning({}, ModelRequestParameters(thinking="none"))
        is _responses_model.omit
    )


def test_responses_reasoning_keeps_summary_when_effort_omitted() -> None:
    """Reasoning summary settings remain independent from effort omission."""
    assert _responses_model._reasoning(
        cast(
            ModelSettings,
            {"openai_reasoning_effort": "none", "openai_reasoning_summary": "auto"},
        ),
        ModelRequestParameters(thinking=False),
    ) == {"summary": "auto"}
