"""Tests for structured output helper behavior."""

from collections.abc import Callable

from custom_components.pydantic_ai_agent.const import (
    OUTPUT_MODE_NATIVE,
    OUTPUT_MODE_PROMPTED,
    OUTPUT_MODE_TOOL,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
)
from custom_components.pydantic_ai_agent.models.model_profiles import (
    ResolvedModelProfile,
)
from custom_components.pydantic_ai_agent.models.structured_output import (
    output_tool_names,
    resolved_structured_output_mode,
    structured_agent_output_type,
    structured_model_request_parameters,
    structured_output_name,
)
from pydantic_ai import ToolDefinition
from pydantic_ai.output import NativeOutput, PromptedOutput, ToolOutput
import pytest


def test_resolved_structured_output_mode_prefers_tools_for_openai_compatible(
    make_profile: Callable[..., ResolvedModelProfile],
) -> None:
    """OpenAI-compatible profiles use tool output when tools are available."""
    profile = make_profile(
        provider_mode=PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        supports_tools=True,
        structured_output_support="json_schema",
    )

    assert resolved_structured_output_mode(profile) == OUTPUT_MODE_TOOL


@pytest.mark.parametrize(
    ("structured_support", "expected"),
    [
        ("json_schema", OUTPUT_MODE_NATIVE),
        ("json_object", OUTPUT_MODE_NATIVE),
        ("none", OUTPUT_MODE_PROMPTED),
    ],
)
def test_resolved_structured_output_mode_uses_native_then_prompted(
    make_profile: Callable[..., ResolvedModelProfile],
    structured_support: str,
    expected: str,
) -> None:
    """OpenAI-compatible profiles without tools select native or prompted mode."""
    profile = make_profile(
        provider_mode=PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        supports_tools=False,
        structured_output_support=structured_support,
    )

    assert resolved_structured_output_mode(profile) == expected


def test_structured_output_name_is_prefixed_bounded_and_collision_safe() -> None:
    """Output names are slugged, length-bounded, and avoid reserved names."""
    output_name = structured_output_name("Friendly Name", "fallback")
    assert output_name == "pydantic_ai_agent_output_friendly_name"

    long_name = structured_output_name("x" * 100, "fallback")
    assert len(long_name) == 64

    colliding = structured_output_name("Friendly Name", "fallback", {output_name})
    assert colliding != output_name
    assert colliding.startswith("pydantic_ai_agent_output_")
    assert len(colliding) <= 64


def test_structured_model_request_parameters_for_tool_mode() -> None:
    """Tool mode exposes the output as an output tool and disallows text."""
    function_tool = ToolDefinition(name="lookup", parameters_json_schema={"type": "object"})
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

    params = structured_model_request_parameters(
        function_tools=[function_tool],
        output_mode=OUTPUT_MODE_TOOL,
        output_name="answer_tool",
        json_schema=schema,
        strict=True,
    )

    assert params.function_tools == [function_tool]
    assert params.output_mode == OUTPUT_MODE_TOOL
    assert [tool.name for tool in params.output_tools] == ["answer_tool"]
    assert params.output_tools[0].parameters_json_schema == schema
    assert params.output_tools[0].strict is True
    assert params.allow_text_output is False
    assert params.output_object is None


def test_structured_model_request_parameters_for_native_mode() -> None:
    """Native mode puts the schema on output_object instead of output tools."""
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

    params = structured_model_request_parameters(
        function_tools=[],
        output_mode=OUTPUT_MODE_NATIVE,
        output_name="answer_object",
        json_schema=schema,
    )

    assert params.output_mode == OUTPUT_MODE_NATIVE
    assert params.output_tools == []
    assert params.output_object is not None
    assert params.output_object.name == "answer_object"
    assert params.output_object.json_schema == schema
    assert params.allow_text_output is True


def test_output_tool_names_only_returns_tool_mode_name() -> None:
    """Only tool structured output reserves an output tool name."""
    assert output_tool_names(OUTPUT_MODE_TOOL, "result") == {"result"}
    assert output_tool_names(OUTPUT_MODE_NATIVE, "result") == set()


@pytest.mark.parametrize(
    ("mode", "expected_type"),
    [
        (OUTPUT_MODE_TOOL, ToolOutput),
        (OUTPUT_MODE_NATIVE, NativeOutput),
        (OUTPUT_MODE_PROMPTED, PromptedOutput),
    ],
)
def test_structured_agent_output_type_returns_mode_specific_wrapper(mode: str, expected_type: type[object]) -> None:
    """Agent output types use the Pydantic AI wrapper for the selected mode."""
    output_type = structured_agent_output_type(
        output_mode=mode,
        output_name="result",
        json_schema={"type": "object"},
        output_tool_retries=2,
    )

    assert isinstance(output_type, expected_type)
