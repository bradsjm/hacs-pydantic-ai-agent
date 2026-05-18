"""Test structured output helper behavior."""

import pytest
from pydantic_ai import ToolDefinition

from custom_components.pydantic_ai_agent.const import (
    DEFAULT_OUTPUT_MODE,
    OUTPUT_MODE_NATIVE,
    OUTPUT_MODE_PROMPTED,
    OUTPUT_MODE_TOOL,
)
from custom_components.pydantic_ai_agent.structured_output import (
    output_tool_names,
    structured_model_request_parameters,
    structured_output_mode,
    structured_output_name,
)


def test_structured_output_mode_defaults_unknown_values() -> None:
    """Test only supported output modes are accepted."""
    assert structured_output_mode(OUTPUT_MODE_TOOL) == OUTPUT_MODE_TOOL
    assert structured_output_mode(OUTPUT_MODE_NATIVE) == OUTPUT_MODE_NATIVE
    assert structured_output_mode(OUTPUT_MODE_PROMPTED) == OUTPUT_MODE_PROMPTED
    assert structured_output_mode("invalid") == DEFAULT_OUTPUT_MODE
    assert structured_output_mode(None) == DEFAULT_OUTPUT_MODE


def test_structured_output_name_is_prefixed_bounded_and_collision_safe() -> None:
    """Test output names avoid HA tool names and provider length limits."""
    output_name = structured_output_name("Kitchen Report", "fallback")
    assert output_name == "pydantic_ai_agent_output_kitchen_report"

    long_name = structured_output_name("x" * 100, "fallback")
    assert len(long_name) == 64

    reserved = {output_name}
    collision_name = structured_output_name(
        "Kitchen Report", "fallback", reserved_names=reserved
    )
    assert collision_name != output_name
    assert collision_name.startswith("pydantic_ai_agent_output_")
    assert collision_name not in reserved


def test_structured_output_name_handles_digest_collision() -> None:
    """Test output names keep searching when the digest fallback is reserved too."""
    first = structured_output_name("Kitchen Report", "fallback")
    digest = structured_output_name(
        "Kitchen Report", "fallback", reserved_names={first}
    )
    assert structured_output_name(
        "Kitchen Report", "fallback", reserved_names={first, digest}
    ).endswith("_2")


def test_structured_model_request_parameters_for_tool_mode() -> None:
    """Test tool output mode creates an output tool and disallows text."""
    function_tool = ToolDefinition(name="HassTurnOn", parameters_json_schema={})

    parameters = structured_model_request_parameters(
        function_tools=[function_tool],
        output_mode=OUTPUT_MODE_TOOL,
        output_name="result",
        json_schema={"type": "object"},
        strict=True,
    )

    assert parameters.function_tools == [function_tool]
    assert parameters.output_mode == OUTPUT_MODE_TOOL
    assert parameters.allow_text_output is False
    assert parameters.output_tools is not None
    assert parameters.output_tools[0].name == "result"
    assert parameters.output_tools[0].strict is True


@pytest.mark.parametrize("output_mode", [OUTPUT_MODE_NATIVE, OUTPUT_MODE_PROMPTED])
def test_structured_model_request_parameters_for_object_modes(output_mode: str) -> None:
    """Test native and prompted modes use output_object."""
    parameters = structured_model_request_parameters(
        function_tools=[],
        output_mode=output_mode,
        output_name="result",
        json_schema={"type": "object"},
    )

    assert parameters.output_mode == output_mode
    assert parameters.output_object is not None
    assert parameters.output_object.name == "result"


def test_structured_model_request_parameters_reject_unknown_mode() -> None:
    """Test unsupported output modes fail loudly."""
    with pytest.raises(ValueError, match="Unsupported structured output mode"):
        structured_model_request_parameters(
            function_tools=[],
            output_mode="invalid",
            output_name="result",
            json_schema={},
        )


def test_output_tool_names_only_returns_tool_mode_name() -> None:
    """Test only tool mode reserves an output tool name."""
    assert output_tool_names(OUTPUT_MODE_TOOL, "result") == {"result"}
    assert output_tool_names(OUTPUT_MODE_NATIVE, "result") == set()
