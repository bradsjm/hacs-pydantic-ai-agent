"""Structured output helpers for Pydantic AI model and Agent requests."""

import hashlib
from collections.abc import Callable, Iterable
from typing import Any, cast

from homeassistant.helpers import llm
from homeassistant.util import slugify
from pydantic_ai import ToolDefinition
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.output import (
    NativeOutput,
    OutputMode,
    OutputObjectDefinition,
    PromptedOutput,
    StructuredDict,
    ToolOutput,
)
from voluptuous_openapi import convert

from .const import (
    DEFAULT_OUTPUT_MODE,
    OUTPUT_MODE_NATIVE,
    OUTPUT_MODE_PROMPTED,
    OUTPUT_MODE_TOOL,
    STRUCTURED_OUTPUT_MODES,
)

_OUTPUT_NAME_PREFIX = "pydantic_ai_agent_output_"
_MAX_OUTPUT_NAME_LENGTH = 64


def structured_output_mode(value: object) -> str:
    """Return a supported structured output mode, defaulting to tool output."""
    if isinstance(value, str) and value in STRUCTURED_OUTPUT_MODES:
        return value
    return DEFAULT_OUTPUT_MODE


def structured_output_name(
    name: str | None, fallback: str, reserved_names: Iterable[str] = ()
) -> str:
    """Return a valid, stable Pydantic AI output name."""
    reserved = set(reserved_names)
    base = slugify(name or fallback) or fallback
    candidate = _bounded_output_name(f"{_OUTPUT_NAME_PREFIX}{base}")
    if candidate not in reserved:
        return candidate

    digest = hashlib.sha1(base.encode()).hexdigest()[:8]
    candidate = _bounded_output_name(f"{_OUTPUT_NAME_PREFIX}{digest}")
    if candidate not in reserved:
        return candidate

    index = 2
    while True:
        candidate = _bounded_output_name(f"{_OUTPUT_NAME_PREFIX}{digest}_{index}")
        if candidate not in reserved:
            return candidate
        index += 1


def _bounded_output_name(value: str) -> str:
    """Return an output name within common provider tool-name limits."""
    return value[:_MAX_OUTPUT_NAME_LENGTH]


def structured_output_json_schema(
    structure: object,
    custom_serializer: Callable[..., Any] | None,
) -> dict[str, Any]:
    """Return a JSON schema for a Home Assistant AI task structure."""
    return convert(structure, custom_serializer=custom_serializer)


def structured_model_request_parameters(
    *,
    function_tools: Iterable[ToolDefinition],
    output_mode: str,
    output_name: str,
    json_schema: dict[str, Any],
    strict: bool | None = None,
) -> ModelRequestParameters:
    """Return request parameters for a selected structured output mode."""
    tools = list(function_tools)
    if output_mode == OUTPUT_MODE_TOOL:
        return ModelRequestParameters(
            function_tools=tools,
            output_mode=cast(OutputMode, OUTPUT_MODE_TOOL),
            output_tools=[
                ToolDefinition(
                    name=output_name,
                    parameters_json_schema=json_schema,
                    strict=strict,
                    kind="output",
                )
            ],
            allow_text_output=False,
        )
    if output_mode in {OUTPUT_MODE_NATIVE, OUTPUT_MODE_PROMPTED}:
        return ModelRequestParameters(
            function_tools=tools,
            output_mode=cast(OutputMode, output_mode),
            output_object=OutputObjectDefinition(
                json_schema=json_schema,
                name=output_name,
                strict=strict,
            ),
        )
    raise ValueError(f"Unsupported structured output mode: {output_mode}")


def output_tool_names(output_mode: str, output_name: str) -> set[str]:
    """Return Pydantic AI output tool names for the selected mode."""
    if output_mode == OUTPUT_MODE_TOOL:
        return {output_name}
    return set()


def default_structure_serializer(
    api_instance: llm.APIInstance | None,
) -> Callable[..., Any] | None:
    """Return the serializer Home Assistant uses for AI task schemas."""
    return api_instance.custom_serializer if api_instance else llm.selector_serializer


def structured_agent_output_type(
    *,
    output_mode: str,
    output_name: str,
    json_schema: dict[str, Any],
) -> object:
    """Return a Pydantic AI Agent output type for a structured HA schema."""
    output_type = StructuredDict(json_schema, name=output_name)
    if output_mode == OUTPUT_MODE_TOOL:
        return ToolOutput(output_type, name=output_name)
    if output_mode == OUTPUT_MODE_NATIVE:
        return NativeOutput(output_type, name=output_name)
    if output_mode == OUTPUT_MODE_PROMPTED:
        return PromptedOutput(output_type, name=output_name)
    raise ValueError(f"Unsupported structured output mode: {output_mode}")
