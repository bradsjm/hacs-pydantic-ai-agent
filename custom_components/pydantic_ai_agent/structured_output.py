"""Structured output helpers for Pydantic AI model and Agent requests."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, cast

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
    OUTPUT_MODE_NATIVE,
    OUTPUT_MODE_PROMPTED,
    OUTPUT_MODE_TOOL,
)
from .openai_compatible_profile import is_openai_compatible_provider_mode
from .provider import model_profile_for_provider_mode

if TYPE_CHECKING:
    from .model_profiles import ResolvedModelProfile

_OUTPUT_NAME_PREFIX = "pydantic_ai_agent_output_"
_MAX_OUTPUT_NAME_LENGTH = 64


def resolved_structured_output_mode(profile: ResolvedModelProfile) -> str:
    """Return the runtime structured output mode for one resolved profile."""
    if is_openai_compatible_provider_mode(profile.provider_mode):
        if profile.supports_tools is True:
            return OUTPUT_MODE_TOOL
        if profile.structured_output_support in {"json_object", "json_schema"}:
            return OUTPUT_MODE_NATIVE
        return OUTPUT_MODE_PROMPTED

    runtime_profile = model_profile_for_provider_mode(
        profile.provider_mode, profile.model_name
    )
    if runtime_profile is not None and runtime_profile.supports_tools:
        return OUTPUT_MODE_TOOL
    if runtime_profile is not None and (
        runtime_profile.supports_json_schema_output
        or runtime_profile.supports_json_object_output
    ):
        return OUTPUT_MODE_NATIVE
    return OUTPUT_MODE_PROMPTED


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
    output_tool_retries: int | None = None,
) -> object:
    """Return a Pydantic AI Agent output type for a structured HA schema."""
    output_type = StructuredDict(json_schema, name=output_name)
    if output_mode == OUTPUT_MODE_TOOL:
        return ToolOutput(
            output_type,
            name=output_name,
            max_retries=output_tool_retries,
        )
    if output_mode == OUTPUT_MODE_NATIVE:
        return NativeOutput(output_type, name=output_name)
    if output_mode == OUTPUT_MODE_PROMPTED:
        return PromptedOutput(output_type, name=output_name)
    raise ValueError(f"Unsupported structured output mode: {output_mode}")
