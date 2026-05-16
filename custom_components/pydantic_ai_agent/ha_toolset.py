"""Home Assistant LLM API tool adapters for Pydantic AI."""

from pydantic_ai import ToolDefinition
from voluptuous_openapi import convert

from homeassistant.helpers import llm


def tool_definitions_from_llm_api(
    api_instance: llm.APIInstance | None,
) -> list[ToolDefinition]:
    """Convert Home Assistant LLM tools into model-visible tool definitions."""
    if api_instance is None:
        return []

    return [
        ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters_json_schema=convert(
                tool.parameters,
                custom_serializer=api_instance.custom_serializer,
            ),
        )
        for tool in api_instance.tools
    ]
