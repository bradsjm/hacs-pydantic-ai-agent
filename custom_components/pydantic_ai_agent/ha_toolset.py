"""Home Assistant LLM API tool adapters for Pydantic AI."""

from typing import Any

from pydantic_ai import RunContext, Tool, ToolDefinition
from pydantic_ai._function_schema import FunctionSchema
from pydantic_core import SchemaValidator
from voluptuous_openapi import convert

from homeassistant.helpers import llm

_ANY_OBJECT_VALIDATOR = SchemaValidator(
    {"type": "dict", "keys_schema": {"type": "str"}, "values_schema": {"type": "any"}}
)


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
            # HA LLM tools expose voluptuous schemas; Pydantic AI needs JSON
            # Schema for model-visible tool definitions.
            parameters_json_schema=convert(
                tool.parameters,
                custom_serializer=api_instance.custom_serializer,
            ),
        )
        for tool in api_instance.tools
    ]


def tools_from_llm_api(api_instance: llm.APIInstance | None) -> list[Tool[Any]]:
    """Convert Home Assistant LLM tools into executable Pydantic AI tools."""
    if api_instance is None:
        return []

    return [_tool_from_ha_tool(api_instance, tool) for tool in api_instance.tools]


def _tool_from_ha_tool(api_instance: llm.APIInstance, tool: llm.Tool) -> Tool[Any]:
    """Return one executable Pydantic AI tool backed by an HA LLM API tool."""
    parameters_json_schema = convert(
        tool.parameters,
        custom_serializer=api_instance.custom_serializer,
    )

    async def execute(ctx: RunContext[Any], **tool_args: Any) -> Any:
        """Execute the Home Assistant LLM tool with model-provided arguments."""
        return await api_instance.async_call_tool(
            llm.ToolInput(
                id=ctx.tool_call_id or "",
                tool_name=tool.name,
                tool_args=dict(tool_args),
            )
        )

    function_schema = FunctionSchema(
        function=execute,
        name=tool.name,
        description=tool.description,
        validator=_ANY_OBJECT_VALIDATOR,
        json_schema=parameters_json_schema,
        takes_ctx=True,
        is_async=True,
    )
    return Tool(execute, function_schema=function_schema)
