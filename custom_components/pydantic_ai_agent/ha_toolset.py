"""Home Assistant LLM API tool adapters for Pydantic AI."""

from typing import Any

from pydantic_ai import RunContext, Tool, ToolDefinition
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
    return tools_from_llm_api_with_diagnostics(api_instance, None)


def tools_from_llm_api_with_diagnostics(
    api_instance: llm.APIInstance | None, run_recorder: Any | None
) -> list[Tool[Any]]:
    """Convert Home Assistant LLM tools and record execution diagnostics."""
    if api_instance is None:
        return []

    return [
        _tool_from_ha_tool(api_instance, tool, run_recorder)
        for tool in api_instance.tools
    ]


def _tool_from_ha_tool(
    api_instance: llm.APIInstance, tool: llm.Tool, run_recorder: Any | None
) -> Tool[Any]:
    """Return one executable Pydantic AI tool backed by an HA LLM API tool."""
    parameters_json_schema = convert(
        tool.parameters,
        custom_serializer=api_instance.custom_serializer,
    )

    async def execute(ctx: RunContext[Any], **tool_args: Any) -> Any:
        """Execute the Home Assistant LLM tool with model-provided arguments."""
        tool_input = llm.ToolInput(
            id=ctx.tool_call_id or "",
            tool_name=tool.name,
            tool_args=dict(tool_args),
        )
        if run_recorder is not None:
            run_recorder.record(
                phase="tool_call",
                source="ha_llm_api",
                event="call_started",
                data={"tool_name": tool.name, "tool_input": tool_input},
            )
        try:
            result = await api_instance.async_call_tool(tool_input)
        except Exception as err:
            if run_recorder is not None:
                run_recorder.record(
                    phase="tool_call",
                    source="ha_llm_api",
                    event="call_failed",
                    data={"tool_name": tool.name, "error": err},
                )
            raise
        if run_recorder is not None:
            run_recorder.record(
                phase="tool_call",
                source="ha_llm_api",
                event="call_finished",
                data={"tool_name": tool.name, "result": result},
            )
        return result

    return Tool.from_schema(
        execute,
        name=tool.name,
        description=tool.description,
        json_schema=parameters_json_schema,
        takes_ctx=True,
    )
