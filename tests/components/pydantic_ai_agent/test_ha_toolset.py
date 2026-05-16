"""Test Home Assistant LLM API tool adapters."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

from pydantic_ai import RunContext
import voluptuous as vol

from homeassistant.helpers import llm

from custom_components.pydantic_ai_agent.ha_toolset import (
    tool_definitions_from_llm_api,
    tools_from_llm_api,
)


class _TestTool(llm.Tool):
    """LLM test tool."""

    name = "turn_on"
    description = "Turn on a device."
    parameters = vol.Schema({vol.Required("entity_id"): str})


def test_tool_definitions_from_llm_api_converts_ha_tools() -> None:
    """Test HA LLM tools convert to model-visible Pydantic AI definitions."""
    api_instance = SimpleNamespace(custom_serializer=None, tools=[_TestTool()])

    definitions = tool_definitions_from_llm_api(cast(llm.APIInstance, api_instance))

    assert len(definitions) == 1
    assert definitions[0].name == "turn_on"
    assert definitions[0].description == "Turn on a device."
    assert "entity_id" in definitions[0].parameters_json_schema["properties"]


async def test_tools_from_llm_api_creates_executable_public_schema_tools() -> None:
    """Test executable HA LLM tools use public Pydantic AI schema APIs."""
    async_call_tool = AsyncMock(return_value={"done": True})
    api_instance = SimpleNamespace(
        async_call_tool=async_call_tool,
        custom_serializer=None,
        tools=[_TestTool()],
    )
    ctx = SimpleNamespace(tool_call_id="call-123")

    tools = tools_from_llm_api(cast(llm.APIInstance, api_instance))
    result = await tools[0].function(
        cast(RunContext[Any], ctx), entity_id="light.kitchen"
    )

    assert result == {"done": True}
    assert tools[0].name == "turn_on"
    assert tools[0].description == "Turn on a device."
    assert "entity_id" in tools[0].function_schema.json_schema["properties"]
    assert async_call_tool.await_args is not None
    tool_input = async_call_tool.await_args.args[0]
    assert isinstance(tool_input, llm.ToolInput)
    assert tool_input.id == "call-123"
    assert tool_input.tool_name == "turn_on"
    assert tool_input.tool_args == {"entity_id": "light.kitchen"}


def test_tools_from_llm_api_returns_empty_without_api() -> None:
    """Test missing HA LLM API produces no Pydantic AI tools."""
    assert tool_definitions_from_llm_api(None) == []
    assert tools_from_llm_api(None) == []
