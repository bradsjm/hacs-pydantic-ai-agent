"""Test Home Assistant LLM API tool adapters."""

import base64
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

from custom_components.pydantic_ai_agent.agent.ha_toolset import (
    tool_definitions_from_llm_api,
    tools_from_llm_api,
)
from custom_components.pydantic_ai_agent.agent.tool_errors import HAToolRetryExhausted
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import llm
from pydantic_ai import BinaryContent, ModelRetry, RunContext
import pytest
import voluptuous as vol


class _TestTool(llm.Tool):
    """LLM test tool."""

    name = "turn_on"
    description = "Turn on a device."
    parameters = vol.Schema({vol.Required("entity_id"): str})

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict[str, Any]:
        """Return a minimal successful tool response."""
        return {"done": True}


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

    assert tools[0].name == "turn_on"
    assert tools[0].description == "Turn on a device."
    assert "entity_id" in tools[0].function_schema.json_schema["properties"]
    assert async_call_tool.await_args is not None
    tool_input = async_call_tool.await_args.args[0]
    assert isinstance(tool_input, llm.ToolInput)
    assert tool_input.id == "call-123"
    assert tool_input.tool_name == "turn_on"
    assert tool_input.tool_args == {"entity_id": "light.kitchen"}
    assert result == {"done": True}


async def test_tools_from_llm_api_normalizes_multimodal_tool_result() -> None:
    """Test recognized multimodal tool results become model-facing content."""
    async_call_tool = AsyncMock(
        return_value={
            "_type": "ha_multimodal_tool_result",
            "text": "Snapshot",
            "attachments": [
                {
                    "kind": "inline_image",
                    "mime_type": "image/jpeg",
                    "base64": base64.b64encode(b"jpeg-bytes").decode(),
                }
            ],
        }
    )
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

    assert result == [
        "Snapshot",
        BinaryContent(data=b"jpeg-bytes", media_type="image/jpeg"),
    ]


async def test_tools_from_llm_api_degrades_invalid_multimodal_tool_result_to_text() -> (
    None
):
    """Test malformed multimodal tool results fall back to safe text."""
    async_call_tool = AsyncMock(
        return_value={
            "_type": "ha_multimodal_tool_result",
            "text": "Snapshot",
            "attachments": [
                {
                    "kind": "inline_image",
                    "mime_type": "image/jpeg",
                    "base64": "not-base64",
                }
            ],
        }
    )
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

    assert result == "Snapshot"


async def test_tools_from_llm_api_wraps_retryable_tool_exception_as_model_retry() -> (
    None
):
    """Test retryable HA tool exceptions become model-visible retry prompts."""
    async_call_tool = AsyncMock(side_effect=ServiceValidationError("invalid target"))
    api_instance = SimpleNamespace(
        async_call_tool=async_call_tool,
        custom_serializer=None,
        tools=[_TestTool()],
    )
    ctx = SimpleNamespace(tool_call_id="call-123", retry=0, max_retries=1)

    tools = tools_from_llm_api(cast(llm.APIInstance, api_instance))
    with pytest.raises(ModelRetry):
        await tools[0].function(cast(RunContext[Any], ctx), entity_id="light.kitchen")


async def test_tools_from_llm_api_raises_owned_error_when_retries_exhausted() -> None:
    """Test final retryable HA tool exceptions use an integration-owned error."""
    async_call_tool = AsyncMock(side_effect=ServiceValidationError("invalid target"))
    api_instance = SimpleNamespace(
        async_call_tool=async_call_tool,
        custom_serializer=None,
        tools=[_TestTool()],
    )
    ctx = SimpleNamespace(tool_call_id="call-123", retry=1, max_retries=1)

    tools = tools_from_llm_api(cast(llm.APIInstance, api_instance))
    with pytest.raises(HAToolRetryExhausted) as exc_info:
        await tools[0].function(cast(RunContext[Any], ctx), entity_id="light.kitchen")

    assert exc_info.value.tool_name == "turn_on"
    assert exc_info.value.attempts == 2
    assert "turn_on" in str(exc_info.value)


async def test_tools_from_llm_api_reraises_non_retryable_tool_exception() -> None:
    """Test unexpected HA tool exceptions still abort immediately."""
    async_call_tool = AsyncMock(side_effect=RuntimeError("device lookup failed"))
    api_instance = SimpleNamespace(
        async_call_tool=async_call_tool,
        custom_serializer=None,
        tools=[_TestTool()],
    )
    ctx = SimpleNamespace(tool_call_id="call-123")

    tools = tools_from_llm_api(cast(llm.APIInstance, api_instance))
    with pytest.raises(RuntimeError):
        await tools[0].function(cast(RunContext[Any], ctx), entity_id="light.kitchen")


def test_tools_from_llm_api_returns_empty_without_api() -> None:
    """Test missing HA LLM API produces no Pydantic AI tools."""
    assert tool_definitions_from_llm_api(None) == []
    assert tools_from_llm_api(None) == []
