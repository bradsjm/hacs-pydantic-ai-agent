"""AI task provider integration tests."""

from typing import Any

import pytest
import voluptuous as vol

from homeassistant.components import ai_task
from homeassistant.core import HomeAssistant

from custom_components.pydantic_ai_agent import entity as agent_entity_module

from .capture import tool_part_names
from .config import (
    AI_TASK_SENTINEL,
    AI_TASK_STRUCTURED_SENTINEL,
    MCP_SENTINEL,
    STRUCTURED_OUTPUT_MODES,
    ProviderIntegrationConfig,
    StructuredOutputSupport,
)
from .entries import ai_task_entity_id, drain_stream_cleanup, mcp_ai_task_entity_id

pytestmark = [
    pytest.mark.provider_integration,
    pytest.mark.usefixtures("socket_enabled"),
]


async def test_ai_task_plain_generation(
    hass: HomeAssistant, provider_config: ProviderIntegrationConfig
) -> None:
    """Test a live provider can generate plain AI task data."""
    entity_id = await ai_task_entity_id(hass, provider_config)

    result = await ai_task.async_generate_data(
        hass,
        task_name="Integration plain task",
        entity_id=entity_id,
        instructions=f"Return exactly {AI_TASK_SENTINEL}. No punctuation.",
    )

    await drain_stream_cleanup(hass)
    assert AI_TASK_SENTINEL in str(result.data)


@pytest.mark.parametrize("output_mode", STRUCTURED_OUTPUT_MODES)
async def test_ai_task_structured_generation(
    hass: HomeAssistant,
    provider_config: ProviderIntegrationConfig,
    structured_output_support: StructuredOutputSupport,
    output_mode: str,
) -> None:
    """Test a live provider can generate schema-validated AI task data."""
    structured_output_support.skip_if_unsupported(output_mode)
    entity_id = await ai_task_entity_id(hass, provider_config, output_mode)

    result = await ai_task.async_generate_data(
        hass,
        task_name="Integration structured task",
        entity_id=entity_id,
        instructions=(
            f"Generate data where result is exactly {AI_TASK_STRUCTURED_SENTINEL}."
        ),
        structure=vol.Schema({vol.Required("result"): str}),
    )

    await drain_stream_cleanup(hass)
    assert isinstance(result.data, dict)
    assert result.data["result"] == AI_TASK_STRUCTURED_SENTINEL


@pytest.mark.parametrize("output_mode", STRUCTURED_OUTPUT_MODES)
async def test_ai_task_uses_hosted_mcp_echo_tool(
    hass: HomeAssistant,
    provider_config: ProviderIntegrationConfig,
    mcp_echo_url: str,
    structured_output_support: StructuredOutputSupport,
    monkeypatch: pytest.MonkeyPatch,
    output_mode: str,
) -> None:
    """Test a live AI task can call a hosted MCP echo tool through Agent."""
    structured_output_support.skip_if_unsupported(output_mode)
    captured_messages: list[object] = []
    original_append = agent_entity_module._append_agent_messages

    async def capture_agent_messages(
        chat_log: Any,
        agent_id: str,
        messages: list[Any],
        output_tool_names: set[str] | None = None,
    ) -> None:
        captured_messages.extend(messages)
        await original_append(chat_log, agent_id, messages, output_tool_names)

    monkeypatch.setattr(
        agent_entity_module,
        "_append_agent_messages",
        capture_agent_messages,
    )
    entity_id = await mcp_ai_task_entity_id(
        hass, provider_config, mcp_echo_url, output_mode
    )

    result = await ai_task.async_generate_data(
        hass,
        task_name="Integration MCP structured task",
        entity_id=entity_id,
        instructions=(
            "Use the available MCP echo tool with message "
            f"{MCP_SENTINEL}. Generate data where result is exactly the tool result. "
            "Do not generate the result without calling the MCP tool."
        ),
        structure=vol.Schema({vol.Required("result"): str}),
    )

    await drain_stream_cleanup(hass)
    assert any(name.endswith("echo") for name in tool_part_names(captured_messages))
    assert isinstance(result.data, dict)
    assert MCP_SENTINEL in result.data["result"]
