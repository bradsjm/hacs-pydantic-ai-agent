"""Conversation provider integration tests."""

from collections.abc import Mapping
from typing import Any

import pytest

from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import llm

from custom_components.pydantic_ai_agent import entity as agent_entity_module

from .config import (
    CONVERSATION_SENTINEL,
    MCP_SECOND_SENTINEL,
    MCP_SENTINEL,
    SKILL_SENTINEL,
    TEST_LLM_API_ID,
    TOOL_SENTINEL,
    WORKSPACE_SKILL_ID,
    ProviderIntegrationConfig,
)
from .entries import (
    conversation_entity_id,
    drain_stream_cleanup,
    skill_conversation_entity_id,
)
from .ha_llm_echo import EchoAPI

pytestmark = [
    pytest.mark.provider_integration,
    pytest.mark.usefixtures("socket_enabled"),
]


async def test_conversation_plain_response(
    hass: HomeAssistant, provider_config: ProviderIntegrationConfig
) -> None:
    """Test a live provider can answer through the HA conversation API."""
    entity_id = await conversation_entity_id(hass, provider_config)

    result = await conversation.async_converse(
        hass,
        f"Reply with exactly {CONVERSATION_SENTINEL}. No punctuation.",
        None,
        Context(),
        agent_id=entity_id,
    )

    await drain_stream_cleanup(hass)
    assert CONVERSATION_SENTINEL in result.response.speech["plain"]["speech"]


async def test_conversation_uses_ha_llm_tool(
    hass: HomeAssistant, provider_config: ProviderIntegrationConfig
) -> None:
    """Test a live provider can call a Home Assistant LLM API tool."""
    tool_calls: list[str] = []
    unregister = llm.async_register_api(hass, EchoAPI(hass, tool_calls))
    entity_id = await conversation_entity_id(
        hass,
        provider_config,
        llm_hass_api=[TEST_LLM_API_ID],
    )

    try:
        result = await conversation.async_converse(
            hass,
            (
                "Call the pydantic_ai_integration_echo tool with token "
                f"{TOOL_SENTINEL}. Then reply with exactly the token returned "
                "by the tool. Do not answer without calling the tool."
            ),
            None,
            Context(),
            agent_id=entity_id,
        )
    finally:
        unregister()

    await drain_stream_cleanup(hass)
    assert tool_calls == [TOOL_SENTINEL]
    speech = result.response.speech["plain"]["speech"]
    assert TOOL_SENTINEL in speech


async def test_conversation_uses_hosted_mcp_echo_tool(
    hass: HomeAssistant,
    provider_config: ProviderIntegrationConfig,
    mcp_echo_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a live provider can call a hosted MCP echo tool through Agent."""
    captured_tool_names: list[str] = []
    tool_result_seen = False
    second_turn_tool_call_seen = False
    original_agent_events_to_chat_deltas = agent_entity_module._agent_events_to_chat_deltas

    async def capture_agent_events_to_chat_deltas(
        events: Any,
        output_tool_names: set[str],
        state: Any,
    ) -> Any:
        nonlocal second_turn_tool_call_seen, tool_result_seen
        async for delta in original_agent_events_to_chat_deltas(
            events, output_tool_names, state
        ):
            if tool_calls := delta.get("tool_calls"):
                for tool_input in tool_calls:
                    tool_name = tool_input.tool_name
                    captured_tool_names.append(tool_name)
                    if tool_result_seen and tool_name.endswith("echo"):
                        second_turn_tool_call_seen = True
            if delta.get("role") == "tool_result":
                tool_name = str(delta["tool_name"])
                captured_tool_names.append(tool_name)
                if tool_name.endswith("echo"):
                    tool_result_seen = True
            yield delta

    monkeypatch.setattr(
        agent_entity_module,
        "_agent_events_to_chat_deltas",
        capture_agent_events_to_chat_deltas,
    )
    entity_id = await conversation_entity_id(
        hass,
        provider_config,
        mcp_echo_url=mcp_echo_url,
    )

    result = await conversation.async_converse(
        hass,
        (
            "Use the available MCP echo tool twice. First call it with message "
            f"{MCP_SENTINEL}. After that tool result returns, call the same "
            f"tool a second time with message {MCP_SECOND_SENTINEL}. Reply "
            "with exactly both tool results. Do not answer without both tool calls."
        ),
        None,
        Context(),
        agent_id=entity_id,
    )

    await drain_stream_cleanup(hass)
    speech = result.response.speech["plain"]["speech"]
    assert sum(name.endswith("echo") for name in captured_tool_names) >= 4, (
        f"captured tools={captured_tool_names!r}; speech={speech!r}"
    )
    assert second_turn_tool_call_seen, captured_tool_names
    assert MCP_SENTINEL in speech
    assert MCP_SECOND_SENTINEL in speech


async def test_conversation_uses_workspace_skill(
    hass: HomeAssistant,
    provider_config: ProviderIntegrationConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a live provider can list and load a selected workspace Skill."""
    captured_tool_names: list[str] = []
    listed_skill_ids: list[str] = []
    original_agent_events_to_chat_deltas = agent_entity_module._agent_events_to_chat_deltas

    async def capture_agent_events_to_chat_deltas(
        events: Any,
        output_tool_names: set[str],
        state: Any,
    ) -> Any:
        async for delta in original_agent_events_to_chat_deltas(
            events, output_tool_names, state
        ):
            if tool_calls := delta.get("tool_calls"):
                captured_tool_names.extend(tool_input.tool_name for tool_input in tool_calls)
            if delta.get("role") == "tool_result":
                tool_name = str(delta["tool_name"])
                captured_tool_names.append(tool_name)
                if tool_name == "list_skills" and isinstance(
                    skill_results := delta.get("tool_result"), list
                ):
                    listed_skill_ids.extend(
                        str(skill["skill_id"])
                        for skill in skill_results
                        if isinstance(skill, Mapping) and "skill_id" in skill
                    )
            yield delta

    monkeypatch.setattr(
        agent_entity_module,
        "_agent_events_to_chat_deltas",
        capture_agent_events_to_chat_deltas,
    )
    entity_id = await skill_conversation_entity_id(hass, provider_config)

    result = await conversation.async_converse(
        hass,
        (
            "Use the selected workspace Skill. First list the available Skills, "
            "then load the relevant Skill, then reply exactly with the workspace "
            "Skill token from the loaded Skill. Do not answer without loading "
            "the Skill."
        ),
        None,
        Context(),
        agent_id=entity_id,
    )

    await drain_stream_cleanup(hass)
    speech = result.response.speech["plain"]["speech"]
    assert listed_skill_ids == [WORKSPACE_SKILL_ID], listed_skill_ids
    assert "list_skills" in captured_tool_names, captured_tool_names
    assert "load_skill" in captured_tool_names, (
        f"captured tools={captured_tool_names!r}; speech={speech!r}"
    )
    assert SKILL_SENTINEL in speech, speech
