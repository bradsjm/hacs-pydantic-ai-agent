"""Conversation provider integration tests."""

from collections.abc import Mapping
from typing import Any

import pytest
from custom_components.pydantic_ai_agent import _entity_runner as agent_runner_module
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import llm

from .config import (
    CONVERSATION_SENTINEL,
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


async def test_conversation_uses_workspace_skill(
    hass: HomeAssistant,
    provider_config: ProviderIntegrationConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a live provider can list and load a selected workspace Skill."""
    captured_tool_names: list[str] = []
    listed_skill_ids: list[str] = []
    original_agent_events_to_chat_deltas = (
        agent_runner_module._agent_events_to_chat_deltas
    )

    async def capture_agent_events_to_chat_deltas(
        events: Any,
        output_tool_names: set[str],
        state: Any,
        trace_recorder: Any = None,
    ) -> Any:
        async for delta in original_agent_events_to_chat_deltas(
            events, output_tool_names, state, trace_recorder
        ):
            if tool_calls := delta.get("tool_calls"):
                captured_tool_names.extend(
                    tool_input.tool_name for tool_input in tool_calls
                )
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
        agent_runner_module,
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
    if listed_skill_ids:
        assert listed_skill_ids == [WORKSPACE_SKILL_ID], listed_skill_ids
    assert "load_skill" in captured_tool_names, (
        f"captured tools={captured_tool_names!r}; speech={speech!r}"
    )
    assert SKILL_SENTINEL in speech, speech
