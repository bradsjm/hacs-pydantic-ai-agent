"""Test system health for Pydantic AI Agent."""

from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.pydantic_ai_agent import PydanticAIAgentRuntimeData
from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_ENABLE_SKILL_SCRIPT_EXECUTION,
    CONF_MCP_HEADERS,
    CONF_MCP_URL,
    CONF_MODEL,
    CONF_PROVIDER_MODE,
    DOMAIN,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_MCP_SERVER,
    SUBENTRY_TYPE_MODEL,
)
from custom_components.pydantic_ai_agent.system_health import system_health_info


async def test_system_health_reports_safe_aggregate_counts(
    hass: HomeAssistant,
) -> None:
    """Test system health exposes aggregate counts without secrets or URLs."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Local Provider",
        data={
            CONF_NAME: "Local Provider",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "sk-secret",
        },
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "data": {CONF_NAME: "Fast GPT", CONF_MODEL: "gpt-test"},
                "subentry_type": SUBENTRY_TYPE_MODEL,
                "title": "Fast GPT",
                "unique_id": None,
            },
            {
                "data": {
                    CONF_AGENT_NAME: "Kitchen Agent",
                    CONF_ENABLE_SKILL_SCRIPT_EXECUTION: True,
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
            },
            {
                "data": {CONF_NAME: "Report Task"},
                "subentry_type": SUBENTRY_TYPE_AI_TASK,
                "title": "Report Task",
                "unique_id": None,
            },
            {
                "data": {
                    CONF_NAME: "Filesystem MCP",
                    CONF_MCP_URL: "https://mcp.example.com/mcp",
                    CONF_MCP_HEADERS: {"Authorization": "Bearer mcp-secret"},
                },
                "subentry_type": SUBENTRY_TYPE_MCP_SERVER,
                "title": "Filesystem MCP",
                "unique_id": None,
            },
        ),
        unique_id=None,
    )
    entry.add_to_hass(hass)
    mcp_subentry_id = next(
        subentry.subentry_id
        for subentry in entry.subentries.values()
        if subentry.subentry_type == SUBENTRY_TYPE_MCP_SERVER
    )
    entry.runtime_data = PydanticAIAgentRuntimeData(
        provider_mode=PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        name="Local Provider",
        api_key="sk-secret",
        base_url="https://provider.example.com/v1",
        logfire_enabled=False,
        logfire_include_content=False,
        mcp_tool_cache={mcp_subentry_id: [{"name": "secret_tool"}]},
    )
    other_entry = MockConfigEntry(
        domain=DOMAIN,
        title="Responses Provider",
        data={
            CONF_NAME: "Responses Provider",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
            CONF_API_KEY: "sk-other",
        },
        source=config_entries.SOURCE_USER,
        unique_id=None,
    )
    other_entry.add_to_hass(hass)

    info = await system_health_info(hass)

    assert info == {
        "configured_entry_count": 2,
        "loaded_entry_count": 1,
        "provider_modes": {
            PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS: 1,
            PROVIDER_OPENAI_COMPATIBLE_RESPONSES: 1,
        },
        "model_profile_count": 1,
        "conversation_count": 1,
        "ai_task_count": 1,
        "mcp_server_count": 1,
        "cached_mcp_server_count": 1,
        "cached_mcp_tool_count": 1,
        "logfire_enabled_count": 0,
        "skill_script_execution_count": 1,
    }
    assert "sk-secret" not in str(info)
    assert "provider.example.com" not in str(info)
    assert "mcp.example.com" not in str(info)
    assert "secret_tool" not in str(info)
