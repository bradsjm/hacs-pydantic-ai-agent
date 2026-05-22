"""Test system health for Pydantic AI Agent."""

from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.pydantic_ai_agent import (
    MCPServerRuntimeData,
    ProviderRuntimeData,
    WorkspaceRuntimeData,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_DEFAULT_MODEL_PROFILE_ID,
    CONF_DISCOVERED,
    CONF_ENABLE_SKILL_SCRIPT_EXECUTION,
    CONF_ENABLED,
    CONF_MCP_HEADERS,
    CONF_MCP_URL,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROVIDER_MODE,
    DOMAIN,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_MCP_SERVER,
    SUBENTRY_TYPE_PROVIDER,
)
from custom_components.pydantic_ai_agent.system_health import system_health_info


async def test_system_health_reports_safe_workspace_aggregate_counts(
    hass: HomeAssistant,
) -> None:
    """Test system health exposes workspace aggregate counts without secrets."""
    entry = MockConfigEntry(
        version=2,
        minor_version=0,
        domain=DOMAIN,
        title="Workspace",
        data={CONF_NAME: "Workspace"},
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "subentry_id": "provider-1",
                "data": {
                    CONF_NAME: "Local Provider",
                    CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
                    CONF_API_KEY: "sk-secret",
                    CONF_DEFAULT_MODEL_PROFILE_ID: "profile-1",
                    CONF_MODEL_PROFILES: {
                        "profile-1": {
                            "id": "profile-1",
                            CONF_NAME: "Fast GPT",
                            CONF_MODEL: "gpt-test",
                            CONF_ENABLED: True,
                            CONF_DISCOVERED: True,
                        }
                    },
                },
                "subentry_type": SUBENTRY_TYPE_PROVIDER,
                "title": "Local Provider",
                "unique_id": None,
            },
            {
                "data": {
                    CONF_AGENT_NAME: "Kitchen Agent",
                    CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
                    CONF_ENABLE_SKILL_SCRIPT_EXECUTION: True,
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
            },
            {
                "data": {
                    CONF_NAME: "Report Task",
                    CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
                },
                "subentry_type": SUBENTRY_TYPE_AI_TASK,
                "title": "Report Task",
                "unique_id": None,
            },
            {
                "subentry_id": "mcp-1",
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
    entry.runtime_data = WorkspaceRuntimeData(
        workspace_name="Workspace",
        providers={
            "provider-1": ProviderRuntimeData(
                provider_subentry_id="provider-1",
                name="Local Provider",
                api_key="sk-secret",
                provider_mode=PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
                base_url="https://provider.example.com/v1",
            )
        },
        mcp_servers={
            "mcp-1": MCPServerRuntimeData(
                subentry_id="mcp-1",
                name="Filesystem MCP",
                url="https://mcp.example.com/mcp",
            )
        },
        mcp_tool_cache={"mcp-1": [{"name": "secret_tool"}]},
    )
    other_entry = MockConfigEntry(
        version=2,
        minor_version=0,
        domain=DOMAIN,
        title="Workspace 2",
        data={CONF_NAME: "Workspace 2"},
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "subentry_id": "provider-2",
                "data": {
                    CONF_NAME: "Responses Provider",
                    CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
                    CONF_API_KEY: "sk-other",
                    CONF_DEFAULT_MODEL_PROFILE_ID: "profile-2",
                    CONF_MODEL_PROFILES: {
                        "profile-2": {
                            "id": "profile-2",
                            CONF_NAME: "Responses GPT",
                            CONF_MODEL: "gpt-responses",
                            CONF_ENABLED: True,
                            CONF_DISCOVERED: True,
                        }
                    },
                },
                "subentry_type": SUBENTRY_TYPE_PROVIDER,
                "title": "Responses Provider",
                "unique_id": None,
            },
        ),
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
        "provider_count": 2,
        "model_profile_count": 2,
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
