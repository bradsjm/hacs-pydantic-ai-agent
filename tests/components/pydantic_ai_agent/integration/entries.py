"""Config-entry builders and setup helpers for provider integration tests."""

import asyncio

from homeassistant import config_entries
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_BASE_URL,
    CONF_DEFAULT_MODEL_PROFILE_ID,
    CONF_DESCRIPTION,
    CONF_DISCOVERED,
    CONF_ENABLED,
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_SERVER_IDS,
    CONF_MCP_URL,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_OUTPUT_MODE,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROVIDER_MODE,
    CONF_SKILL_CONTENT,
    CONF_SKILL_REFERENCES,
    CONF_SKILLS,
    DOMAIN,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_MCP_SERVER,
    SUBENTRY_TYPE_PROVIDER,
    SUBENTRY_TYPE_SKILL,
)

from .config import (
    MCP_ECHO_SERVER_ID,
    MODEL_PROFILE_ID,
    MODEL_REF,
    PROVIDER_ID,
    PROVIDER_INTEGRATION_TIMEOUT,
    ProviderIntegrationConfig,
    UNSELECTED_WORKSPACE_SKILL_ID,
    WORKSPACE_SKILL_ID,
)


def conversation_subentry(
    llm_hass_api: list[str] | None = None,
    mcp_server_ids: list[str] | None = None,
    skill_ids: list[str] | None = None,
) -> dict[str, object]:
    """Return a provider integration conversation subentry."""
    data: dict[str, object] = {
        CONF_AGENT_NAME: "Integration Conversation Agent",
        CONF_PRIMARY_MODEL_REF: MODEL_REF,
    }
    if llm_hass_api is not None:
        data[CONF_LLM_HASS_API] = llm_hass_api
    if mcp_server_ids is not None:
        data[CONF_MCP_SERVER_IDS] = mcp_server_ids
    if skill_ids is not None:
        data[CONF_SKILLS] = skill_ids

    return {
        "data": data,
        "subentry_type": SUBENTRY_TYPE_CONVERSATION,
        "title": "Integration Conversation Agent",
        "unique_id": None,
    }


def ai_task_subentry(output_mode: str | None = None) -> dict[str, object]:
    """Return a provider integration AI task subentry."""
    data: dict[str, object] = {CONF_PRIMARY_MODEL_REF: MODEL_REF}
    if output_mode is not None:
        data[CONF_OUTPUT_MODE] = output_mode
    return {
        "data": data,
        "subentry_type": SUBENTRY_TYPE_AI_TASK,
        "title": "Integration AI Task",
        "unique_id": None,
    }


def mcp_ai_task_subentry(output_mode: str | None = None) -> dict[str, object]:
    """Return a provider integration AI task subentry with MCP echo access."""
    data: dict[str, object] = {
        CONF_PRIMARY_MODEL_REF: MODEL_REF,
        CONF_MCP_SERVER_IDS: [MCP_ECHO_SERVER_ID],
    }
    if output_mode is not None:
        data[CONF_OUTPUT_MODE] = output_mode
    return {
        "data": data,
        "subentry_type": SUBENTRY_TYPE_AI_TASK,
        "title": "Integration MCP AI Task",
        "unique_id": None,
    }


def mcp_echo_subentry(mcp_echo_url: str) -> dict[str, object]:
    """Return a hosted MCP echo server subentry."""
    return {
        "data": {
            CONF_MCP_URL: mcp_echo_url,
            CONF_MCP_ALLOWED_TOOLS: ["echo"],
        },
        "subentry_id": MCP_ECHO_SERVER_ID,
        "subentry_type": SUBENTRY_TYPE_MCP_SERVER,
        "title": "Hosted MCP Echo",
        "unique_id": None,
    }


def skill_subentry() -> dict[str, object]:
    """Return a native workspace Skill subentry."""
    return {
        "data": {
            CONF_NAME: "Integration Skill",
            CONF_DESCRIPTION: "Provides the integration Skill token.",
            CONF_SKILL_CONTENT: (
                "Workspace Skill token: PAI_E2E_SKILL_OK\n"
                "When asked for the workspace Skill token, reply exactly with it."
            ),
            CONF_SKILL_REFERENCES: [],
        },
        "subentry_id": WORKSPACE_SKILL_ID,
        "subentry_type": SUBENTRY_TYPE_SKILL,
        "title": "Integration Skill",
        "unique_id": None,
    }


def unselected_skill_subentry() -> dict[str, object]:
    """Return an unselected native workspace Skill subentry."""
    return {
        "data": {
            CONF_NAME: "Unselected Integration Skill",
            CONF_DESCRIPTION: "Must not be exposed to this integration agent.",
            CONF_SKILL_CONTENT: "This unselected Skill must not be loaded.",
            CONF_SKILL_REFERENCES: [],
        },
        "subentry_id": UNSELECTED_WORKSPACE_SKILL_ID,
        "subentry_type": SUBENTRY_TYPE_SKILL,
        "title": "Unselected Integration Skill",
        "unique_id": None,
    }


def provider_subentry(
    provider_config: ProviderIntegrationConfig,
) -> dict[str, object]:
    """Return a provider subentry with one model profile."""
    return {
        "subentry_id": PROVIDER_ID,
        "data": {
            CONF_NAME: "Integration Model Profile",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: provider_config.api_key,
            CONF_BASE_URL: provider_config.base_url,
            CONF_DEFAULT_MODEL_PROFILE_ID: MODEL_PROFILE_ID,
            CONF_MODEL_PROFILES: {
                MODEL_PROFILE_ID: {
                    "id": MODEL_PROFILE_ID,
                    CONF_NAME: "Integration Model Profile",
                    CONF_MODEL: provider_config.model,
                    CONF_MODEL_SETTINGS: {"timeout": PROVIDER_INTEGRATION_TIMEOUT},
                    CONF_ENABLED: True,
                    CONF_DISCOVERED: False,
                }
            },
        },
        "subentry_type": SUBENTRY_TYPE_PROVIDER,
        "title": "Live OpenAI-compatible Provider",
        "unique_id": None,
    }


def entry(
    provider_config: ProviderIntegrationConfig, *subentries: dict[str, object]
) -> MockConfigEntry:
    """Return a config entry for provider integration subentries."""
    return MockConfigEntry(
        version=2,
        minor_version=1,
        domain=DOMAIN,
        title="Integration Workspace",
        data={CONF_NAME: "Integration Workspace"},
        source=config_entries.SOURCE_USER,
        subentries_data=(provider_subentry(provider_config), *subentries),
        options={},
        unique_id=None,
    )


async def setup_entry(hass: HomeAssistant, config_entry: MockConfigEntry) -> None:
    """Add and load a provider integration config entry."""
    config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()
    assert config_entry.state is ConfigEntryState.LOADED


async def drain_stream_cleanup(hass: HomeAssistant) -> None:
    """Let async stream finalizers finish before HA cleanup assertions run."""
    await hass.async_block_till_done()
    await asyncio.sleep(0)
    await hass.async_block_till_done()


async def conversation_entity_id(
    hass: HomeAssistant,
    provider_config: ProviderIntegrationConfig,
    llm_hass_api: list[str] | None = None,
    mcp_echo_url: str | None = None,
) -> str:
    """Set up a conversation agent and return its entity ID."""
    mcp_server_ids = [MCP_ECHO_SERVER_ID] if mcp_echo_url is not None else None
    subentries = [conversation_subentry(llm_hass_api, mcp_server_ids)]
    if mcp_echo_url is not None:
        subentries.append(mcp_echo_subentry(mcp_echo_url))
    await setup_entry(hass, entry(provider_config, *subentries))
    entity_ids = [
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id != "conversation.home_assistant"
    ]
    assert len(entity_ids) == 1
    return entity_ids[0]


async def skill_conversation_entity_id(
    hass: HomeAssistant, provider_config: ProviderIntegrationConfig
) -> str:
    """Set up a conversation agent with a selected workspace Skill."""
    await setup_entry(
        hass,
        entry(
            provider_config,
            conversation_subentry(skill_ids=[WORKSPACE_SKILL_ID]),
            skill_subentry(),
            unselected_skill_subentry(),
        ),
    )
    entity_ids = [
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id != "conversation.home_assistant"
    ]
    assert len(entity_ids) == 1
    return entity_ids[0]


async def ai_task_entity_id(
    hass: HomeAssistant,
    provider_config: ProviderIntegrationConfig,
    output_mode: str | None = None,
) -> str:
    """Set up an AI task entity and return its entity ID."""
    await setup_entry(hass, entry(provider_config, ai_task_subentry(output_mode)))
    entity_ids = [state.entity_id for state in hass.states.async_all("ai_task")]
    assert len(entity_ids) == 1
    return entity_ids[0]


async def mcp_ai_task_entity_id(
    hass: HomeAssistant,
    provider_config: ProviderIntegrationConfig,
    mcp_echo_url: str,
    output_mode: str | None = None,
) -> str:
    """Set up an AI task entity with hosted MCP echo access."""
    await setup_entry(
        hass,
        entry(
            provider_config,
            mcp_ai_task_subentry(output_mode),
            mcp_echo_subentry(mcp_echo_url),
        ),
    )
    entity_ids = [state.entity_id for state in hass.states.async_all("ai_task")]
    assert len(entity_ids) == 1
    return entity_ids[0]
