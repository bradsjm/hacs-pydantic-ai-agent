"""Tests for workspace entry creation and subentry creation flows."""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import voluptuous_serialize
from custom_components.pydantic_ai_agent import (
    ProviderRuntimeData,
    WorkspaceRuntimeData,
)
from custom_components.pydantic_ai_agent.config_flows import mcp_server_flow
from custom_components.pydantic_ai_agent.config_flows.workspace_flow import (
    PydanticAIAgentConfigFlow,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_API_KEY,
    CONF_ENABLED,
    CONF_FALLBACK_MODEL_REFS,
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_URL,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_NAME,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROVIDER_MODE,
    CONF_SKILL_CONTENT,
    CONF_SKILL_REFERENCES,
    DOMAIN,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_MCP_SERVER,
    SUBENTRY_TYPE_PROVIDER,
    SUBENTRY_TYPE_SKILL,
)
from custom_components.pydantic_ai_agent.conversation import (
    PydanticAIConversationEntity,
)
from custom_components.pydantic_ai_agent.mcp import MCPValidationError
from homeassistant import config_entries
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import config_validation as cv
from tests.components.pydantic_ai_agent.support.wizard import (
    entry_flow_configure_result,
    entry_flow_init_result,
    loaded_workspace_entry,
    subentry_configure_result,
    subentry_init_result,
)


async def test_conversation_entity_streaming_supports_model_profile_ref(
    hass: HomeAssistant,
) -> None:
    """Test conversation entity streaming support with provider-owned profiles."""
    provider_subentry_id = "provider-1"
    default_profile_id = "profile-1"
    profile_ref = f"{provider_subentry_id}:{default_profile_id}"
    entry = await loaded_workspace_entry(
        hass,
        (
            {
                "subentry_id": provider_subentry_id,
                "subentry_type": SUBENTRY_TYPE_PROVIDER,
                "title": "OpenAI-compatible",
                "unique_id": None,
                "data": {
                    CONF_NAME: "OpenAI-compatible",
                    CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
                    CONF_API_KEY: "sk-test",
                    CONF_MODEL_PROFILES: {
                        default_profile_id: {
                            "id": default_profile_id,
                            CONF_NAME: "GPT Mini",
                            CONF_MODEL: "gpt-4.1-mini",
                            CONF_ENABLED: True,
                        }
                    },
                },
            },
        ),
    )
    entry.runtime_data = WorkspaceRuntimeData(workspace_name="Workspace", providers={})

    result = await subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await subentry_configure_result(
        hass,
        result["flow_id"],
        {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_PRIMARY_MODEL_REF: profile_ref,
        },
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_PRIMARY_MODEL_REF] == profile_ref
    assert CONF_LLM_HASS_API not in result["data"]
    entry.runtime_data = WorkspaceRuntimeData(
        workspace_name="Workspace",
        providers={
            provider_subentry_id: ProviderRuntimeData(
                provider_subentry_id=provider_subentry_id,
                name="OpenAI-compatible",
                api_key="sk-test",
                provider_mode=PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
                base_url=None,
                provider_headers={},
            )
        },
    )
    plain_data = dict(result["data"])
    plain_data.pop(CONF_LLM_HASS_API, None)

    plain_subentry = cast(
        ConfigSubentry,
        SimpleNamespace(
            data=plain_data,
            subentry_id="conversation_plain",
            subentry_type=SUBENTRY_TYPE_CONVERSATION,
        ),
    )
    tool_subentry = cast(
        ConfigSubentry,
        SimpleNamespace(
            data=plain_data | {CONF_LLM_HASS_API: ["assist"]},
            subentry_id="conversation_tools",
            subentry_type=SUBENTRY_TYPE_CONVERSATION,
        ),
    )

    assert (
        PydanticAIConversationEntity(entry, plain_subentry).supports_streaming is True
    )
    assert PydanticAIConversationEntity(entry, plain_subentry).supported_features == 0
    assert PydanticAIConversationEntity(entry, tool_subentry).supports_streaming is True


async def test_conversation_disabled_skills_ignores_invalid_folder(
    hass: HomeAssistant,
) -> None:
    """Test disabled skills do not require or validate the skills folder."""
    provider_subentry_id = "provider-1"
    default_profile_id = "profile-1"
    profile_ref = f"{provider_subentry_id}:{default_profile_id}"
    entry = await loaded_workspace_entry(
        hass,
        (
            {
                "subentry_id": provider_subentry_id,
                "subentry_type": SUBENTRY_TYPE_PROVIDER,
                "title": "OpenAI-compatible",
                "unique_id": None,
                "data": {
                    CONF_NAME: "OpenAI-compatible",
                    CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
                    CONF_API_KEY: "sk-test",
                    CONF_MODEL_PROFILES: {
                        default_profile_id: {
                            "id": default_profile_id,
                            CONF_NAME: "GPT Mini",
                            CONF_MODEL: "gpt-4.1-mini",
                            CONF_ENABLED: True,
                        }
                    },
                },
            },
        ),
    )

    result = await subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await subentry_configure_result(
        hass,
        result["flow_id"],
        {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_PRIMARY_MODEL_REF: profile_ref,
            "fallback_models": {CONF_FALLBACK_MODEL_REFS: []},
            "skill_settings": {
                "enable_skills": False,
                "skills_folder": "/tmp/skills",
            },
        },
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_PRIMARY_MODEL_REF] == profile_ref
    assert "enable_skills" not in result["data"]
    assert "skills_folder" not in result["data"]


async def test_skill_subentry_uses_template_editor_and_stores_raw_text(
    hass: HomeAssistant,
) -> None:
    """Test native Skill subentries store raw content from the template editor."""
    entry = await loaded_workspace_entry(hass)

    result = await subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_SKILL),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM
    schema = voluptuous_serialize.convert(
        result["data_schema"], custom_serializer=cv.custom_serializer
    )
    assert isinstance(schema, list)
    content_field = next(
        field for field in schema if field["name"] == CONF_SKILL_CONTENT
    )
    assert content_field["selector"] == {"template": {}}

    result = await subentry_configure_result(
        hass,
        result["flow_id"],
        {
            CONF_NAME: "Kitchen Skill",
            "description": "Kitchen guidance",
            CONF_SKILL_CONTENT: "Use {{ states('sensor.mode') }} as literal text.",
        },
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Kitchen Skill"
    assert result["data"][CONF_SKILL_CONTENT] == (
        "Use {{ states('sensor.mode') }} as literal text."
    )
    assert result["data"][CONF_SKILL_REFERENCES] == []


async def test_workspace_flow_offers_mcp_subentries(
    hass: HomeAssistant,
) -> None:
    """Test workspace subentry menus expose MCP server support."""
    entry = await loaded_workspace_entry(hass)

    supported = PydanticAIAgentConfigFlow.async_get_supported_subentry_types(entry)

    assert "mcp_server" in supported


async def test_mcp_server_validation_success_advances_to_tools(
    hass: HomeAssistant,
) -> None:
    """Test MCP validation success advances from progress to tool selection."""
    entry = await loaded_workspace_entry(hass)

    async def discover_tools(
        *_args: object, **_kwargs: object
    ) -> list[dict[str, object]]:
        return [
            {
                "name": "echo",
                "description": "Echo a message",
                "input_schema": {"type": "object"},
            }
        ]

    result = await subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_MCP_SERVER),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM

    with patch(
        "custom_components.pydantic_ai_agent.config_flows.mcp_server_flow.async_discover_mcp_tools_from_config",
        new=discover_tools,
    ):
        result = await subentry_configure_result(
            hass,
            result["flow_id"],
            {CONF_NAME: "Echo MCP", CONF_MCP_URL: "https://mcp.example.com/mcp"},
        )
        assert result["type"] is FlowResultType.SHOW_PROGRESS

        await hass.async_block_till_done()
        result = await subentry_configure_result(hass, result["flow_id"])

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "tools"
    assert result.get("errors") is None

    result = await subentry_configure_result(
        hass,
        result["flow_id"],
        {CONF_MCP_ALLOWED_TOOLS: ["echo"]},
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Echo MCP"
    assert result["data"][CONF_MCP_ALLOWED_TOOLS] == ["echo"]


async def test_mcp_server_validation_known_failure_returns_form_error(
    hass: HomeAssistant,
) -> None:
    """Test MCP validation errors return to the form instead of hanging."""
    entry = await loaded_workspace_entry(hass)

    async def fail_discovery(*_args: object, **_kwargs: object) -> list[dict[str, str]]:
        raise MCPValidationError(
            "cannot_connect",
            "Could not connect to the MCP server.",
            status_code=502,
        )

    result = await subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_MCP_SERVER),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM

    with patch(
        "custom_components.pydantic_ai_agent.config_flows.mcp_server_flow.async_discover_mcp_tools_from_config",
        new=fail_discovery,
    ):
        result = await subentry_configure_result(
            hass,
            result["flow_id"],
            {CONF_NAME: "Echo MCP", CONF_MCP_URL: "https://mcp.example.com/mcp"},
        )
        assert result["type"] is FlowResultType.SHOW_PROGRESS

        await hass.async_block_till_done()
        result = await subentry_configure_result(hass, result["flow_id"])

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"
    assert result["errors"] == {"base": "cannot_connect"}
    assert result["description_placeholders"] == {
        "error_message": "Could not connect to the MCP server.",
        "status_code": "502",
    }


async def test_mcp_server_validation_hard_timeout_returns_form_error(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test flow-level MCP validation timeout returns to the form."""
    entry = await loaded_workspace_entry(hass)
    monkeypatch.setattr(mcp_server_flow, "DEFAULT_MCP_TIMEOUT", 0.001)

    async def hang_discovery(*_args: object, **_kwargs: object) -> list[dict[str, str]]:
        await asyncio.sleep(60)
        return []

    result = await subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_MCP_SERVER),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM

    with patch(
        "custom_components.pydantic_ai_agent.config_flows.mcp_server_flow.async_discover_mcp_tools_from_config",
        new=hang_discovery,
    ):
        result = await subentry_configure_result(
            hass,
            result["flow_id"],
            {CONF_NAME: "Echo MCP", CONF_MCP_URL: "https://mcp.example.com/mcp"},
        )
        assert result["type"] is FlowResultType.SHOW_PROGRESS

        await hass.async_block_till_done()
        result = await subentry_configure_result(hass, result["flow_id"])

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"
    assert result["errors"] == {"base": "timeout"}


async def test_mcp_server_validation_exception_returns_form_error(
    hass: HomeAssistant,
) -> None:
    """Test MCP validation task failures return to the form instead of hanging."""
    entry = await loaded_workspace_entry(hass)

    async def fail_discovery(*_args: object, **_kwargs: object) -> list[dict[str, str]]:
        raise RuntimeError("boom")

    result = await subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_MCP_SERVER),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM

    with patch(
        "custom_components.pydantic_ai_agent.config_flows.mcp_server_flow.async_discover_mcp_tools_from_config",
        new=fail_discovery,
    ):
        result = await subentry_configure_result(
            hass,
            result["flow_id"],
            {CONF_NAME: "Echo MCP", CONF_MCP_URL: "https://mcp.example.com/mcp"},
        )
        assert result["type"] is FlowResultType.SHOW_PROGRESS

        await hass.async_block_till_done()
        result = await subentry_configure_result(hass, result["flow_id"])

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"
    assert result["errors"] == {"base": "unknown"}


async def test_create_workspace_entry(hass: HomeAssistant) -> None:
    """Test the parent flow creates a workspace entry."""
    result = await entry_flow_init_result(
        hass, DOMAIN, {"source": config_entries.SOURCE_USER}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    result = await entry_flow_configure_result(
        hass, result["flow_id"], {CONF_NAME: "Living Room Workspace"}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Living Room Workspace"
    assert result["data"] == {CONF_NAME: "Living Room Workspace"}
