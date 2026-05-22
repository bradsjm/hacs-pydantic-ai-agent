"""Smoke tests for the workspace-first config flow."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from homeassistant import config_entries
from homeassistant.config_entries import ConfigSubentry, SubentryFlowResult
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.pydantic_ai_agent import (
    ProviderRuntimeData,
    WorkspaceRuntimeData,
)
from custom_components.pydantic_ai_agent.conversation import (
    PydanticAIConversationEntity,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_CUSTOM_MODEL_NAMES,
    CONF_ENABLED,
    CONF_DEFAULT_SKILLS_FOLDER,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROVIDER_MODE,
    DEFAULT_SKILLS_FOLDER,
    DOMAIN,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_PROVIDER,
)


async def _finish_subentry_progress(
    hass: HomeAssistant, flow_id: str
) -> SubentryFlowResult:
    """Advance a subentry flow until it leaves progress states."""
    result = await hass.config_entries.subentries.async_configure(flow_id)
    for _ in range(10):
        if result["type"] is FlowResultType.SHOW_PROGRESS:
            await hass.async_block_till_done()
            result = await hass.config_entries.subentries.async_configure(flow_id)
            continue
        if result["type"] is FlowResultType.SHOW_PROGRESS_DONE:
            result = await hass.config_entries.subentries.async_configure(flow_id)
            continue
        return result
    raise AssertionError(f"Subentry flow did not leave progress state: {result}")


async def _loaded_workspace_entry(
    hass: HomeAssistant, subentries_data: tuple[dict[str, object], ...] = ()
) -> MockConfigEntry:
    """Return a loaded workspace entry for subentry flow tests."""
    entry = MockConfigEntry(
        version=2,
        minor_version=0,
        domain=DOMAIN,
        title="Workspace",
        data={
            CONF_NAME: "Workspace",
            CONF_DEFAULT_SKILLS_FOLDER: DEFAULT_SKILLS_FOLDER,
        },
        subentries_data=subentries_data,
        source=config_entries.SOURCE_USER,
        unique_id=None,
    )
    entry.add_to_hass(hass)
    with patch(
        "custom_components.pydantic_ai_agent.async_setup_entry",
        return_value=True,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    return entry


async def test_create_workspace_entry(hass: HomeAssistant) -> None:
    """Test the parent flow creates a workspace entry."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": config_entries.SOURCE_USER},
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_NAME: "Living Room Workspace"},
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Living Room Workspace"
    assert result["data"] == {
        CONF_NAME: "Living Room Workspace",
        CONF_DEFAULT_SKILLS_FOLDER: DEFAULT_SKILLS_FOLDER,
    }


async def test_create_provider_subentry_with_disabled_custom_profile(
    hass: HomeAssistant,
) -> None:
    """Test provider creation stores custom profiles disabled by default."""
    entry = await _loaded_workspace_entry(hass)

    with (
        patch(
            "custom_components.pydantic_ai_agent.config_flow.async_list_provider_model_names",
            new=AsyncMock(return_value=["gpt-4.1-mini"]),
        ) as list_model_names,
        patch(
            "custom_components.pydantic_ai_agent.config_flow.async_probe_model",
            new=AsyncMock(),
        ) as probe_model,
    ):
        result = await hass.config_entries.subentries.async_init(
            (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
            context={"source": config_entries.SOURCE_USER},
        )
        assert result["type"] is FlowResultType.FORM

        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"],
            {
                CONF_NAME: "OpenAI-compatible",
                CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
                CONF_API_KEY: "sk-test",
                "customize_model_list": {CONF_CUSTOM_MODEL_NAMES: "gpt-4.1-mini"},
            },
        )
        assert result["type"] is FlowResultType.SHOW_PROGRESS

        result = await _finish_subentry_progress(hass, result["flow_id"])

    assert result["type"] is FlowResultType.CREATE_ENTRY
    list_model_names.assert_not_called()
    probe_model.assert_not_called()
    provider_data = cast(dict[str, Any], result["data"])
    assert provider_data[CONF_NAME] == "OpenAI-compatible"
    assert provider_data[CONF_PROVIDER_MODE] == PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS
    assert provider_data[CONF_MODEL_PROFILES]
    model_profiles = cast(dict[str, dict[str, Any]], provider_data[CONF_MODEL_PROFILES])
    profile = next(iter(model_profiles.values()))
    assert profile[CONF_MODEL] == "gpt-4.1-mini"
    assert profile[CONF_ENABLED] is False
    assert provider_data[CONF_CUSTOM_MODEL_NAMES] == ["gpt-4.1-mini"]


async def test_conversation_entity_streaming_supports_model_profile_ref(
    hass: HomeAssistant,
) -> None:
    """Test conversation entity streaming support with provider-owned profiles."""
    provider_subentry_id = "provider-1"
    default_profile_id = "profile-1"
    profile_ref = f"{provider_subentry_id}:{default_profile_id}"
    entry = await _loaded_workspace_entry(
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

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_PRIMARY_MODEL_REF: profile_ref,
        },
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_PRIMARY_MODEL_REF] == profile_ref
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
    assert (
        PydanticAIConversationEntity(entry, tool_subentry).supports_streaming is False
    )


async def test_provider_subentry_base_url_endpoint_returns_form_error(
    hass: HomeAssistant,
) -> None:
    """Test provider URL endpoint validation replays as a form error."""
    entry = await _loaded_workspace_entry(hass)

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "OpenAI-compatible",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "sk-test",
            "base_url": "https://api.example.com/v1/chat/completions",
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "invalid_base_url_endpoint"}
