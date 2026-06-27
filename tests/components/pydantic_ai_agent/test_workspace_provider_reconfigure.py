"""Tests for provider reconfigure forms and provider edit flows."""

from hashlib import sha256
import json
from typing import Any, cast
from unittest.mock import patch

from custom_components.pydantic_ai_agent.config_flows.common import (
    _SECTION_HASS_CONTROL,
)
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.const import (
    CONF_CATALOG_PROVIDER_ID,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_AI_TASK_NAME,
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_CUSTOM_MODEL_NAMES,
    CONF_DISCOVERED,
    CONF_DISCOVERED_MODELS,
    CONF_DISCOVERED_MODELS_AT,
    CONF_DISCOVERED_MODELS_CACHE_KEY,
    CONF_ENABLED,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_NAME,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_METADATA,
    CONF_PROVIDER_MODE,
    DOMAIN,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_PROVIDER,
)
from homeassistant import config_entries
from homeassistant.config_entries import SubentryFlowContext
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry
from tests.components.pydantic_ai_agent.support.builders import model_profile_data
from tests.components.pydantic_ai_agent.support.schemas import (
    schema_key_names as _schema_key_names,
)
from tests.components.pydantic_ai_agent.support.schemas import (
    section_default as _section_default,
)
from tests.components.pydantic_ai_agent.support.schemas import (
    section_field_suggested_value as _section_field_suggested_value,
)
from tests.components.pydantic_ai_agent.support.schemas import (
    serialized_section_default as _serialized_section_default,
)

type _FlowResultDict = dict[str, Any]


async def _subentry_init_result(
    hass: HomeAssistant,
    flow_key: tuple[str, str],
    context: SubentryFlowContext,
) -> _FlowResultDict:
    return cast(
        _FlowResultDict,
        await hass.config_entries.subentries.async_init(flow_key, context=context),
    )


async def _subentry_configure_result(
    hass: HomeAssistant, flow_id: str, user_input: dict[str, object] | None = None
) -> _FlowResultDict:
    return cast(
        _FlowResultDict,
        await hass.config_entries.subentries.async_configure(flow_id, user_input),
    )


async def _loaded_workspace_entry(
    hass: HomeAssistant, subentries_data: tuple[dict[str, object], ...] = ()
) -> MockConfigEntry:
    entry = MockConfigEntry(
        version=2,
        minor_version=2,
        domain=DOMAIN,
        title="Workspace",
        data={CONF_NAME: "Workspace"},
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


def _provider_subentry_data() -> dict[str, object]:
    """Return a provider subentry with one enabled profile for flow tests."""
    return {
        "subentry_id": "provider-1",
        "subentry_type": SUBENTRY_TYPE_PROVIDER,
        "title": "OpenAI-compatible",
        "unique_id": None,
        "data": {
            CONF_NAME: "OpenAI-compatible",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "sk-test",
            CONF_PROVIDER_METADATA: {CONF_CATALOG_PROVIDER_ID: "openai"},
            CONF_MODEL_PROFILES: {
                "profile-1": model_profile_data(
                    profile_id="profile-1",
                    name="GPT Mini",
                    model="gpt-4.1-mini",
                )
            },
        },
    }


async def test_reconfigure_forms_prefill_llm_hass_api_in_control_section(
    hass: HomeAssistant,
) -> None:
    """Test agent reconfigure forms prefill HA control values in their section."""
    profile_ref = "provider-1:profile-1"
    conversation_subentry_id = "conversation-1"
    ai_task_subentry_id = "ai-task-1"
    entry = await _loaded_workspace_entry(
        hass,
        (
            _provider_subentry_data(),
            {
                "subentry_id": conversation_subentry_id,
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
                "data": {
                    CONF_AGENT_NAME: "Kitchen Agent",
                    CONF_PRIMARY_MODEL_REF: profile_ref,
                    CONF_LLM_HASS_API: ["assist"],
                },
            },
            {
                "subentry_id": ai_task_subentry_id,
                "subentry_type": SUBENTRY_TYPE_AI_TASK,
                "title": "Summary Task",
                "unique_id": None,
                "data": {
                    CONF_AI_TASK_NAME: "Summary Task",
                    CONF_PRIMARY_MODEL_REF: profile_ref,
                    CONF_LLM_HASS_API: ["assist"],
                },
            },
        ),
    )

    conversation_result = await _subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": conversation_subentry_id,
        },
    )
    ai_task_result = await _subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_AI_TASK),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": ai_task_subentry_id,
        },
    )

    assert conversation_result["type"] is FlowResultType.FORM
    assert _SECTION_HASS_CONTROL in _schema_key_names(
        conversation_result["data_schema"]
    )
    assert _section_default(
        conversation_result["data_schema"], _SECTION_HASS_CONTROL
    ) == {CONF_LLM_HASS_API: ["assist"]}
    assert _serialized_section_default(
        conversation_result["data_schema"], _SECTION_HASS_CONTROL
    ) == {CONF_LLM_HASS_API: ["assist"]}
    assert _section_field_suggested_value(
        conversation_result["data_schema"],
        _SECTION_HASS_CONTROL,
        CONF_LLM_HASS_API,
    ) == ["assist"]
    assert ai_task_result["type"] is FlowResultType.FORM
    assert _SECTION_HASS_CONTROL in _schema_key_names(ai_task_result["data_schema"])
    assert _section_default(ai_task_result["data_schema"], _SECTION_HASS_CONTROL) == {
        CONF_LLM_HASS_API: ["assist"]
    }


async def test_conversation_reconfigure_assist_round_trips_to_form_section(
    hass: HomeAssistant,
) -> None:
    """Test enabling Assist in reconfigure remains selected on the next edit."""
    profile_ref = "provider-1:profile-1"
    conversation_subentry_id = "conversation-1"
    entry = await _loaded_workspace_entry(
        hass,
        (
            _provider_subentry_data(),
            {
                "subentry_id": conversation_subentry_id,
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
                "data": {
                    CONF_AGENT_NAME: "Kitchen Agent",
                    CONF_PRIMARY_MODEL_REF: profile_ref,
                },
            },
        ),
    )

    result = await _subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": conversation_subentry_id,
        },
    )
    assert result["type"] is FlowResultType.FORM
    assert CONF_LLM_HASS_API not in entry.subentries[conversation_subentry_id].data
    assert _section_default(result["data_schema"], _SECTION_HASS_CONTROL) == {}
    assert (
        _serialized_section_default(result["data_schema"], _SECTION_HASS_CONTROL) == {}
    )

    result = await _subentry_configure_result(
        hass,
        result["flow_id"],
        {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_PRIMARY_MODEL_REF: profile_ref,
            _SECTION_HASS_CONTROL: {CONF_LLM_HASS_API: ["assist"]},
        },
    )

    assert result["type"] is FlowResultType.ABORT
    assert entry.subentries[conversation_subentry_id].data[CONF_LLM_HASS_API] == [
        "assist"
    ]

    result = await _subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": conversation_subentry_id,
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert _section_default(result["data_schema"], _SECTION_HASS_CONTROL) == {
        CONF_LLM_HASS_API: ["assist"]
    }


async def test_provider_reconfigure_menu_exposes_model_management(
    hass: HomeAssistant,
) -> None:
    """Test provider reconfigure separates connection, availability, and editing."""
    entry = await _loaded_workspace_entry(hass, (_provider_subentry_data(),))
    provider_subentry = next(iter(entry.subentries.values()))

    result = await _subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": provider_subentry.subentry_id,
        },
    )

    assert result["type"] is FlowResultType.MENU
    assert result["menu_options"] == [
        "edit_connection",
        "manage_models",
        "customize_model_profile",
    ]


async def test_provider_edit_connection_preserves_catalog_metadata(
    hass: HomeAssistant,
) -> None:
    """Test editing a guided provider keeps catalog metadata for profile filters."""
    entry = await _loaded_workspace_entry(hass, (_provider_subentry_data(),))
    provider_subentry = next(iter(entry.subentries.values()))

    result = await _subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": provider_subentry.subentry_id,
        },
    )
    assert result["type"] is FlowResultType.MENU

    result = await _subentry_configure_result(
        hass, result["flow_id"], {"next_step_id": "edit_connection"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "edit_connection"

    result = await _subentry_configure_result(
        hass,
        result["flow_id"],
        {
            CONF_NAME: "OpenAI-compatible",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "sk-updated",
        },
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    updated_subentry = entry.subentries[provider_subentry.subentry_id]
    assert updated_subentry.data[CONF_PROVIDER_METADATA] == {
        CONF_CATALOG_PROVIDER_ID: "openai"
    }


async def test_provider_edit_connection_preserves_catalog_metadata_for_default_url(
    hass: HomeAssistant,
) -> None:
    """Test explicit default URLs keep guided provider catalog metadata."""
    entry = await _loaded_workspace_entry(hass, (_provider_subentry_data(),))
    provider_subentry = next(iter(entry.subentries.values()))

    result = await _subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": provider_subentry.subentry_id,
        },
    )
    assert result["type"] is FlowResultType.MENU

    result = await _subentry_configure_result(
        hass, result["flow_id"], {"next_step_id": "edit_connection"}
    )
    assert result["type"] is FlowResultType.FORM

    result = await _subentry_configure_result(
        hass,
        result["flow_id"],
        {
            CONF_NAME: "OpenAI-compatible",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "sk-updated",
            CONF_BASE_URL: "https://api.openai.com/v1",
        },
    )

    assert result["type"] is FlowResultType.ABORT
    updated_subentry = entry.subentries[provider_subentry.subentry_id]
    assert updated_subentry.data[CONF_PROVIDER_METADATA] == {
        CONF_CATALOG_PROVIDER_ID: "openai"
    }


async def test_provider_edit_connection_clears_catalog_metadata_when_repointed(
    hass: HomeAssistant,
) -> None:
    """Test repointing a guided provider clears stale catalog metadata."""
    entry = await _loaded_workspace_entry(hass, (_provider_subentry_data(),))
    provider_subentry = next(iter(entry.subentries.values()))

    result = await _subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": provider_subentry.subentry_id,
        },
    )
    result = await _subentry_configure_result(
        hass, result["flow_id"], {"next_step_id": "edit_connection"}
    )
    result = await _subentry_configure_result(
        hass,
        result["flow_id"],
        {
            CONF_NAME: "OpenAI-compatible",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "sk-updated",
            CONF_BASE_URL: "https://api.deepseek.com/v1",
        },
    )

    assert result["type"] is FlowResultType.ABORT
    updated_subentry = entry.subentries[provider_subentry.subentry_id]
    assert CONF_PROVIDER_METADATA not in updated_subentry.data


async def test_provider_edit_connection_preserves_model_management_data(
    hass: HomeAssistant,
) -> None:
    """Test connection edits preserve custom names, profiles, and valid cache data."""
    provider_data = _provider_subentry_data()
    provider_config = cast(dict[str, Any], provider_data["data"])
    provider_config[CONF_CUSTOM_MODEL_NAMES] = ["local/custom-model"]
    provider_config[CONF_MODEL_PROFILES]["profile-custom"] = {
        "id": "profile-custom",
        CONF_NAME: "Local Custom",
        CONF_MODEL: "local/custom-model",
        CONF_ENABLED: True,
        CONF_DISCOVERED: False,
    }
    provider_config[CONF_DISCOVERED_MODELS] = ["gpt-4.1-mini"]
    provider_config[CONF_DISCOVERED_MODELS_AT] = dt_util.utcnow().isoformat()
    provider_config[CONF_DISCOVERED_MODELS_CACHE_KEY] = json.dumps(
        {
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: sha256(b"sk-test").hexdigest(),
            CONF_BASE_URL: None,
            CONF_PROVIDER_HEADERS: {},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    entry = await _loaded_workspace_entry(hass, (provider_data,))
    provider_subentry = next(iter(entry.subentries.values()))

    result = await _subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": provider_subentry.subentry_id,
        },
    )
    result = await _subentry_configure_result(
        hass, result["flow_id"], {"next_step_id": "edit_connection"}
    )
    assert CONF_CUSTOM_MODEL_NAMES not in _schema_key_names(result["data_schema"])

    result = await _subentry_configure_result(
        hass,
        result["flow_id"],
        {
            CONF_NAME: "OpenAI-compatible",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "sk-test",
        },
    )

    assert result["type"] is FlowResultType.ABORT
    updated_subentry = entry.subentries[provider_subentry.subentry_id]
    assert updated_subentry.data[CONF_CUSTOM_MODEL_NAMES] == ["local/custom-model"]
    assert (
        updated_subentry.data[CONF_MODEL_PROFILES]["profile-custom"][CONF_ENABLED]
        is True
    )
    assert updated_subentry.data[CONF_DISCOVERED_MODELS] == ["gpt-4.1-mini"]
    assert CONF_DISCOVERED_MODELS_CACHE_KEY in updated_subentry.data
