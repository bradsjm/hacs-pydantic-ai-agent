"""Tests for provider model profile customization flows."""

from typing import Any, cast
from unittest.mock import patch

from custom_components.pydantic_ai_agent.config_flows._constants import (
    _SECTION_ADVANCED_MODEL_SETTINGS,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
    CONF_DISCOVERED,
    CONF_ENABLED,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_NAME,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROVIDER_METADATA,
    CONF_PROVIDER_MODE,
    CONF_TEMPLATED_EXTRA_BODY,
    DOMAIN,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_PROVIDER,
)
from homeassistant import config_entries
from homeassistant.config_entries import SubentryFlowContext
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry
from tests.components.pydantic_ai_agent.support.schemas import (
    schema_key_names as _schema_key_names,
)
from tests.components.pydantic_ai_agent.support.schemas import (
    schema_select_options as _schema_select_options,
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
            CONF_PROVIDER_METADATA: {"catalog_provider_id": "openai"},
            CONF_MODEL_PROFILES: {
                "profile-1": {
                    "id": "profile-1",
                    CONF_NAME: "GPT Mini",
                    CONF_MODEL: "gpt-4.1-mini",
                    CONF_ENABLED: True,
                }
            },
        },
    }


async def test_provider_edit_model_picker_shows_only_enabled_profiles(
    hass: HomeAssistant,
) -> None:
    """Test model customization only offers available profiles."""
    provider_data = _provider_subentry_data()
    provider_config = cast(dict[str, Any], provider_data["data"])
    provider_config[CONF_MODEL_PROFILES]["profile-2"] = {
        "id": "profile-2",
        CONF_NAME: "Disabled Model",
        CONF_MODEL: "gpt-4.1",
        CONF_ENABLED: False,
    }
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
        hass, result["flow_id"], {"next_step_id": "customize_model_profile"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "pick_model_profile"
    assert _schema_select_options(result["data_schema"], "model_profile_id") == [
        {"label": "GPT Mini", "value": "profile-1"}
    ]


async def test_provider_edit_discovered_model_profile_hides_model_identifier(
    hass: HomeAssistant,
) -> None:
    """Test catalog/discovered profile edits preserve hidden model identifiers."""
    provider_data = _provider_subentry_data()
    provider_config = cast(dict[str, Any], provider_data["data"])
    provider_config[CONF_MODEL_PROFILES]["profile-1"][CONF_DISCOVERED] = True
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
        hass, result["flow_id"], {"next_step_id": "customize_model_profile"}
    )
    result = await _subentry_configure_result(
        hass, result["flow_id"], {"model_profile_id": "profile-1"}
    )

    assert result["type"] is FlowResultType.FORM
    assert CONF_NAME in _schema_key_names(result["data_schema"])
    assert CONF_MODEL not in _schema_key_names(result["data_schema"])

    result = await _subentry_configure_result(
        hass, result["flow_id"], {CONF_NAME: "Friendly GPT"}
    )

    assert result["type"] is FlowResultType.ABORT
    updated_profile = entry.subentries[provider_subentry.subentry_id].data[
        CONF_MODEL_PROFILES
    ]["profile-1"]
    assert updated_profile[CONF_NAME] == "Friendly GPT"
    assert updated_profile[CONF_MODEL] == "gpt-4.1-mini"

    result = await _subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": config_entries.SOURCE_USER},
    )
    assert _schema_select_options(result["data_schema"], CONF_PRIMARY_MODEL_REF) == [
        {"label": "OpenAI-compatible / Friendly GPT", "value": "provider-1:profile-1"}
    ]

    result = await _subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_AI_TASK),
        context={"source": config_entries.SOURCE_USER},
    )
    assert _schema_select_options(result["data_schema"], CONF_PRIMARY_MODEL_REF) == [
        {"label": "OpenAI-compatible / Friendly GPT", "value": "provider-1:profile-1"}
    ]


async def test_provider_edit_manual_model_profile_allows_model_identifier(
    hass: HomeAssistant,
) -> None:
    """Test manual/custom profile edits can still change model identifiers."""
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
        hass, result["flow_id"], {"next_step_id": "customize_model_profile"}
    )
    result = await _subentry_configure_result(
        hass, result["flow_id"], {"model_profile_id": "profile-1"}
    )

    assert result["type"] is FlowResultType.FORM
    assert CONF_MODEL in _schema_key_names(result["data_schema"])

    result = await _subentry_configure_result(
        hass,
        result["flow_id"],
        {CONF_NAME: "Manual GPT", CONF_MODEL: "local/manual-model"},
    )

    assert result["type"] is FlowResultType.ABORT
    updated_profile = entry.subentries[provider_subentry.subentry_id].data[
        CONF_MODEL_PROFILES
    ]["profile-1"]
    assert updated_profile[CONF_NAME] == "Manual GPT"
    assert updated_profile[CONF_MODEL] == "local/manual-model"


async def test_provider_edit_model_profile_templated_extra_body_round_trip(
    hass: HomeAssistant,
) -> None:
    """Test templated extra body persists and reloads in the advanced section."""
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
        hass, result["flow_id"], {"next_step_id": "customize_model_profile"}
    )
    result = await _subentry_configure_result(
        hass, result["flow_id"], {"model_profile_id": "profile-1"}
    )

    templated_extra_body = [
        {
            CONF_CHAT_TEMPLATE_KWARG_KEY: "chat_template_kwargs.enable_thinking",
            CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ true }}",
        }
    ]
    result = await _subentry_configure_result(
        hass,
        result["flow_id"],
        {
            CONF_NAME: "GPT Mini",
            CONF_MODEL: "gpt-4.1-mini",
            _SECTION_ADVANCED_MODEL_SETTINGS: {
                CONF_TEMPLATED_EXTRA_BODY: templated_extra_body
            },
        },
    )

    assert result["type"] is FlowResultType.ABORT
    updated_profile = entry.subentries[provider_subentry.subentry_id].data[
        CONF_MODEL_PROFILES
    ]["profile-1"]
    assert updated_profile[CONF_MODEL_SETTINGS][CONF_TEMPLATED_EXTRA_BODY] == (
        templated_extra_body
    )

    result = await _subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": provider_subentry.subentry_id,
        },
    )
    result = await _subentry_configure_result(
        hass, result["flow_id"], {"next_step_id": "customize_model_profile"}
    )
    result = await _subentry_configure_result(
        hass, result["flow_id"], {"model_profile_id": "profile-1"}
    )

    assert result["type"] is FlowResultType.FORM
    assert _serialized_section_default(
        result["data_schema"], _SECTION_ADVANCED_MODEL_SETTINGS
    ) == {CONF_TEMPLATED_EXTRA_BODY: templated_extra_body}
