"""Tests for conversation reconfigure streaming state."""

from custom_components.pydantic_ai_agent.config_flows.common import (
    _SECTION_ADVANCED_OPTIONS,
    _SECTION_RUN_SETTINGS,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_PRIMARY_MODEL_REF,
    CONF_STREAMING_ENABLED,
    CONF_TIMEOUT,
    SUBENTRY_TYPE_CONVERSATION,
)
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from tests.components.pydantic_ai_agent.test_workspace_provider_reconfigure import (
    _loaded_workspace_entry,
    _provider_subentry_data,
    _serialized_section_default,
    _subentry_configure_result,
    _subentry_init_result,
)


async def test_conversation_reconfigure_preserves_streaming_toggle_on_validation_error(
    hass: HomeAssistant,
) -> None:
    """Test invalid run settings do not reset the advanced streaming toggle."""
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
                    CONF_STREAMING_ENABLED: False,
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

    result = await _subentry_configure_result(
        hass,
        result["flow_id"],
        {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_PRIMARY_MODEL_REF: profile_ref,
            _SECTION_ADVANCED_OPTIONS: {CONF_STREAMING_ENABLED: False},
            _SECTION_RUN_SETTINGS: {CONF_TIMEOUT: -1},
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {CONF_TIMEOUT: "positive_number"}
    assert _serialized_section_default(
        result["data_schema"], _SECTION_ADVANCED_OPTIONS
    ) == {CONF_STREAMING_ENABLED: False}
