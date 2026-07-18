"""Tests for agent subentry validation form behavior."""

from typing import Any

from custom_components.pydantic_ai_agent.config_flows.ai_task_flow import (
    AITaskDataSubentryFlowHandler,
)
from custom_components.pydantic_ai_agent.config_flows.common import (
    CONF_AGENT_NAME,
    CONF_AI_TASK_NAME,
    CONF_PRIMARY_MODEL_REF,
    CONF_TODO_LIST_ENTITY_ID,
)
from custom_components.pydantic_ai_agent.config_flows.conversation_flow import (
    ConversationSubentryFlowHandler,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_ENABLED,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_PROVIDER_MODE,
    CONF_STRUCTURED_OUTPUT_SUPPORT,
    CONF_SUPPORTS_TOOLS,
    CONF_THINKING_SUPPORT,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_PROVIDER,
)
from homeassistant.config_entries import SOURCE_USER, ConfigEntryState
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType


def _loaded_entry_with_model(
    hass: HomeAssistant, make_config_entry: Any, make_subentry: Any
) -> Any:
    """Add a loaded workspace with one selectable model profile to HA."""
    provider = make_subentry(
        subentry_id="provider-1",
        subentry_type=SUBENTRY_TYPE_PROVIDER,
        title="Provider",
        data={
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_MODEL_PROFILES: {
                "default": {
                    CONF_ENABLED: True,
                    CONF_NAME: "Configured Model",
                    CONF_MODEL: "configured-model",
                    CONF_STRUCTURED_OUTPUT_SUPPORT: "none",
                    CONF_SUPPORTS_TOOLS: True,
                    CONF_THINKING_SUPPORT: False,
                }
            },
        },
    )
    entry = make_config_entry(
        subentries=(provider,),
        state=ConfigEntryState.LOADED,
    )
    entry.add_to_hass(hass)
    return entry


async def test_conversation_invalid_model_rerenders_init_form(
    hass: HomeAssistant, make_config_entry: Any, make_subentry: Any
) -> None:
    """An unavailable conversation model remains attached to the model field."""
    entry = _loaded_entry_with_model(hass, make_config_entry, make_subentry)
    flow = ConversationSubentryFlowHandler()
    flow.hass = hass
    flow.handler = (entry.entry_id, "conversation")

    flow.context = {"source": SOURCE_USER}
    result = await flow.async_step_user(
        {
            CONF_AGENT_NAME: "Conversation",
            CONF_PRIMARY_MODEL_REF: "provider-1:missing",
        }
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"
    assert result["errors"] == {CONF_PRIMARY_MODEL_REF: "model_profile_not_found"}


async def test_ai_task_invalid_todo_workspace_rerenders_init_form(
    hass: HomeAssistant, make_config_entry: Any, make_subentry: Any
) -> None:
    """A missing todo workspace remains attached to the todo entity field."""
    entry = _loaded_entry_with_model(hass, make_config_entry, make_subentry)
    flow = AITaskDataSubentryFlowHandler()
    flow.hass = hass
    flow.handler = (entry.entry_id, "ai_task_data")

    flow.context = {"source": SOURCE_USER}
    result = await flow.async_step_user(
        {
            CONF_AI_TASK_NAME: "AI Task",
            CONF_PRIMARY_MODEL_REF: "provider-1:default",
            CONF_TODO_LIST_ENTITY_ID: "todo.missing",
        }
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"
    assert result["errors"] == {CONF_TODO_LIST_ENTITY_ID: "todo_list_not_found"}
