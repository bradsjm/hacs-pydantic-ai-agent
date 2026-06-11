"""Conversation support for Pydantic AI Agent."""

from typing import Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import PydanticAIAgentConfigEntry
from .agent_subentries import iter_valid_agent_subentries
from .const import (
    CONF_AGENT_NAME,
    CONF_PROMPT,
    CONF_WEB_FETCH_ENABLED,
    DOMAIN,
    SUBENTRY_TYPE_CONVERSATION,
)
from .entity import PydanticAIBaseLLMEntity
from .model_profiles import model_display_names, model_profile_chain
from .virtual_workspace import virtual_workspace_enabled


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: PydanticAIAgentConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    for valid in iter_valid_agent_subentries(
        config_entry,
        subentry_type=SUBENTRY_TYPE_CONVERSATION,
        platform=conversation.DOMAIN,
        resolver=model_profile_chain,
    ):
        async_add_entities(
            [PydanticAIConversationEntity(config_entry, valid.subentry)],
            config_subentry_id=valid.subentry.subentry_id,
        )


class PydanticAIConversationEntity(
    PydanticAIBaseLLMEntity, conversation.ConversationEntity
):
    """Conversation entity backed by Pydantic AI direct model streaming."""

    _attr_has_entity_name = True
    _attr_icon = "mdi:message-processing-outline"
    _attr_name = None

    def __init__(
        self, entry: PydanticAIAgentConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the conversation entity."""
        name = subentry.data[CONF_AGENT_NAME]
        super().__init__(entry, subentry, name=name, device_name=name)
        self._attr_supports_streaming = True
        if subentry.data.get(CONF_LLM_HASS_API):
            # CONTROL means the agent can call HA tools/services, which is only
            # true when an HA LLM API is attached to this subentry.
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )
        profiles = model_profile_chain(self.entry, self.subentry)
        self._attr_extra_state_attributes = {
            "provider_mode": profiles[0].provider_mode,
            "model": profiles[0].model_name,
            "model_profile": profiles[0].title,
            "fallback_model_profiles": model_display_names(profiles[1:]),
            "ha_tools_enabled": bool(self.subentry.data.get(CONF_LLM_HASS_API)),
            "ha_llm_api": self.subentry.data.get(CONF_LLM_HASS_API),
            "web_fetch_enabled": bool(
                self.subentry.data.get(CONF_WEB_FETCH_ENABLED, False)
            ),
            "virtual_workspace_enabled": virtual_workspace_enabled(self.subentry.data),
        }

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return supported languages."""
        return MATCH_ALL

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Prepare HA LLM context and stream the model response into ChatLog."""
        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                self.subentry.data.get(CONF_LLM_HASS_API),
                self.subentry.data.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        await self._async_handle_chat_log(
            chat_log,
            stream=True,
        )
        return conversation.async_get_result_from_chat_log(user_input, chat_log)
