"""Conversation support for Pydantic AI Agent."""

from typing import Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import PydanticAIAgentConfigEntry
from .const import (
    CONF_AGENT_NAME,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_WEB_FETCH_ENABLED,
    DOMAIN,
    SUBENTRY_TYPE_CONVERSATION,
)
from .entity import PydanticAIBaseLLMEntity


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: PydanticAIAgentConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    for subentry_id, subentry in config_entry.subentries.items():
        if subentry.subentry_type != SUBENTRY_TYPE_CONVERSATION:
            continue
        async_add_entities(
            [PydanticAIConversationEntity(config_entry, subentry)],
            config_subentry_id=subentry_id,
        )


class PydanticAIConversationEntity(
    PydanticAIBaseLLMEntity, conversation.ConversationEntity
):
    """Conversation entity backed by Pydantic AI direct model streaming."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = False

    def __init__(
        self, entry: PydanticAIAgentConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the conversation entity."""
        super().__init__(entry, subentry, name=subentry.data[CONF_AGENT_NAME])
        if subentry.data.get(CONF_LLM_HASS_API):
            # CONTROL means the agent can call HA tools/services, which is only
            # true when an HA LLM API is attached to this subentry.
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return supported languages."""
        return MATCH_ALL

    @property
    def extra_state_attributes(self) -> dict[str, str | bool | list[str] | None]:
        """Return observability attributes."""
        return {
            "provider_mode": self.entry.runtime_data.provider_mode,
            "model": self.subentry.data[CONF_MODEL],
            "ha_tools_enabled": bool(self.subentry.data.get(CONF_LLM_HASS_API)),
            "ha_llm_api": self.subentry.data.get(CONF_LLM_HASS_API),
            "web_fetch_enabled": bool(
                self.subentry.data.get(CONF_WEB_FETCH_ENABLED, False)
            ),
        }

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

        await self._async_handle_chat_log(chat_log)
        return conversation.async_get_result_from_chat_log(user_input, chat_log)
