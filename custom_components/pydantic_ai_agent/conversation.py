"""Conversation support for Pydantic AI Agent."""

from typing import Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, intent
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import ulid

from . import PydanticAIAgentConfigEntry
from .const import CONF_AGENT_NAME, CONF_MODEL, DOMAIN, SUBENTRY_TYPE_CONVERSATION


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


class PydanticAIConversationEntity(conversation.ConversationEntity):
    """Pydantic AI conversation agent foundation."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = False

    def __init__(
        self, entry: PydanticAIAgentConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the conversation entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.data[CONF_AGENT_NAME],
            manufacturer="Pydantic AI",
            model=subentry.data[CONF_MODEL],
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if subentry.data.get(CONF_LLM_HASS_API):
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
        }

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process user input with a foundation placeholder response."""
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(
            "Pydantic AI Agent is configured, but provider chat runtime is not implemented yet."
        )
        return conversation.ConversationResult(
            response=response,
            conversation_id=user_input.conversation_id or ulid.ulid_now(),
        )
