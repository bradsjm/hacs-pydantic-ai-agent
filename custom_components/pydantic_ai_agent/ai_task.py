"""AI task support for Pydantic AI Agent."""

import json

import voluptuous as vol

from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_MODEL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import PydanticAIAgentConfigEntry
from .const import SUBENTRY_TYPE_AI_TASK
from .entity import PydanticAIBaseLLMEntity


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: PydanticAIAgentConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI task entities."""
    for subentry_id, subentry in config_entry.subentries.items():
        if subentry.subentry_type != SUBENTRY_TYPE_AI_TASK:
            continue
        async_add_entities(
            [PydanticAIAgentAITaskEntity(config_entry, subentry)],
            config_subentry_id=subentry_id,
        )


class PydanticAIAgentAITaskEntity(
    PydanticAIBaseLLMEntity, ai_task.AITaskEntity
):
    """Pydantic AI data-generation task entity."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supported_features = (
        ai_task.AITaskEntityFeature.GENERATE_DATA
        | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
    )

    def __init__(
        self, entry: PydanticAIAgentConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the AI task entity."""
        super().__init__(entry, subentry, name=subentry.title)

    @property
    def extra_state_attributes(self) -> dict[str, str]:
        """Return observability attributes."""
        return {
            "provider_mode": self.entry.runtime_data.provider_mode,
            "model": self.subentry.data[CONF_MODEL],
        }

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Generate data for an AI task."""
        await self._async_handle_chat_log(
            chat_log,
            structure_name=task.name,
            structure=task.structure,
            max_iterations=1000,
        )

        last_content = chat_log.content[-1]
        if not isinstance(last_content, conversation.AssistantContent):
            raise HomeAssistantError("Provider did not return an assistant response")

        data: object = last_content.content or ""
        if task.structure is not None:
            try:
                data = json.loads(last_content.content or "")
                data = task.structure(data)
            except json.JSONDecodeError as err:
                raise HomeAssistantError(
                    "Provider returned malformed structured data"
                ) from err
            except vol.Invalid as err:
                raise HomeAssistantError(
                    "Provider returned structured data that does not match the schema"
                ) from err

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )
