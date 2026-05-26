"""AI task support for Pydantic AI Agent."""

import json

import voluptuous as vol

from homeassistant.components import ai_task, conversation
from homeassistant.components.ai_task.const import (
    DEFAULT_SYSTEM_PROMPT,
    DOMAIN as AI_TASK_DOMAIN,
)
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import PydanticAIAgentConfigEntry
from .const import (
    CONF_AI_TASK_NAME,
    CONF_OUTPUT_MODE,
    CONF_TODO_LIST_ENTITY_ID,
    CONF_WEB_FETCH_ENABLED,
    SUBENTRY_TYPE_AI_TASK,
)
from .entity import AgentRunOutcome, PydanticAIBaseLLMEntity
from .ha_todo_tools import TodoWorkspace, todo_workspace_lock
from .metrics import EVENT_STRUCTURED_AI_TASK_OUTPUT_GENERATED, fire_integration_event
from .model_profiles import model_display_names, model_profile_chain
from .structured_output import structured_output_mode


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: PydanticAIAgentConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI task entities."""
    for subentry_id, subentry in config_entry.subentries.items():
        if subentry.subentry_type != SUBENTRY_TYPE_AI_TASK:
            continue
        try:
            model_profile_chain(config_entry, subentry)
        except Exception:
            continue
        async_add_entities(
            [PydanticAIAgentAITaskEntity(config_entry, subentry)],
            config_subentry_id=subentry_id,
        )


class PydanticAIAgentAITaskEntity(PydanticAIBaseLLMEntity, ai_task.AITaskEntity):
    """AI task entity that requests structured output from the model."""

    _attr_has_entity_name = True
    _attr_icon = "mdi:robot-outline"
    _attr_name = None
    _attr_supported_features = (
        ai_task.AITaskEntityFeature.GENERATE_DATA
        | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
    )

    def __init__(
        self, entry: PydanticAIAgentConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the AI task entity."""
        name = str(subentry.data.get(CONF_AI_TASK_NAME, subentry.title))
        super().__init__(
            entry,
            subentry,
            name=name,
            device_name=f"{name} Configuration",
        )

    @property
    def extra_state_attributes(self) -> dict[str, str | bool | list[str] | None]:
        """Return observability attributes."""
        profiles = model_profile_chain(self.entry, self.subentry)
        return {
            "provider_mode": profiles[0].provider_mode,
            "model": profiles[0].model_name,
            "model_profile": profiles[0].title,
            "fallback_model_profiles": model_display_names(profiles[1:]),
            "output_mode": structured_output_mode(
                self.subentry.data.get(CONF_OUTPUT_MODE)
            ),
            "web_fetch_enabled": bool(
                self.subentry.data.get(CONF_WEB_FETCH_ENABLED, False)
            ),
            "todo_workspace_enabled": bool(
                self.subentry.data.get(CONF_TODO_LIST_ENTITY_ID)
            ),
            "ha_tools_enabled": bool(self.subentry.data.get(CONF_LLM_HASS_API)),
            "ha_llm_api": self.subentry.data.get(CONF_LLM_HASS_API),
        }

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Generate task data and validate structured responses when requested."""
        if task.llm_api is None and self.subentry.data.get(CONF_LLM_HASS_API):
            await chat_log.async_provide_llm_data(
                llm.LLMContext(
                    platform=getattr(
                        getattr(self, "platform", None), "domain", AI_TASK_DOMAIN
                    ),
                    context=None,
                    language=None,
                    # Assist entity exposure defaults are keyed to the
                    # conversation assistant, even when AI Task owns the run.
                    assistant=conversation.DOMAIN,
                    device_id=None,
                ),
                user_llm_hass_api=self.subentry.data.get(CONF_LLM_HASS_API),
                user_llm_prompt=DEFAULT_SYSTEM_PROMPT,
            )
        todo_entity_id = self.subentry.data.get(CONF_TODO_LIST_ENTITY_ID)
        if isinstance(todo_entity_id, str) and todo_entity_id:
            workspace = TodoWorkspace(self.hass, todo_entity_id)
            lock = todo_workspace_lock(self.hass, todo_entity_id)
            async with lock:
                await workspace.prepare_run()
                initial_state = await workspace.read_items()
                outcome = await self._async_handle_chat_log(
                    chat_log,
                    structure_name=task.name,
                    structure=task.structure,
                    max_iterations=30,
                    record_success=task.structure is None,
                    extra_toolsets=(workspace.toolset(),),
                    extra_instructions=workspace.instructions(initial_state),
                )
        else:
            outcome = await self._async_handle_chat_log(
                chat_log,
                structure_name=task.name,
                structure=task.structure,
                max_iterations=30,
                record_success=task.structure is None,
            )

        # After all tool calls resolve, ChatLog's final assistant message carries
        # the model output that Home Assistant expects for the task result.
        last_content = chat_log.content[-1]
        if not isinstance(last_content, conversation.AssistantContent):
            raise HomeAssistantError("Provider did not return an assistant response")

        data: object = last_content.content or ""
        if task.structure is not None:
            try:
                # HA receives streamed assistant content for every structured
                # output mode and validates the final JSON before returning it.
                data = json.loads(last_content.content or "")
                data = task.structure(data)
            except json.JSONDecodeError as err:
                self._record_agent_run_failure(err)
                raise HomeAssistantError(
                    "Provider returned malformed structured data"
                ) from err
            except vol.Invalid as err:
                self._record_agent_run_failure(err)
                raise HomeAssistantError(
                    "Provider returned structured data that does not match the schema"
                ) from err
            if not isinstance(outcome, AgentRunOutcome):
                raise HomeAssistantError("Provider did not return run metrics")
            self._record_agent_run_success(outcome)
            fire_integration_event(
                self.hass,
                EVENT_STRUCTURED_AI_TASK_OUTPUT_GENERATED,
                {
                    "config_entry_id": self.entry.entry_id,
                    "subentry_id": self.subentry.subentry_id,
                    "entity_id": self.entity_id,
                    "task_name": task.name,
                },
            )

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )
