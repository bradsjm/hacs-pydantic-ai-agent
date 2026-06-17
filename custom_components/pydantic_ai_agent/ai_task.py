"""AI task support for Pydantic AI Agent."""

import json

import voluptuous as vol
from homeassistant.components import ai_task, conversation
from homeassistant.components.ai_task.const import (
    DEFAULT_SYSTEM_PROMPT,
)
from homeassistant.components.ai_task.const import (
    DOMAIN as AI_TASK_DOMAIN,
)
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import PydanticAIAgentConfigEntry
from .agent._entity_auth import _clear_runtime_auth_failure_for_ref
from .agent.agent_subentries import iter_valid_agent_subentries
from .agent.ha_todo_tools import TodoWorkspace, todo_workspace_lock
from .agent.run_state import AgentRunOutcome
from .const import (
    CONF_AI_TASK_NAME,
    CONF_TODO_LIST_ENTITY_ID,
    CONF_WEB_FETCH_ENABLED,
    CONF_WEB_SEARCH_ENABLED,
    SUBENTRY_TYPE_AI_TASK,
)
from .entity import PydanticAIBaseLLMEntity
from .models.model_profiles import model_display_names, model_profile_chain
from .models.structured_output import resolved_structured_output_mode
from .observability.metrics import (
    EVENT_STRUCTURED_AI_TASK_OUTPUT_GENERATED,
    fire_integration_event,
)
from .virtual_workspace import virtual_workspace_enabled


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: PydanticAIAgentConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI task entities."""
    for valid in iter_valid_agent_subentries(
        config_entry,
        subentry_type=SUBENTRY_TYPE_AI_TASK,
        platform=AI_TASK_DOMAIN,
        resolver=model_profile_chain,
    ):
        async_add_entities(
            [PydanticAIAgentAITaskEntity(config_entry, valid.subentry)],
            config_subentry_id=valid.subentry.subentry_id,
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
        super().__init__(entry, subentry, name=name, device_name=name)
        profiles = model_profile_chain(self.entry, self.subentry)
        self._attr_extra_state_attributes = {
            "provider_mode": profiles[0].provider_mode,
            "model": profiles[0].model_name,
            "model_profile": profiles[0].title,
            "fallback_model_profiles": model_display_names(profiles[1:]),
            "structured_output_mode": resolved_structured_output_mode(profiles[0]),
            "web_fetch_enabled": bool(
                self.subentry.data.get(CONF_WEB_FETCH_ENABLED, False)
            ),
            "web_search_enabled": bool(
                self.subentry.data.get(CONF_WEB_SEARCH_ENABLED, False)
            ),
            "virtual_workspace_enabled": virtual_workspace_enabled(self.subentry.data),
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

        data = await self._async_finalize_structured_output(outcome, task, chat_log)

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )

    async def _async_finalize_structured_output(
        self,
        outcome: object | None,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> object:
        """Validate structured output or return raw assistant content."""
        structured_outcome: AgentRunOutcome | None = None
        if task.structure is not None:
            if not isinstance(outcome, AgentRunOutcome):
                raise HomeAssistantError("Provider did not return run metrics")
            structured_outcome = outcome
            _clear_runtime_auth_failure_for_ref(
                self.hass,
                self.entry,
                outcome.provider_subentry_id,
                outcome.model_profile_ref,
            )

        last_content = chat_log.content[-1]
        if not isinstance(last_content, conversation.AssistantContent):
            raise HomeAssistantError("Provider did not return an assistant response")

        data: object = last_content.content or ""
        if task.structure is not None:
            assert structured_outcome is not None
            data = await self._async_validate_structured_data(
                structured_outcome, last_content, task.structure
            )
            if structured_outcome.run_recorder is not None:
                structured_outcome.run_recorder.record(
                    phase="output_validation",
                    event="structured_output_validated",
                    data={"validated_data": data},
                )
                self._store_run_diagnostics(
                    structured_outcome.run_recorder,
                    status="success",
                    summary={
                        "output": structured_outcome.output,
                        "validated_data": data,
                        "usage": structured_outcome.usage,
                        "model_profile": structured_outcome.model_profile,
                        "duration": structured_outcome.duration,
                    },
                )
            self._record_agent_run_success(structured_outcome)
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

        return data

    async def _async_validate_structured_data(
        self,
        structured_outcome: AgentRunOutcome,
        last_content: conversation.AssistantContent,
        structure: vol.Schema,
    ) -> object:
        """Validate structured data from model output against the schema."""
        content_str = last_content.content or ""
        try:
            data = json.loads(content_str)
            data = structure(data)
        except json.JSONDecodeError as err:
            self._record_agent_run_failure(err)
            if structured_outcome.run_recorder:
                structured_outcome.run_recorder.record(
                    phase="output_validation",
                    event="json_decode_failed",
                    data={"error": err, "content": content_str},
                )
                self._store_run_diagnostics(
                    structured_outcome.run_recorder,
                    status="failed",
                    summary={
                        "error": err,
                        "model_profile": structured_outcome.model_profile,
                        "output": structured_outcome.output,
                    },
                )
            raise HomeAssistantError(
                "Provider returned malformed structured data"
            ) from err
        except vol.Invalid as err:
            self._record_agent_run_failure(err)
            if structured_outcome.run_recorder:
                structured_outcome.run_recorder.record(
                    phase="output_validation",
                    event="schema_validation_failed",
                    data={"error": err, "content": content_str},
                )
                self._store_run_diagnostics(
                    structured_outcome.run_recorder,
                    status="failed",
                    summary={
                        "error": err,
                        "model_profile": structured_outcome.model_profile,
                        "output": structured_outcome.output,
                    },
                )
            raise HomeAssistantError(
                "Provider returned structured data that does not match the schema"
            ) from err
        return data
