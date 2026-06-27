"""Conversation support for Pydantic AI Agent."""

from typing import Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import PydanticAIAgentConfigEntry
from .agent._entity_auth import _clear_runtime_auth_failure_for_ref
from .agent.agent_subentries import iter_valid_agent_subentries
from .agent.chat_deltas import _append_text
from .agent.run_state import AgentRunOutcome
from .const import (
    CONF_AGENT_NAME,
    CONF_PROMPT,
    CONF_STREAMING_ENABLED,
    CONF_WEB_FETCH_ENABLED,
    CONF_WEB_SEARCH_ENABLED,
    DOMAIN,
    SUBENTRY_TYPE_CONVERSATION,
)
from .entity import PydanticAIBaseLLMEntity
from .models.model_profiles import model_display_names, model_profile_chain
from .observability.run_failures import _AgentRunFailed
from .virtual_workspace import virtual_workspace_enabled

_TOOL_RESUME_SEPARATOR = "\n\n"
_CONVERSATION_FAILURE_PREFIX = (
    "Sorry, I couldn't get a response from the language model."
)


class _NoAssistantResponseError(HomeAssistantError):
    """Raised when a completed conversation run produced no assistant response."""


async def async_setup_entry(
    _hass: HomeAssistant,
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
        self._streaming_enabled = (
            subentry.data.get(CONF_STREAMING_ENABLED, True) is not False
        )
        self._attr_supports_streaming = self._streaming_enabled
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
            "streaming_enabled": self._streaming_enabled,
            "web_fetch_enabled": bool(
                self.subentry.data.get(CONF_WEB_FETCH_ENABLED, False)
            ),
            "web_search_enabled": bool(
                self.subentry.data.get(CONF_WEB_SEARCH_ENABLED, False)
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

        outcome: object | None = None
        try:
            outcome = await self._async_handle_chat_log(
                chat_log,
                record_success=False,
                stream=self._streaming_enabled,
            )
            result = _async_get_result_from_chat_log(user_input, chat_log)
            if not isinstance(outcome, AgentRunOutcome):
                raise _NoAssistantResponseError(
                    "the provider did not return run metadata."
                )
            _clear_runtime_auth_failure_for_ref(
                self.hass,
                self.entry,
                outcome.provider_subentry_id,
                outcome.model_profile_ref,
            )
            self._record_agent_run_success(outcome, self.entity_id)
            if outcome.run_recorder is not None:
                self._store_run_diagnostics(
                    outcome.run_recorder,
                    status="success",
                    summary={
                        "output": outcome.output,
                        "usage": outcome.usage,
                        "model_profile": outcome.model_profile,
                        "duration": outcome.duration,
                    },
                )
            return result
        except _AgentRunFailed as err:
            return await _failure_result_from_chat_log(
                user_input,
                chat_log,
                self.entity_id,
                err.failure.user_message,
                partial_response=err.failure.partial_response,
            )
        except _NoAssistantResponseError as err:
            self._record_agent_run_failure(err, self.entity_id)
            if (
                isinstance(outcome, AgentRunOutcome)
                and outcome.run_recorder is not None
            ):
                outcome.run_recorder.record(
                    phase="failure",
                    event="no_assistant_response",
                    data={"error": err, "model_profile": outcome.model_profile},
                )
                self._store_run_diagnostics(
                    outcome.run_recorder,
                    status="failed",
                    summary={
                        "error": err,
                        "model_profile": outcome.model_profile,
                        "output": outcome.output,
                    },
                )
            return await _failure_result_from_chat_log(
                user_input, chat_log, self.entity_id, str(err)
            )


def _async_get_result_from_chat_log(
    user_input: conversation.ConversationInput,
    chat_log: conversation.ChatLog,
) -> conversation.ConversationResult:
    """Build the final result from all assistant text emitted this turn."""
    tool_results = [
        content.tool_result
        for content in chat_log.content[chat_log.llm_input_provided_index :]
        if isinstance(content, conversation.ToolResultContent)
        and isinstance(content.tool_result, llm.IntentResponseDict)
    ]

    if tool_results:
        intent_response = tool_results[-1].original
    else:
        intent_response = intent.IntentResponse(language=user_input.language)

    assistant_messages = [
        content
        for content in chat_log.content[chat_log.llm_input_provided_index :]
        if isinstance(content, conversation.AssistantContent)
    ]
    if not assistant_messages:
        raise _NoAssistantResponseError("the provider did not return a response.")

    intent_response.async_set_speech(_merged_assistant_speech(assistant_messages))

    return conversation.ConversationResult(
        response=intent_response,
        conversation_id=chat_log.conversation_id,
        continue_conversation=chat_log.continue_conversation,
    )


async def _failure_result_from_chat_log(
    user_input: conversation.ConversationInput,
    chat_log: conversation.ChatLog,
    agent_id: str,
    reason: str,
    *,
    partial_response: bool = False,
) -> conversation.ConversationResult:
    """Build a user-facing result for a failed provider request."""
    failure_text = f"{_CONVERSATION_FAILURE_PREFIX} {reason}"
    if partial_response:
        failure_text = (
            f"{_TOOL_RESUME_SEPARATOR}Sorry, I couldn't finish the response. {reason}"
        )

    await _append_text(chat_log, agent_id, failure_text)
    result = _async_get_result_from_chat_log(user_input, chat_log)
    return conversation.ConversationResult(
        response=result.response,
        conversation_id=result.conversation_id,
        continue_conversation=False,
    )


def _merged_assistant_speech(
    assistant_messages: list[conversation.AssistantContent],
) -> str:
    """Merge streamed assistant text segments into the final speech string."""
    speech = ""
    for message in assistant_messages:
        content = message.content or ""
        if not content:
            continue
        if not speech:
            speech = content
            continue
        if content.startswith(_TOOL_RESUME_SEPARATOR):
            resumed_content = content.removeprefix(_TOOL_RESUME_SEPARATOR)
            if resumed_content.startswith(speech):
                speech = resumed_content
            else:
                speech += content
            continue
        speech = content
    return speech
