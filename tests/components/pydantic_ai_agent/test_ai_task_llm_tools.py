"""Test AI task Home Assistant LLM API tool support."""

from typing import Any, cast

from custom_components.pydantic_ai_agent.ai_task import PydanticAIAgentAITaskEntity
from custom_components.pydantic_ai_agent.const import (
    SUBENTRY_TYPE_AI_TASK,
)
from homeassistant.components import ai_task, conversation
from homeassistant.components.ai_task.const import (
    DEFAULT_SYSTEM_PROMPT,
)
from homeassistant.components.ai_task.const import (
    DOMAIN as AI_TASK_DOMAIN,
)
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.helpers import llm
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from tests.components.pydantic_ai_agent.support.builders import (
    ai_task_subentry_data,
    provider_subentry_data,
    workspace_entry,
)

_PROVIDER_SUBENTRY_ID = "provider"
_MODEL_PROFILE_ID = "primary"
_MODEL_PROFILE_REF = f"{_PROVIDER_SUBENTRY_ID}:{_MODEL_PROFILE_ID}"


class _ChatLog:
    """Minimal ChatLog test double."""

    def __init__(self) -> None:
        """Initialize recorded LLM data calls."""
        self.content: list[object] = [
            conversation.SystemContent("system"),
            conversation.UserContent("user"),
        ]
        self.conversation_id = "ai-task-test"
        self.provide_calls: list[dict[str, object]] = []

    async def async_provide_llm_data(
        self,
        llm_context: llm.LLMContext,
        user_llm_hass_api: str | list[str] | llm.API | None = None,
        user_llm_prompt: str | None = None,
        user_extra_system_prompt: str | None = None,
    ) -> None:
        """Record supplied LLM data."""
        self.provide_calls.append(
            {
                "llm_context": llm_context,
                "user_llm_hass_api": user_llm_hass_api,
                "user_llm_prompt": user_llm_prompt,
                "user_extra_system_prompt": user_extra_system_prompt,
            }
        )


def _entry(llm_hass_api: list[str] | None = None) -> MockConfigEntry:
    """Return a config entry with one provider and one AI task subentry."""
    extra_data: dict[str, object] = {}
    if llm_hass_api is not None:
        extra_data[CONF_LLM_HASS_API] = llm_hass_api
    return workspace_entry(
        (
            provider_subentry_data(
                subentry_id=_PROVIDER_SUBENTRY_ID,
                title="Provider",
                profile_id=_MODEL_PROFILE_ID,
                profile_name="Primary",
            ),
            ai_task_subentry_data(
                _MODEL_PROFILE_REF,
                title="Report task",
                task_name="Report task",
                extra_data=extra_data,
            ),
        )
    )


def _ai_task_entity(
    llm_hass_api: list[str] | None = None,
) -> PydanticAIAgentAITaskEntity:
    """Return an AI task entity for the configured API selection."""
    entry = _entry(llm_hass_api)
    subentry = next(
        subentry
        for subentry in entry.subentries.values()
        if subentry.subentry_type == SUBENTRY_TYPE_AI_TASK
    )
    return PydanticAIAgentAITaskEntity(entry, subentry)


async def _fake_handle_chat_log(
    _self: PydanticAIAgentAITaskEntity,
    chat_log: _ChatLog,
    **_kwargs: Any,
) -> str:
    """Append the assistant response expected by AI task result handling."""
    chat_log.content.append(conversation.AssistantContent("agent", "plain result"))
    return "plain result"


def test_ai_task_entity_observability_attributes_include_ha_tools() -> None:
    """Test AI task state attributes expose configured HA LLM APIs."""
    entity = _ai_task_entity([llm.LLM_API_ASSIST])
    attributes = entity.extra_state_attributes

    assert attributes is not None
    assert attributes["ha_tools_enabled"] is True
    assert attributes["ha_llm_api"] == [llm.LLM_API_ASSIST]


def test_ai_task_entity_observability_attributes_without_ha_tools() -> None:
    """Test AI task state attributes report missing HA LLM API selection."""
    entity = _ai_task_entity()
    attributes = entity.extra_state_attributes

    assert attributes is not None
    assert attributes["ha_tools_enabled"] is False
    assert attributes["ha_llm_api"] is None


async def test_ai_task_uses_configured_llm_api_when_task_has_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test AI task subentry LLM API is provided when the call has no API."""
    monkeypatch.setattr(
        PydanticAIAgentAITaskEntity,
        "_async_handle_chat_log",
        _fake_handle_chat_log,
    )
    entity = _ai_task_entity([llm.LLM_API_ASSIST])
    chat_log = _ChatLog()

    result = await entity._async_generate_data(
        ai_task.GenDataTask("Plain task", "Generate text"),
        cast(conversation.ChatLog, chat_log),
    )

    assert result.data == "plain result"
    assert len(chat_log.provide_calls) == 1
    call = chat_log.provide_calls[0]
    assert call["user_llm_hass_api"] == [llm.LLM_API_ASSIST]
    assert call["user_llm_prompt"] == DEFAULT_SYSTEM_PROMPT
    llm_context = cast(llm.LLMContext, call["llm_context"])
    assert llm_context.platform == AI_TASK_DOMAIN
    assert llm_context.assistant == conversation.DOMAIN


async def test_ai_task_preserves_per_call_llm_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test caller-provided AI task LLM API is not overwritten by subentry config."""
    monkeypatch.setattr(
        PydanticAIAgentAITaskEntity,
        "_async_handle_chat_log",
        _fake_handle_chat_log,
    )
    entity = _ai_task_entity([llm.LLM_API_ASSIST])
    chat_log = _ChatLog()

    result = await entity._async_generate_data(
        ai_task.GenDataTask(
            "Plain task",
            "Generate text",
            llm_api=cast(llm.API, object()),
        ),
        cast(conversation.ChatLog, chat_log),
    )

    assert result.data == "plain result"
    assert chat_log.provide_calls == []
