"""Test Pydantic AI Agent conversation runtime execution."""

from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import AsyncMock, patch

from custom_components.pydantic_ai_agent.const import (
    CONF_MAX_ITERATIONS,
    CONF_THINKING,
    PROVIDER_ANTHROPIC,
)
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from pydantic_ai import (
    ModelMessage,
    ModelResponse,
    TextPart,
)
from pydantic_ai.capabilities import WebFetch
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from tests.components.pydantic_ai_agent.support.builders import (
    provider_runtime_data,
    provider_subentry_data,
)
from tests.components.pydantic_ai_agent.support.pydantic_ai import (
    Agent as _Agent,
)
from tests.components.pydantic_ai_agent.support.pydantic_ai import (
    request_limit_from_kwargs,
)
from tests.components.pydantic_ai_agent.support.runtime import (
    assert_has_context_management_capability,
    first_non_default_conversation_entity_id,
    loaded_conversation_entry,
    thinking_capabilities,
)

_PROVIDER_SUBENTRY_ID = "provider-1"
_MODEL_PROFILE_ID = "model-profile-1"
_MODEL_PROFILE_REF = f"{_PROVIDER_SUBENTRY_ID}:{_MODEL_PROFILE_ID}"


async def test_conversation_runtime_uses_configured_max_iterations(
    hass: HomeAssistant,
    mock_probe_model: AsyncMock,
    mock_chat_model_for_profile: TestModel,
) -> None:
    """Test conversation runs use the configured run iteration limit."""
    del mock_probe_model, mock_chat_model_for_profile
    entry = loaded_conversation_entry(
        extra_data={CONF_MAX_ITERATIONS: 24},
        provider_subentry=provider_subentry_data(
            subentry_id=_PROVIDER_SUBENTRY_ID,
            title="Hosted OpenAI",
            profile_id=_MODEL_PROFILE_ID,
            provider_mode=PROVIDER_ANTHROPIC,
            model="claude-sonnet-4",
        ),
        provider_runtime=provider_runtime_data(
            subentry_id=_PROVIDER_SUBENTRY_ID,
            name="Hosted OpenAI",
            provider_mode=PROVIDER_ANTHROPIC,
        ),
    )
    entry.add_to_hass(hass)
    agent = _Agent()

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    entity_id = first_non_default_conversation_entity_id(hass)
    with patch("custom_components.pydantic_ai_agent.entity.Agent", return_value=agent):
        await conversation.async_converse(
            hass,
            "hello",
            None,
            Context(),
            agent_id=entity_id,
        )

    assert request_limit_from_kwargs(agent.run_kwargs) == 24


async def test_conversation_runtime_uses_thinking_capability(
    hass: HomeAssistant,
    mock_probe_model: AsyncMock,
    mock_chat_model_for_profile: TestModel,
) -> None:
    """Test configured thinking is represented as a Pydantic AI capability."""
    del mock_probe_model, mock_chat_model_for_profile
    entry = loaded_conversation_entry(
        extra_data={CONF_THINKING: "high"},
        provider_subentry=provider_subentry_data(
            subentry_id=_PROVIDER_SUBENTRY_ID,
            title="Hosted OpenAI",
            profile_id=_MODEL_PROFILE_ID,
            provider_mode=PROVIDER_ANTHROPIC,
            model="claude-sonnet-4",
        ),
        provider_runtime=provider_runtime_data(
            subentry_id=_PROVIDER_SUBENTRY_ID,
            name="Hosted OpenAI",
            provider_mode=PROVIDER_ANTHROPIC,
        ),
    )
    entry.add_to_hass(hass)
    agent = _Agent()

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    entity_id = first_non_default_conversation_entity_id(hass)
    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            return_value=agent,
        ) as agent_class,
    ):
        await conversation.async_converse(
            hass,
            "hello",
            None,
            Context(),
            agent_id=entity_id,
        )

    capabilities = agent_class.call_args.kwargs["capabilities"]
    assert_has_context_management_capability(capabilities)
    thinking = thinking_capabilities(capabilities)
    assert len(thinking) == 1
    assert thinking[0].effort == "high"


async def test_conversation_runtime_keeps_explicit_disabled_thinking_capability(
    hass: HomeAssistant,
    mock_probe_model: AsyncMock,
    mock_chat_model_for_profile: TestModel,
) -> None:
    """Test explicit disabled thinking is still passed as a capability."""
    del mock_probe_model, mock_chat_model_for_profile
    entry = loaded_conversation_entry(
        extra_data={CONF_THINKING: False},
        provider_subentry=provider_subentry_data(
            subentry_id=_PROVIDER_SUBENTRY_ID,
            title="Hosted OpenAI",
            profile_id=_MODEL_PROFILE_ID,
            provider_mode=PROVIDER_ANTHROPIC,
            model="claude-sonnet-4",
        ),
        provider_runtime=provider_runtime_data(
            subentry_id=_PROVIDER_SUBENTRY_ID,
            name="Hosted OpenAI",
            provider_mode=PROVIDER_ANTHROPIC,
        ),
    )
    entry.add_to_hass(hass)
    agent = _Agent()

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    entity_id = first_non_default_conversation_entity_id(hass)
    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            return_value=agent,
        ) as agent_class,
    ):
        await conversation.async_converse(
            hass,
            "hello",
            None,
            Context(),
            agent_id=entity_id,
        )

    capabilities = agent_class.call_args.kwargs["capabilities"]
    thinking = thinking_capabilities(capabilities)
    assert len(thinking) == 1
    assert thinking[0].effort is False


async def test_conversation_runtime_supports_test_model_without_patching_agent_run(
    hass: HomeAssistant,
    mock_probe_model: AsyncMock,
) -> None:
    """Test a deterministic Pydantic AI TestModel runtime path."""
    del mock_probe_model
    entry = loaded_conversation_entry()
    entry.add_to_hass(hass)

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    entity_id = first_non_default_conversation_entity_id(hass)
    with patch(
        "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
        return_value=TestModel(custom_output_text="test model response"),
    ):
        result = await conversation.async_converse(
            hass,
            "hello",
            None,
            Context(),
            agent_id=entity_id,
        )

    assert result.response.speech["plain"]["speech"] == "test model response"


async def test_conversation_runtime_supports_function_model_without_patching_agent_run(
    hass: HomeAssistant,
    mock_probe_model: AsyncMock,
) -> None:
    """Test a deterministic Pydantic AI FunctionModel runtime path."""
    del mock_probe_model
    entry = loaded_conversation_entry()
    entry.add_to_hass(hass)
    captured_messages: list[ModelResponse] = []

    async def model_function(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        captured_messages.extend(cast(list[ModelResponse], messages))
        assert info.function_tools == []
        return ModelResponse(parts=[TextPart(content="function model response")])

    async def stream_function(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str]:
        captured_messages.extend(cast(list[ModelResponse], messages))
        assert info.function_tools == []
        yield "function model response"

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    entity_id = first_non_default_conversation_entity_id(hass)
    with patch(
        "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
        return_value=FunctionModel(model_function, stream_function=stream_function),
    ):
        result = await conversation.async_converse(
            hass,
            "hello from function model",
            None,
            Context(),
            agent_id=entity_id,
        )

    assert result.response.speech["plain"]["speech"] == "function model response"
    assert captured_messages
    assert "hello from function model" in repr(captured_messages[-1])


async def test_conversation_runtime_defaults_max_iterations(
    hass: HomeAssistant,
    mock_probe_model: AsyncMock,
    mock_chat_model_for_profile: TestModel,
) -> None:
    """Test conversation runs keep the default iteration limit when unset."""
    del mock_probe_model, mock_chat_model_for_profile
    entry = loaded_conversation_entry()
    entry.add_to_hass(hass)
    agent = _Agent()

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    entity_id = first_non_default_conversation_entity_id(hass)
    with patch("custom_components.pydantic_ai_agent.entity.Agent", return_value=agent):
        await conversation.async_converse(
            hass,
            "hello",
            None,
            Context(),
            agent_id=entity_id,
        )

    assert request_limit_from_kwargs(agent.run_kwargs) == 10


async def test_conversation_runtime_passes_selected_skills_capabilities(
    hass: HomeAssistant,
    mock_probe_model: AsyncMock,
    mock_chat_model_for_profile: TestModel,
) -> None:
    """Test selected conversation skills become Agent capabilities."""
    del mock_probe_model, mock_chat_model_for_profile
    entry = loaded_conversation_entry(skills=["kitchen-skill"])
    entry.add_to_hass(hass)
    capability = {"skill": "kitchen-skill"}
    agent = _Agent()

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    entity_id = first_non_default_conversation_entity_id(hass)
    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.async_skills_capabilities",
            new_callable=AsyncMock,
            return_value=[capability],
        ) as skills_capabilities,
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            return_value=agent,
        ) as agent_class,
    ):
        await conversation.async_converse(
            hass,
            "hello",
            None,
            Context(),
            agent_id=entity_id,
        )

    assert skills_capabilities.call_args.args[0] is hass
    assert skills_capabilities.call_args.args[1] is entry
    assert skills_capabilities.call_args.args[2] == ["kitchen-skill"]
    capabilities = agent_class.call_args.kwargs["capabilities"]
    assert capability in capabilities
    assert_has_context_management_capability(capabilities)
    assert agent.run_stream_events_calls == 1
    assert agent.run_calls == 0


async def test_conversation_runtime_adds_web_fetch_capability(
    hass: HomeAssistant,
    mock_probe_model: AsyncMock,
    mock_chat_model_for_profile: TestModel,
) -> None:
    """Test WebFetch-enabled conversation agents get the WebFetch capability."""
    del mock_probe_model, mock_chat_model_for_profile
    entry = loaded_conversation_entry(web_fetch_enabled=True)
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    entity_id = first_non_default_conversation_entity_id(hass)
    with patch(
        "custom_components.pydantic_ai_agent.entity.Agent",
        return_value=_Agent(),
    ) as agent_class:
        await conversation.async_converse(
            hass,
            "fetch https://example.com",
            None,
            Context(),
            agent_id=entity_id,
        )

    assert agent_class.call_args is not None
    capabilities = agent_class.call_args.kwargs["capabilities"]
    assert any(isinstance(capability, WebFetch) for capability in capabilities)
    assert_has_context_management_capability(capabilities)
