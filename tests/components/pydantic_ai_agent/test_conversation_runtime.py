"""Test Pydantic AI Agent conversation runtime execution."""

from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import AsyncMock, patch

from custom_components.pydantic_ai_agent.const import (
    CONF_MAX_ITERATIONS,
    CONF_THINKING,
)
from custom_components.pydantic_ai_agent.context_management import (
    SlidingWindowContextCapability,
)
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from pydantic_ai import (
    ModelMessage,
    ModelResponse,
    TextPart,
)
from pydantic_ai.capabilities import Thinking, WebFetch
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pytest_homeassistant_custom_component.common import MockConfigEntry
from tests.components.pydantic_ai_agent.support.builders import (
    conversation_subentry_data,
    provider_runtime_data,
    provider_subentry_data,
    workspace_entry,
    workspace_runtime_data,
)
from tests.components.pydantic_ai_agent.support.pydantic_ai import (
    Agent as _Agent,
)
from tests.components.pydantic_ai_agent.support.pydantic_ai import (
    Usage,
    request_limit_from_kwargs,
)

_PROVIDER_SUBENTRY_ID = "provider-1"
_MODEL_PROFILE_ID = "model-profile-1"
_MODEL_PROFILE_REF = f"{_PROVIDER_SUBENTRY_ID}:{_MODEL_PROFILE_ID}"


def _entry(
    llm_hass_api: list[str] | None = None,
    skills: list[str] | None = None,
    *,
    virtual_workspace_enabled: bool = False,
    web_fetch_enabled: bool = False,
    extra_data: dict[str, object] | None = None,
) -> MockConfigEntry:
    """Return a config entry with one conversation subentry."""
    entry = workspace_entry(
        (
            conversation_subentry_data(
                _MODEL_PROFILE_REF,
                llm_hass_api=llm_hass_api,
                skills=skills,
                virtual_workspace_enabled=virtual_workspace_enabled,
                web_fetch_enabled=web_fetch_enabled,
                extra_data=extra_data,
            ),
            provider_subentry_data(
                subentry_id=_PROVIDER_SUBENTRY_ID,
                title="Hosted OpenAI",
                profile_id=_MODEL_PROFILE_ID,
            ),
        )
    )
    entry.runtime_data = workspace_runtime_data(
        providers={
            _PROVIDER_SUBENTRY_ID: provider_runtime_data(
                subentry_id=_PROVIDER_SUBENTRY_ID, name="Hosted OpenAI"
            )
        },
    )
    return entry


class _ResultWithMessages:
    """Minimal Agent result with explicit final messages."""

    def __init__(self, output: str, messages: list[ModelResponse]) -> None:
        """Initialize the result."""
        self.output = output
        self.usage = Usage()
        self._messages = messages

    def new_messages(self) -> list[ModelResponse]:
        """Return final Agent messages."""
        return self._messages


class _Span:
    """Synchronous context manager returned by the Logfire span mock."""

    def __init__(self) -> None:
        """Initialize recorded attributes."""
        self.attributes: dict[str, int] = {}

    def __enter__(self) -> None:
        """Enter the mocked span."""

    def __exit__(self, *_args: object) -> None:
        """Exit the mocked span."""

    def set_attributes(self, attributes: dict[str, int]) -> None:
        """Record attributes set on the mocked span."""
        self.attributes.update(attributes)


def _assert_context_management_capability(capabilities: list[object]) -> None:
    """Assert the automatic context management capability is attached."""
    assert any(
        isinstance(capability, SlidingWindowContextCapability)
        for capability in capabilities
    )


def _thinking_capabilities(capabilities: list[object]) -> list[Thinking]:
    """Return Thinking capabilities from an Agent constructor call."""
    return [
        capability for capability in capabilities if isinstance(capability, Thinking)
    ]


async def test_conversation_runtime_uses_configured_max_iterations(
    hass: HomeAssistant,
) -> None:
    """Test conversation runs use the configured run iteration limit."""
    entry = _entry(extra_data={CONF_MAX_ITERATIONS: 24})
    entry.add_to_hass(hass)
    agent = _Agent()

    with patch(
        "custom_components.pydantic_ai_agent._model_validation.async_probe_model",
        new_callable=AsyncMock,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    entity_id = next(
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id != "conversation.home_assistant"
    )
    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            return_value=agent,
        ),
    ):
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
) -> None:
    """Test configured thinking is represented as a Pydantic AI capability."""
    entry = _entry(extra_data={CONF_THINKING: "high"})
    entry.add_to_hass(hass)
    agent = _Agent()

    with patch(
        "custom_components.pydantic_ai_agent._model_validation.async_probe_model",
        new_callable=AsyncMock,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    entity_id = next(
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id != "conversation.home_assistant"
    )
    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
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
    _assert_context_management_capability(capabilities)
    thinking = _thinking_capabilities(capabilities)
    assert len(thinking) == 1
    assert thinking[0].effort == "high"


async def test_conversation_runtime_keeps_explicit_disabled_thinking_capability(
    hass: HomeAssistant,
) -> None:
    """Test explicit disabled thinking is still passed as a capability."""
    entry = _entry(extra_data={CONF_THINKING: False})
    entry.add_to_hass(hass)
    agent = _Agent()

    with patch(
        "custom_components.pydantic_ai_agent._model_validation.async_probe_model",
        new_callable=AsyncMock,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    entity_id = next(
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id != "conversation.home_assistant"
    )
    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
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
    thinking = _thinking_capabilities(capabilities)
    assert len(thinking) == 1
    assert thinking[0].effort is False


async def test_conversation_runtime_supports_test_model_without_patching_agent_run(
    hass: HomeAssistant,
) -> None:
    """Test a deterministic Pydantic AI TestModel runtime path."""
    entry = _entry()
    entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent._model_validation.async_probe_model",
        new_callable=AsyncMock,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    entity_id = next(
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id != "conversation.home_assistant"
    )
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
) -> None:
    """Test a deterministic Pydantic AI FunctionModel runtime path."""
    entry = _entry()
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

    with patch(
        "custom_components.pydantic_ai_agent._model_validation.async_probe_model",
        new_callable=AsyncMock,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    entity_id = next(
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id != "conversation.home_assistant"
    )
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
) -> None:
    """Test conversation runs keep the default iteration limit when unset."""
    entry = _entry()
    entry.add_to_hass(hass)
    agent = _Agent()

    with patch(
        "custom_components.pydantic_ai_agent._model_validation.async_probe_model",
        new_callable=AsyncMock,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    entity_id = next(
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id != "conversation.home_assistant"
    )
    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            return_value=agent,
        ),
    ):
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
) -> None:
    """Test selected conversation skills become Agent capabilities."""
    entry = _entry(skills=["kitchen-skill"])
    entry.add_to_hass(hass)
    capability = object()
    agent = _Agent()

    with patch(
        "custom_components.pydantic_ai_agent._model_validation.async_probe_model",
        new_callable=AsyncMock,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    entity_id = next(
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id != "conversation.home_assistant"
    )
    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
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
    _assert_context_management_capability(capabilities)
    assert agent.run_stream_events_calls == 1
    assert agent.run_calls == 0


async def test_conversation_runtime_adds_web_fetch_capability(
    hass: HomeAssistant,
) -> None:
    """Test WebFetch-enabled conversation agents get the WebFetch capability."""
    entry = _entry(web_fetch_enabled=True)
    entry.add_to_hass(hass)
    with patch(
        "custom_components.pydantic_ai_agent._model_validation.async_probe_model",
        new_callable=AsyncMock,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    entity_id = next(
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id != "conversation.home_assistant"
    )
    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            return_value=_Agent(),
        ) as agent_class,
    ):
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
    _assert_context_management_capability(capabilities)
