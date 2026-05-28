"""Test Pydantic AI Agent conversation entities."""

from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_NAME, __version__
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.entity import Entity
from homeassistant.helpers import llm
from homeassistant.util import slugify
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pydantic_ai import (
    FunctionToolset,
    ModelRequest,
    ModelResponse,
    PartStartEvent,
    TextPart,
)
from pydantic_ai.capabilities import Thinking, ToolSearch
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.exceptions import UsageLimitExceeded

from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_FALLBACK_MODEL_REFS,
    CONF_LOGFIRE_INCLUDE_CONTENT,
    CONF_LOGFIRE_TOKEN,
    CONF_MAX_ITERATIONS,
    CONF_THINKING,
    CONF_VIRTUAL_WORKSPACE_ENABLED,
    DOMAIN,
    OUTPUT_MODE_TOOL,
)
from custom_components.pydantic_ai_agent.context_management import (
    SlidingWindowContextCapability,
)
from custom_components.pydantic_ai_agent.conversation import (
    PydanticAIConversationEntity,
    async_setup_entry,
)
from custom_components.pydantic_ai_agent.entity import unique_id_for_subentry_entity
from custom_components.pydantic_ai_agent.metrics import (
    EVENT_AGENT_RUN_COMPLETED,
    EVENT_AGENT_RUN_FAILED,
)
from tests.components.pydantic_ai_agent.support.builders import (
    conversation_subentry_data,
    model_profile_data,
    provider_runtime_data,
    provider_subentry_data,
    workspace_entry,
    workspace_runtime_data,
)
from tests.components.pydantic_ai_agent.support.pydantic_ai import (
    ConversationAgent as _Agent,
)

_PROVIDER_SUBENTRY_ID = "provider-1"
_MODEL_PROFILE_ID = "model-profile-1"
_MODEL_PROFILE_REF = f"{_PROVIDER_SUBENTRY_ID}:{_MODEL_PROFILE_ID}"


def _entry(
    llm_hass_api: list[str] | None,
    skills: list[str] | None = None,
    *,
    mcp_server_ids: list[str] | None = None,
    virtual_workspace_enabled: bool = False,
    web_fetch_enabled: bool = False,
    model_settings: dict[str, object] | None = None,
    extra_data: dict[str, object] | None = None,
) -> MockConfigEntry:
    """Return a config entry with one conversation subentry."""
    entry = workspace_entry(
        (
            conversation_subentry_data(
                _MODEL_PROFILE_REF,
                llm_hass_api=llm_hass_api,
                skills=skills,
                mcp_server_ids=mcp_server_ids,
                virtual_workspace_enabled=virtual_workspace_enabled,
                web_fetch_enabled=web_fetch_enabled,
                extra_data=extra_data,
            ),
            provider_subentry_data(
                subentry_id=_PROVIDER_SUBENTRY_ID,
                title="Hosted OpenAI",
                profile_id=_MODEL_PROFILE_ID,
                model_settings=model_settings,
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


def _entry_with_conversation_subentries(*, logfire: bool = False) -> MockConfigEntry:
    """Return a config entry with two conversation subentries."""
    data: dict[str, object] = {CONF_NAME: "Workspace"}
    if logfire:
        data[CONF_LOGFIRE_TOKEN] = "lf-token"
        data[CONF_LOGFIRE_INCLUDE_CONTENT] = True
    kitchen_profile_id = "kitchen-model"
    garage_profile_id = "garage-model"
    entry = workspace_entry(
        (
            conversation_subentry_data(
                f"{_PROVIDER_SUBENTRY_ID}:{kitchen_profile_id}",
                title="Kitchen Agent",
                agent_name="Kitchen Agent",
            ),
            conversation_subentry_data(
                f"{_PROVIDER_SUBENTRY_ID}:{garage_profile_id}",
                title="Garage Agent",
                agent_name="Garage Agent",
            ),
            provider_subentry_data(
                subentry_id=_PROVIDER_SUBENTRY_ID,
                title="Hosted OpenAI",
                model_profiles={
                    kitchen_profile_id: model_profile_data(
                        profile_id=kitchen_profile_id,
                        name="Kitchen Model",
                        model="gpt-kitchen",
                    ),
                    garage_profile_id: model_profile_data(
                        profile_id=garage_profile_id,
                        name="Garage Model",
                        model="gpt-garage",
                    ),
                },
                default_model_profile_id=kitchen_profile_id,
            ),
        ),
        data=data,
    )
    entry.runtime_data = workspace_runtime_data(
        providers={
            _PROVIDER_SUBENTRY_ID: provider_runtime_data(
                subentry_id=_PROVIDER_SUBENTRY_ID, name="Hosted OpenAI"
            )
        },
        logfire_enabled=logfire,
        logfire_include_content=logfire,
    )
    return entry


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


class _FailingSetAttributesSpan(_Span):
    """Span that fails when usage attributes are copied."""

    def set_attributes(self, attributes: dict[str, int]) -> None:
        """Raise while setting attributes."""
        del attributes
        raise RuntimeError("set attributes failed")


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


def _state(hass: HomeAssistant, entity_id: str) -> str:
    """Return a state value for an expected entity."""
    state = hass.states.get(entity_id)
    assert state is not None
    return state.state


def _enable_diagnostic_entities(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    subentry: ConfigSubentry,
    entity_domain: str,
    keys: tuple[str, ...],
) -> None:
    """Pre-enable diagnostic entities that default disabled."""
    entity_registry = er.async_get(hass)
    name = str(subentry.data[CONF_AGENT_NAME])
    for key in keys:
        entity_registry.async_get_or_create(
            entity_domain,
            DOMAIN,
            unique_id_for_subentry_entity(entry, subentry, key),
            suggested_object_id=slugify(f"{name} {key}"),
        )


def test_conversation_entity_controls_home_assistant_with_llm_api() -> None:
    """Test LLM API selection enables Home Assistant control support."""
    entry = _entry([llm.LLM_API_ASSIST])
    subentry = next(iter(entry.subentries.values()))

    entity = PydanticAIConversationEntity(entry, subentry)

    assert entity.supported_features == conversation.ConversationEntityFeature.CONTROL
    assert entity.extra_state_attributes["ha_tools_enabled"] is True
    assert entity.extra_state_attributes["ha_llm_api"] == [llm.LLM_API_ASSIST]
    assert entity.extra_state_attributes["web_fetch_enabled"] is False
    assert entity.extra_state_attributes["virtual_workspace_enabled"] is False


def test_conversation_entity_advertises_streaming() -> None:
    """Test Agent-backed conversation entities advertise streaming."""
    entry = _entry(None)
    subentry = next(iter(entry.subentries.values()))

    entity = PydanticAIConversationEntity(entry, subentry)

    assert entity.supports_streaming is True


@pytest.mark.parametrize(
    (
        "llm_hass_api",
        "mcp_server_ids",
        "virtual_workspace_enabled",
        "web_fetch_enabled",
        "skills",
        "supported_features",
    ),
    [
        (
            [llm.LLM_API_ASSIST],
            None,
            False,
            False,
            None,
            conversation.ConversationEntityFeature.CONTROL,
        ),
        (None, ["mcp-server-1"], False, False, None, 0),
        (None, None, True, False, None, 0),
        (None, None, False, True, None, 0),
        (None, None, False, False, ["kitchen-skill"], 0),
    ],
)
def test_conversation_entity_advertises_streaming_for_tool_sources(
    llm_hass_api: list[str] | None,
    mcp_server_ids: list[str] | None,
    virtual_workspace_enabled: bool,
    web_fetch_enabled: bool,
    skills: list[str] | None,
    supported_features: conversation.ConversationEntityFeature | int,
) -> None:
    """Test tool-capable conversations still advertise streaming."""
    entry = _entry(
        llm_hass_api,
        skills=skills,
        mcp_server_ids=mcp_server_ids,
        virtual_workspace_enabled=virtual_workspace_enabled,
        web_fetch_enabled=web_fetch_enabled,
    )
    subentry = next(iter(entry.subentries.values()))

    entity = PydanticAIConversationEntity(entry, subentry)

    assert entity.supports_streaming is True
    assert entity.supported_features == supported_features


def test_conversation_entity_without_llm_api_has_no_control() -> None:
    """Test missing LLM API leaves Home Assistant control support disabled."""
    entry = _entry(None)
    subentry = next(iter(entry.subentries.values()))

    entity = PydanticAIConversationEntity(entry, subentry)

    assert entity.supported_features == 0
    assert entity.extra_state_attributes["ha_tools_enabled"] is False
    assert entity.extra_state_attributes["ha_llm_api"] is None
    assert entity.extra_state_attributes["web_fetch_enabled"] is False
    assert entity.extra_state_attributes["virtual_workspace_enabled"] is False


def test_conversation_entity_requires_literal_virtual_workspace_true() -> None:
    """Test truthy persisted values do not report virtual workspace enabled."""
    entry = _entry(None, extra_data={CONF_VIRTUAL_WORKSPACE_ENABLED: "true"})
    subentry = next(iter(entry.subentries.values()))

    entity = PydanticAIConversationEntity(entry, subentry)

    assert entity.extra_state_attributes["virtual_workspace_enabled"] is False


async def test_conversation_subentries_add_separate_entity_agents(
    hass: HomeAssistant,
) -> None:
    """Test each conversation subentry is exposed as a separate entity agent."""
    entry = _entry_with_conversation_subentries()
    added_entities: list[tuple[list[Entity], str | None]] = []

    def async_add_entities(
        new_entities: Iterable[Entity],
        update_before_add: bool = False,
        *,
        config_subentry_id: str | None = None,
    ) -> None:
        del update_before_add
        added_entities.append((list(new_entities), config_subentry_id))

    await async_setup_entry(hass, entry, async_add_entities)

    subentries = list(entry.subentries.values())
    assert [item[1] for item in added_entities] == [
        subentries[0].subentry_id,
        subentries[1].subentry_id,
    ]
    assert [item[0][0].unique_id for item in added_entities] == [
        f"{DOMAIN}_{entry.entry_id}_{subentries[0].subentry_type}_{subentries[0].subentry_id}",
        f"{DOMAIN}_{entry.entry_id}_{subentries[1].subentry_type}_{subentries[1].subentry_id}",
    ]
    assert [item[0][0].device_info for item in added_entities] == [
        {
            "identifiers": {
                (
                    DOMAIN,
                    f"{entry.entry_id}:{subentries[0].subentry_type}:{subentries[0].subentry_id}",
                )
            },
            "name": "Kitchen Agent Configuration",
            "manufacturer": "Pydantic AI",
            "model": "gpt-kitchen",
            "entry_type": dr.DeviceEntryType.SERVICE,
        },
        {
            "identifiers": {
                (
                    DOMAIN,
                    f"{entry.entry_id}:{subentries[1].subentry_type}:{subentries[1].subentry_id}",
                )
            },
            "name": "Garage Agent Configuration",
            "manufacturer": "Pydantic AI",
            "model": "gpt-garage",
            "entry_type": dr.DeviceEntryType.SERVICE,
        },
    ]


async def test_conversation_entity_id_dispatches_assist_agent(
    hass: HomeAssistant,
) -> None:
    """Test conversation entity IDs are valid Assist agent IDs."""
    entry = _entry_with_conversation_subentries()
    entry.add_to_hass(hass)
    garage_subentry = next(
        subentry
        for subentry in entry.subentries.values()
        if subentry.data.get(CONF_AGENT_NAME) == "Garage Agent"
    )
    _enable_diagnostic_entities(
        hass,
        entry,
        garage_subentry,
        "sensor",
        (
            "last_run_model_profile",
            "last_run_input_tokens",
            "last_run_output_tokens",
            "last_run_total_tokens",
            "last_run_model_request_count",
            "last_run_tool_use_count",
            "cumulative_input_tokens",
            "cumulative_output_tokens",
            "cumulative_total_tokens",
            "consecutive_failures",
            "last_error_type",
        ),
    )
    _enable_diagnostic_entities(
        hass,
        entry,
        garage_subentry,
        "binary_sensor",
        ("provider_healthy", "last_run_succeeded", "assist_enabled"),
    )

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
        new_callable=AsyncMock,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    entity_ids = sorted(
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id != "conversation.home_assistant"
    )

    assert len(entity_ids) == 2
    assert all(
        conversation.async_get_agent(hass, entity_id) for entity_id in entity_ids
    )

    garage_entity_id = next(
        entity_id for entity_id in entity_ids if entity_id.endswith("garage_agent")
    )
    events: list[dict[str, object]] = []
    hass.bus.async_listen(
        f"{DOMAIN}_{EVENT_AGENT_RUN_COMPLETED}",
        lambda event: events.append(dict(event.data)),
    )
    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ) as chat_model,
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            return_value=_Agent(),
        ) as agent_class,
    ):
        result = await conversation.async_converse(
            hass,
            "hello",
            None,
            Context(),
            agent_id=garage_entity_id,
        )
        await hass.async_block_till_done()

    assert result.response.speech["plain"]["speech"] == "runtime response"
    assert chat_model.call_args is not None
    assert chat_model.call_args.args[2].model_name == "gpt-garage"
    assert agent_class.call_args.kwargs["output_type"] is str
    _assert_context_management_capability(agent_class.call_args.kwargs["capabilities"])
    assert _state(hass, "sensor.garage_agent_last_run_model_profile") == "Garage Model"
    assert _state(hass, "sensor.garage_agent_last_run_input_tokens") == "10"
    assert _state(hass, "sensor.garage_agent_last_run_output_tokens") == "2"
    assert _state(hass, "sensor.garage_agent_last_run_total_tokens") == "12"
    assert _state(hass, "sensor.garage_agent_last_run_model_request_count") == "1"
    assert _state(hass, "sensor.garage_agent_last_run_tool_use_count") == "3"
    assert _state(hass, "sensor.garage_agent_cumulative_input_tokens") == "10"
    assert _state(hass, "sensor.garage_agent_cumulative_output_tokens") == "2"
    assert _state(hass, "sensor.garage_agent_cumulative_total_tokens") == "12"
    assert _state(hass, "sensor.garage_agent_consecutive_failures") == "0"
    assert _state(hass, "sensor.garage_agent_last_error_type") == "unknown"
    assert _state(hass, "binary_sensor.garage_agent_provider_healthy") == "on"
    assert _state(hass, "binary_sensor.garage_agent_last_run_succeeded") == "on"
    assert _state(hass, "binary_sensor.garage_agent_assist_enabled") == "off"
    assert events == [
        {
            "config_entry_id": entry.entry_id,
            "subentry_id": garage_subentry.subentry_id,
            "entity_id": garage_entity_id,
            "model_profile": "Garage Model",
        }
    ]


async def test_diagnostic_entities_are_disabled_by_default(
    hass: HomeAssistant,
) -> None:
    """Test per-agent diagnostic entities are registry-disabled by default."""
    entry = _entry(None)
    entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
        new_callable=AsyncMock,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    entity_registry = er.async_get(hass)
    for entity_id in (
        "sensor.kitchen_agent_last_run_model_profile",
        "sensor.kitchen_agent_primary_language_model",
        "binary_sensor.kitchen_agent_provider_healthy",
        "binary_sensor.kitchen_agent_assist_enabled",
    ):
        registry_entry = entity_registry.async_get(entity_id)
        assert registry_entry is not None
        assert registry_entry.disabled_by is er.RegistryEntryDisabler.INTEGRATION
        assert hass.states.get(entity_id) is None


async def test_conversation_runtime_uses_configured_max_iterations(
    hass: HomeAssistant,
) -> None:
    """Test conversation runs use the configured run iteration limit."""
    entry = _entry(None, extra_data={CONF_MAX_ITERATIONS: 24})
    entry.add_to_hass(hass)
    agent = _Agent()

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
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

    assert getattr(agent.run_kwargs["usage_limits"], "request_limit") == 24


async def test_streaming_iteration_failure_updates_chat_and_sensors(
    hass: HomeAssistant,
) -> None:
    """Test streaming usage-limit failures stay actionable after partial output."""
    entry = _entry(None, extra_data={CONF_MAX_ITERATIONS: 24})
    entry.add_to_hass(hass)
    subentry = next(iter(entry.subentries.values()))
    _enable_diagnostic_entities(
        hass,
        entry,
        subentry,
        "sensor",
        ("consecutive_failures", "last_error_type"),
    )
    _enable_diagnostic_entities(
        hass,
        entry,
        subentry,
        "binary_sensor",
        ("provider_healthy", "last_run_succeeded"),
    )

    class FailingAfterPartialAgent(_Agent):
        @asynccontextmanager
        async def run_stream_events(
            self, *_args: object, **kwargs: object
        ) -> AsyncIterator[AsyncIterator[object]]:
            self.run_stream_events_calls += 1
            self.run_kwargs = kwargs

            async def stream() -> AsyncIterator[object]:
                yield PartStartEvent(index=0, part=TextPart(content="partial"))
                raise UsageLimitExceeded(
                    "The next request would exceed the request_limit of 24"
                )

            yield stream()

    agent = FailingAfterPartialAgent()
    events: list[dict[str, object]] = []
    hass.bus.async_listen(
        f"{DOMAIN}_{EVENT_AGENT_RUN_FAILED}",
        lambda event: events.append(dict(event.data)),
    )

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
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
        result = await conversation.async_converse(
            hass,
            "hello",
            None,
            Context(),
            agent_id=entity_id,
        )
        await hass.async_block_till_done()

    speech = result.response.speech["plain"]["speech"]
    assert "configured maximum of 24 iterations" in speech
    assert "Streaming model failed after sending a partial response" not in speech
    assert _state(hass, "sensor.kitchen_agent_consecutive_failures") == "1"
    assert _state(hass, "sensor.kitchen_agent_last_error_type") == "UsageLimitExceeded"
    assert _state(hass, "binary_sensor.kitchen_agent_provider_healthy") == "off"
    assert _state(hass, "binary_sensor.kitchen_agent_last_run_succeeded") == "off"
    assert events[-1]["error_type"] == "UsageLimitExceeded"
    assert events[-1]["partial_response"] is True
    assert "configured maximum of 24 iterations" in str(events[-1]["error_message"])


async def test_conversation_runtime_uses_thinking_capability(
    hass: HomeAssistant,
) -> None:
    """Test configured thinking is represented as a Pydantic AI capability."""
    entry = _entry(None, extra_data={CONF_THINKING: "high"})
    entry.add_to_hass(hass)
    agent = _Agent()

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
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


async def test_conversation_runtime_supports_test_model_without_patching_agent_run(
    hass: HomeAssistant,
) -> None:
    """Test a deterministic Pydantic AI TestModel runtime path."""
    entry = _entry(None)
    entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
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
    entry = _entry(None)
    entry.add_to_hass(hass)
    captured_messages: list[ModelRequest | ModelResponse] = []

    async def model_function(
        messages: list[ModelRequest | ModelResponse], info: AgentInfo
    ) -> ModelResponse:
        captured_messages.extend(messages)
        assert info.function_tools == []
        return ModelResponse(parts=[TextPart(content="function model response")])

    async def stream_function(
        messages: list[ModelRequest | ModelResponse], info: AgentInfo
    ) -> AsyncIterator[str]:
        captured_messages.extend(messages)
        assert info.function_tools == []
        yield "function model response"

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
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
    entry = _entry(None)
    entry.add_to_hass(hass)
    agent = _Agent()

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
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

    assert getattr(agent.run_kwargs["usage_limits"], "request_limit") == 10


async def test_conversation_runtime_passes_selected_skills_capabilities(
    hass: HomeAssistant,
) -> None:
    """Test selected conversation skills become Agent capabilities."""
    entry = _entry(None, skills=["kitchen-skill"])
    entry.add_to_hass(hass)
    capability = object()
    agent = _Agent()

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
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
    entry = _entry(None, web_fetch_enabled=True)
    entry.add_to_hass(hass)
    web_fetch_capability = object()

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
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
            "custom_components.pydantic_ai_agent.entity.WebFetch",
            return_value=web_fetch_capability,
        ) as web_fetch,
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

    web_fetch.assert_called_once_with(local=True)
    capabilities = agent_class.call_args.kwargs["capabilities"]
    assert web_fetch_capability in capabilities
    _assert_context_management_capability(capabilities)


async def test_conversation_runtime_adds_virtual_workspace_tools(
    hass: HomeAssistant,
) -> None:
    """Test virtual workspace-enabled conversations add tools/instructions."""
    entry = _entry(None, virtual_workspace_enabled=True)
    entry.add_to_hass(hass)
    fake_toolset = object()

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
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
            "custom_components.pydantic_ai_agent.entity.virtual_workspace_parts",
            return_value=SimpleNamespace(
                toolsets=(fake_toolset,), instructions="virtual instructions"
            ),
        ) as workspace_parts,
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
            "use a workspace",
            None,
            Context(),
            agent_id=entity_id,
        )

    workspace_parts.assert_called_once_with()
    assert agent_class.call_args.kwargs["toolsets"] == [fake_toolset]
    assert agent_class.call_args.kwargs["instructions"] == "virtual instructions"


async def test_conversation_runtime_recreates_virtual_workspace_for_fallback(
    hass: HomeAssistant,
) -> None:
    """Test fallback model attempts do not reuse a failed workspace."""
    primary_ref = f"{_PROVIDER_SUBENTRY_ID}:primary-model"
    fallback_ref = f"{_PROVIDER_SUBENTRY_ID}:fallback-model"
    entry = workspace_entry(
        (
            conversation_subentry_data(
                primary_ref,
                virtual_workspace_enabled=True,
                extra_data={CONF_FALLBACK_MODEL_REFS: [fallback_ref]},
            ),
            provider_subentry_data(
                subentry_id=_PROVIDER_SUBENTRY_ID,
                title="Hosted OpenAI",
                default_model_profile_id="primary-model",
                model_profiles={
                    "primary-model": model_profile_data(
                        profile_id="primary-model",
                        name="Primary",
                        model="primary-model",
                    ),
                    "fallback-model": model_profile_data(
                        profile_id="fallback-model",
                        name="Fallback",
                        model="fallback-model",
                    ),
                },
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
    entry.add_to_hass(hass)

    class FailingAgent:
        async def __aenter__(self) -> "FailingAgent":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def run_stream_events(self, *_args: object, **_kwargs: object) -> object:
            class FailingEvents:
                async def __aenter__(self) -> object:
                    raise TimeoutError

                async def __aexit__(self, *_args: object) -> None:
                    return None

            return FailingEvents()

    first_toolset = object()
    second_toolset = object()

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
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
            "custom_components.pydantic_ai_agent.entity.virtual_workspace_parts",
            side_effect=[
                SimpleNamespace(toolsets=(first_toolset,), instructions="first"),
                SimpleNamespace(toolsets=(second_toolset,), instructions="second"),
            ],
        ) as workspace_parts,
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            side_effect=[FailingAgent(), _Agent()],
        ) as agent_class,
    ):
        await conversation.async_converse(
            hass,
            "use a workspace",
            None,
            Context(),
            agent_id=entity_id,
        )

    assert workspace_parts.call_count == 2
    assert agent_class.call_args_list[0].kwargs["toolsets"] == [first_toolset]
    assert agent_class.call_args_list[1].kwargs["toolsets"] == [second_toolset]


async def test_conversation_runtime_adds_keyword_tool_search_for_deferred_mcp(
    hass: HomeAssistant,
) -> None:
    """Test deferred MCP toolsets force local keyword tool search."""
    entry = _entry(None)
    entry.add_to_hass(hass)
    deferred_toolset = FunctionToolset().defer_loading()

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
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
            "custom_components.pydantic_ai_agent.entity.async_runtime_mcp_toolsets",
            new_callable=AsyncMock,
            return_value=[deferred_toolset],
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            return_value=_Agent(),
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
    assert any(
        isinstance(capability, ToolSearch) and capability.strategy == "keywords"
        for capability in capabilities
    )
    _assert_context_management_capability(capabilities)


async def test_conversation_logfire_instruments_agent_with_ha_metadata(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test active Logfire entries instrument agents and add HA trace metadata."""
    instrument = Mock()
    span = Mock(return_value=_Span())
    monkeypatch.setitem(
        sys.modules,
        "logfire",
        SimpleNamespace(configure=Mock(), instrument_pydantic_ai=instrument, span=span),
    )
    entry = _entry_with_conversation_subentries(logfire=True)
    entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
        new_callable=AsyncMock,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    entity_ids = sorted(
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id != "conversation.home_assistant"
    )
    kitchen_entity_id = next(
        entity_id for entity_id in entity_ids if entity_id.endswith("kitchen_agent")
    )
    agent = _Agent()
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
            "conversation-test",
            Context(),
            agent_id=kitchen_entity_id,
        )

    instrument.assert_called_once_with(agent, include_content=True)
    span.assert_called_once()
    assert span.call_args.args == ("Run Pydantic AI agent",)
    assert span.call_args.kwargs["ha.domain"] == DOMAIN
    assert span.call_args.kwargs["ha.version"] == __version__
    assert span.call_args.kwargs["ha.entry_id"] == entry.entry_id
    assert span.call_args.kwargs["ha.subentry_title"] == "Kitchen Agent"
    assert span.call_args.kwargs["ha.model"] == "gpt-kitchen"
    assert span.call_args.kwargs["ha.structured_output_mode"] == OUTPUT_MODE_TOOL
    assert "ha.output_mode" not in span.call_args.kwargs
    assert span.call_args.kwargs["ha.entity_id"] == kitchen_entity_id
    assert span.call_args.kwargs["ha.conversation_id"] == "conversation-test"
    assert span.call_args.kwargs["ha.logfire_include_content"] is True
    assert span.return_value.attributes == {
        "gen_ai.usage.input_tokens": 10,
        "gen_ai.usage.output_tokens": 2,
    }


async def test_conversation_logfire_usage_failures_do_not_block_agent_run(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test optional wrapper usage attributes do not block provider responses."""
    monkeypatch.setitem(
        sys.modules,
        "logfire",
        SimpleNamespace(
            configure=Mock(),
            instrument_pydantic_ai=Mock(),
            span=Mock(return_value=_FailingSetAttributesSpan()),
        ),
    )
    entry = _entry_with_conversation_subentries(logfire=True)
    entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
        new_callable=AsyncMock,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    entity_id = next(
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id.endswith("kitchen_agent")
    )
    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            return_value=_Agent(),
        ),
    ):
        result = await conversation.async_converse(
            hass,
            "hello",
            "conversation-test",
            Context(),
            agent_id=entity_id,
        )

    assert result.response.speech["plain"]["speech"] == "runtime response"


async def test_conversation_logfire_failures_do_not_block_agent_run(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test optional Logfire runtime failures do not block the provider call."""
    monkeypatch.setitem(
        sys.modules,
        "logfire",
        SimpleNamespace(
            configure=Mock(),
            instrument_pydantic_ai=Mock(side_effect=RuntimeError("instrument failed")),
            span=Mock(side_effect=RuntimeError("span failed")),
        ),
    )
    entry = _entry_with_conversation_subentries(logfire=True)
    entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
        new_callable=AsyncMock,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    entity_ids = sorted(
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id != "conversation.home_assistant"
    )
    kitchen_entity_id = next(
        entity_id for entity_id in entity_ids if entity_id.endswith("kitchen_agent")
    )
    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            return_value=_Agent(),
        ),
    ):
        result = await conversation.async_converse(
            hass,
            "hello",
            "conversation-test",
            Context(),
            agent_id=kitchen_entity_id,
        )

    assert result.response.speech["plain"]["speech"] == "runtime response"
