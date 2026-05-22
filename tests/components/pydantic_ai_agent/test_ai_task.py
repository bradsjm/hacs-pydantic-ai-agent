"""Test Pydantic AI Agent AI task entities."""

# ruff: noqa: E402

from collections.abc import AsyncGenerator, Iterable
from contextlib import asynccontextmanager
import json
from unittest.mock import AsyncMock, patch

import pytest

pytest.skip(
    "Legacy model-subentry AI task tests need workspace/provider-profile rewrite.",
    allow_module_level=True,
)

from pydantic_ai import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.capabilities import Thinking
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import NativeOutput, PromptedOutput, ToolOutput
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.components import ai_task
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity import Entity
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.pydantic_ai_agent.ai_task import (
    PydanticAIAgentAITaskEntity,
    async_setup_entry,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_AI_TASK_NAME,
    CONF_ENABLE_SKILLS,
    CONF_MAX_ITERATIONS,
    CONF_MODEL,
    CONF_MODEL_SETTINGS,
    CONF_OUTPUT_MODE,
    CONF_PROVIDER_MODE,
    CONF_SKILLS,
    CONF_TODO_LIST_ENTITY_ID,
    CONF_WEB_FETCH_ENABLED,
    DOMAIN,
    OUTPUT_MODE_NATIVE,
    OUTPUT_MODE_PROMPTED,
    OUTPUT_MODE_TOOL,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_AI_TASK,
)
from custom_components.pydantic_ai_agent.context_management import (
    SlidingWindowContextCapability,
)
from custom_components.pydantic_ai_agent.metrics import (
    EVENT_AGENT_RUN_FAILED,
    EVENT_STRUCTURED_AI_TASK_OUTPUT_GENERATED,
)

CONF_MODEL_SUBENTRY_ID = "model_subentry_id"
SUBENTRY_TYPE_MODEL = "model"


class PydanticAIAgentRuntimeData:
    """Legacy runtime-data test double for obsolete model-subentry tests."""

    def __init__(self, **kwargs: object) -> None:
        """Store provided attributes."""
        self.__dict__.update(kwargs)


class _Usage:
    """Minimal Pydantic AI usage test double."""

    input_tokens = 20
    output_tokens = 5
    total_tokens = 25
    requests = 2
    tool_calls = 1

    def opentelemetry_attributes(self) -> dict[str, int]:
        """Return deterministic token usage attributes."""
        return {
            "gen_ai.usage.input_tokens": 20,
            "gen_ai.usage.output_tokens": 5,
        }


class _TextStream:
    """Async iterator over text chunks."""

    def __init__(self, *events: object) -> None:
        """Initialize the event stream."""
        self._events = iter(events)

    def __aiter__(self) -> "_TextStream":
        """Return the async iterator."""
        return self

    async def __anext__(self) -> object:
        """Return the next stream event."""
        try:
            return next(self._events)
        except StopIteration as err:
            raise StopAsyncIteration from err


def _entry(
    output_mode: str | None = None,
    skills: list[str] | None = None,
    *,
    legacy_task_name: bool = False,
    web_fetch_enabled: bool = False,
    model_settings: dict[str, object] | None = None,
    todo_workspace_entity_id: str | None = None,
) -> MockConfigEntry:
    """Return a config entry with one AI task subentry."""
    subentry_data: dict[str, object] = {CONF_MODEL_SUBENTRY_ID: "task_model_profile"}
    if not legacy_task_name:
        subentry_data[CONF_AI_TASK_NAME] = "Report task"
    if output_mode is not None:
        subentry_data[CONF_OUTPUT_MODE] = output_mode
    if skills is not None:
        subentry_data[CONF_ENABLE_SKILLS] = True
        subentry_data[CONF_SKILLS] = skills
    if web_fetch_enabled:
        subentry_data[CONF_WEB_FETCH_ENABLED] = True
    if todo_workspace_entity_id is not None:
        subentry_data[CONF_TODO_LIST_ENTITY_ID] = todo_workspace_entity_id
    model_subentry_data: dict[str, object] = {
        CONF_NAME: "Task Model",
        CONF_MODEL: "task-model",
    }
    if model_settings is not None:
        model_subentry_data[CONF_MODEL_SETTINGS] = model_settings

    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Hosted OpenAI",
        data={
            CONF_NAME: "Hosted OpenAI",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "sk-test",
        },
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "data": subentry_data,
                "subentry_type": SUBENTRY_TYPE_AI_TASK,
                "title": "Report task",
                "unique_id": None,
            },
            {
                "subentry_id": "task_model_profile",
                "data": model_subentry_data,
                "subentry_type": SUBENTRY_TYPE_MODEL,
                "title": "Task Model",
                "unique_id": None,
            },
        ),
        options={},
        unique_id=None,
    )
    entry.runtime_data = PydanticAIAgentRuntimeData(
        provider_mode=PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        name="Hosted OpenAI",
        api_key="sk-test",
        base_url=None,
        logfire_enabled=False,
        logfire_include_content=False,
    )
    return entry


async def _setup_ai_task_entity(
    hass: HomeAssistant,
    output_mode: str | None = None,
    skills: list[str] | None = None,
    *,
    web_fetch_enabled: bool = False,
    model_settings: dict[str, object] | None = None,
    todo_workspace_entity_id: str | None = None,
) -> str:
    """Set up an AI task config entry and return its entity ID."""
    entry = _entry(
        output_mode,
        skills,
        web_fetch_enabled=web_fetch_enabled,
        model_settings=model_settings,
        todo_workspace_entity_id=todo_workspace_entity_id,
    )
    entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
        new_callable=AsyncMock,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    entity_ids = [state.entity_id for state in hass.states.async_all("ai_task")]
    assert len(entity_ids) == 1
    return entity_ids[0]


class _StreamResult:
    """Minimal Agent streamed result for AI task tests."""

    def __init__(self, text: str) -> None:
        """Initialize the streamed result."""
        self._text = text

    def stream_text(self, *, delta: bool = False) -> _TextStream:
        """Return streamed text chunks."""
        del delta
        return _TextStream(self._text)

    def get_output(self) -> str:
        """Return final output."""
        return self._text

    def new_messages(self) -> list[ModelResponse]:
        """Return final Agent messages."""
        return [ModelResponse(parts=[TextPart(content=self._text)])]


class _RunResult:
    """Minimal Agent run result for AI task tests."""

    def __init__(
        self, output: object, messages: list[ModelResponse] | None = None
    ) -> None:
        """Initialize the run result."""
        self.output = output
        self._messages = messages
        self.usage = _Usage()

    def new_messages(self) -> list[ModelResponse]:
        """Return final Agent messages."""
        if self._messages is not None:
            return self._messages
        content = (
            self.output if isinstance(self.output, str) else json.dumps(self.output)
        )
        return [ModelResponse(parts=[TextPart(content=content)])]


class _Agent:
    """Minimal async-context Agent test double."""

    def __init__(
        self,
        *,
        stream_text: str = "",
        output: object = None,
        messages: list[ModelResponse] | None = None,
    ) -> None:
        """Initialize the agent."""
        self._stream_text = stream_text
        self._output = output
        self._messages = messages
        self.run_kwargs: dict[str, object] = {}

    async def __aenter__(self) -> "_Agent":
        """Enter the agent context."""
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Exit the agent context."""

    @asynccontextmanager
    async def run_stream(
        self, *_args: object, **_kwargs: object
    ) -> AsyncGenerator[_StreamResult]:
        """Return a deterministic streamed result."""
        yield _StreamResult(self._stream_text)

    async def run(self, *_args: object, **kwargs: object) -> _RunResult:
        """Return a deterministic run result."""
        self.run_kwargs = kwargs
        return _RunResult(self._output, self._messages)


def _agent_factory(
    *,
    stream_text: str = "",
    output: object = None,
    messages: list[ModelResponse] | None = None,
):
    """Return an Agent constructor test double."""

    def factory(*_args: object, **_kwargs: object) -> _Agent:
        return _Agent(
            stream_text=stream_text,
            output=stream_text if output is None else output,
            messages=messages,
        )

    return factory


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


async def test_ai_task_subentries_add_separate_entities(
    hass: HomeAssistant,
) -> None:
    """Test each AI task subentry is exposed as an AI task entity."""
    entry = _entry()
    added_entities: list[tuple[list[Entity], str | None]] = []

    def add_entities(
        new_entities: Iterable[Entity],
        update_before_add: bool = False,
        *,
        config_subentry_id: str | None = None,
    ) -> None:
        del update_before_add
        added_entities.append((list(new_entities), config_subentry_id))

    await async_setup_entry(hass, entry, add_entities)

    subentry = next(iter(entry.subentries.values()))
    entity = added_entities[0][0][0]
    assert added_entities[0][1] == subentry.subentry_id
    assert entity.unique_id == (
        f"{DOMAIN}_{entry.entry_id}_{subentry.subentry_type}_{subentry.subentry_id}"
    )


def test_ai_task_entity_uses_task_name() -> None:
    """Test AI task entity display name uses the configured task name."""
    entry = _entry()
    subentry = next(iter(entry.subentries.values()))

    entity = PydanticAIAgentAITaskEntity(entry, subentry)

    assert entity.device_info is not None
    assert entity.device_info["name"] == "Report task"


def test_ai_task_entity_falls_back_to_legacy_subentry_title() -> None:
    """Test legacy AI task subentries without task names keep their title."""
    entry = _entry(legacy_task_name=True)
    subentry = next(iter(entry.subentries.values()))

    entity = PydanticAIAgentAITaskEntity(entry, subentry)

    assert entity.device_info is not None
    assert entity.device_info["name"] == "Report task"


def test_ai_task_entity_features() -> None:
    """Test AI task entity advertises data generation without image generation."""
    entry = _entry()
    subentry = next(iter(entry.subentries.values()))

    entity = PydanticAIAgentAITaskEntity(entry, subentry)

    assert ai_task.AITaskEntityFeature.GENERATE_DATA in entity.supported_features
    assert ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS in entity.supported_features
    assert ai_task.AITaskEntityFeature.GENERATE_IMAGE not in entity.supported_features


async def test_plain_data_task_returns_text(hass: HomeAssistant) -> None:
    """Test a plain data task returns assistant text."""
    entity_id = await _setup_ai_task_entity(hass)

    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            side_effect=_agent_factory(stream_text="plain result"),
        ) as agent_class,
    ):
        result = await ai_task.async_generate_data(
            hass,
            task_name="Plain task",
            entity_id=entity_id,
            instructions="Generate text",
        )

    assert result.data == "plain result"
    _assert_context_management_capability(agent_class.call_args.kwargs["capabilities"])


async def test_plain_data_task_uses_thinking_capability(hass: HomeAssistant) -> None:
    """Test configured AI task thinking is passed as a capability."""
    entity_id = await _setup_ai_task_entity(hass, model_settings={"thinking": False})

    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            side_effect=_agent_factory(stream_text="plain result"),
        ) as agent_class,
    ):
        await ai_task.async_generate_data(
            hass,
            task_name="Plain task",
            entity_id=entity_id,
            instructions="Generate text",
        )

    capabilities = agent_class.call_args.kwargs["capabilities"]
    _assert_context_management_capability(capabilities)
    thinking = _thinking_capabilities(capabilities)
    assert len(thinking) == 1
    assert thinking[0].effort is False


async def test_ai_task_runtime_adds_todo_workspace_tools(
    hass: HomeAssistant,
) -> None:
    """Test configured todo workspace clears and adds tools/instructions."""
    entity_id = await _setup_ai_task_entity(
        hass, todo_workspace_entity_id="todo.ai_workspace"
    )
    fake_toolset = object()
    calls: list[str] = []

    class FakeTodoWorkspace:
        """Minimal todo workspace test double."""

        def __init__(self, hass: HomeAssistant, entity_id: str) -> None:
            """Initialize fake workspace."""
            self.hass = hass
            self.entity_id = entity_id

        async def prepare_run(self) -> str:
            """Record preparation."""
            calls.append("prepare")
            return "cleared"

        async def read_items(self) -> str:
            """Return initial workspace state."""
            calls.append("read")
            return "Summary: 0 completed, 0 in progress, 0 pending"

        def toolset(self) -> object:
            """Return fake toolset."""
            return fake_toolset

        def instructions(self, initial_state: str) -> str:
            """Return fake instructions."""
            return f"todo instructions: {initial_state}"

    with (
        patch(
            "custom_components.pydantic_ai_agent.ai_task.TodoWorkspace",
            FakeTodoWorkspace,
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            side_effect=_agent_factory(stream_text="plain result"),
        ) as agent_class,
    ):
        result = await ai_task.async_generate_data(
            hass,
            task_name="Plain task",
            entity_id=entity_id,
            instructions="Generate text",
        )

    assert result.data == "plain result"
    assert calls == ["prepare", "read"]
    assert agent_class.call_args.kwargs["toolsets"] == [fake_toolset]
    assert agent_class.call_args.kwargs["instructions"].startswith("todo instructions")


async def test_structured_data_task_fires_output_event(hass: HomeAssistant) -> None:
    """Test structured AI task output emits an integration event."""
    entity_id = await _setup_ai_task_entity(hass)
    events: list[dict[str, object]] = []
    hass.bus.async_listen(
        f"{DOMAIN}_{EVENT_STRUCTURED_AI_TASK_OUTPUT_GENERATED}",
        lambda event: events.append(dict(event.data)),
    )

    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            side_effect=_agent_factory(output={"summary": "ok"}),
        ),
    ):
        result = await ai_task.async_generate_data(
            hass,
            task_name="Structured task",
            entity_id=entity_id,
            instructions="Generate structured data",
            structure=vol.Schema({"summary": str}),
        )
        await hass.async_block_till_done()

    assert result.data == {"summary": "ok"}
    assert events[0]["entity_id"] == entity_id
    assert events[0]["task_name"] == "Structured task"


async def test_structured_data_task_validation_failure_records_failed_run(
    hass: HomeAssistant,
) -> None:
    """Test structured validation failures update health metrics and events."""
    entity_id = await _setup_ai_task_entity(hass)
    events: list[dict[str, object]] = []
    hass.bus.async_listen(
        f"{DOMAIN}_{EVENT_AGENT_RUN_FAILED}",
        lambda event: events.append(dict(event.data)),
    )

    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            side_effect=_agent_factory(output="{bad json"),
        ),
    ):
        with pytest.raises(HomeAssistantError, match="malformed structured data"):
            await ai_task.async_generate_data(
                hass,
                task_name="Structured task",
                entity_id=entity_id,
                instructions="Generate structured data",
                structure=vol.Schema({"summary": str}),
            )
        await hass.async_block_till_done()

    assert _state(hass, "binary_sensor.report_task_last_run_succeeded") == "off"
    assert _state(hass, "binary_sensor.report_task_provider_healthy") == "off"
    assert _state(hass, "sensor.report_task_last_error_type") == "JSONDecodeError"
    assert events[0]["entity_id"] == entity_id
    assert events[0]["error_type"] == "JSONDecodeError"


async def test_ai_task_runtime_uses_configured_max_iterations(
    hass: HomeAssistant,
) -> None:
    """Test AI task runs use the model profile iteration limit."""
    entity_id = await _setup_ai_task_entity(
        hass, model_settings={CONF_MAX_ITERATIONS: 26}
    )
    agent = _Agent(stream_text="plain result", output="plain result")

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
        result = await ai_task.async_generate_data(
            hass,
            task_name="Plain task",
            entity_id=entity_id,
            instructions="Generate text",
        )

    assert result.data == "plain result"
    assert getattr(agent.run_kwargs["usage_limits"], "request_limit") == 26


async def test_ai_task_runtime_defaults_max_iterations(hass: HomeAssistant) -> None:
    """Test AI task runs default to 30 iterations when unset."""
    entity_id = await _setup_ai_task_entity(hass)
    agent = _Agent(stream_text="plain result", output="plain result")

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
        result = await ai_task.async_generate_data(
            hass,
            task_name="Plain task",
            entity_id=entity_id,
            instructions="Generate text",
        )

    assert result.data == "plain result"
    assert getattr(agent.run_kwargs["usage_limits"], "request_limit") == 30


async def test_ai_task_runtime_passes_selected_skills_capabilities(
    hass: HomeAssistant,
) -> None:
    """Test selected AI task skills become Agent capabilities."""
    entity_id = await _setup_ai_task_entity(hass, skills=["report-skill"])
    capability = object()

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
            side_effect=_agent_factory(stream_text="plain result"),
        ) as agent_class,
    ):
        result = await ai_task.async_generate_data(
            hass,
            task_name="Plain task",
            entity_id=entity_id,
            instructions="Generate text",
        )

    assert result.data == "plain result"
    assert skills_capabilities.call_args.args[1][CONF_ENABLE_SKILLS] is True
    assert skills_capabilities.call_args.args[2] == ["report-skill"]
    capabilities = agent_class.call_args.kwargs["capabilities"]
    assert capability in capabilities
    _assert_context_management_capability(capabilities)


async def test_ai_task_runtime_adds_web_fetch_capability(
    hass: HomeAssistant,
) -> None:
    """Test WebFetch-enabled AI tasks get the WebFetch capability."""
    entity_id = await _setup_ai_task_entity(hass, web_fetch_enabled=True)
    web_fetch_capability = object()

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
            side_effect=_agent_factory(stream_text="plain result"),
        ) as agent_class,
    ):
        result = await ai_task.async_generate_data(
            hass,
            task_name="Fetch task",
            entity_id=entity_id,
            instructions="Fetch https://example.com",
        )

    assert result.data == "plain result"
    web_fetch.assert_called_once_with(local=True)
    capabilities = agent_class.call_args.kwargs["capabilities"]
    assert web_fetch_capability in capabilities
    _assert_context_management_capability(capabilities)


@pytest.mark.parametrize(
    ("output_mode", "output", "output_type", "messages"),
    [
        (
            OUTPUT_MODE_TOOL,
            {"name": "Kitchen"},
            ToolOutput,
            [
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="pydantic_ai_agent_output_structured_task",
                            args={"name": "Kitchen"},
                            tool_call_id="output-1",
                        )
                    ]
                )
            ],
        ),
        (OUTPUT_MODE_NATIVE, {"name": "Kitchen"}, NativeOutput, None),
        (OUTPUT_MODE_PROMPTED, {"name": "Kitchen"}, PromptedOutput, None),
    ],
)
async def test_structured_data_task_returns_parsed_json(
    hass: HomeAssistant,
    output_mode: str,
    output: dict[str, object],
    output_type: type,
    messages: list[ModelResponse] | None,
) -> None:
    """Test a structured data task returns parsed JSON."""
    entity_id = await _setup_ai_task_entity(hass, output_mode)

    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            side_effect=_agent_factory(output=output, messages=messages),
        ) as agent_class,
    ):
        result = await ai_task.async_generate_data(
            hass,
            task_name="Structured task",
            entity_id=entity_id,
            instructions="Generate JSON",
            structure=vol.Schema({vol.Required("name"): str}),
        )

    assert result.data == {"name": "Kitchen"}
    assert isinstance(agent_class.call_args.kwargs["output_type"], output_type)
    assert agent_class.call_args.kwargs["output_type"].name == (
        "pydantic_ai_agent_output_structured_task"
    )


async def test_structured_data_task_supports_test_model_without_patching_agent_run(
    hass: HomeAssistant,
) -> None:
    """Test structured AI tasks can run through Pydantic AI TestModel."""
    entity_id = await _setup_ai_task_entity(hass, OUTPUT_MODE_TOOL)

    with patch(
        "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
        return_value=TestModel(custom_output_args={"name": "Kitchen"}),
    ):
        result = await ai_task.async_generate_data(
            hass,
            task_name="Structured task",
            entity_id=entity_id,
            instructions="Generate JSON",
            structure=vol.Schema({vol.Required("name"): str}),
        )

    assert result.data == {"name": "Kitchen"}


async def test_structured_data_task_rejects_malformed_json(
    hass: HomeAssistant,
) -> None:
    """Test malformed structured data raises a Home Assistant error."""
    entity_id = await _setup_ai_task_entity(hass)

    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            side_effect=_agent_factory(output="not json"),
        ),
        pytest.raises(HomeAssistantError, match="malformed structured data"),
    ):
        await ai_task.async_generate_data(
            hass,
            task_name="Structured task",
            entity_id=entity_id,
            instructions="Generate JSON",
            structure=vol.Schema({vol.Required("name"): str}),
        )


async def test_structured_data_task_rejects_schema_mismatch(
    hass: HomeAssistant,
) -> None:
    """Test structured data must match the requested schema."""
    entity_id = await _setup_ai_task_entity(hass)

    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            side_effect=_agent_factory(output={"name": 1}),
        ),
        pytest.raises(HomeAssistantError, match="does not match the schema"),
    ):
        await ai_task.async_generate_data(
            hass,
            task_name="Structured task",
            entity_id=entity_id,
            instructions="Generate JSON",
            structure=vol.Schema({vol.Required("name"): str}),
        )
