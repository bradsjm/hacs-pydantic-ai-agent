"""Test Pydantic AI Agent AI task entities."""

from collections.abc import Iterable
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.capabilities import Thinking
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import NativeOutput, PromptedOutput, ToolOutput
import voluptuous as vol

from homeassistant.components import ai_task
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity import Entity
from homeassistant.util import slugify
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.pydantic_ai_agent.ai_task import (
    PydanticAIAgentAITaskEntity,
    async_setup_entry,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_AI_TASK_NAME,
    CONF_MAX_ITERATIONS,
    CONF_THINKING,
    CONF_VIRTUAL_WORKSPACE_ENABLED,
    DOMAIN,
    OUTPUT_MODE_NATIVE,
    OUTPUT_MODE_PROMPTED,
    OUTPUT_MODE_TOOL,
)
from custom_components.pydantic_ai_agent.context_management import (
    SlidingWindowContextCapability,
)
from custom_components.pydantic_ai_agent.entity import unique_id_for_subentry_entity
from custom_components.pydantic_ai_agent.metrics import (
    EVENT_AGENT_RUN_FAILED,
    EVENT_STRUCTURED_AI_TASK_OUTPUT_GENERATED,
)
from tests.components.pydantic_ai_agent.support.builders import (
    ai_task_subentry_data,
    provider_runtime_data,
    provider_subentry_data,
    workspace_entry,
    workspace_runtime_data,
)
from tests.components.pydantic_ai_agent.support.pydantic_ai import (
    Agent as _Agent,
    agent_factory as _agent_factory,
)

_PROVIDER_SUBENTRY_ID = "provider-1"
_MODEL_PROFILE_ID = "task-profile"
_MODEL_PROFILE_REF = f"{_PROVIDER_SUBENTRY_ID}:{_MODEL_PROFILE_ID}"


def _entry(
    output_mode: str | None = None,
    skills: list[str] | None = None,
    *,
    include_task_name: bool = True,
    subentry_title: str = "AI task subentry title",
    virtual_workspace_enabled: bool = False,
    web_fetch_enabled: bool = False,
    model_settings: dict[str, object] | None = None,
    todo_workspace_entity_id: str | None = None,
    extra_data: dict[str, object] | None = None,
) -> MockConfigEntry:
    """Return a config entry with one AI task subentry."""
    entry = workspace_entry(
        (
            ai_task_subentry_data(
                _MODEL_PROFILE_REF,
                title=subentry_title,
                task_name="Report task" if include_task_name else None,
                output_mode=output_mode,
                skills=skills,
                virtual_workspace_enabled=virtual_workspace_enabled,
                web_fetch_enabled=web_fetch_enabled,
                todo_workspace_entity_id=todo_workspace_entity_id,
                extra_data=extra_data,
            ),
            provider_subentry_data(
                subentry_id=_PROVIDER_SUBENTRY_ID,
                title="Hosted OpenAI",
                profile_id=_MODEL_PROFILE_ID,
                profile_name="Task Model",
                model="task-model",
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


async def _setup_ai_task_entity(
    hass: HomeAssistant,
    output_mode: str | None = None,
    skills: list[str] | None = None,
    *,
    virtual_workspace_enabled: bool = False,
    web_fetch_enabled: bool = False,
    model_settings: dict[str, object] | None = None,
    todo_workspace_entity_id: str | None = None,
    extra_data: dict[str, object] | None = None,
    enable_diagnostics: bool = False,
) -> str:
    """Set up an AI task config entry and return its entity ID."""
    entry = _entry(
        output_mode,
        skills,
        virtual_workspace_enabled=virtual_workspace_enabled,
        web_fetch_enabled=web_fetch_enabled,
        model_settings=model_settings,
        todo_workspace_entity_id=todo_workspace_entity_id,
        extra_data=extra_data,
    )
    entry.add_to_hass(hass)
    if enable_diagnostics:
        subentry = next(iter(entry.subentries.values()))
        _enable_diagnostic_entities(
            hass,
            entry,
            subentry,
            "sensor",
            ("last_error_type",),
        )
        _enable_diagnostic_entities(
            hass,
            entry,
            subentry,
            "binary_sensor",
            ("provider_healthy", "last_run_succeeded"),
        )

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
        new_callable=AsyncMock,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    entity_ids = [state.entity_id for state in hass.states.async_all("ai_task")]
    assert len(entity_ids) == 1
    return entity_ids[0]


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
    name = str(subentry.data.get(CONF_AI_TASK_NAME, subentry.title))
    for key in keys:
        entity_registry.async_get_or_create(
            entity_domain,
            DOMAIN,
            unique_id_for_subentry_entity(entry, subentry, key),
            suggested_object_id=slugify(f"{name} {key}"),
        )


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
    assert entity.device_info["name"] == "Report task Configuration"
    assert entity.name == "Report task"


def test_ai_task_entity_falls_back_to_subentry_title() -> None:
    """Test nameless AI task subentries use their subentry title."""
    entry = _entry(include_task_name=False, subentry_title="Title-only task")
    subentry = next(iter(entry.subentries.values()))

    entity = PydanticAIAgentAITaskEntity(entry, subentry)

    assert entity.device_info is not None
    assert entity.device_info["name"] == "Title-only task Configuration"
    assert entity.name == "Title-only task"


def test_ai_task_entity_features() -> None:
    """Test AI task entity advertises data generation without image generation."""
    entry = _entry()
    subentry = next(iter(entry.subentries.values()))

    entity = PydanticAIAgentAITaskEntity(entry, subentry)

    assert ai_task.AITaskEntityFeature.GENERATE_DATA in entity.supported_features
    assert ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS in entity.supported_features
    assert ai_task.AITaskEntityFeature.GENERATE_IMAGE not in entity.supported_features


def test_ai_task_entity_reports_virtual_workspace_attribute() -> None:
    """Test AI task attributes expose virtual workspace state."""
    entry = _entry(virtual_workspace_enabled=True)
    subentry = next(iter(entry.subentries.values()))

    entity = PydanticAIAgentAITaskEntity(entry, subentry)

    assert entity.extra_state_attributes["virtual_workspace_enabled"] is True


def test_ai_task_entity_requires_literal_virtual_workspace_true() -> None:
    """Test truthy persisted values do not report virtual workspace enabled."""
    entry = _entry(extra_data={CONF_VIRTUAL_WORKSPACE_ENABLED: "true"})
    subentry = next(iter(entry.subentries.values()))

    entity = PydanticAIAgentAITaskEntity(entry, subentry)

    assert entity.extra_state_attributes["virtual_workspace_enabled"] is False


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
    entity_id = await _setup_ai_task_entity(hass, extra_data={CONF_THINKING: False})

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


async def test_ai_task_runtime_composes_virtual_workspace_and_todo_tools(
    hass: HomeAssistant,
) -> None:
    """Test AI tasks compose virtual workspace before todo tools/instructions."""
    entity_id = await _setup_ai_task_entity(
        hass,
        virtual_workspace_enabled=True,
        todo_workspace_entity_id="todo.ai_workspace",
    )
    virtual_toolset = object()
    todo_toolset = object()

    class FakeTodoWorkspace:
        """Minimal todo workspace test double."""

        def __init__(self, hass: HomeAssistant, entity_id: str) -> None:
            """Initialize fake workspace."""
            self.hass = hass
            self.entity_id = entity_id

        async def prepare_run(self) -> str:
            """Prepare the fake workspace."""
            return "cleared"

        async def read_items(self) -> str:
            """Return initial workspace state."""
            return "Summary: 0 completed, 0 in progress, 0 pending"

        def toolset(self) -> object:
            """Return fake toolset."""
            return todo_toolset

        def instructions(self, initial_state: str) -> str:
            """Return fake instructions."""
            return f"todo instructions: {initial_state}"

    with (
        patch(
            "custom_components.pydantic_ai_agent.ai_task.TodoWorkspace",
            FakeTodoWorkspace,
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.virtual_workspace_parts",
            return_value=SimpleNamespace(
                toolsets=(virtual_toolset,), instructions="virtual instructions"
            ),
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
        await ai_task.async_generate_data(
            hass,
            task_name="Plain task",
            entity_id=entity_id,
            instructions="Generate text",
        )

    assert agent_class.call_args.kwargs["toolsets"] == [virtual_toolset, todo_toolset]
    assert agent_class.call_args.kwargs["instructions"] == (
        "virtual instructions\n\ntodo instructions: "
        "Summary: 0 completed, 0 in progress, 0 pending"
    )


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
    entity_id = await _setup_ai_task_entity(hass, enable_diagnostics=True)
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
    """Test AI task runs use the configured run iteration limit."""
    entity_id = await _setup_ai_task_entity(hass, extra_data={CONF_MAX_ITERATIONS: 26})
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
    assert skills_capabilities.call_args.args[0] is hass
    assert skills_capabilities.call_args.args[1].domain == DOMAIN
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
