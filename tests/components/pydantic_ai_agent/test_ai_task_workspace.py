"""Test Pydantic AI Agent AI task workspace and capability integration."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from custom_components.pydantic_ai_agent.context_management import (
    SlidingWindowContextCapability,
)
from homeassistant.components import ai_task
from homeassistant.core import HomeAssistant
from pydantic_ai.capabilities import WebFetch
from pytest_homeassistant_custom_component.common import MockConfigEntry
from tests.components.pydantic_ai_agent.support.builders import (
    ai_task_subentry_data,
    provider_runtime_data,
    provider_subentry_data,
    workspace_entry,
    workspace_runtime_data,
)
from tests.components.pydantic_ai_agent.support.pydantic_ai import (
    agent_factory as _agent_factory,
)

_PROVIDER_SUBENTRY_ID = "provider-1"
_MODEL_PROFILE_ID = "task-profile"
_MODEL_PROFILE_REF = f"{_PROVIDER_SUBENTRY_ID}:{_MODEL_PROFILE_ID}"


def _entry(
    *,
    skills: list[str] | None = None,
    virtual_workspace_enabled: bool = False,
    web_fetch_enabled: bool = False,
    todo_workspace_entity_id: str | None = None,
    extra_data: dict[str, object] | None = None,
) -> MockConfigEntry:
    """Return a config entry with one AI task subentry."""
    entry = workspace_entry(
        (
            ai_task_subentry_data(
                _MODEL_PROFILE_REF,
                task_name="Report task",
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
    *,
    skills: list[str] | None = None,
    virtual_workspace_enabled: bool = False,
    web_fetch_enabled: bool = False,
    todo_workspace_entity_id: str | None = None,
) -> str:
    """Set up an AI task config entry and return its entity ID."""
    entry = _entry(
        skills=skills,
        virtual_workspace_enabled=virtual_workspace_enabled,
        web_fetch_enabled=web_fetch_enabled,
        todo_workspace_entity_id=todo_workspace_entity_id,
    )
    entry.add_to_hass(hass)

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
            self.hass = hass
            self.entity_id = entity_id

        async def prepare_run(self) -> str:
            calls.append("prepare")
            return "cleared"

        async def read_items(self) -> str:
            calls.append("read")
            return "Summary: 0 completed, 0 in progress, 0 pending"

        def toolset(self) -> object:
            return fake_toolset

        def instructions(self, initial_state: str) -> str:
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
        def __init__(self, hass: HomeAssistant, entity_id: str) -> None:
            self.hass = hass
            self.entity_id = entity_id

        async def prepare_run(self) -> str:
            return "cleared"

        async def read_items(self) -> str:
            return "Summary: 0 completed, 0 in progress, 0 pending"

        def toolset(self) -> object:
            return todo_toolset

        def instructions(self, initial_state: str) -> str:
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
    assert skills_capabilities.call_args.args[2] == ["report-skill"]
    capabilities = agent_class.call_args.kwargs["capabilities"]
    assert capability in capabilities
    _assert_context_management_capability(capabilities)


async def test_ai_task_runtime_adds_web_fetch_capability(
    hass: HomeAssistant,
) -> None:
    """Test WebFetch-enabled AI tasks get the WebFetch capability."""
    entity_id = await _setup_ai_task_entity(hass, web_fetch_enabled=True)

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
            task_name="Fetch task",
            entity_id=entity_id,
            instructions="Fetch https://example.com",
        )

    assert result.data == "plain result"
    assert agent_class.call_args is not None
    capabilities = agent_class.call_args.kwargs["capabilities"]
    assert any(isinstance(capability, WebFetch) for capability in capabilities)
    _assert_context_management_capability(capabilities)
