"""Test Pydantic AI Agent AI task entity setup and data generation."""

from unittest.mock import AsyncMock, patch

import voluptuous as vol
from custom_components.pydantic_ai_agent.ai_task import (
    PydanticAIAgentAITaskEntity,
    async_setup_entry,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_AI_TASK_NAME,
    CONF_MAX_ITERATIONS,
    DOMAIN,
    OUTPUT_MODE_TOOL,
)
from custom_components.pydantic_ai_agent.context_management import (
    SlidingWindowContextCapability,
)
from custom_components.pydantic_ai_agent.entity import unique_id_for_subentry_entity
from homeassistant.components import ai_task
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity import Entity
from homeassistant.util import slugify
from pydantic_ai.capabilities import Thinking
from pydantic_ai.models.test import TestModel
from pytest_homeassistant_custom_component.common import MockConfigEntry
from tests.components.pydantic_ai_agent.support.builders import (
    ai_task_subentry_data,
    provider_runtime_data,
    provider_subentry_data,
    workspace_entry,
    workspace_runtime_data,
)
from tests.components.pydantic_ai_agent.support.pydantic_ai import (
    Agent as _Agent,
)
from tests.components.pydantic_ai_agent.support.pydantic_ai import (
    agent_factory as _agent_factory,
)
from tests.components.pydantic_ai_agent.support.pydantic_ai import (
    request_limit_from_kwargs,
)

_PROVIDER_SUBENTRY_ID = "provider-1"
_MODEL_PROFILE_ID = "task-profile"
_MODEL_PROFILE_REF = f"{_PROVIDER_SUBENTRY_ID}:{_MODEL_PROFILE_ID}"


def _entry(
    output_mode: str | None = None,
    *,
    include_task_name: bool = True,
    subentry_title: str = "AI task subentry title",
    extra_data: dict[str, object] | None = None,
) -> MockConfigEntry:
    entry = workspace_entry(
        (
            ai_task_subentry_data(
                _MODEL_PROFILE_REF,
                title=subentry_title,
                task_name="Report task" if include_task_name else None,
                output_mode=output_mode,
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
    output_mode: str | None = None,
    *,
    extra_data: dict[str, object] | None = None,
    enable_diagnostics: bool = False,
) -> str:
    entry = _entry(output_mode, extra_data=extra_data)
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
        "custom_components.pydantic_ai_agent._model_validation.async_probe_model",
        new_callable=AsyncMock,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    entity_ids = [state.entity_id for state in hass.states.async_all("ai_task")]
    assert len(entity_ids) == 1
    return entity_ids[0]


def _assert_context_management_capability(capabilities: list[object]) -> None:
    assert any(
        isinstance(capability, SlidingWindowContextCapability)
        for capability in capabilities
    )


def _thinking_capabilities(capabilities: list[object]) -> list[Thinking]:
    return [
        capability for capability in capabilities if isinstance(capability, Thinking)
    ]


def _state(hass: HomeAssistant, entity_id: str) -> str:
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
    entry = _entry()
    added_entities: list[tuple[list[Entity], str | None]] = []

    def add_entities(new_entities, update_before_add=False, *, config_subentry_id=None):
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
    entry = _entry()
    subentry = next(iter(entry.subentries.values()))
    entity = PydanticAIAgentAITaskEntity(entry, subentry)
    assert entity.device_info is not None
    assert entity.device_info.get("name") == "Report task"
    assert entity._attr_name is None


def test_ai_task_entity_falls_back_to_subentry_title() -> None:
    entry = _entry(include_task_name=False, subentry_title="Title-only task")
    subentry = next(iter(entry.subentries.values()))
    entity = PydanticAIAgentAITaskEntity(entry, subentry)
    assert entity.device_info is not None
    assert entity.device_info.get("name") == "Title-only task"
    assert entity._attr_name is None


def test_ai_task_entity_features() -> None:
    entry = _entry()
    subentry = next(iter(entry.subentries.values()))
    entity = PydanticAIAgentAITaskEntity(entry, subentry)
    assert ai_task.AITaskEntityFeature.GENERATE_DATA in entity.supported_features
    assert ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS in entity.supported_features
    assert ai_task.AITaskEntityFeature.GENERATE_IMAGE not in entity.supported_features


async def test_plain_data_task_returns_text(hass: HomeAssistant) -> None:
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


async def test_ai_task_runtime_uses_configured_max_iterations(
    hass: HomeAssistant,
) -> None:
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
    assert request_limit_from_kwargs(agent.run_kwargs) == 26


async def test_ai_task_runtime_defaults_max_iterations(hass: HomeAssistant) -> None:
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
    assert request_limit_from_kwargs(agent.run_kwargs) == 30


async def test_structured_data_task_supports_test_model_without_patching_agent_run(
    hass: HomeAssistant,
) -> None:
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
