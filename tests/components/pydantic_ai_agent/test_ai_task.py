"""Test Pydantic AI Agent AI task entity setup and data generation."""

from unittest.mock import patch

from custom_components.pydantic_ai_agent.ai_task import (
    PydanticAIAgentAITaskEntity,
    async_setup_entry,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_AI_TASK_NAME,
    CONF_MAX_ITERATIONS,
    CONF_THINKING,
    CONF_WEB_SEARCH_ENABLED,
    DOMAIN,
    OUTPUT_MODE_NATIVE,
    PROVIDER_ANTHROPIC,
)
from homeassistant.components import ai_task
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import Entity
from pydantic_ai.models.test import TestModel
from tests.components.pydantic_ai_agent.support.builders import (
    model_profile_data,
    provider_runtime_data,
    provider_subentry_data,
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
from tests.components.pydantic_ai_agent.support.runtime import (
    assert_has_context_management_capability,
    enable_diagnostic_entities,
    loaded_ai_task_entry,
    only_entity_id,
    thinking_capabilities,
)
import voluptuous as vol

_PROVIDER_SUBENTRY_ID = "provider-1"
_MODEL_PROFILE_ID = "task-profile"
_MODEL_PROFILE_REF = f"{_PROVIDER_SUBENTRY_ID}:{_MODEL_PROFILE_ID}"


async def _setup_ai_task_entity(
    hass: HomeAssistant,
    *,
    extra_data: dict[str, object] | None = None,
    enable_diagnostics: bool = False,
) -> str:
    entry = loaded_ai_task_entry(
        extra_data=extra_data,
        provider_subentry=provider_subentry_data(
            subentry_id=_PROVIDER_SUBENTRY_ID,
            title="Hosted OpenAI",
            profile_id=_MODEL_PROFILE_ID,
            profile_name="Task Model",
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
    if enable_diagnostics:
        subentry = next(iter(entry.subentries.values()))
        enable_diagnostic_entities(
            hass,
            entry,
            subentry,
            "sensor",
            ("last_error_type",),
            name=str(subentry.data.get(CONF_AI_TASK_NAME, subentry.title)),
        )
        enable_diagnostic_entities(
            hass,
            entry,
            subentry,
            "binary_sensor",
            ("provider_healthy", "last_run_succeeded"),
            name=str(subentry.data.get(CONF_AI_TASK_NAME, subentry.title)),
        )

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    return only_entity_id(hass, "ai_task")


async def test_ai_task_subentries_add_separate_entities(
    hass: HomeAssistant,
) -> None:
    entry = loaded_ai_task_entry()
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
    entry = loaded_ai_task_entry()
    subentry = next(iter(entry.subentries.values()))
    entity = PydanticAIAgentAITaskEntity(entry, subentry)
    assert entity.device_info is not None
    assert entity.device_info.get("name") == "Report task"
    assert entity._attr_name is None


def test_ai_task_entity_falls_back_to_subentry_title() -> None:
    entry = loaded_ai_task_entry(
        include_task_name=False,
        subentry_title="Title-only task",
    )
    subentry = next(iter(entry.subentries.values()))
    entity = PydanticAIAgentAITaskEntity(entry, subentry)
    assert entity.device_info is not None
    assert entity.device_info.get("name") == "Title-only task"


def test_ai_task_entity_exposes_web_search_attribute() -> None:
    entry = loaded_ai_task_entry(web_search_enabled=True)
    subentry = next(iter(entry.subentries.values()))
    entity = PydanticAIAgentAITaskEntity(entry, subentry)
    attributes = entity.extra_state_attributes
    assert attributes is not None
    assert attributes[CONF_WEB_SEARCH_ENABLED] is True
    assert entity._attr_name is None


def test_ai_task_entity_features() -> None:
    entry = loaded_ai_task_entry()
    subentry = next(iter(entry.subentries.values()))
    entity = PydanticAIAgentAITaskEntity(entry, subentry)
    assert ai_task.AITaskEntityFeature.GENERATE_DATA in entity.supported_features
    assert ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS in entity.supported_features
    assert ai_task.AITaskEntityFeature.GENERATE_IMAGE not in entity.supported_features


def test_ai_task_entity_uses_runtime_structured_output_mode() -> None:
    entry = loaded_ai_task_entry(
        provider_subentry=provider_subentry_data(
            subentry_id=_PROVIDER_SUBENTRY_ID,
            title="Hosted OpenAI",
            profile_id=_MODEL_PROFILE_ID,
            profile_name="Task Model",
            model="gpt-test",
            model_profiles={
                _MODEL_PROFILE_ID: model_profile_data(
                    profile_id=_MODEL_PROFILE_ID,
                    name="Task Model",
                    model="gpt-test",
                    extra_data={
                        "supports_tools": False,
                        "structured_output_support": "json_schema",
                        "openai_supports_strict_tool_definition": False,
                    },
                )
            },
        ),
    )
    subentry = next(iter(entry.subentries.values()))
    entity = PydanticAIAgentAITaskEntity(entry, subentry)

    assert entity.extra_state_attributes is not None
    assert entity.extra_state_attributes["structured_output_mode"] == OUTPUT_MODE_NATIVE


async def test_plain_data_task_returns_text(
    hass: HomeAssistant,
    mock_chat_model_for_profile: TestModel,
) -> None:
    del mock_chat_model_for_profile
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
    assert_has_context_management_capability(
        agent_class.call_args.kwargs["capabilities"]
    )


async def test_ai_task_runtime_uses_configured_max_iterations(
    hass: HomeAssistant,
    mock_chat_model_for_profile: TestModel,
) -> None:
    del mock_chat_model_for_profile
    entity_id = await _setup_ai_task_entity(
        hass,
        extra_data={CONF_MAX_ITERATIONS: 26},
    )
    agent = _Agent(stream_text="plain result", output="plain result")

    with patch("custom_components.pydantic_ai_agent.entity.Agent", return_value=agent):
        result = await ai_task.async_generate_data(
            hass,
            task_name="Plain task",
            entity_id=entity_id,
            instructions="Generate text",
        )

    assert result.data == "plain result"
    assert request_limit_from_kwargs(agent.run_kwargs) == 26


async def test_ai_task_runtime_keeps_explicit_disabled_thinking_capability(
    hass: HomeAssistant,
    mock_chat_model_for_profile: TestModel,
) -> None:
    del mock_chat_model_for_profile
    entity_id = await _setup_ai_task_entity(
        hass,
        extra_data={CONF_THINKING: False},
    )
    agent = _Agent(stream_text="plain result", output="plain result")

    with patch(
        "custom_components.pydantic_ai_agent.entity.Agent",
        return_value=agent,
    ) as agent_class:
        result = await ai_task.async_generate_data(
            hass,
            task_name="Plain task",
            entity_id=entity_id,
            instructions="Generate text",
        )

    assert result.data == "plain result"
    thinking = thinking_capabilities(agent_class.call_args.kwargs["capabilities"])
    assert len(thinking) == 1
    assert thinking[0].effort is False


async def test_structured_data_task_supports_test_model_without_patching_agent_run(
    hass: HomeAssistant,
) -> None:
    entity_id = await _setup_ai_task_entity(hass)

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
