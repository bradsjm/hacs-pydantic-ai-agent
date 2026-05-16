"""Test Pydantic AI Agent AI task entities."""

from collections.abc import AsyncGenerator, Iterable
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai import PartEndEvent, PartStartEvent, TextPart, ToolCallPart
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.components import ai_task
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity import Entity
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.pydantic_ai_agent import PydanticAIAgentRuntimeData
from custom_components.pydantic_ai_agent.ai_task import (
    PydanticAIAgentAITaskEntity,
    async_setup_entry,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_MODEL,
    CONF_OUTPUT_MODE,
    CONF_PROVIDER_MODE,
    DOMAIN,
    OUTPUT_MODE_NATIVE,
    OUTPUT_MODE_PROMPTED,
    OUTPUT_MODE_TOOL,
    PROVIDER_OPENAI,
    SUBENTRY_TYPE_AI_TASK,
)


class _EventStream:
    """Async iterator over Pydantic AI stream events."""

    def __init__(self, *events: object) -> None:
        """Initialize the event stream."""
        self._events = iter(events)

    def __aiter__(self) -> "_EventStream":
        """Return the async iterator."""
        return self

    async def __anext__(self) -> object:
        """Return the next stream event."""
        try:
            return next(self._events)
        except StopIteration as err:
            raise StopAsyncIteration from err


def _entry(output_mode: str | None = None) -> MockConfigEntry:
    """Return a config entry with one AI task subentry."""
    subentry_data = {CONF_MODEL: "task-model"}
    if output_mode is not None:
        subentry_data[CONF_OUTPUT_MODE] = output_mode
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Hosted OpenAI",
        data={
            CONF_NAME: "Hosted OpenAI",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI,
            CONF_API_KEY: "sk-test",
        },
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "data": subentry_data,
                "subentry_type": SUBENTRY_TYPE_AI_TASK,
                "title": "Task Model",
                "unique_id": None,
            },
        ),
        options={},
        unique_id=None,
    )
    entry.runtime_data = PydanticAIAgentRuntimeData(
        provider_mode=PROVIDER_OPENAI,
        name="Hosted OpenAI",
        api_key="sk-test",
        base_url=None,
    )
    return entry


async def _setup_ai_task_entity(
    hass: HomeAssistant, output_mode: str | None = None
) -> str:
    """Set up an AI task config entry and return its entity ID."""
    entry = _entry(output_mode)
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


def _model_stream_factory(text: str):
    """Return a model stream side effect for the given text."""

    @asynccontextmanager
    async def stream(*_args: object, **_kwargs: object) -> AsyncGenerator[_EventStream]:
        yield _EventStream(
            PartStartEvent(index=0, part=TextPart(content=text)),
            PartEndEvent(index=0, part=TextPart(content=text)),
        )

    return stream


def _tool_output_stream_factory(args: dict[str, object]):
    """Return a model stream side effect for output tool data."""

    @asynccontextmanager
    async def stream(*_args: object, **_kwargs: object) -> AsyncGenerator[_EventStream]:
        yield _EventStream(
            PartEndEvent(
                index=0,
                part=ToolCallPart(
                    tool_name="pydantic_ai_agent_output_structured_task",
                    args=args,
                    tool_call_id="tool-1",
                ),
            )
        )

    return stream


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
    assert entity.unique_id == subentry.subentry_id


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
            "custom_components.pydantic_ai_agent.entity.openai_chat_model",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.model_request_stream",
            side_effect=_model_stream_factory("plain result"),
        ),
    ):
        result = await ai_task.async_generate_data(
            hass,
            task_name="Plain task",
            entity_id=entity_id,
            instructions="Generate text",
        )

    assert result.data == "plain result"


@pytest.mark.parametrize(
    ("output_mode", "stream_factory"),
    [
        (OUTPUT_MODE_TOOL, _tool_output_stream_factory({"name": "Kitchen"})),
        (OUTPUT_MODE_NATIVE, _model_stream_factory('{"name":"Kitchen"}')),
        (OUTPUT_MODE_PROMPTED, _model_stream_factory('{"name":"Kitchen"}')),
    ],
)
async def test_structured_data_task_returns_parsed_json(
    hass: HomeAssistant,
    output_mode: str,
    stream_factory: object,
) -> None:
    """Test a structured data task returns parsed JSON."""
    entity_id = await _setup_ai_task_entity(hass, output_mode)

    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.openai_chat_model",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.model_request_stream",
            side_effect=stream_factory,
        ) as model_request_stream,
    ):
        result = await ai_task.async_generate_data(
            hass,
            task_name="Structured task",
            entity_id=entity_id,
            instructions="Generate JSON",
            structure=vol.Schema({vol.Required("name"): str}),
        )

    assert result.data == {"name": "Kitchen"}
    request_parameters = model_request_stream.call_args.kwargs[
        "model_request_parameters"
    ]
    assert request_parameters.output_mode == output_mode
    if output_mode == OUTPUT_MODE_TOOL:
        assert request_parameters.output_tools[0].kind == "output"
    else:
        assert request_parameters.output_object.name == (
            "pydantic_ai_agent_output_structured_task"
        )


async def test_structured_data_task_rejects_malformed_json(
    hass: HomeAssistant,
) -> None:
    """Test malformed structured data raises a Home Assistant error."""
    entity_id = await _setup_ai_task_entity(hass)

    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.openai_chat_model",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.model_request_stream",
            side_effect=_model_stream_factory("not json"),
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
            "custom_components.pydantic_ai_agent.entity.openai_chat_model",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.model_request_stream",
            side_effect=_model_stream_factory('{"name":1}'),
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
