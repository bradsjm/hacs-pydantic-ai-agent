"""Test Pydantic AI Agent conversation entities."""

from unittest.mock import AsyncMock, patch

from homeassistant import config_entries
from homeassistant.components import conversation
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import llm
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.pydantic_ai_agent import PydanticAIAgentRuntimeData
from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_MODEL,
    CONF_PROVIDER_MODE,
    DOMAIN,
    PROVIDER_OPENAI,
    SUBENTRY_TYPE_CONVERSATION,
)
from custom_components.pydantic_ai_agent.conversation import (
    PydanticAIConversationEntity,
    async_setup_entry,
)


def _entry(llm_hass_api: list[str] | None) -> MockConfigEntry:
    """Return a config entry with one conversation subentry."""
    subentry_data: dict[str, object] = {
        CONF_AGENT_NAME: "Kitchen Agent",
        CONF_MODEL: "gpt-test",
    }
    if llm_hass_api is not None:
        subentry_data[CONF_LLM_HASS_API] = llm_hass_api

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
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
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


def _entry_with_conversation_subentries() -> MockConfigEntry:
    """Return a config entry with two conversation subentries."""
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
                "data": {
                    CONF_AGENT_NAME: "Kitchen Agent",
                    CONF_MODEL: "gpt-kitchen",
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
            },
            {
                "data": {
                    CONF_AGENT_NAME: "Garage Agent",
                    CONF_MODEL: "gpt-garage",
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Garage Agent",
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


def test_conversation_entity_controls_home_assistant_with_llm_api() -> None:
    """Test LLM API selection enables Home Assistant control support."""
    entry = _entry([llm.LLM_API_ASSIST])
    subentry = next(iter(entry.subentries.values()))

    entity = PydanticAIConversationEntity(entry, subentry)

    assert entity.supported_features == conversation.ConversationEntityFeature.CONTROL
    assert entity.extra_state_attributes["ha_tools_enabled"] is True
    assert entity.extra_state_attributes["ha_llm_api"] == [llm.LLM_API_ASSIST]


def test_conversation_entity_without_llm_api_has_no_control() -> None:
    """Test missing LLM API leaves Home Assistant control support disabled."""
    entry = _entry(None)
    subentry = next(iter(entry.subentries.values()))

    entity = PydanticAIConversationEntity(entry, subentry)

    assert entity.supported_features == 0
    assert entity.extra_state_attributes["ha_tools_enabled"] is False
    assert entity.extra_state_attributes["ha_llm_api"] is None


async def test_conversation_subentries_add_separate_entity_agents(
    hass: HomeAssistant,
) -> None:
    """Test each conversation subentry is exposed as a separate entity agent."""
    entry = _entry_with_conversation_subentries()
    added_entities: list[tuple[list[PydanticAIConversationEntity], str | None]] = []

    def async_add_entities(
        entities: list[PydanticAIConversationEntity],
        _update_before_add: bool = False,
        config_subentry_id: str | None = None,
    ) -> None:
        added_entities.append((entities, config_subentry_id))

    await async_setup_entry(hass, entry, async_add_entities)

    subentries = list(entry.subentries.values())
    assert [item[1] for item in added_entities] == [
        subentries[0].subentry_id,
        subentries[1].subentry_id,
    ]
    assert [item[0][0].unique_id for item in added_entities] == [
        subentries[0].subentry_id,
        subentries[1].subentry_id,
    ]
    assert [item[0][0].device_info for item in added_entities] == [
        {
            "identifiers": {(DOMAIN, subentries[0].subentry_id)},
            "name": "Kitchen Agent",
            "manufacturer": "Pydantic AI",
            "model": "gpt-kitchen",
            "entry_type": dr.DeviceEntryType.SERVICE,
        },
        {
            "identifiers": {(DOMAIN, subentries[1].subentry_id)},
            "name": "Garage Agent",
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

    result = await conversation.async_converse(
        hass,
        "hello",
        None,
        Context(),
        agent_id=entity_ids[0],
    )

    assert result.response.speech["plain"]["speech"] == (
        "Pydantic AI Agent is configured, but provider chat runtime is not implemented yet."
    )
