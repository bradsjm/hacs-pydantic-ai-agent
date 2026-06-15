"""Test Pydantic AI Agent conversation entities."""

import pytest
from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_LOGFIRE_INCLUDE_CONTENT,
    CONF_LOGFIRE_TOKEN,
    CONF_STREAMING_ENABLED,
    CONF_VIRTUAL_WORKSPACE_ENABLED,
    DOMAIN,
)
from custom_components.pydantic_ai_agent.conversation import (
    PydanticAIConversationEntity,
    async_setup_entry,
)
from custom_components.pydantic_ai_agent.entity import unique_id_for_subentry_entity
from homeassistant.components import conversation
from homeassistant.const import CONF_NAME
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import llm
from homeassistant.util import slugify
from tests.components.pydantic_ai_agent.support.builders import (
    conversation_subentry_data,
    model_profile_data,
    provider_runtime_data,
    provider_subentry_data,
    workspace_entry,
    workspace_runtime_data,
)

_PROVIDER_SUBENTRY_ID = "provider-1"
_MODEL_PROFILE_ID = "model-profile-1"
_MODEL_PROFILE_REF = f"{_PROVIDER_SUBENTRY_ID}:{_MODEL_PROFILE_ID}"


def _entry(
    llm_hass_api=None,
    skills=None,
    *,
    streaming_enabled=True,
    virtual_workspace_enabled=False,
    web_fetch_enabled=False,
    model_settings=None,
    extra_data=None,
):
    entry = workspace_entry(
        (
            conversation_subentry_data(
                _MODEL_PROFILE_REF,
                llm_hass_api=llm_hass_api,
                skills=skills,
                streaming_enabled=streaming_enabled,
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


def _entry_with_conversation_subentries(*, logfire=False):
    data: dict[str, object] = {CONF_NAME: "Workspace"}
    if logfire:
        data[CONF_LOGFIRE_TOKEN] = "lf-token"
        data[CONF_LOGFIRE_INCLUDE_CONTENT] = True
    entry = workspace_entry(
        (
            conversation_subentry_data(
                f"{_PROVIDER_SUBENTRY_ID}:kitchen-model",
                title="Kitchen Agent",
                agent_name="Kitchen Agent",
            ),
            conversation_subentry_data(
                f"{_PROVIDER_SUBENTRY_ID}:garage-model",
                title="Garage Agent",
                agent_name="Garage Agent",
            ),
            provider_subentry_data(
                subentry_id=_PROVIDER_SUBENTRY_ID,
                title="Hosted OpenAI",
                model_profiles={
                    "kitchen-model": model_profile_data(
                        profile_id="kitchen-model",
                        name="Kitchen Model",
                        model="gpt-kitchen",
                    ),
                    "garage-model": model_profile_data(
                        profile_id="garage-model",
                        name="Garage Model",
                        model="gpt-garage",
                    ),
                },
                default_model_profile_id="kitchen-model",
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


def _state(hass, entity_id):
    state = hass.states.get(entity_id)
    assert state is not None
    return state.state


def _enable_diagnostic_entities(hass, entry, subentry, entity_domain, keys):
    ereg = er.async_get(hass)
    name = str(subentry.data[CONF_AGENT_NAME])
    for key in keys:
        ereg.async_get_or_create(
            entity_domain,
            DOMAIN,
            unique_id_for_subentry_entity(entry, subentry, key),
            suggested_object_id=slugify(f"{name} {key}"),
        )


def test_conversation_entity_controls_home_assistant_with_llm_api():
    entry = _entry([llm.LLM_API_ASSIST])
    entity = PydanticAIConversationEntity(entry, next(iter(entry.subentries.values())))
    attributes = entity.extra_state_attributes
    assert entity.supported_features == conversation.ConversationEntityFeature.CONTROL
    assert attributes is not None
    assert attributes["ha_tools_enabled"] is True


def test_conversation_entity_advertises_streaming():
    entry = _entry(None)
    entity = PydanticAIConversationEntity(entry, next(iter(entry.subentries.values())))
    assert entity.supports_streaming is True
    attributes = entity.extra_state_attributes
    assert attributes is not None
    assert attributes[CONF_STREAMING_ENABLED] is True


def test_conversation_entity_disables_streaming_when_configured():
    entry = _entry(None, streaming_enabled=False)
    entity = PydanticAIConversationEntity(entry, next(iter(entry.subentries.values())))
    assert entity.supports_streaming is False
    attributes = entity.extra_state_attributes
    assert attributes is not None
    assert attributes[CONF_STREAMING_ENABLED] is False


@pytest.mark.parametrize(
    ("llm_hass_api", "vw", "wf", "skills", "features"),
    [
        (
            [llm.LLM_API_ASSIST],
            False,
            False,
            None,
            conversation.ConversationEntityFeature.CONTROL,
        ),
        (None, True, False, None, 0),
        (None, False, True, None, 0),
        (None, False, False, ["kitchen-skill"], 0),
    ],
)
def test_conversation_entity_advertises_streaming_for_tool_sources(
    llm_hass_api, vw, wf, skills, features
):
    entity = PydanticAIConversationEntity(
        _entry(
            llm_hass_api,
            skills=skills,
            virtual_workspace_enabled=vw,
            web_fetch_enabled=wf,
        ),
        next(
            iter(
                _entry(
                    llm_hass_api,
                    skills=skills,
                    virtual_workspace_enabled=vw,
                    web_fetch_enabled=wf,
                ).subentries.values()
            )
        ),
    )
    assert entity.supports_streaming is True
    assert entity.supported_features == features


def test_conversation_entity_without_llm_api_has_no_control():
    entity = PydanticAIConversationEntity(
        _entry(None), next(iter(_entry(None).subentries.values()))
    )
    attributes = entity.extra_state_attributes
    assert entity.supported_features == 0
    assert attributes is not None
    assert attributes["ha_tools_enabled"] is False


def test_conversation_entity_requires_literal_virtual_workspace_true():
    entity = PydanticAIConversationEntity(
        _entry(None, extra_data={CONF_VIRTUAL_WORKSPACE_ENABLED: "true"}),
        next(iter(_entry(None).subentries.values())),
    )
    attributes = entity.extra_state_attributes
    assert attributes is not None
    assert attributes["virtual_workspace_enabled"] is False


async def test_conversation_subentries_add_separate_entity_agents(hass):
    entry = _entry_with_conversation_subentries()
    added = []

    def async_add_entities(
        new_entities, update_before_add=False, *, config_subentry_id=None
    ):
        added.append((list(new_entities), config_subentry_id))

    await async_setup_entry(hass, entry, async_add_entities)
    subentries = list(entry.subentries.values())
    assert [item[1] for item in added] == [
        subentries[0].subentry_id,
        subentries[1].subentry_id,
    ]


async def test_diagnostic_entity_defaults_are_respected(hass):
    entry = _entry(None, skills=["skill-1", "skill-2"], virtual_workspace_enabled=True)
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    ereg = er.async_get(hass)
    for eid in (
        "sensor.kitchen_agent_last_run_model_profile",
        "sensor.kitchen_agent_primary_language_model",
    ):
        entry = ereg.async_get(eid)
        assert entry is not None
        assert entry.disabled_by is er.RegistryEntryDisabler.INTEGRATION
        assert hass.states.get(eid) is None
    for eid in (
        "binary_sensor.kitchen_agent_provider_healthy",
        "binary_sensor.kitchen_agent_assist_enabled",
    ):
        entry = ereg.async_get(eid)
        assert entry is not None
        assert entry.disabled_by is None
    assert _state(hass, "sensor.kitchen_agent_skills_enabled") == "2"
