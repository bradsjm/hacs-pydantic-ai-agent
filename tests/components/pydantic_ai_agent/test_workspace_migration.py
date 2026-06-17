"""Tests for config-entry migration and removal."""

from typing import Any, cast

from custom_components.pydantic_ai_agent import (
    async_migrate_entry,
    async_remove_entry,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_AI_TASK_NAME,
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
    CONF_CONTEXT_MANAGEMENT_MODE,
    CONF_CONTEXT_SUMMARIZATION_MODEL_REF,
    CONF_CONTEXT_WINDOW_SOURCE,
    CONF_CONTEXT_WINDOW_TOKENS,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_PRIMARY_MODEL_REF,
    CONF_TEMPLATED_EXTRA_BODY,
    CONTEXT_MANAGEMENT_CONTEXT_MANAGER,
    CONTEXT_MANAGEMENT_SLIDING_WINDOW,
    CONTEXT_WINDOW_SOURCE_DEFAULT,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    DOMAIN,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_PROVIDER,
)
from homeassistant import config_entries
from homeassistant.const import CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.storage import Store
from pytest_homeassistant_custom_component.common import MockConfigEntry


async def test_migration_removes_removed_in_repo_llm_api_refs(
    hass: HomeAssistant,
) -> None:
    """Test v2.0 migration removes API IDs for the deleted in-repo semantic API."""
    entry = MockConfigEntry(
        version=2,
        minor_version=0,
        domain=DOMAIN,
        title="Workspace",
        data={CONF_NAME: "Workspace"},
        subentries_data=(
            {
                "subentry_id": "conversation-1",
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Agent",
                "unique_id": None,
                "data": {
                    CONF_AGENT_NAME: "Agent",
                    CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
                    CONF_LLM_HASS_API: [
                        "pydantic_ai_agent_home_semantic_workspace",
                        "external_llm_api",
                    ],
                },
            },
            {
                "subentry_id": "task-1",
                "subentry_type": SUBENTRY_TYPE_AI_TASK,
                "title": "Task",
                "unique_id": None,
                "data": {
                    CONF_AI_TASK_NAME: "Task",
                    CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
                    CONF_LLM_HASS_API: [
                        "pydantic_ai_agent_home_semantic_workspace",
                    ],
                },
            },
        ),
        source=config_entries.SOURCE_USER,
        unique_id=None,
    )
    entry.add_to_hass(hass)

    assert await async_migrate_entry(hass, cast(Any, entry))

    assert entry.minor_version == 4
    assert entry.subentries["conversation-1"].data[CONF_LLM_HASS_API] == [
        "external_llm_api"
    ]
    assert CONF_LLM_HASS_API not in entry.subentries["task-1"].data


async def test_migration_removes_removed_diagnostic_entities(
    hass: HomeAssistant,
) -> None:
    """Test v2.0 migration removes registry entries for deleted diagnostics."""
    entry = MockConfigEntry(
        version=2,
        minor_version=0,
        domain=DOMAIN,
        title="Workspace",
        data={CONF_NAME: "Workspace"},
        source=config_entries.SOURCE_USER,
        unique_id=None,
    )
    entry.add_to_hass(hass)
    entity_registry = er.async_get(hass)
    device_registry = dr.async_get(hass)
    device_registry.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={(DOMAIN, entry.entry_id)},
        name="Workspace",
    )
    removed_entities = (
        ("binary_sensor", "semantic_index_ready"),
        ("sensor", "semantic_index_generation"),
        ("sensor", "semantic_document_count"),
        ("sensor", "semantic_last_refresh_duration"),
    )
    for domain, key in removed_entities:
        entity_registry.async_get_or_create(
            domain,
            DOMAIN,
            f"{DOMAIN}_{entry.entry_id}_{key}",
            config_entry=entry,
            suggested_object_id=f"workspace_{key}",
        )

    assert await async_migrate_entry(hass, cast(Any, entry))

    assert (
        device_registry.async_get_device(identifiers={(DOMAIN, entry.entry_id)}) is None
    )
    for domain, key in removed_entities:
        assert (
            entity_registry.async_get_entity_id(
                domain,
                DOMAIN,
                f"{DOMAIN}_{entry.entry_id}_{key}",
            )
            is None
        )


async def test_migration_moves_chat_template_kwargs_to_templated_extra_body(
    hass: HomeAssistant,
) -> None:
    """Test v2.1 migration rewrites old chat template kwargs rows."""
    entry = MockConfigEntry(
        version=2,
        minor_version=1,
        domain=DOMAIN,
        title="Workspace",
        data={CONF_NAME: "Workspace"},
        subentries_data=(
            {
                "subentry_id": "provider-1",
                "subentry_type": SUBENTRY_TYPE_PROVIDER,
                "title": "Provider",
                "unique_id": None,
                "data": {
                    CONF_MODEL_PROFILES: {
                        "profile-1": {
                            "id": "profile-1",
                            CONF_NAME: "Fast GPT",
                            CONF_MODEL: "gpt-test",
                            CONF_MODEL_SETTINGS: {
                                "chat_template_kwargs": [
                                    {
                                        CONF_CHAT_TEMPLATE_KWARG_KEY: "enable_thinking",
                                        CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: (
                                            "{{ true }}"
                                        ),
                                    }
                                ],
                                "temperature": 0.2,
                            },
                        }
                    }
                },
            },
        ),
        source=config_entries.SOURCE_USER,
        unique_id=None,
    )
    entry.add_to_hass(hass)

    assert await async_migrate_entry(hass, cast(Any, entry))

    migrated_profile = entry.subentries["provider-1"].data[CONF_MODEL_PROFILES][
        "profile-1"
    ]
    assert entry.minor_version == 4
    assert "chat_template_kwargs" not in migrated_profile[CONF_MODEL_SETTINGS]
    assert migrated_profile[CONF_MODEL_SETTINGS][CONF_TEMPLATED_EXTRA_BODY] == [
        {
            CONF_CHAT_TEMPLATE_KWARG_KEY: "chat_template_kwargs.enable_thinking",
            CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ true }}",
        }
    ]
    assert migrated_profile[CONF_MODEL_SETTINGS]["temperature"] == 0.2


async def test_migration_strips_legacy_ai_task_output_mode(
    hass: HomeAssistant,
) -> None:
    """Test v2.2 migration strips stored output_mode from AI task subentries."""
    entry = MockConfigEntry(
        version=2,
        minor_version=2,
        domain=DOMAIN,
        title="Workspace",
        data={CONF_NAME: "Workspace"},
        subentries_data=(
            {
                "subentry_id": "task-legacy",
                "subentry_type": SUBENTRY_TYPE_AI_TASK,
                "title": "Legacy Task",
                "unique_id": None,
                "data": {
                    CONF_AI_TASK_NAME: "Legacy Task",
                    CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
                    "output_mode": "tool",
                },
            },
            {
                "subentry_id": "task-clean",
                "subentry_type": SUBENTRY_TYPE_AI_TASK,
                "title": "Clean Task",
                "unique_id": None,
                "data": {
                    CONF_AI_TASK_NAME: "Clean Task",
                    CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
                },
            },
            {
                "subentry_id": "conversation-1",
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Agent",
                "unique_id": None,
                "data": {
                    CONF_AGENT_NAME: "Agent",
                    CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
                },
            },
        ),
        source=config_entries.SOURCE_USER,
        unique_id=None,
    )
    entry.add_to_hass(hass)

    assert await async_migrate_entry(hass, cast(Any, entry))

    assert entry.minor_version == 4

    legacy_data = entry.subentries["task-legacy"].data
    assert "output_mode" not in legacy_data
    assert legacy_data[CONF_AI_TASK_NAME] == "Legacy Task"
    assert legacy_data[CONF_PRIMARY_MODEL_REF] == "provider-1:profile-1"

    clean_data = entry.subentries["task-clean"].data
    assert "output_mode" not in clean_data
    assert clean_data[CONF_AI_TASK_NAME] == "Clean Task"


async def test_migration_backfills_context_management_defaults(
    hass: HomeAssistant,
) -> None:
    """Test v2.4 migration adds context settings to existing subentries."""
    entry = MockConfigEntry(
        version=2,
        minor_version=3,
        domain=DOMAIN,
        title="Workspace",
        data={CONF_NAME: "Workspace"},
        subentries_data=(
            {
                "subentry_id": "provider-1",
                "subentry_type": SUBENTRY_TYPE_PROVIDER,
                "title": "Provider",
                "unique_id": None,
                "data": {
                    CONF_MODEL_PROFILES: {
                        "missing": {"id": "missing", CONF_MODEL: "missing"},
                        "string": {
                            "id": "string",
                            CONF_MODEL: "string",
                            CONF_CONTEXT_WINDOW_TOKENS: "200",
                        },
                        "negative": {
                            "id": "negative",
                            CONF_MODEL: "negative",
                            CONF_CONTEXT_WINDOW_TOKENS: -1,
                        },
                    }
                },
            },
            {
                "subentry_id": "conversation-1",
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Agent",
                "unique_id": None,
                "data": {
                    CONF_AGENT_NAME: "Agent",
                    CONF_PRIMARY_MODEL_REF: "provider-1:missing",
                    CONF_CONTEXT_SUMMARIZATION_MODEL_REF: "",
                },
            },
            {
                "subentry_id": "task-1",
                "subentry_type": SUBENTRY_TYPE_AI_TASK,
                "title": "Task",
                "unique_id": None,
                "data": {
                    CONF_AI_TASK_NAME: "Task",
                    CONF_PRIMARY_MODEL_REF: "provider-1:missing",
                },
            },
        ),
        source=config_entries.SOURCE_USER,
        unique_id=None,
    )
    entry.add_to_hass(hass)

    assert await async_migrate_entry(hass, cast(Any, entry))

    assert entry.minor_version == 4
    profiles = entry.subentries["provider-1"].data[CONF_MODEL_PROFILES]
    assert profiles["missing"][CONF_CONTEXT_WINDOW_TOKENS] == (
        DEFAULT_CONTEXT_WINDOW_TOKENS
    )
    assert profiles["string"][CONF_CONTEXT_WINDOW_TOKENS] == 200
    assert profiles["negative"][CONF_CONTEXT_WINDOW_TOKENS] == (
        DEFAULT_CONTEXT_WINDOW_TOKENS
    )
    assert profiles["missing"][CONF_CONTEXT_WINDOW_SOURCE] == (
        CONTEXT_WINDOW_SOURCE_DEFAULT
    )
    assert entry.subentries["conversation-1"].data[CONF_CONTEXT_MANAGEMENT_MODE] == (
        CONTEXT_MANAGEMENT_CONTEXT_MANAGER
    )
    assert (
        CONF_CONTEXT_SUMMARIZATION_MODEL_REF
        not in entry.subentries["conversation-1"].data
    )
    assert entry.subentries["task-1"].data[CONF_CONTEXT_MANAGEMENT_MODE] == (
        CONTEXT_MANAGEMENT_SLIDING_WINDOW
    )

    subentry_data = {key: dict(value.data) for key, value in entry.subentries.items()}
    assert await async_migrate_entry(hass, cast(Any, entry))
    assert {key: dict(value.data) for key, value in entry.subentries.items()} == (
        subentry_data
    )


async def test_remove_entry_removes_removed_memory_store(
    hass: HomeAssistant,
) -> None:
    """Test workspace removal deletes obsolete in-repo semantic memory storage."""
    entry = MockConfigEntry(
        version=2,
        minor_version=2,
        domain=DOMAIN,
        title="Workspace",
        data={CONF_NAME: "Workspace"},
        source=config_entries.SOURCE_USER,
        unique_id=None,
    )
    entry.add_to_hass(hass)
    store_key = f"{DOMAIN}.home_semantic.{entry.entry_id}"
    store: Store[dict[str, Any]] = Store(hass, 1, store_key)
    await store.async_save({"schema_version": 1})

    await async_remove_entry(hass, cast(Any, entry))

    assert await Store[dict[str, Any]](hass, 1, store_key).async_load() is None
