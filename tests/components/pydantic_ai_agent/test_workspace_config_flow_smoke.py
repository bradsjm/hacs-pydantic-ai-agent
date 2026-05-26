"""Smoke tests for the workspace-first config flow."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.pydantic_ai_agent import (
    ProviderRuntimeData,
    WorkspaceRuntimeData,
)
from custom_components.pydantic_ai_agent.conversation import (
    PydanticAIConversationEntity,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_AI_TASK_NAME,
    CONF_BASE_URL,
    CONF_CUSTOM_MODEL_NAMES,
    CONF_ENABLE_SKILLS,
    CONF_ENABLED,
    CONF_FALLBACK_MODEL_REFS,
    CONF_DEFAULT_SKILLS_FOLDER,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_METADATA,
    CONF_PROVIDER_MODE,
    CONF_SKILLS_FOLDER,
    DEFAULT_SKILLS_FOLDER,
    DOMAIN,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_PROVIDER,
)
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.const import (
    CATALOG_RETRY_PROVIDER_ID,
    CONF_CATALOG_PROVIDER_ID,
    CONF_DRIVER,
    CONF_PROVIDER_ID,
    CONF_SELECTED_MODEL_IDS,
    CUSTOM_PROVIDER_ID,
)
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.models_dev import (
    CatalogLoadError,
)
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.catalog_cache import (
    catalog_manager,
)
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.types import (
    CatalogModelOption,
    CatalogProviderOption,
    CompactCatalog,
)

_TRANSLATIONS_PATH = (
    Path(__file__).parents[3]
    / "custom_components"
    / "pydantic_ai_agent"
    / "translations"
    / "en.json"
)


def _schema_default(data_schema: vol.Schema | None, field: str) -> Any:
    """Return a voluptuous top-level field default from a flow schema."""
    assert data_schema is not None
    for key in data_schema.schema:
        if key.schema == field:
            return key.default()
    raise AssertionError(f"Schema field {field} not found")


def _schema_select_options(data_schema: vol.Schema | None, field: str) -> list[Any]:
    """Return selector options for a voluptuous top-level field."""
    assert data_schema is not None
    for key, selector in data_schema.schema.items():
        if key.schema == field:
            return list(selector.config["options"])
    raise AssertionError(f"Schema field {field} not found")


def test_provider_edit_connection_translations_cover_rendered_schema() -> None:
    """Test edit-connection fields and sections have translations."""
    translations = json.loads(_TRANSLATIONS_PATH.read_text(encoding="utf-8"))
    step = translations["config_subentries"]["provider"]["step"]["edit_connection"]

    assert set(step["data"]) >= {
        CONF_NAME,
        CONF_PROVIDER_MODE,
        CONF_API_KEY,
        CONF_BASE_URL,
        CONF_CUSTOM_MODEL_NAMES,
        CONF_PROVIDER_EXTRA_BODY,
        CONF_PROVIDER_HEADERS,
    }
    assert set(step["sections"]) >= {"advanced_options", "customize_model_list"}
    assert step["sections"]["advanced_options"]["name"] == "Advanced Options"
    assert set(step["sections"]["advanced_options"]["data"]) >= {
        CONF_PROVIDER_EXTRA_BODY,
        CONF_PROVIDER_HEADERS,
    }
    assert (
        step["sections"]["customize_model_list"]["name"]
        == "Override provider model list"
    )
    assert set(step["sections"]["customize_model_list"]["data"]) >= {
        CONF_CUSTOM_MODEL_NAMES,
    }


def test_provider_wizard_translations_cover_rendered_steps() -> None:
    """Test guided provider wizard steps and errors have translations."""
    translations = json.loads(_TRANSLATIONS_PATH.read_text(encoding="utf-8"))
    provider = translations["config_subentries"]["provider"]
    steps = provider["step"]

    assert set(steps) >= {
        "pick_provider",
        "pick_driver",
        "wizard_connection",
        "model_filters",
        "pick_models",
    }
    assert provider["progress"]["load_model_catalog"]
    assert set(provider["error"]) >= {
        "model_catalog_unavailable",
        "model_required",
        "no_models_available",
    }
    assert steps["pick_provider"]["data"][CONF_PROVIDER_ID]
    assert steps["pick_driver"]["data"][CONF_DRIVER]
    assert steps["wizard_connection"]["data"][CONF_API_KEY]
    assert steps["model_filters"]["sections"]["advanced_filters"]["name"] == (
        "Advanced Filters"
    )
    assert steps["model_filters"]["sections"]["advanced_filters"]["data"][
        "hide_without_tool_call"
    ]
    assert steps["pick_models"]["data"][CONF_SELECTED_MODEL_IDS]


async def _loaded_workspace_entry(
    hass: HomeAssistant, subentries_data: tuple[dict[str, object], ...] = ()
) -> MockConfigEntry:
    """Return a loaded workspace entry for subentry flow tests."""
    entry = MockConfigEntry(
        version=2,
        minor_version=0,
        domain=DOMAIN,
        title="Workspace",
        data={
            CONF_NAME: "Workspace",
            CONF_DEFAULT_SKILLS_FOLDER: DEFAULT_SKILLS_FOLDER,
        },
        subentries_data=subentries_data,
        source=config_entries.SOURCE_USER,
        unique_id=None,
    )
    entry.add_to_hass(hass)
    with patch(
        "custom_components.pydantic_ai_agent.async_setup_entry",
        return_value=True,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    return entry


def _cache_provider_catalog(hass: HomeAssistant, catalog: CompactCatalog) -> None:
    """Cache a provider wizard catalog for flow tests."""
    manager = catalog_manager(hass)
    now = dt_util.utcnow()
    manager.catalog = catalog
    manager.loaded_at = now
    manager.last_used_at = now


def _provider_subentry_data() -> dict[str, object]:
    """Return a provider subentry with one enabled profile for flow tests."""
    return {
        "subentry_id": "provider-1",
        "subentry_type": SUBENTRY_TYPE_PROVIDER,
        "title": "OpenAI-compatible",
        "unique_id": None,
        "data": {
            CONF_NAME: "OpenAI-compatible",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "sk-test",
            CONF_PROVIDER_METADATA: {CONF_CATALOG_PROVIDER_ID: "openai"},
            CONF_MODEL_PROFILES: {
                "profile-1": {
                    "id": "profile-1",
                    CONF_NAME: "GPT Mini",
                    CONF_MODEL: "gpt-4.1-mini",
                    CONF_ENABLED: True,
                }
            },
        },
    }


async def test_provider_edit_connection_preserves_catalog_metadata(
    hass: HomeAssistant,
) -> None:
    """Test editing a guided provider keeps catalog metadata for profile filters."""
    entry = await _loaded_workspace_entry(hass, (_provider_subentry_data(),))
    provider_subentry = next(iter(entry.subentries.values()))

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": provider_subentry.subentry_id,
        },
    )
    assert result["type"] is FlowResultType.MENU

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"next_step_id": "edit_connection"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "edit_connection"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "OpenAI-compatible",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "sk-updated",
        },
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    updated_subentry = entry.subentries[provider_subentry.subentry_id]
    assert updated_subentry.data[CONF_PROVIDER_METADATA] == {
        CONF_CATALOG_PROVIDER_ID: "openai"
    }


async def test_provider_edit_connection_preserves_catalog_metadata_for_default_url(
    hass: HomeAssistant,
) -> None:
    """Test explicit default URLs keep guided provider catalog metadata."""
    entry = await _loaded_workspace_entry(hass, (_provider_subentry_data(),))
    provider_subentry = next(iter(entry.subentries.values()))

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": provider_subentry.subentry_id,
        },
    )
    assert result["type"] is FlowResultType.MENU

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"next_step_id": "edit_connection"}
    )
    assert result["type"] is FlowResultType.FORM

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "OpenAI-compatible",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "sk-updated",
            CONF_BASE_URL: "https://api.openai.com/v1",
        },
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    updated_subentry = entry.subentries[provider_subentry.subentry_id]
    assert updated_subentry.data[CONF_PROVIDER_METADATA] == {
        CONF_CATALOG_PROVIDER_ID: "openai"
    }


async def test_provider_edit_connection_clears_catalog_metadata_when_repointed(
    hass: HomeAssistant,
) -> None:
    """Test repointing a guided provider clears stale catalog metadata."""
    entry = await _loaded_workspace_entry(hass, (_provider_subentry_data(),))
    provider_subentry = next(iter(entry.subentries.values()))

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": provider_subentry.subentry_id,
        },
    )
    assert result["type"] is FlowResultType.MENU

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"next_step_id": "edit_connection"}
    )
    assert result["type"] is FlowResultType.FORM

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "OpenAI-compatible",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "sk-updated",
            CONF_BASE_URL: "https://api.deepseek.com/v1",
        },
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    updated_subentry = entry.subentries[provider_subentry.subentry_id]
    assert CONF_PROVIDER_METADATA not in updated_subentry.data


async def test_provider_reconfigure_menu_exposes_model_management(
    hass: HomeAssistant,
) -> None:
    """Test provider reconfigure separates connection, availability, and editing."""
    entry = await _loaded_workspace_entry(hass, (_provider_subentry_data(),))
    provider_subentry = next(iter(entry.subentries.values()))

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": provider_subentry.subentry_id,
        },
    )

    assert result["type"] is FlowResultType.MENU
    assert result["menu_options"] == [
        "edit_connection",
        "manage_models",
        "customize_model_profile",
    ]


async def test_provider_manage_models_preselects_enabled_profiles(
    hass: HomeAssistant,
) -> None:
    """Test availability management preselects enabled catalog profiles."""
    entry = await _loaded_workspace_entry(hass, (_provider_subentry_data(),))
    provider_subentry = next(iter(entry.subentries.values()))
    _cache_provider_catalog(hass, _wizard_catalog())

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": provider_subentry.subentry_id,
        },
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"next_step_id": "manage_models"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "manage_models"
    assert _schema_default(result["data_schema"], CONF_SELECTED_MODEL_IDS) == [
        "gpt-4.1-mini"
    ]
    assert _schema_select_options(result["data_schema"], CONF_SELECTED_MODEL_IDS) == [
        {"label": "GPT 4.1 (128K context)", "value": "gpt-4.1"},
        {"label": "GPT 4.1 Mini (128K context)", "value": "gpt-4.1-mini"},
    ]


async def test_provider_manage_models_includes_enabled_models_hidden_by_filters(
    hass: HomeAssistant,
) -> None:
    """Test enabled models hidden by default filters stay manageable."""
    provider_data = _provider_subentry_data()
    provider_config = cast(dict[str, Any], provider_data["data"])
    provider_config[CONF_MODEL_PROFILES]["profile-2"] = {
        "id": "profile-2",
        CONF_NAME: "No Tools",
        CONF_MODEL: "gpt-4.1-no-tools",
        CONF_ENABLED: True,
    }
    entry = await _loaded_workspace_entry(hass, (provider_data,))
    provider_subentry = next(iter(entry.subentries.values()))
    _cache_provider_catalog(hass, _wizard_catalog(hidden_openai_model=True))

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": provider_subentry.subentry_id,
        },
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"next_step_id": "manage_models"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "manage_model_filters"

    result = await hass.config_entries.subentries.async_configure(result["flow_id"], {})

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "manage_models"
    assert _schema_default(result["data_schema"], CONF_SELECTED_MODEL_IDS) == [
        "gpt-4.1-mini",
        "gpt-4.1-no-tools",
    ]
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_SELECTED_MODEL_IDS: ["gpt-4.1-mini"]}
    )

    assert result["type"] is FlowResultType.ABORT
    updated_profiles = entry.subentries[provider_subentry.subentry_id].data[
        CONF_MODEL_PROFILES
    ]
    assert updated_profiles["profile-2"][CONF_ENABLED] is False


async def test_provider_manage_models_matches_catalog_after_metadata_cleared(
    hass: HomeAssistant,
) -> None:
    """Test stale profiles do not block catalog matching without metadata."""
    provider_data = _provider_subentry_data()
    provider_config = cast(dict[str, Any], provider_data["data"])
    provider_config.pop(CONF_PROVIDER_METADATA)
    provider_config[CONF_MODEL_PROFILES]["profile-old"] = {
        "id": "profile-old",
        CONF_NAME: "Old Model",
        CONF_MODEL: "gpt-removed",
        CONF_ENABLED: True,
    }
    entry = await _loaded_workspace_entry(hass, (provider_data,))
    provider_subentry = next(iter(entry.subentries.values()))
    _cache_provider_catalog(hass, _wizard_catalog())

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": provider_subentry.subentry_id,
        },
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"next_step_id": "manage_models"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "manage_models"
    assert _schema_default(result["data_schema"], CONF_SELECTED_MODEL_IDS) == [
        "gpt-4.1-mini",
        "gpt-removed",
    ]


async def test_provider_manage_models_discovers_manual_provider_models(
    hass: HomeAssistant,
) -> None:
    """Test manual providers without a model list manage discovered models."""
    provider_data = _provider_subentry_data()
    provider_config = cast(dict[str, Any], provider_data["data"])
    provider_config.pop(CONF_PROVIDER_METADATA)
    provider_config[CONF_MODEL_PROFILES] = {}
    entry = await _loaded_workspace_entry(hass, (provider_data,))
    provider_subentry = next(iter(entry.subentries.values()))

    async def fake_list_provider_model_names(
        _hass: HomeAssistant, _data: dict[str, Any]
    ) -> list[str]:
        return ["gpt-4.1-mini", "manual-model"]

    with patch(
        "custom_components.pydantic_ai_agent.config_flows.provider_flow.async_list_provider_model_names",
        new=fake_list_provider_model_names,
    ):
        result = await hass.config_entries.subentries.async_init(
            (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
            context={
                "source": config_entries.SOURCE_RECONFIGURE,
                "subentry_id": provider_subentry.subentry_id,
            },
        )
        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"], {"next_step_id": "manage_models"}
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "manage_models"
    assert _schema_select_options(result["data_schema"], CONF_SELECTED_MODEL_IDS) == [
        {"label": "gpt-4.1-mini", "value": "gpt-4.1-mini"},
        {"label": "manual-model", "value": "manual-model"}
    ]

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_SELECTED_MODEL_IDS: ["gpt-4.1-mini", "manual-model"]}
    )

    assert result["type"] is FlowResultType.ABORT
    updated_profiles = entry.subentries[provider_subentry.subentry_id].data[
        CONF_MODEL_PROFILES
    ]
    assert {profile[CONF_MODEL] for profile in updated_profiles.values()} == {
        "gpt-4.1-mini",
        "manual-model",
    }
    assert all(profile[CONF_ENABLED] is True for profile in updated_profiles.values())

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": provider_subentry.subentry_id,
        },
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"next_step_id": "edit_connection"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "OpenAI-compatible",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "sk-updated",
        },
    )

    assert result["type"] is FlowResultType.ABORT
    updated_profiles = entry.subentries[provider_subentry.subentry_id].data[
        CONF_MODEL_PROFILES
    ]
    assert {profile[CONF_MODEL] for profile in updated_profiles.values()} == {
        "gpt-4.1-mini",
        "manual-model",
    }


async def test_provider_manage_models_ignores_stale_profiles_with_catalog_metadata(
    hass: HomeAssistant,
) -> None:
    """Test stale profile IDs do not block catalog-backed availability edits."""
    provider_data = _provider_subentry_data()
    provider_config = cast(dict[str, Any], provider_data["data"])
    provider_config[CONF_MODEL_PROFILES]["profile-old"] = {
        "id": "profile-old",
        CONF_NAME: "Old Model",
        CONF_MODEL: "gpt-removed",
        CONF_ENABLED: True,
    }
    entry = await _loaded_workspace_entry(hass, (provider_data,))
    provider_subentry = next(iter(entry.subentries.values()))
    _cache_provider_catalog(hass, _wizard_catalog())

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": provider_subentry.subentry_id,
        },
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"next_step_id": "manage_models"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "manage_models"
    assert _schema_default(result["data_schema"], CONF_SELECTED_MODEL_IDS) == [
        "gpt-4.1-mini",
        "gpt-removed",
    ]

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_SELECTED_MODEL_IDS: ["gpt-4.1-mini"]}
    )

    assert result["type"] is FlowResultType.ABORT
    updated_profiles = entry.subentries[provider_subentry.subentry_id].data[
        CONF_MODEL_PROFILES
    ]
    assert updated_profiles["profile-old"][CONF_ENABLED] is False


async def test_provider_manage_models_creates_selected_catalog_profile(
    hass: HomeAssistant,
) -> None:
    """Test selected catalog models are created with catalog display names."""
    entry = await _loaded_workspace_entry(hass, (_provider_subentry_data(),))
    provider_subentry = next(iter(entry.subentries.values()))
    _cache_provider_catalog(hass, _wizard_catalog())

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": provider_subentry.subentry_id,
        },
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"next_step_id": "manage_models"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_SELECTED_MODEL_IDS: ["gpt-4.1-mini", "gpt-4.1"]}
    )

    assert result["type"] is FlowResultType.ABORT
    updated_subentry = entry.subentries[provider_subentry.subentry_id]
    model_profiles = cast(
        dict[str, dict[str, Any]], updated_subentry.data[CONF_MODEL_PROFILES]
    )
    created_profile = next(
        profile for profile in model_profiles.values() if profile[CONF_MODEL] == "gpt-4.1"
    )
    assert created_profile[CONF_NAME] == "GPT 4.1"
    assert created_profile[CONF_ENABLED] is True


async def test_provider_manage_models_can_disable_last_unrefenced_profile(
    hass: HomeAssistant,
) -> None:
    """Test availability management can leave a provider with no enabled models."""
    entry = await _loaded_workspace_entry(hass, (_provider_subentry_data(),))
    provider_subentry = next(iter(entry.subentries.values()))
    _cache_provider_catalog(hass, _wizard_catalog())

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": provider_subentry.subentry_id,
        },
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"next_step_id": "manage_models"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_SELECTED_MODEL_IDS: []}
    )

    assert result["type"] is FlowResultType.ABORT
    updated_subentry = entry.subentries[provider_subentry.subentry_id]
    model_profiles = cast(
        dict[str, dict[str, Any]], updated_subentry.data[CONF_MODEL_PROFILES]
    )
    assert model_profiles["profile-1"][CONF_ENABLED] is False


async def test_provider_manage_models_preserves_existing_profile_customization(
    hass: HomeAssistant,
) -> None:
    """Test enabling an existing profile preserves custom name and settings."""
    provider_data = _provider_subentry_data()
    provider_config = cast(dict[str, Any], provider_data["data"])
    provider_config[CONF_MODEL_PROFILES]["profile-2"] = {
        "id": "profile-2",
        CONF_NAME: "Fast Custom",
        CONF_MODEL: "gpt-4.1",
        CONF_ENABLED: False,
        CONF_MODEL_SETTINGS: {"temperature": 0.2},
    }
    entry = await _loaded_workspace_entry(hass, (provider_data,))
    provider_subentry = next(iter(entry.subentries.values()))
    _cache_provider_catalog(hass, _wizard_catalog())

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": provider_subentry.subentry_id,
        },
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"next_step_id": "manage_models"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_SELECTED_MODEL_IDS: ["gpt-4.1-mini", "gpt-4.1"]}
    )

    assert result["type"] is FlowResultType.ABORT
    updated_subentry = entry.subentries[provider_subentry.subentry_id]
    model_profiles = cast(
        dict[str, dict[str, Any]], updated_subentry.data[CONF_MODEL_PROFILES]
    )
    assert model_profiles["profile-2"][CONF_NAME] == "Fast Custom"
    assert model_profiles["profile-2"][CONF_ENABLED] is True
    assert model_profiles["profile-2"][CONF_MODEL_SETTINGS] == {"temperature": 0.2}


async def test_provider_manage_models_rejects_disabling_referenced_profile(
    hass: HomeAssistant,
) -> None:
    """Test referenced model profiles cannot be removed from availability."""
    entry = await _loaded_workspace_entry(
        hass,
        (
            _provider_subentry_data(),
            {
                "subentry_id": "agent-1",
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
                "data": {
                    CONF_AGENT_NAME: "Kitchen Agent",
                    CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
                },
            },
        ),
    )
    provider_subentry = entry.subentries["provider-1"]
    _cache_provider_catalog(hass, _wizard_catalog())

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": provider_subentry.subentry_id,
        },
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"next_step_id": "manage_models"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_SELECTED_MODEL_IDS: ["gpt-4.1"]}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "manage_models"
    assert result["errors"] == {"base": "model_profile_in_use"}


async def test_provider_edit_model_picker_shows_only_enabled_profiles(
    hass: HomeAssistant,
) -> None:
    """Test model customization only offers available profiles."""
    provider_data = _provider_subentry_data()
    provider_config = cast(dict[str, Any], provider_data["data"])
    provider_config[CONF_MODEL_PROFILES]["profile-2"] = {
        "id": "profile-2",
        CONF_NAME: "Disabled Model",
        CONF_MODEL: "gpt-4.1",
        CONF_ENABLED: False,
    }
    entry = await _loaded_workspace_entry(hass, (provider_data,))
    provider_subentry = next(iter(entry.subentries.values()))

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": provider_subentry.subentry_id,
        },
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"next_step_id": "customize_model_profile"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "pick_model_profile"
    assert _schema_select_options(result["data_schema"], "model_profile_id") == [
        {"label": "GPT Mini", "value": "profile-1"}
    ]


async def test_create_workspace_entry(hass: HomeAssistant) -> None:
    """Test the parent flow creates a workspace entry."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": config_entries.SOURCE_USER},
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_NAME: "Living Room Workspace"},
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Living Room Workspace"
    assert result["data"] == {
        CONF_NAME: "Living Room Workspace",
        CONF_DEFAULT_SKILLS_FOLDER: DEFAULT_SKILLS_FOLDER,
    }


async def test_new_workspace_default_title_is_generated(
    hass: HomeAssistant,
) -> None:
    """Test new workspace setup uses a generated default title."""
    with patch(
        "custom_components.pydantic_ai_agent.generated_titles.generate_name",
        return_value="brave_turing",
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": config_entries.SOURCE_USER},
        )

    assert result["type"] is FlowResultType.FORM
    assert _schema_default(result["data_schema"], CONF_NAME) == "Brave Turing Workspace"


async def test_new_custom_provider_default_title_is_generated(
    hass: HomeAssistant,
) -> None:
    """Test custom provider setup uses a generated default service title."""
    entry = await _loaded_workspace_entry(hass)
    _cache_provider_catalog(hass, _wizard_catalog())

    with patch(
        "custom_components.pydantic_ai_agent.generated_titles.generate_name",
        return_value="clever_matsumoto",
    ):
        result = await hass.config_entries.subentries.async_init(
            (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
            context={"source": config_entries.SOURCE_USER},
        )
        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"], {CONF_PROVIDER_ID: CUSTOM_PROVIDER_ID}
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"
    assert _schema_default(result["data_schema"], CONF_NAME) == "Clever Matsumoto Service"


async def test_guided_provider_default_title_stays_provider_name(
    hass: HomeAssistant,
) -> None:
    """Test guided provider setup keeps provider-specific default names."""
    entry = await _loaded_workspace_entry(hass)
    _cache_provider_catalog(hass, _wizard_catalog())

    with patch(
        "custom_components.pydantic_ai_agent.generated_titles.generate_name",
        return_value="clever_matsumoto",
    ):
        result = await hass.config_entries.subentries.async_init(
            (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
            context={"source": config_entries.SOURCE_USER},
        )
        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"], {CONF_PROVIDER_ID: "openai"}
        )
        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"], {CONF_DRIVER: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS}
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "wizard_connection"
    assert _schema_default(result["data_schema"], CONF_NAME) == "OpenAI"


async def test_new_conversation_default_title_is_generated(
    hass: HomeAssistant,
) -> None:
    """Test new conversation setup uses a generated default agent title."""
    entry = await _loaded_workspace_entry(hass, (_provider_subentry_data(),))

    with patch(
        "custom_components.pydantic_ai_agent.generated_titles.generate_name",
        return_value="fervent_ardinghelli",
    ):
        result = await hass.config_entries.subentries.async_init(
            (entry.entry_id, SUBENTRY_TYPE_CONVERSATION),
            context={"source": config_entries.SOURCE_USER},
        )

    assert result["type"] is FlowResultType.FORM
    assert _schema_default(result["data_schema"], CONF_AGENT_NAME) == (
        "Fervent Ardinghelli Agent"
    )


async def test_new_ai_task_default_title_is_generated(
    hass: HomeAssistant,
) -> None:
    """Test new AI task setup uses a generated default AI task title."""
    entry = await _loaded_workspace_entry(hass, (_provider_subentry_data(),))

    with patch(
        "custom_components.pydantic_ai_agent.generated_titles.generate_name",
        return_value="trusting_knuth",
    ):
        result = await hass.config_entries.subentries.async_init(
            (entry.entry_id, SUBENTRY_TYPE_AI_TASK),
            context={"source": config_entries.SOURCE_USER},
        )

    assert result["type"] is FlowResultType.FORM
    assert _schema_default(result["data_schema"], CONF_AI_TASK_NAME) == (
        "Trusting Knuth AI Task"
    )


async def test_create_provider_subentry_with_disabled_custom_profile(
    hass: HomeAssistant,
) -> None:
    """Test provider creation stores custom profiles disabled by default."""
    entry = await _loaded_workspace_entry(hass)
    _cache_provider_catalog(hass, _wizard_catalog())

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "pick_provider"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_PROVIDER_ID: CUSTOM_PROVIDER_ID}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "OpenAI-compatible",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "sk-test",
            "customize_model_list": {CONF_CUSTOM_MODEL_NAMES: "gpt-4.1-mini"},
        },
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    provider_data = cast(dict[str, Any], result["data"])
    assert provider_data[CONF_NAME] == "OpenAI-compatible"
    assert provider_data[CONF_PROVIDER_MODE] == PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS
    assert provider_data[CONF_MODEL_PROFILES]
    model_profiles = cast(dict[str, dict[str, Any]], provider_data[CONF_MODEL_PROFILES])
    profile = next(iter(model_profiles.values()))
    assert profile[CONF_MODEL] == "gpt-4.1-mini"
    assert profile[CONF_ENABLED] is False
    assert provider_data[CONF_CUSTOM_MODEL_NAMES] == ["gpt-4.1-mini"]


async def test_provider_catalog_failure_shows_fallback_picker(
    hass: HomeAssistant,
) -> None:
    """Test catalog failures still allow custom provider setup or retry."""
    entry = await _loaded_workspace_entry(hass)

    async def fake_fetch_catalog(_hass: HomeAssistant) -> CompactCatalog:
        raise CatalogLoadError("failed")

    with patch(
        "custom_components.pydantic_ai_agent.config_flows.provider_wizard.catalog_cache.async_fetch_catalog",
        new=fake_fetch_catalog,
    ):
        result = await hass.config_entries.subentries.async_init(
            (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
            context={"source": config_entries.SOURCE_USER},
        )
        assert result["type"] is FlowResultType.SHOW_PROGRESS

        await hass.async_block_till_done()
        result = await hass.config_entries.subentries.async_configure(result["flow_id"])

        assert result["type"] is FlowResultType.FORM
        assert result["step_id"] == "pick_provider"
        assert result["errors"] == {"base": "model_catalog_unavailable"}
        assert _schema_select_options(result["data_schema"], CONF_PROVIDER_ID) == [
            {"label": "Try loading catalog again", "value": CATALOG_RETRY_PROVIDER_ID},
            {"label": "Custom provider", "value": CUSTOM_PROVIDER_ID},
        ]

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_PROVIDER_ID: CUSTOM_PROVIDER_ID}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"


async def test_provider_catalog_failure_can_retry_catalog_load(
    hass: HomeAssistant,
) -> None:
    """Test catalog failure fallback can retry guided provider setup."""
    entry = await _loaded_workspace_entry(hass)
    catalog = _wizard_catalog()
    calls = 0

    async def fake_fetch_catalog(_hass: HomeAssistant) -> CompactCatalog:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise CatalogLoadError("failed")
        return catalog

    with patch(
        "custom_components.pydantic_ai_agent.config_flows.provider_wizard.catalog_cache.async_fetch_catalog",
        new=fake_fetch_catalog,
    ):
        result = await hass.config_entries.subentries.async_init(
            (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
            context={"source": config_entries.SOURCE_USER},
        )
        await hass.async_block_till_done()
        result = await hass.config_entries.subentries.async_configure(result["flow_id"])

        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"], {CONF_PROVIDER_ID: CATALOG_RETRY_PROVIDER_ID}
        )
        assert result["type"] is FlowResultType.SHOW_PROGRESS

        await hass.async_block_till_done()
        result = await hass.config_entries.subentries.async_configure(result["flow_id"])

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "pick_provider"
    assert result["errors"] == {}
    assert _schema_select_options(result["data_schema"], CONF_PROVIDER_ID) == [
        {"label": "OpenAI", "value": "openai"},
        {"label": "Custom provider", "value": CUSTOM_PROVIDER_ID},
    ]
    assert calls == 2


async def test_guided_provider_subentry_creates_enabled_profile(
    hass: HomeAssistant,
) -> None:
    """Test guided provider creation stores selected profiles enabled."""
    entry = await _loaded_workspace_entry(hass)
    _cache_provider_catalog(hass, _wizard_catalog())

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "pick_provider"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_PROVIDER_ID: "openai"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "pick_driver"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_DRIVER: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "wizard_connection"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_NAME: "OpenAI", CONF_API_KEY: "sk-test"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "pick_models"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_SELECTED_MODEL_IDS: ["gpt-4.1-mini"]}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    provider_data = cast(dict[str, Any], result["data"])
    assert provider_data[CONF_NAME] == "OpenAI"
    assert provider_data[CONF_PROVIDER_MODE] == PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS
    assert CONF_BASE_URL not in provider_data
    model_profiles = cast(dict[str, dict[str, Any]], provider_data[CONF_MODEL_PROFILES])
    profile = next(iter(model_profiles.values()))
    assert profile[CONF_NAME] == "GPT 4.1 Mini"
    assert profile[CONF_MODEL] == "gpt-4.1-mini"
    assert profile[CONF_ENABLED] is True


async def test_guided_provider_hidden_models_shows_filter_step(
    hass: HomeAssistant,
) -> None:
    """Test hidden default-filtered models remain reachable in guided setup."""
    entry = await _loaded_workspace_entry(hass)
    _cache_provider_catalog(hass, _wizard_catalog(hidden_openai_model=True))

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_PROVIDER_ID: "openai"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_DRIVER: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_NAME: "OpenAI", CONF_API_KEY: "sk-test"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "model_filters"


async def test_guided_single_visible_model_with_hidden_model_does_not_auto_create(
    hass: HomeAssistant,
) -> None:
    """Test hidden models prevent the single-model auto-finish shortcut."""
    entry = await _loaded_workspace_entry(hass)
    _cache_provider_catalog(
        hass, _wizard_catalog(single_anthropic=True, hidden_anthropic_model=True)
    )

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_PROVIDER_ID: "anthropic"}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_NAME: "Anthropic", CONF_API_KEY: "sk-ant"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "model_filters"


async def test_guided_single_driver_single_model_skips_extra_steps(
    hass: HomeAssistant,
) -> None:
    """Test guided setup skips driver and model steps when choices are singular."""
    entry = await _loaded_workspace_entry(hass)
    _cache_provider_catalog(hass, _wizard_catalog(single_anthropic=True))

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_PROVIDER_ID: "anthropic"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "wizard_connection"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_NAME: "Anthropic", CONF_API_KEY: "sk-ant"}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    provider_data = cast(dict[str, Any], result["data"])
    assert provider_data[CONF_PROVIDER_MODE] == "anthropic"
    model_profiles = cast(dict[str, dict[str, Any]], provider_data[CONF_MODEL_PROFILES])
    assert next(iter(model_profiles.values()))[CONF_ENABLED] is True


async def test_conversation_entity_streaming_supports_model_profile_ref(
    hass: HomeAssistant,
) -> None:
    """Test conversation entity streaming support with provider-owned profiles."""
    provider_subentry_id = "provider-1"
    default_profile_id = "profile-1"
    profile_ref = f"{provider_subentry_id}:{default_profile_id}"
    entry = await _loaded_workspace_entry(
        hass,
        (
            {
                "subentry_id": provider_subentry_id,
                "subentry_type": SUBENTRY_TYPE_PROVIDER,
                "title": "OpenAI-compatible",
                "unique_id": None,
                "data": {
                    CONF_NAME: "OpenAI-compatible",
                    CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
                    CONF_API_KEY: "sk-test",
                    CONF_MODEL_PROFILES: {
                        default_profile_id: {
                            "id": default_profile_id,
                            CONF_NAME: "GPT Mini",
                            CONF_MODEL: "gpt-4.1-mini",
                            CONF_ENABLED: True,
                        }
                    },
                },
            },
        ),
    )
    entry.runtime_data = WorkspaceRuntimeData(workspace_name="Workspace", providers={})

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_PRIMARY_MODEL_REF: profile_ref,
        },
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_PRIMARY_MODEL_REF] == profile_ref
    entry.runtime_data = WorkspaceRuntimeData(
        workspace_name="Workspace",
        providers={
            provider_subentry_id: ProviderRuntimeData(
                provider_subentry_id=provider_subentry_id,
                name="OpenAI-compatible",
                api_key="sk-test",
                provider_mode=PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
                base_url=None,
                provider_headers={},
            )
        },
    )
    plain_data = dict(result["data"])
    plain_data.pop(CONF_LLM_HASS_API, None)

    plain_subentry = cast(
        ConfigSubentry,
        SimpleNamespace(
            data=plain_data,
            subentry_id="conversation_plain",
            subentry_type=SUBENTRY_TYPE_CONVERSATION,
        ),
    )
    tool_subentry = cast(
        ConfigSubentry,
        SimpleNamespace(
            data=plain_data | {CONF_LLM_HASS_API: ["assist"]},
            subentry_id="conversation_tools",
            subentry_type=SUBENTRY_TYPE_CONVERSATION,
        ),
    )

    assert (
        PydanticAIConversationEntity(entry, plain_subentry).supports_streaming is True
    )
    assert (
        PydanticAIConversationEntity(entry, tool_subentry).supports_streaming is False
    )


async def test_conversation_disabled_skills_ignores_invalid_folder(
    hass: HomeAssistant,
) -> None:
    """Test disabled skills do not require or validate the skills folder."""
    provider_subentry_id = "provider-1"
    default_profile_id = "profile-1"
    profile_ref = f"{provider_subentry_id}:{default_profile_id}"
    entry = await _loaded_workspace_entry(
        hass,
        (
            {
                "subentry_id": provider_subentry_id,
                "subentry_type": SUBENTRY_TYPE_PROVIDER,
                "title": "OpenAI-compatible",
                "unique_id": None,
                "data": {
                    CONF_NAME: "OpenAI-compatible",
                    CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
                    CONF_API_KEY: "sk-test",
                    CONF_MODEL_PROFILES: {
                        default_profile_id: {
                            "id": default_profile_id,
                            CONF_NAME: "GPT Mini",
                            CONF_MODEL: "gpt-4.1-mini",
                            CONF_ENABLED: True,
                        }
                    },
                },
            },
        ),
    )

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_PRIMARY_MODEL_REF: profile_ref,
            "fallback_models": {CONF_FALLBACK_MODEL_REFS: []},
            "skill_settings": {
                CONF_ENABLE_SKILLS: False,
                CONF_SKILLS_FOLDER: "/tmp/skills",
            },
        },
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_PRIMARY_MODEL_REF] == profile_ref
    assert CONF_ENABLE_SKILLS not in result["data"]
    assert CONF_SKILLS_FOLDER not in result["data"]


async def test_provider_subentry_base_url_endpoint_returns_form_error(
    hass: HomeAssistant,
) -> None:
    """Test provider URL endpoint validation replays as a form error."""
    entry = await _loaded_workspace_entry(hass)
    _cache_provider_catalog(hass, _wizard_catalog())

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_PROVIDER_ID: CUSTOM_PROVIDER_ID}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "OpenAI-compatible",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "sk-test",
            "base_url": "https://api.example.com/v1/chat/completions",
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "invalid_base_url_endpoint"}


def _wizard_catalog(
    *,
    single_anthropic: bool = False,
    hidden_openai_model: bool = False,
    hidden_anthropic_model: bool = False,
) -> CompactCatalog:
    """Return a compact catalog fixture for guided flow tests."""
    openai_provider = CatalogProviderOption(
        id="openai",
        name="OpenAI",
        doc_url="https://models.dev/providers/openai",
        api_key_hints=("OPENAI_API_KEY",),
        default_base_url=None,
        supported_drivers=(
            PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            "openai_compatible_responses",
        ),
        model_count=2,
        families=("gpt",),
    )
    openai_models = [
        _wizard_model("gpt-4.1-mini", "GPT 4.1 Mini", "openai"),
        _wizard_model("gpt-4.1", "GPT 4.1", "openai"),
    ]
    if hidden_openai_model:
        openai_models.append(
            _wizard_model("gpt-4.1-no-tools", "GPT 4.1 No Tools", "openai", tool_call=False)
        )
    if not single_anthropic:
        return CompactCatalog(
            providers={"openai": openai_provider},
            models_by_provider={"openai": tuple(openai_models)},
        )
    anthropic_provider = CatalogProviderOption(
        id="anthropic",
        name="Anthropic",
        doc_url="https://models.dev/providers/anthropic",
        api_key_hints=("ANTHROPIC_API_KEY",),
        default_base_url=None,
        supported_drivers=("anthropic",),
        model_count=1,
        families=("claude",),
    )
    anthropic_models = [
        _wizard_model("claude-sonnet-4", "Claude Sonnet 4", "anthropic"),
    ]
    if hidden_anthropic_model:
        anthropic_models.append(
            _wizard_model(
                "claude-sonnet-4-no-tools",
                "Claude Sonnet 4 No Tools",
                "anthropic",
                tool_call=False,
            )
        )
    return CompactCatalog(
        providers={"anthropic": anthropic_provider, "openai": openai_provider},
        models_by_provider={
            "anthropic": tuple(anthropic_models),
            "openai": tuple(openai_models),
        },
    )


def _wizard_model(
    model_id: str,
    name: str,
    provider_id: str,
    *,
    tool_call: bool = True,
) -> CatalogModelOption:
    """Return a compact catalog model fixture."""
    return CatalogModelOption(
        id=model_id,
        name=name,
        provider_id=provider_id,
        family="gpt",
        tool_call=tool_call,
        structured_output=True,
        reasoning=False,
        attachment=False,
        text_output=True,
        context_limit=128000,
        output_limit=16000,
        status=None,
    )
