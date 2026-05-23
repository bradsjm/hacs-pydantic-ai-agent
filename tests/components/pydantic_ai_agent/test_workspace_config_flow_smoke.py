"""Smoke tests for the workspace-first config flow."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

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
    CONF_BASE_URL,
    CONF_CUSTOM_MODEL_NAMES,
    CONF_ENABLE_SKILLS,
    CONF_ENABLED,
    CONF_FALLBACK_MODEL_REFS,
    CONF_DEFAULT_SKILLS_FOLDER,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_MODE,
    CONF_SKILLS_FOLDER,
    DEFAULT_SKILLS_FOLDER,
    DOMAIN,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_PROVIDER,
)
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.const import (
    CONF_DRIVER,
    CONF_PROVIDER_ID,
    CONF_SELECTED_MODEL_IDS,
    CONF_SETUP_METHOD,
    SETUP_METHOD_CUSTOM,
    SETUP_METHOD_GUIDED,
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
    assert step["sections"]["advanced_options"]["name"] == "Advanced options"
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


async def test_create_provider_subentry_with_disabled_custom_profile(
    hass: HomeAssistant,
) -> None:
    """Test provider creation stores custom profiles disabled by default."""
    entry = await _loaded_workspace_entry(hass)

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "setup_method"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_SETUP_METHOD: SETUP_METHOD_CUSTOM}
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


async def test_guided_provider_subentry_creates_enabled_profile(
    hass: HomeAssistant,
) -> None:
    """Test guided provider creation stores selected profiles enabled."""
    entry = await _loaded_workspace_entry(hass)
    manager = catalog_manager(hass)
    now = dt_util.utcnow()
    manager.catalog = _wizard_catalog()
    manager.loaded_at = now
    manager.last_used_at = now

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "setup_method"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_SETUP_METHOD: SETUP_METHOD_GUIDED}
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


async def test_guided_single_driver_single_model_skips_extra_steps(
    hass: HomeAssistant,
) -> None:
    """Test guided setup skips driver and model steps when choices are singular."""
    entry = await _loaded_workspace_entry(hass)
    manager = catalog_manager(hass)
    now = dt_util.utcnow()
    manager.catalog = _wizard_catalog(single_anthropic=True)
    manager.loaded_at = now
    manager.last_used_at = now

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_SETUP_METHOD: SETUP_METHOD_GUIDED}
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

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_SETUP_METHOD: SETUP_METHOD_CUSTOM}
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


def _wizard_catalog(*, single_anthropic: bool = False) -> CompactCatalog:
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
    openai_models = (
        _wizard_model("gpt-4.1-mini", "GPT 4.1 Mini", "openai"),
        _wizard_model("gpt-4.1", "GPT 4.1", "openai"),
    )
    if not single_anthropic:
        return CompactCatalog(
            providers={"openai": openai_provider},
            models_by_provider={"openai": openai_models},
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
    return CompactCatalog(
        providers={"anthropic": anthropic_provider, "openai": openai_provider},
        models_by_provider={
            "anthropic": (_wizard_model("claude-sonnet-4", "Claude Sonnet 4", "anthropic"),),
            "openai": openai_models,
        },
    )


def _wizard_model(model_id: str, name: str, provider_id: str) -> CatalogModelOption:
    """Return a compact catalog model fixture."""
    return CatalogModelOption(
        id=model_id,
        name=name,
        provider_id=provider_id,
        family="gpt",
        tool_call=True,
        structured_output=True,
        reasoning=False,
        attachment=False,
        text_output=True,
        context_limit=128000,
        output_limit=16000,
        status=None,
    )
