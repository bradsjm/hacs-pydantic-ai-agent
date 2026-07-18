"""Tests for public provider subentry workflow paths."""

from collections.abc import Callable
from typing import Any

from custom_components.pydantic_ai_agent.config_flows._provider_data import (
    _store_provider_model_cache,
)
from custom_components.pydantic_ai_agent.config_flows.provider_flow import (
    ProviderSubentryFlowHandler,
)
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.catalog_cache import (
    catalog_manager,
)
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.types import (
    CatalogModelOption,
    CatalogProviderOption,
    CompactCatalog,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_ENABLED,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_NAME,
    CONF_PROVIDER_MODE,
    CONF_STRUCTURED_OUTPUT_SUPPORT,
    CONF_SUPPORTS_TOOLS,
    CONF_THINKING_SUPPORT,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_PROVIDER,
)
from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
    SOURCE_USER,
    ConfigEntryState,
)
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry


def _provider_data() -> dict[str, Any]:
    """Return minimal valid persisted provider data."""
    return {
        CONF_NAME: "Provider",
        CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        CONF_API_KEY: "test-key",
        CONF_MODEL_PROFILES: {
            "default": {
                "id": "default",
                CONF_NAME: "Configured Model",
                CONF_MODEL: "configured-model",
                CONF_ENABLED: True,
                CONF_STRUCTURED_OUTPUT_SUPPORT: "none",
                CONF_SUPPORTS_TOOLS: True,
                CONF_THINKING_SUPPORT: False,
            }
        },
    }


def _flow(
    entry: MockConfigEntry,
    hass: HomeAssistant,
    subentry_id: str | None = None,
) -> ProviderSubentryFlowHandler:
    """Return a provider flow attached to one workspace entry."""
    flow = ProviderSubentryFlowHandler()
    flow.hass = hass
    flow.handler = (entry.entry_id, SUBENTRY_TYPE_PROVIDER)
    flow.context = (
        {"source": SOURCE_USER}
        if subentry_id is None
        else {"source": SOURCE_RECONFIGURE, "subentry_id": subentry_id}
    )
    return flow


def _catalog() -> CompactCatalog:
    """Return a compact one-model catalog for guided-flow coverage."""
    provider = CatalogProviderOption(
        id="catalog-provider",
        name="Catalog Provider",
        doc_url="https://example.com/docs",
        api_key_hints=(),
        default_base_url="https://api.example.com/v1",
        supported_drivers=(PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,),
        model_count=1,
        families=(),
    )
    model = CatalogModelOption(
        id="catalog-model",
        name="Catalog Model",
        provider_id=provider.id,
        family=None,
        tool_call=True,
        structured_output=None,
        reasoning=False,
        attachment=False,
        input_modalities=(),
        text_output=True,
        context_limit=8192,
        output_limit=1024,
        status=None,
    )
    return CompactCatalog(
        providers={provider.id: provider}, models_by_provider={provider.id: (model,)}
    )


def _prime_catalog(hass: HomeAssistant) -> None:
    """Seed the process-wide catalog cache for deterministic flow tests."""
    manager = catalog_manager(hass)
    manager.catalog = _catalog()
    manager.loaded_at = manager.last_used_at = dt_util.utcnow()


async def test_manual_custom_provider_creation(
    hass: HomeAssistant, make_config_entry: Callable[..., MockConfigEntry]
) -> None:
    """A custom provider can be created through the public manual path."""
    entry = make_config_entry(state=ConfigEntryState.LOADED)
    entry.add_to_hass(hass)
    _prime_catalog(hass)
    flow = _flow(entry, hass)

    await flow.async_step_user()
    manual = await flow.async_step_pick_provider({"provider_id": "custom"})
    result = await flow.async_step_init(
        {
            CONF_NAME: "Manual Provider",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "manual-key",
            "base_url": "https://manual.example.com/v1",
        }
    )

    assert manual["type"] is FlowResultType.FORM
    assert manual["step_id"] == "init"
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_NAME] == "Manual Provider"
    assert result["data"][CONF_API_KEY] == "manual-key"
    assert result["data"][CONF_PROVIDER_MODE] == PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS
    assert result["data"][CONF_BASE_URL] == "https://manual.example.com/v1"


async def test_guided_provider_creation(
    hass: HomeAssistant, make_config_entry: Callable[..., MockConfigEntry]
) -> None:
    """A catalog provider creates its selected model profile through public steps."""
    entry = make_config_entry(state=ConfigEntryState.LOADED)
    entry.add_to_hass(hass)
    _prime_catalog(hass)
    flow = _flow(entry, hass)

    picker = await flow.async_step_user()
    connection = await flow.async_step_pick_provider(
        {"provider_id": "catalog-provider"}
    )
    result = await flow.async_step_wizard_connection(
        {CONF_NAME: "Guided Provider", CONF_API_KEY: "guided-key"}
    )

    assert picker["step_id"] == "pick_provider"
    assert connection["step_id"] == "wizard_connection"
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_MODEL_PROFILES]


async def test_catalog_failure_keeps_manual_provider_fallback(
    hass: HomeAssistant,
    make_config_entry: Callable[..., MockConfigEntry],
) -> None:
    """The custom provider option remains available after catalog failure."""
    entry = make_config_entry(state=ConfigEntryState.LOADED)
    entry.add_to_hass(hass)
    _prime_catalog(hass)
    flow = _flow(entry, hass)

    await flow.async_step_user()
    flow._wizard_catalog = None
    flow._wizard_catalog_error = "model_catalog_unavailable"
    picker = await flow.async_step_pick_provider()

    result = await flow.async_step_pick_provider({"provider_id": "custom"})

    assert picker["errors"] == {"base": "model_catalog_unavailable"}
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"


async def test_duplicate_provider_is_rejected(
    hass: HomeAssistant,
    make_config_entry: Callable[..., MockConfigEntry],
    make_subentry: Callable[..., Any],
) -> None:
    """Manual creation rejects a provider connection already owned by the workspace."""
    provider = make_subentry(
        subentry_id="provider-1",
        subentry_type=SUBENTRY_TYPE_PROVIDER,
        data=_provider_data(),
    )
    entry = make_config_entry(
        subentries=(provider,), state=ConfigEntryState.LOADED
    )
    entry.add_to_hass(hass)
    _prime_catalog(hass)
    flow = _flow(entry, hass)

    await flow.async_step_user()
    await flow.async_step_pick_provider({"provider_id": "custom"})
    result = await flow.async_step_init(_provider_data())

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_reconfigure_manages_provider_models(
    hass: HomeAssistant,
    make_config_entry: Callable[..., MockConfigEntry],
    make_subentry: Callable[..., Any],
) -> None:
    """Reconfiguration saves the selected provider-owned models."""
    data = _provider_data()
    _store_provider_model_cache(data, ["configured-model"])
    provider = make_subentry(
        subentry_id="provider-1",
        subentry_type=SUBENTRY_TYPE_PROVIDER,
        data=data,
    )
    entry = make_config_entry(
        subentries=(provider,), state=ConfigEntryState.LOADED
    )
    entry.add_to_hass(hass)
    flow = _flow(entry, hass, provider.subentry_id)

    menu = await flow.async_step_reconfigure()
    result = await flow.async_step_manage_models()
    saved = await flow.async_step_manage_models(
        {"selected_model_ids": ["configured-model"]}
    )

    assert menu["type"] is FlowResultType.MENU
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "manage_models"
    assert saved["type"] is FlowResultType.ABORT
    assert saved["reason"] == "reconfigure_successful"


async def test_profile_edit_attaches_context_window_error_to_field(
    hass: HomeAssistant,
    make_config_entry: Callable[..., MockConfigEntry],
    make_subentry: Callable[..., Any],
) -> None:
    """Profile validation keeps an invalid context window attached to its field."""
    provider = make_subentry(
        subentry_id="provider-1",
        subentry_type=SUBENTRY_TYPE_PROVIDER,
        data=_provider_data(),
    )
    entry = make_config_entry(
        subentries=(provider,), state=ConfigEntryState.LOADED
    )
    entry.add_to_hass(hass)
    flow = _flow(entry, hass, provider.subentry_id)

    await flow.async_step_reconfigure()
    await flow.async_step_customize_model_profile()
    await flow.async_step_pick_model_profile({"model_profile_id": "default"})
    result = await flow.async_step_edit_model_profile({"context_window_tokens": 0})

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "edit_model_profile"
    assert result["errors"] == {"context_window_tokens": "positive_number"}


async def test_unloaded_workspace_aborts_provider_setup(
    hass: HomeAssistant, make_config_entry: Callable[..., MockConfigEntry]
) -> None:
    """Provider setup does not run against an unloaded workspace."""
    entry = make_config_entry(state=ConfigEntryState.NOT_LOADED)
    entry.add_to_hass(hass)
    flow = _flow(entry, hass)

    result = await flow.async_step_user()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "entry_not_loaded"
