"""Provider wizard catalog helpers for config-flow tests."""

from typing import Any, cast
from unittest.mock import patch

from custom_components.pydantic_ai_agent.config_flows.provider_wizard import (
    catalog_cache,
)
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.types import (
    CatalogModelOption,
    CatalogProviderOption,
    CompactCatalog,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_API_KEY,
    CONF_ENABLED,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_NAME,
    CONF_PROVIDER_METADATA,
    CONF_PROVIDER_MODE,
    DOMAIN,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_PROVIDER,
)
from homeassistant import config_entries
from homeassistant.config_entries import SubentryFlowContext
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry


def cache_provider_catalog(hass: HomeAssistant, catalog: CompactCatalog) -> None:
    """Cache a provider wizard catalog for flow tests."""
    manager = catalog_cache.catalog_manager(hass)
    now = dt_util.utcnow()
    manager.catalog = catalog
    manager.loaded_at = now
    manager.last_used_at = now


def wizard_catalog(
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
        wizard_model("gpt-4.1-mini", "GPT 4.1 Mini", "openai"),
        wizard_model("gpt-4.1", "GPT 4.1", "openai"),
    ]
    if hidden_openai_model:
        openai_models.append(
            wizard_model(
                "gpt-4.1-no-tools", "GPT 4.1 No Tools", "openai", tool_call=False
            )
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
        wizard_model("claude-sonnet-4", "Claude Sonnet 4", "anthropic"),
    ]
    if hidden_anthropic_model:
        anthropic_models.append(
            wizard_model(
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


def wizard_model(
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
        input_modalities=(),
        text_output=True,
        context_limit=128000,
        output_limit=16000,
        status=None,
        thinking_support="none",
        structured_output_support="json_object",
    )


type _FlowResultDict = dict[str, Any]


async def subentry_init_result(
    hass: HomeAssistant,
    flow_key: tuple[str, str],
    context: SubentryFlowContext,
) -> _FlowResultDict:
    """Return a subentry init result as a plain dictionary."""
    return cast(
        _FlowResultDict,
        await hass.config_entries.subentries.async_init(flow_key, context=context),
    )


async def subentry_configure_result(
    hass: HomeAssistant, flow_id: str, user_input: dict[str, object] | None = None
) -> _FlowResultDict:
    """Return a subentry configure result as a plain dictionary."""
    return cast(
        _FlowResultDict,
        await hass.config_entries.subentries.async_configure(flow_id, user_input),
    )


async def entry_flow_init_result(
    hass: HomeAssistant,
    domain: str,
    context: dict[str, str],
) -> _FlowResultDict:
    """Return a parent config-flow init result as a plain dictionary."""
    return cast(
        _FlowResultDict,
        await hass.config_entries.flow.async_init(
            domain,
            context=context,  # type: ignore[arg-type]
        ),
    )


async def entry_flow_configure_result(
    hass: HomeAssistant, flow_id: str, user_input: dict[str, object]
) -> _FlowResultDict:
    """Return a parent config-flow configure result as a plain dictionary."""
    return cast(
        _FlowResultDict,
        await hass.config_entries.flow.async_configure(flow_id, user_input),
    )


async def loaded_workspace_entry(
    hass: HomeAssistant, subentries_data: tuple[dict[str, object], ...] = ()
) -> MockConfigEntry:
    """Return a loaded workspace entry for subentry flow tests."""
    entry = MockConfigEntry(
        version=2,
        minor_version=4,
        domain=DOMAIN,
        title="Workspace",
        data={CONF_NAME: "Workspace"},
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


def provider_subentry_data(
    *,
    profile_name: str = "GPT Mini",
    model: str = "gpt-4.1-mini",
) -> dict[str, object]:
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
            CONF_PROVIDER_METADATA: {"catalog_provider_id": "openai"},
            CONF_MODEL_PROFILES: {
                "profile-1": {
                    "id": "profile-1",
                    CONF_NAME: profile_name,
                    CONF_MODEL: model,
                    CONF_ENABLED: True,
                }
            },
        },
    }
