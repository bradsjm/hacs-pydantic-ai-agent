"""Tests for guided/custom provider creation and catalog fallback flows."""

from typing import Any, cast
from unittest.mock import patch

from custom_components.pydantic_ai_agent.config_flows.provider_wizard import models_dev
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.const import (
    CATALOG_RETRY_PROVIDER_ID,
    CONF_PROVIDER_ID,
    CUSTOM_PROVIDER_ID,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_BASE_URL,
    CONF_CUSTOM_MODEL_NAMES,
    CONF_ENABLED,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_OPENAI_SUPPORTS_ENCRYPTED_REASONING_CONTENT,
    CONF_OPENAI_SUPPORTS_STRICT_TOOL_DEFINITION,
    CONF_PROVIDER_MODE,
    CONF_STRUCTURED_OUTPUT_SUPPORT,
    CONF_SUPPORTS_TOOLS,
    CONF_THINKING_SUPPORT,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_PROVIDER,
)
from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from tests.components.pydantic_ai_agent.support.schemas import (
    schema_select_custom_value as _schema_select_custom_value,
)
from tests.components.pydantic_ai_agent.support.schemas import (
    schema_select_options as _schema_select_options,
)
from tests.components.pydantic_ai_agent.support.wizard import (
    cache_provider_catalog,
    loaded_workspace_entry,
    subentry_configure_result,
    subentry_init_result,
    wizard_catalog,
)


async def test_create_custom_provider_subentry_without_model_profiles(
    hass: HomeAssistant,
) -> None:
    """Test provider creation leaves model availability management separate."""
    entry = await loaded_workspace_entry(hass)
    cache_provider_catalog(hass, wizard_catalog())

    result = await subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "pick_provider"

    result = await subentry_configure_result(
        hass, result["flow_id"], {CONF_PROVIDER_ID: CUSTOM_PROVIDER_ID}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"

    result = await subentry_configure_result(
        hass,
        result["flow_id"],
        {
            CONF_NAME: "OpenAI-compatible",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "sk-test",
        },
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    provider_data = cast(dict[str, Any], result["data"])
    assert provider_data[CONF_NAME] == "OpenAI-compatible"
    assert provider_data[CONF_PROVIDER_MODE] == PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS
    assert provider_data[CONF_MODEL_PROFILES] == {}
    assert CONF_CUSTOM_MODEL_NAMES not in provider_data


async def test_provider_catalog_failure_shows_fallback_picker(
    hass: HomeAssistant,
) -> None:
    """Test catalog failures still allow custom provider setup or retry."""
    entry = await loaded_workspace_entry(hass)

    async def fake_fetch_catalog(_hass: HomeAssistant):
        raise models_dev.CatalogLoadError("failed")

    with patch(
        "custom_components.pydantic_ai_agent.config_flows.provider_wizard.catalog_cache.async_fetch_catalog",
        new=fake_fetch_catalog,
    ):
        result = await subentry_init_result(
            hass,
            (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
            context={"source": config_entries.SOURCE_USER},
        )
        assert result["type"] is FlowResultType.SHOW_PROGRESS

        await hass.async_block_till_done()
        result = await subentry_configure_result(hass, result["flow_id"])

        assert result["type"] is FlowResultType.FORM
        assert result["step_id"] == "pick_provider"
        assert result["errors"] == {"base": "model_catalog_unavailable"}
        assert _schema_select_options(result["data_schema"], CONF_PROVIDER_ID) == [
            {"label": "Try loading catalog again", "value": CATALOG_RETRY_PROVIDER_ID},
            {"label": "Custom provider", "value": CUSTOM_PROVIDER_ID},
        ]

    result = await subentry_configure_result(
        hass, result["flow_id"], {CONF_PROVIDER_ID: CUSTOM_PROVIDER_ID}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"


async def test_provider_catalog_failure_can_retry_catalog_load(
    hass: HomeAssistant,
) -> None:
    """Test catalog failure fallback can retry guided provider setup."""
    entry = await loaded_workspace_entry(hass)
    catalog = wizard_catalog()
    calls = 0

    async def fake_fetch_catalog(_hass: HomeAssistant):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise models_dev.CatalogLoadError("failed")
        return catalog

    with patch(
        "custom_components.pydantic_ai_agent.config_flows.provider_wizard.catalog_cache.async_fetch_catalog",
        new=fake_fetch_catalog,
    ):
        result = await subentry_init_result(
            hass,
            (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
            context={"source": config_entries.SOURCE_USER},
        )
        await hass.async_block_till_done()
        result = await subentry_configure_result(hass, result["flow_id"])

        result = await subentry_configure_result(
            hass, result["flow_id"], {CONF_PROVIDER_ID: CATALOG_RETRY_PROVIDER_ID}
        )
        assert result["type"] is FlowResultType.SHOW_PROGRESS

        await hass.async_block_till_done()
        result = await subentry_configure_result(hass, result["flow_id"])

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
    entry = await loaded_workspace_entry(hass)
    cache_provider_catalog(hass, wizard_catalog())

    result = await subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "pick_provider"

    result = await subentry_configure_result(
        hass, result["flow_id"], {CONF_PROVIDER_ID: "openai"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "pick_driver"

    result = await subentry_configure_result(
        hass,
        result["flow_id"],
        {"driver": PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "wizard_connection"

    result = await subentry_configure_result(
        hass, result["flow_id"], {CONF_NAME: "OpenAI", CONF_API_KEY: "sk-test"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "pick_models"
    assert (
        _schema_select_custom_value(result["data_schema"], "selected_model_ids")
        is False
    )

    result = await subentry_configure_result(
        hass, result["flow_id"], {"selected_model_ids": ["gpt-4.1-mini"]}
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
    assert profile[CONF_THINKING_SUPPORT] == "none"
    assert profile[CONF_STRUCTURED_OUTPUT_SUPPORT] == "json_object"
    assert profile[CONF_SUPPORTS_TOOLS] is True
    assert profile[CONF_OPENAI_SUPPORTS_STRICT_TOOL_DEFINITION] is True
    assert profile[CONF_OPENAI_SUPPORTS_ENCRYPTED_REASONING_CONTENT] is False


async def test_guided_provider_hidden_models_shows_filter_step(
    hass: HomeAssistant,
) -> None:
    """Test hidden default-filtered models remain reachable in guided setup."""
    entry = await loaded_workspace_entry(hass)
    cache_provider_catalog(hass, wizard_catalog(hidden_openai_model=True))

    result = await subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await subentry_configure_result(
        hass, result["flow_id"], {CONF_PROVIDER_ID: "openai"}
    )
    result = await subentry_configure_result(
        hass,
        result["flow_id"],
        {"driver": PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS},
    )
    result = await subentry_configure_result(
        hass, result["flow_id"], {CONF_NAME: "OpenAI", CONF_API_KEY: "sk-test"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "model_filters"


async def test_guided_single_visible_model_with_hidden_model_does_not_auto_create(
    hass: HomeAssistant,
) -> None:
    """Test hidden models prevent the single-model auto-finish shortcut."""
    entry = await loaded_workspace_entry(hass)
    cache_provider_catalog(
        hass, wizard_catalog(single_anthropic=True, hidden_anthropic_model=True)
    )

    result = await subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await subentry_configure_result(
        hass, result["flow_id"], {CONF_PROVIDER_ID: "anthropic"}
    )
    result = await subentry_configure_result(
        hass, result["flow_id"], {CONF_NAME: "Anthropic", CONF_API_KEY: "sk-ant"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "model_filters"


async def test_guided_single_driver_single_model_skips_extra_steps(
    hass: HomeAssistant,
) -> None:
    """Test guided setup skips driver and model steps when choices are singular."""
    entry = await loaded_workspace_entry(hass)
    cache_provider_catalog(hass, wizard_catalog(single_anthropic=True))

    result = await subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await subentry_configure_result(
        hass, result["flow_id"], {CONF_PROVIDER_ID: "anthropic"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "wizard_connection"

    result = await subentry_configure_result(
        hass, result["flow_id"], {CONF_NAME: "Anthropic", CONF_API_KEY: "sk-ant"}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    provider_data = cast(dict[str, Any], result["data"])
    assert provider_data[CONF_PROVIDER_MODE] == "anthropic"
    model_profiles = cast(dict[str, dict[str, Any]], provider_data[CONF_MODEL_PROFILES])
    assert next(iter(model_profiles.values()))[CONF_ENABLED] is True


async def test_provider_subentry_base_url_endpoint_returns_form_error(
    hass: HomeAssistant,
) -> None:
    """Test provider URL endpoint validation replays as a form error."""
    entry = await loaded_workspace_entry(hass)
    cache_provider_catalog(hass, wizard_catalog())

    result = await subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await subentry_configure_result(
        hass, result["flow_id"], {CONF_PROVIDER_ID: CUSTOM_PROVIDER_ID}
    )
    result = await subentry_configure_result(
        hass,
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
