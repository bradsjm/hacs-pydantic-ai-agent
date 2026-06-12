"""Tests for provider model availability management."""

from typing import Any, cast
from unittest.mock import patch

from custom_components.pydantic_ai_agent.config_flows.provider_wizard.const import (
    CONF_SELECTED_MODEL_IDS,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_CUSTOM_MODEL_NAMES,
    CONF_DISCOVERED,
    CONF_ENABLED,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROVIDER_METADATA,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_PROVIDER,
)
from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.data_entry_flow import FlowResultType
from tests.components.pydantic_ai_agent.support.schemas import (
    schema_default as _sd,
)
from tests.components.pydantic_ai_agent.support.schemas import (
    schema_key_names as _skn,
)
from tests.components.pydantic_ai_agent.support.schemas import (
    schema_select_custom_value as _ssc,
)
from tests.components.pydantic_ai_agent.support.schemas import (
    schema_select_options as _sso,
)
from tests.components.pydantic_ai_agent.support.wizard import (
    cache_provider_catalog,
    loaded_workspace_entry,
    wizard_catalog,
)
from tests.components.pydantic_ai_agent.support.wizard import (
    provider_subentry_data as _bpsd,
)
from tests.components.pydantic_ai_agent.support.wizard import (
    subentry_configure_result as _scr,
)
from tests.components.pydantic_ai_agent.support.wizard import (
    subentry_init_result as _sir,
)


def _psd():
    return _bpsd()


async def _init_manage_models(hass, entry, subentry):
    r = await _sir(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": subentry.subentry_id,
        },
    )
    return await _scr(hass, r["flow_id"], {"next_step_id": "manage_models"})


async def test_provider_manage_models_preselects_enabled_profiles(hass):
    entry = await loaded_workspace_entry(hass, (_psd(),))
    cache_provider_catalog(hass, wizard_catalog())
    r = await _init_manage_models(hass, entry, next(iter(entry.subentries.values())))
    assert _sd(r["data_schema"], CONF_SELECTED_MODEL_IDS) == ["gpt-4.1-mini"]
    assert _sso(r["data_schema"], CONF_SELECTED_MODEL_IDS) == [
        {"label": "GPT 4.1 (128K context)", "value": "gpt-4.1"},
        {"label": "GPT 4.1 Mini (128K context)", "value": "gpt-4.1-mini"},
    ]


async def test_provider_manage_models_includes_enabled_models_hidden_by_filters(hass):
    pd = _psd()
    cast(dict[str, Any], pd["data"])[CONF_MODEL_PROFILES]["profile-2"] = {
        "id": "profile-2",
        CONF_NAME: "No Tools",
        CONF_MODEL: "gpt-4.1-no-tools",
        CONF_ENABLED: True,
    }
    entry = await loaded_workspace_entry(hass, (pd,))
    cache_provider_catalog(hass, wizard_catalog(hidden_openai_model=True))
    r = await _init_manage_models(hass, entry, next(iter(entry.subentries.values())))
    assert r["step_id"] == "manage_model_filters"
    r = await _scr(hass, r["flow_id"], {})
    assert r["step_id"] == "manage_models"
    assert set(_sd(r["data_schema"], CONF_SELECTED_MODEL_IDS)) == {
        "gpt-4.1-mini",
        "gpt-4.1-no-tools",
    }


async def test_provider_manage_models_matches_catalog_after_metadata_cleared(hass):
    pd = _psd()
    pc = cast(dict[str, Any], pd["data"])
    pc.pop(CONF_PROVIDER_METADATA)
    pc[CONF_MODEL_PROFILES]["profile-old"] = {
        "id": "profile-old",
        CONF_NAME: "Old",
        CONF_MODEL: "gpt-removed",
        CONF_ENABLED: True,
    }
    entry = await loaded_workspace_entry(hass, (pd,))
    cache_provider_catalog(hass, wizard_catalog())
    r = await _init_manage_models(hass, entry, next(iter(entry.subentries.values())))
    assert _sd(r["data_schema"], CONF_SELECTED_MODEL_IDS) == [
        "gpt-4.1-mini",
        "gpt-removed",
    ]


async def test_provider_manage_models_ignores_stale_profiles_with_catalog_metadata(
    hass,
):
    pd = _psd()
    cast(dict[str, Any], pd["data"])[CONF_MODEL_PROFILES]["profile-old"] = {
        "id": "profile-old",
        CONF_NAME: "Old",
        CONF_MODEL: "gpt-removed",
        CONF_ENABLED: True,
    }
    entry = await loaded_workspace_entry(hass, (pd,))
    cache_provider_catalog(hass, wizard_catalog())
    r = await _init_manage_models(hass, entry, next(iter(entry.subentries.values())))
    assert _sd(r["data_schema"], CONF_SELECTED_MODEL_IDS) == [
        "gpt-4.1-mini",
        "gpt-removed",
    ]


async def test_provider_manage_models_discovers_manual_provider_models(hass):
    pd = _psd()
    pc = cast(dict[str, Any], pd["data"])
    pc.pop(CONF_PROVIDER_METADATA)
    pc[CONF_MODEL_PROFILES] = {}
    entry = await loaded_workspace_entry(hass, (pd,))

    async def fake_list(_h, _d):
        return ["gpt-4.1-mini", "manual-model"]

    with patch(
        "custom_components.pydantic_ai_agent.provider_validation.async_list_provider_model_names",
        new=fake_list,
    ):
        r = await _init_manage_models(
            hass, entry, next(iter(entry.subentries.values()))
        )
    assert _sso(r["data_schema"], CONF_SELECTED_MODEL_IDS) == [
        {"label": "gpt-4.1-mini", "value": "gpt-4.1-mini"},
        {"label": "manual-model", "value": "manual-model"},
    ]


async def test_provider_manage_models_creates_selected_catalog_profile(hass):
    entry = await loaded_workspace_entry(hass, (_psd(),))
    ps = next(iter(entry.subentries.values()))
    cache_provider_catalog(hass, wizard_catalog())
    r = await _init_manage_models(hass, entry, ps)
    r = await _scr(
        hass, r["flow_id"], {CONF_SELECTED_MODEL_IDS: ["gpt-4.1-mini", "gpt-4.1"]}
    )
    assert r["type"] is FlowResultType.ABORT
    profiles = cast(dict, entry.subentries[ps.subentry_id].data[CONF_MODEL_PROFILES])
    cp = next(p for p in profiles.values() if p[CONF_MODEL] == "gpt-4.1")
    assert cp[CONF_NAME] == "GPT 4.1"
    assert cp[CONF_ENABLED] is True


async def test_provider_manage_models_can_disable_last_unrefenced_profile(hass):
    entry = await loaded_workspace_entry(hass, (_psd(),))
    ps = next(iter(entry.subentries.values()))
    cache_provider_catalog(hass, wizard_catalog())
    r = await _init_manage_models(hass, entry, ps)
    r = await _scr(hass, r["flow_id"], {CONF_SELECTED_MODEL_IDS: []})
    assert r["type"] is FlowResultType.ABORT
    profiles = cast(dict, entry.subentries[ps.subentry_id].data[CONF_MODEL_PROFILES])
    assert profiles["profile-1"][CONF_ENABLED] is False


async def test_provider_manage_models_preserves_existing_profile_customization(hass):
    pd = _psd()
    cast(dict[str, Any], pd["data"])[CONF_MODEL_PROFILES]["profile-2"] = {
        "id": "profile-2",
        CONF_NAME: "Fast Custom",
        CONF_MODEL: "gpt-4.1",
        CONF_ENABLED: False,
        CONF_MODEL_SETTINGS: {"temperature": 0.2},
    }
    entry = await loaded_workspace_entry(hass, (pd,))
    ps = next(iter(entry.subentries.values()))
    cache_provider_catalog(hass, wizard_catalog())
    r = await _init_manage_models(hass, entry, ps)
    r = await _scr(
        hass, r["flow_id"], {CONF_SELECTED_MODEL_IDS: ["gpt-4.1-mini", "gpt-4.1"]}
    )
    assert r["type"] is FlowResultType.ABORT
    profiles = cast(dict, entry.subentries[ps.subentry_id].data[CONF_MODEL_PROFILES])
    assert profiles["profile-2"][CONF_NAME] == "Fast Custom"
    assert profiles["profile-2"][CONF_ENABLED] is True
    assert profiles["profile-2"][CONF_MODEL_SETTINGS] == {"temperature": 0.2}


async def test_provider_manage_models_persists_and_enables_new_custom_model(hass):
    entry = await loaded_workspace_entry(hass, (_psd(),))
    ps = next(iter(entry.subentries.values()))
    cache_provider_catalog(hass, wizard_catalog())
    r = await _init_manage_models(hass, entry, ps)
    assert _skn(r["data_schema"]) == {CONF_SELECTED_MODEL_IDS}
    assert _ssc(r["data_schema"], CONF_SELECTED_MODEL_IDS) is True
    r = await _scr(
        hass,
        r["flow_id"],
        {CONF_SELECTED_MODEL_IDS: ["gpt-4.1-mini", "local/custom-model"]},
    )
    assert r["type"] is FlowResultType.ABORT
    profiles = cast(dict, entry.subentries[ps.subentry_id].data[CONF_MODEL_PROFILES])
    cp = next(p for p in profiles.values() if p[CONF_MODEL] == "local/custom-model")
    assert cp[CONF_ENABLED] is True
    assert cp[CONF_DISCOVERED] is False


async def test_provider_manage_models_keeps_catalog_option_for_matching_custom_name(
    hass,
):
    pd = _psd()
    cast(dict[str, Any], pd["data"])[CONF_MODEL_PROFILES] = {}
    entry = await loaded_workspace_entry(hass, (pd,))
    cache_provider_catalog(hass, wizard_catalog())
    r = await _init_manage_models(hass, entry, next(iter(entry.subentries.values())))
    r = await _scr(
        hass,
        r["flow_id"],
        {CONF_SELECTED_MODEL_IDS: ["gpt-4.1"]},
    )
    assert r["type"] is FlowResultType.ABORT
    profiles = cast(dict, entry.subentries["provider-1"].data[CONF_MODEL_PROFILES])
    p = next(v for v in profiles.values() if v[CONF_MODEL] == "gpt-4.1")
    assert p[CONF_DISCOVERED] is True


async def test_provider_manage_models_preserves_disabled_custom_and_clears_enabled(
    hass,
):
    pd = _psd()
    pc = cast(dict[str, Any], pd["data"])
    pc[CONF_CUSTOM_MODEL_NAMES] = ["local/disabled-model", "local/enabled-model"]
    pc[CONF_MODEL_PROFILES]["profile-disabled-custom"] = {
        "id": "profile-disabled-custom",
        CONF_NAME: "Disabled",
        CONF_MODEL: "local/disabled-model",
        CONF_ENABLED: False,
        CONF_DISCOVERED: False,
    }
    pc[CONF_MODEL_PROFILES]["profile-enabled-custom"] = {
        "id": "profile-enabled-custom",
        CONF_NAME: "Enabled",
        CONF_MODEL: "local/enabled-model",
        CONF_ENABLED: True,
        CONF_DISCOVERED: False,
    }
    entry = await loaded_workspace_entry(hass, (pd,))
    ps = next(iter(entry.subentries.values()))
    cache_provider_catalog(hass, wizard_catalog())
    r = await _init_manage_models(hass, entry, ps)
    assert set(_sd(r["data_schema"], CONF_SELECTED_MODEL_IDS)) == {
        "gpt-4.1-mini",
        "local/enabled-model",
    }
    r = await _scr(hass, r["flow_id"], {CONF_SELECTED_MODEL_IDS: ["gpt-4.1-mini"]})
    assert r["type"] is FlowResultType.ABORT
    us = entry.subentries[ps.subentry_id]
    assert us.data[CONF_CUSTOM_MODEL_NAMES] == ["local/disabled-model"]
    assert (
        us.data[CONF_MODEL_PROFILES]["profile-disabled-custom"][CONF_ENABLED] is False
    )
    assert us.data[CONF_MODEL_PROFILES]["profile-enabled-custom"][CONF_ENABLED] is False


async def test_provider_manage_models_reenables_disabled_manual_custom_profile(hass):
    pd = _psd()
    pc = cast(dict[str, Any], pd["data"])
    pc[CONF_MODEL_PROFILES]["profile-manual-custom"] = {
        "id": "profile-manual-custom",
        CONF_NAME: "Manual Custom",
        CONF_MODEL: "local/manual-custom",
        CONF_ENABLED: False,
        CONF_DISCOVERED: False,
        CONF_MODEL_SETTINGS: {"temperature": 0.3},
    }
    entry = await loaded_workspace_entry(hass, (pd,))
    ps = next(iter(entry.subentries.values()))
    cache_provider_catalog(hass, wizard_catalog())
    r = await _init_manage_models(hass, entry, ps)
    r = await _scr(
        hass,
        r["flow_id"],
        {CONF_SELECTED_MODEL_IDS: ["gpt-4.1-mini", "local/manual-custom"]},
    )
    assert r["type"] is FlowResultType.ABORT
    profiles = cast(dict, entry.subentries[ps.subentry_id].data[CONF_MODEL_PROFILES])
    assert profiles["profile-manual-custom"][CONF_ENABLED] is True
    assert profiles["profile-manual-custom"][CONF_DISCOVERED] is False
    assert profiles["profile-manual-custom"][CONF_MODEL_SETTINGS] == {
        "temperature": 0.3
    }
    assert [
        profile_id
        for profile_id, profile in profiles.items()
        if profile[CONF_MODEL] == "local/manual-custom"
    ] == ["profile-manual-custom"]


async def test_provider_manage_models_rejects_disabling_referenced_profile(hass):
    entry = await loaded_workspace_entry(
        hass,
        (
            _psd(),
            {
                "subentry_id": "agent-1",
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Agent",
                "unique_id": None,
                "data": {
                    CONF_AGENT_NAME: "Agent",
                    CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
                },
            },
        ),
    )
    cache_provider_catalog(hass, wizard_catalog())
    r = await _init_manage_models(hass, entry, entry.subentries["provider-1"])
    r = await _scr(hass, r["flow_id"], {CONF_SELECTED_MODEL_IDS: ["gpt-4.1"]})
    assert r["type"] is FlowResultType.FORM
    assert r["errors"] == {"base": "model_profile_in_use"}
    assert _sd(r["data_schema"], CONF_SELECTED_MODEL_IDS) == ["gpt-4.1"]


async def test_provider_manage_models_rejects_clearing_referenced_custom_model(hass):
    pd = _psd()
    pc = cast(dict[str, Any], pd["data"])
    pc[CONF_CUSTOM_MODEL_NAMES] = ["local/custom-model"]
    pc[CONF_MODEL_PROFILES]["profile-custom"] = {
        "id": "profile-custom",
        CONF_NAME: "Local",
        CONF_MODEL: "local/custom-model",
        CONF_ENABLED: True,
        CONF_DISCOVERED: False,
    }
    entry = await loaded_workspace_entry(
        hass,
        (
            pd,
            {
                "subentry_id": "agent-1",
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Agent",
                "unique_id": None,
                "data": {
                    CONF_AGENT_NAME: "Agent",
                    CONF_PRIMARY_MODEL_REF: "provider-1:profile-custom",
                },
            },
        ),
    )
    cache_provider_catalog(hass, wizard_catalog())
    r = await _init_manage_models(hass, entry, entry.subentries["provider-1"])
    r = await _scr(hass, r["flow_id"], {CONF_SELECTED_MODEL_IDS: ["gpt-4.1-mini"]})
    assert r["errors"] == {"base": "model_profile_in_use"}
    assert _sd(r["data_schema"], CONF_SELECTED_MODEL_IDS) == ["gpt-4.1-mini"]
