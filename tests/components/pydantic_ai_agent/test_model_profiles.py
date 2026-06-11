"""Test workspace-local model profile helpers."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
from custom_components.pydantic_ai_agent import (
    ProviderRuntimeData,
    WorkspaceRuntimeData,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_BASE_URL,
    CONF_DEFAULT_MODEL_PROFILE_ID,
    CONF_DISCOVERED,
    CONF_ENABLED,
    CONF_FALLBACK_MODEL_REFS,
    CONF_MAX_ITERATIONS,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_MODEL_PRICING,
    CONF_MODEL_PROFILES,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_MODE,
    CONF_THINKING,
    CONF_TIMEOUT,
    DOMAIN,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_PROVIDER,
)
from custom_components.pydantic_ai_agent.model_profiles import (
    ModelProfile,
    chat_model_for_profile,
    configured_model_profile_exists,
    model_display_names,
    model_profile_chain,
    model_profile_exists,
    model_profile_ref,
    model_settings,
    parse_model_profile_ref,
    resolve_model_profile,
    thinking_capability,
)
from homeassistant import config_entries
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from pytest_homeassistant_custom_component.common import MockConfigEntry


def _provider_subentry(
    subentry_id: str,
    *,
    title: str,
    model_profiles: dict[str, dict[str, object]],
    api_key: str = "sk-test",
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, object]:
    """Return one provider subentry payload."""
    data: dict[str, object] = {
        CONF_NAME: title,
        CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        CONF_API_KEY: api_key,
        CONF_MODEL_PROFILES: model_profiles,
        CONF_DEFAULT_MODEL_PROFILE_ID: next(iter(model_profiles)),
    }
    if base_url is not None:
        data[CONF_BASE_URL] = base_url
    if headers is not None:
        data[CONF_PROVIDER_HEADERS] = headers
    return {
        "subentry_id": subentry_id,
        "subentry_type": SUBENTRY_TYPE_PROVIDER,
        "title": title,
        "unique_id": None,
        "data": data,
    }


def _workspace_entry() -> MockConfigEntry:
    """Return a workspace entry with one provider runtime."""
    provider_subentry_id = "provider-1"
    model_profiles = {
        "primary-profile": {
            "id": "primary-profile",
            CONF_NAME: "Fast GPT",
            CONF_MODEL: "gpt-test",
            CONF_MODEL_PRICING: {"input": 0.4, "output": 1.6, "cache_read": 0.1},
            CONF_ENABLED: True,
            CONF_DISCOVERED: True,
        },
        "fallback-profile": {
            "id": "fallback-profile",
            CONF_NAME: "Fallback GPT",
            CONF_MODEL: "gpt-fallback",
            CONF_ENABLED: True,
            CONF_DISCOVERED: False,
        },
    }
    entry = MockConfigEntry(
        version=2,
        minor_version=1,
        domain=DOMAIN,
        title="Workspace",
        data={CONF_NAME: "Workspace"},
        source=config_entries.SOURCE_USER,
        subentries_data=(
            _provider_subentry(
                provider_subentry_id,
                title="Local Provider",
                model_profiles=model_profiles,
                base_url="https://provider.example.com/v1",
                headers={"X-Test": "provider"},
            ),
        ),
        options={},
        unique_id=None,
    )
    entry.runtime_data = WorkspaceRuntimeData(
        workspace_name="Workspace",
        providers={
            provider_subentry_id: ProviderRuntimeData(
                provider_subentry_id=provider_subentry_id,
                name="Local Provider",
                api_key="sk-test",
                provider_mode=PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
                base_url="https://provider.example.com/v1",
                provider_headers={"X-Test": "provider"},
            )
        },
    )
    return entry


def test_parse_model_profile_ref_uses_workspace_local_format() -> None:
    """Test provider/profile refs use the new workspace-local format."""
    assert parse_model_profile_ref("provider-1:primary-profile") == (
        "provider-1",
        "primary-profile",
    )


def test_resolve_model_profile_reads_provider_owned_profile() -> None:
    """Test resolving a workspace-local ref reads provider-owned profile data."""
    entry = _workspace_entry()

    profile = resolve_model_profile(entry, "provider-1:primary-profile")

    assert profile.ref == "provider-1:primary-profile"
    assert profile.provider_subentry_id == "provider-1"
    assert profile.profile_id == "primary-profile"
    assert profile.provider_title == "Local Provider"
    assert profile.title == "Fast GPT"
    assert profile.provider_mode == PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS
    assert profile.model_name == "gpt-test"
    assert profile.model_pricing == {"input": 0.4, "output": 1.6, "cache_read": 0.1}
    assert model_profile_exists(entry, profile.ref) is True


def test_configured_model_profile_exists_ignores_runtime_provider() -> None:
    """Test config validation can use persisted profiles before reload completes."""
    entry = _workspace_entry()
    entry.runtime_data = WorkspaceRuntimeData(workspace_name="Workspace", providers={})

    assert configured_model_profile_exists(entry, "provider-1:primary-profile") is True
    assert model_profile_exists(entry, "provider-1:primary-profile") is False
    with pytest.raises(HomeAssistantError):
        resolve_model_profile(entry, "provider-1:primary-profile")


def test_chat_model_for_profile_uses_provider_runtime_credentials(
    hass: HomeAssistant,
) -> None:
    """Test chat model construction uses workspace runtime provider credentials."""
    entry = _workspace_entry()
    profile = resolve_model_profile(entry, "provider-1:primary-profile")
    model = object()

    with patch(
        "custom_components.pydantic_ai_agent.model_profiles.openai_compatible_completions_model",
        return_value=model,
    ) as completions_model:
        result = chat_model_for_profile(hass, entry, profile)

    assert result is model
    completions_model.assert_called_once_with(
        hass,
        api_key="sk-test",
        base_url="https://provider.example.com/v1",
        headers={"X-Test": "provider"},
        model_name="gpt-test",
    )


def test_model_profile_chain_keeps_primary_then_ordered_fallback() -> None:
    """Test model chains resolve provider/profile refs within one workspace."""
    entry = _workspace_entry()
    owner_subentry = SimpleNamespace(
        subentry_id="conversation-1",
        data={
            CONF_PRIMARY_MODEL_REF: "provider-1:primary-profile",
            CONF_FALLBACK_MODEL_REFS: ["provider-1:fallback-profile"],
        },
    )

    profiles = model_profile_chain(entry, cast(ConfigSubentry, owner_subentry))

    assert [profile.ref for profile in profiles] == [
        "provider-1:primary-profile",
        "provider-1:fallback-profile",
    ]
    assert model_display_names(profiles) == [
        "Local Provider / Fast GPT",
        "Local Provider / Fallback GPT",
    ]


def test_run_settings_override_legacy_profile_run_settings() -> None:
    """Test run settings own operational request controls."""
    profile = ModelProfile(
        ref=model_profile_ref("provider-1", "profile-1"),
        provider_subentry_id="provider-1",
        profile_id="profile-1",
        title="Fast GPT",
        provider_title="Provider",
        provider_mode=PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        model_name="gpt-test",
        model_settings={
            "temperature": 0.7,
            CONF_MAX_ITERATIONS: 5,
            CONF_MAX_TOKENS: 99,
            CONF_THINKING: "low",
            CONF_TIMEOUT: 20.0,
        },
    )

    settings = model_settings(
        profile,
        {CONF_MAX_TOKENS: 256, CONF_THINKING: "high", CONF_TIMEOUT: 12.5},
    )
    thinking = thinking_capability({CONF_THINKING: "high"})

    assert settings.get("temperature") == 0.7
    assert settings.get(CONF_MAX_TOKENS) == 256
    assert settings.get(CONF_TIMEOUT) == 12.5
    assert thinking is not None
    assert thinking.effort == "high"


def test_thinking_capability_keeps_explicit_false() -> None:
    """Test explicit thinking=False creates a capability."""
    thinking = thinking_capability({CONF_THINKING: False})

    assert thinking is not None
    assert thinking.effort is False


def test_thinking_capability_absent_when_unconfigured() -> None:
    """Test absent thinking means no Thinking capability."""
    assert thinking_capability({}) is None
