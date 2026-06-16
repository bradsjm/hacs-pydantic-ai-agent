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
    CONF_STRUCTURED_OUTPUT_SUPPORT,
    CONF_SUPPORTS_TOOLS,
    CONF_THINKING,
    CONF_THINKING_SUPPORT,
    CONF_TIMEOUT,
    DOMAIN,
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_PROVIDER,
)
from custom_components.pydantic_ai_agent.models.model_profiles import (
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
from custom_components.pydantic_ai_agent.models.openai_compatible_profile import (
    default_openai_compatible_profile_data,
)
from custom_components.pydantic_ai_agent.models.provider import (
    openai_compatible_model_profile,
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
            CONF_THINKING_SUPPORT: "supported",
            CONF_STRUCTURED_OUTPUT_SUPPORT: "json_schema",
            CONF_SUPPORTS_TOOLS: True,
        },
        "fallback-profile": {
            "id": "fallback-profile",
            CONF_NAME: "Fallback GPT",
            CONF_MODEL: "gpt-fallback",
            CONF_ENABLED: True,
            CONF_DISCOVERED: False,
            **default_openai_compatible_profile_data(),
        },
    }
    entry = MockConfigEntry(
        version=2,
        minor_version=2,
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
    assert profile.thinking_support == "supported"
    assert profile.structured_output_support == "json_schema"
    assert profile.supports_tools is True
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
        "custom_components.pydantic_ai_agent.models.model_profiles.openai_compatible_completions_model",
        return_value=model,
    ) as completions_model:
        result = chat_model_for_profile(hass, entry, profile)

    assert result is model
    profile_arg = completions_model.call_args.kwargs["profile"]
    completions_model.assert_called_once_with(
        hass,
        api_key="sk-test",
        base_url="https://provider.example.com/v1",
        headers={"X-Test": "provider"},
        model_name="gpt-test",
        profile=profile_arg,
    )
    assert profile_arg.supports_thinking is True
    assert profile_arg.supports_json_schema_output is True
    assert profile_arg.supports_json_object_output is True
    assert profile_arg.openai_supports_strict_tool_definition is True


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
        thinking_support="supported",
        structured_output_support="json_schema",
        supports_tools=True,
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


def test_thinking_capability_omits_unsupported_effective_profile() -> None:
    """Test unsupported effective profiles suppress thinking capability."""
    profile = ModelProfile(
        ref=model_profile_ref("provider-1", "profile-1"),
        provider_subentry_id="provider-1",
        profile_id="profile-1",
        title="DeepSeek Flash",
        provider_title="Provider",
        provider_mode=PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        model_name="deepseek-v4-flash",
        model_settings={},
        thinking_support="none",
        structured_output_support="none",
        supports_tools=True,
    )

    assert thinking_capability({CONF_THINKING: "high"}, profile) is None


def test_openai_profile_mapping_uses_persisted_capabilities_not_model_name() -> None:
    """Test OpenAI-compatible profile mapping uses persisted capability data."""
    profile = ModelProfile(
        ref=model_profile_ref("provider-1", "profile-1"),
        provider_subentry_id="provider-1",
        profile_id="profile-1",
        title="DeepSeek R1",
        provider_title="Provider",
        provider_mode=PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        model_name="deepseek-r1",
        model_settings={},
        thinking_support="supported",
        structured_output_support="json_object",
        supports_tools=False,
    )
    runtime_profile = openai_compatible_model_profile(
        {
            CONF_THINKING_SUPPORT: profile.thinking_support,
            CONF_STRUCTURED_OUTPUT_SUPPORT: profile.structured_output_support,
            CONF_SUPPORTS_TOOLS: profile.supports_tools,
        }
    )

    assert runtime_profile.supports_thinking is True
    assert runtime_profile.thinking_always_enabled is False
    assert runtime_profile.supports_json_schema_output is False
    assert runtime_profile.supports_json_object_output is True
    assert runtime_profile.supports_tools is False
    assert runtime_profile.openai_supports_strict_tool_definition is True


@pytest.mark.parametrize(
    "missing_key",
    [
        CONF_THINKING_SUPPORT,
        CONF_STRUCTURED_OUTPUT_SUPPORT,
        CONF_SUPPORTS_TOOLS,
    ],
)
def test_incomplete_old_openai_profile_is_not_usable(missing_key: str) -> None:
    """Test incomplete OpenAI-compatible profiles require reconfiguration."""
    entry = _workspace_entry()
    provider = entry.subentries["provider-1"]
    profile = provider.data[CONF_MODEL_PROFILES]["primary-profile"]
    del profile[missing_key]

    assert configured_model_profile_exists(entry, "provider-1:primary-profile") is False
    with pytest.raises(HomeAssistantError):
        resolve_model_profile(entry, "provider-1:primary-profile")


def test_thinking_capability_keeps_supported_effective_profile() -> None:
    """Test supported effective profiles still emit thinking capability."""
    profile = ModelProfile(
        ref=model_profile_ref("provider-1", "profile-1"),
        provider_subentry_id="provider-1",
        profile_id="profile-1",
        title="Claude Sonnet 4",
        provider_title="Provider",
        provider_mode=PROVIDER_ANTHROPIC,
        model_name="claude-sonnet-4",
        model_settings={},
    )

    thinking = thinking_capability({CONF_THINKING: "high"}, profile)

    assert thinking is not None
    assert thinking.effort == "high"


def test_thinking_capability_absent_when_unconfigured() -> None:
    """Test absent thinking means no Thinking capability."""
    assert thinking_capability({}) is None
