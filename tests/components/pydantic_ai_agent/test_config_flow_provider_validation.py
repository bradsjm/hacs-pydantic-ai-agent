"""Test config-flow provider validation behavior."""

import pytest
from custom_components.pydantic_ai_agent.config_flows.common import (
    _SECTION_EXTERNAL_TOOLS,
    _ai_task_data_schema,
    _conversation_schema,
    _normalise_provider_model_profiles,
    _provider_data_matches,
    _provider_model_profiles_for_discovery_mode,
    _validate_provider_data,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_DISCOVERED,
    CONF_ENABLED,
    CONF_MODEL,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_MODE,
    CONF_TODO_LIST_ENTITY_ID,
    CONF_VIRTUAL_WORKSPACE_ENABLED,
    CONF_WEB_FETCH_ENABLED,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE_GEMINI,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
)
from custom_components.pydantic_ai_agent.models.provider_validation import (
    ProviderValidationError,
)
from homeassistant.core import HomeAssistant
from tests.components.pydantic_ai_agent.support.builders import (
    provider_subentry_data,
    skill_subentry_data,
    workspace_entry,
)


def _section_key_names(data_schema, section_name):
    """Return field names from a sectioned flow schema."""
    for section_key, section_value in data_schema.schema.items():
        if section_key.schema == section_name:
            return {key.schema for key in section_value.schema.schema}
    raise AssertionError(f"Section {section_name} not found")


def test_provider_base_url_rejects_endpoint_suffix(hass: HomeAssistant) -> None:
    """Test provider base URLs cannot point at generated API endpoints."""
    with pytest.raises(ProviderValidationError) as err:
        _validate_provider_data(
            hass,
            {
                CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
                CONF_BASE_URL: "https://api.example.com/openai/chat/completions",
            },
        )

    assert err.value.reason == "invalid_base_url_endpoint"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.anthropic.com/v1/messages",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini:generateContent",
    ],
)
def test_provider_base_url_rejects_non_openai_endpoint_suffixes(
    hass: HomeAssistant, base_url: str
) -> None:
    """Test endpoint URL validation covers native provider request endpoints."""
    with pytest.raises(ProviderValidationError) as err:
        _validate_provider_data(
            hass,
            {
                CONF_PROVIDER_MODE: PROVIDER_GOOGLE_GEMINI,
                CONF_BASE_URL: base_url,
            },
        )

    assert err.value.reason == "invalid_base_url_endpoint"


def test_provider_base_url_allows_non_v1_base(hass: HomeAssistant) -> None:
    """Test endpoint validation does not require a v1 suffix."""
    _validate_provider_data(
        hass,
        {
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_BASE_URL: "https://api.example.com/openai/deployments/gpt-test",
        },
    )


def test_provider_extra_body_rejects_gemini_provider(hass: HomeAssistant) -> None:
    """Test provider extra body cannot be configured for unsupported providers."""
    with pytest.raises(ProviderValidationError) as err:
        _validate_provider_data(
            hass,
            {
                CONF_PROVIDER_MODE: PROVIDER_GOOGLE_GEMINI,
                CONF_PROVIDER_EXTRA_BODY: {"service_tier": "flex"},
            },
        )

    assert err.value.reason == "provider_extra_body_unsupported"


def test_provider_extra_body_allows_anthropic_provider(hass: HomeAssistant) -> None:
    """Test Anthropic can use provider extra body fields."""
    _validate_provider_data(
        hass,
        {
            CONF_PROVIDER_MODE: PROVIDER_ANTHROPIC,
            CONF_PROVIDER_EXTRA_BODY: {"anthropic_beta": ["feature-test"]},
        },
    )


def test_normalise_provider_model_profiles_adds_new_profiles_disabled() -> None:
    """Test newly discovered model profiles require explicit enablement."""
    profiles = _normalise_provider_model_profiles({}, ["gpt-test"], ["gpt-test"])

    profile = next(iter(profiles.values()))
    assert profile[CONF_MODEL] == "gpt-test"
    assert profile[CONF_ENABLED] is False
    assert profile[CONF_DISCOVERED] is True


def test_normalise_provider_model_profiles_uses_catalog_display_name() -> None:
    """Test catalog names replace default identifier-derived profile names."""
    profiles = _normalise_provider_model_profiles(
        {
            "profile-1": {
                "id": "profile-1",
                "name": "deepseek-v4-pro",
                CONF_MODEL: "deepseek-v4-pro",
                CONF_ENABLED: False,
                CONF_DISCOVERED: True,
            },
            "profile-2": {
                "id": "profile-2",
                "name": "Custom Display Name",
                CONF_MODEL: "deepseek-v4-flash",
                CONF_ENABLED: False,
                CONF_DISCOVERED: True,
            },
        },
        ["deepseek-v4-pro", "deepseek-v4-flash"],
        ["deepseek-v4-pro", "deepseek-v4-flash"],
        model_labels={
            "deepseek-v4-pro": "Deepseek V4 Pro",
            "deepseek-v4-flash": "Deepseek V4 Flash",
        },
    )

    assert profiles["profile-1"]["name"] == "Deepseek V4 Pro"
    assert profiles["profile-2"]["name"] == "Custom Display Name"


def test_normalise_provider_model_profiles_keeps_referenced_missing_profile() -> None:
    """Test refresh pruning keeps disappeared models still referenced by agents."""
    profiles = _normalise_provider_model_profiles(
        {
            "referenced": {
                "id": "referenced",
                CONF_MODEL: "gpt-old",
                CONF_ENABLED: True,
                CONF_DISCOVERED: True,
            },
            "unreferenced": {
                "id": "unreferenced",
                CONF_MODEL: "gpt-removed",
                CONF_ENABLED: True,
                CONF_DISCOVERED: True,
            },
        },
        ["gpt-new"],
        ["gpt-new"],
        keep_profile_ids={"referenced"},
    )

    assert "referenced" in profiles
    assert profiles["referenced"][CONF_MODEL] == "gpt-old"
    assert "unreferenced" not in profiles
    assert any(profile[CONF_MODEL] == "gpt-new" for profile in profiles.values())


def test_provider_data_identity_includes_provider_extra_body() -> None:
    """Test provider-level body settings distinguish provider subentries."""
    base_data = {
        CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        CONF_API_KEY: "sk-test",
    }

    assert not _provider_data_matches(
        base_data | {CONF_PROVIDER_EXTRA_BODY: {"service_tier": "flex"}},
        base_data | {CONF_PROVIDER_EXTRA_BODY: {"service_tier": "default"}},
    )


def test_discovery_mode_profiles_drop_unreferenced_custom_profiles() -> None:
    """Test clearing custom names removes old custom profiles unless referenced."""
    profiles = _provider_model_profiles_for_discovery_mode(
        {
            "discovered": {
                "id": "discovered",
                CONF_MODEL: "gpt-listed",
                CONF_ENABLED: True,
                CONF_DISCOVERED: True,
            },
            "referenced-custom": {
                "id": "referenced-custom",
                CONF_MODEL: "gpt-custom-used",
                CONF_ENABLED: True,
                CONF_DISCOVERED: False,
            },
            "removed-custom": {
                "id": "removed-custom",
                CONF_MODEL: "gpt-custom-removed",
                CONF_ENABLED: True,
                CONF_DISCOVERED: False,
            },
        },
        keep_profile_ids={"referenced-custom"},
    )

    assert set(profiles) == {"discovered", "referenced-custom"}


def test_conversation_and_ai_task_schemas_hide_private_mcp_selection(
    hass: HomeAssistant,
) -> None:
    """Test agent schemas no longer expose repo-owned MCP selectors."""
    entry = workspace_entry((provider_subentry_data(), skill_subentry_data()))

    conversation_fields = _section_key_names(
        _conversation_schema(
            hass, {CONF_PRIMARY_MODEL_REF: "provider-1:profile-1"}, entry
        ),
        _SECTION_EXTERNAL_TOOLS,
    )
    ai_task_fields = _section_key_names(
        _ai_task_data_schema(
            hass, {CONF_PRIMARY_MODEL_REF: "provider-1:profile-1"}, entry
        ),
        _SECTION_EXTERNAL_TOOLS,
    )

    assert conversation_fields == {
        CONF_VIRTUAL_WORKSPACE_ENABLED,
        CONF_WEB_FETCH_ENABLED,
    }
    assert ai_task_fields == {
        CONF_TODO_LIST_ENTITY_ID,
        CONF_VIRTUAL_WORKSPACE_ENABLED,
        CONF_WEB_FETCH_ENABLED,
    }
