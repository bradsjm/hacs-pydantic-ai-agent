"""Test model profile helpers."""

from unittest.mock import patch
from types import SimpleNamespace
from typing import cast

from homeassistant import config_entries
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.pydantic_ai_agent import PydanticAIAgentRuntimeData
from custom_components.pydantic_ai_agent.const import (
    CONF_FALLBACK_MODEL_SUBENTRY_IDS,
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_MODEL_SUBENTRY_ID,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_MODE,
    DOMAIN,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_MODEL,
)
from custom_components.pydantic_ai_agent.model_profiles import (
    ModelProfile,
    chat_model_for_profile,
    model_profile,
    model_profile_chain,
    model_profile_ref,
    model_settings,
    parse_model_profile_ref,
    resolve_model_profile_ref,
    thinking_capability,
)


async def _loaded_model_entry(
    hass: HomeAssistant,
    *,
    title: str,
    subentry_id: str,
    api_key: str = "sk-test",
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
) -> MockConfigEntry:
    """Return a loaded provider entry with one model profile."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title=title,
        data={
            CONF_NAME: title,
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: api_key,
            CONF_BASE_URL: base_url,
            CONF_PROVIDER_HEADERS: headers or {},
        },
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "subentry_id": subentry_id,
                "subentry_type": SUBENTRY_TYPE_MODEL,
                "title": f"{title} Model",
                "unique_id": None,
                "data": {CONF_NAME: f"{title} Model", CONF_MODEL: f"{title}-model"},
            },
        ),
        options={},
        unique_id=None,
    )
    entry.runtime_data = PydanticAIAgentRuntimeData(
        provider_mode=PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        name=title,
        api_key=api_key,
        base_url=base_url,
        provider_headers=headers or {},
        logfire_enabled=False,
        logfire_include_content=False,
    )
    entry.add_to_hass(hass)
    with patch(
        "custom_components.pydantic_ai_agent.async_setup_entry", return_value=True
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    return entry


async def test_model_profile_ref_resolves_cross_provider_entry(
    hass: HomeAssistant,
) -> None:
    """Test canonical and legacy fallback refs resolve to model subentries."""
    current_entry = await _loaded_model_entry(
        hass, title="Primary", subentry_id="primary-model"
    )
    fallback_entry = await _loaded_model_entry(
        hass, title="Fallback", subentry_id="fallback-model"
    )

    assert parse_model_profile_ref(current_entry, "primary-model") == (
        current_entry.entry_id,
        "primary-model",
    )
    ref = model_profile_ref(fallback_entry.entry_id, "fallback-model")
    owner_entry, subentry = resolve_model_profile_ref(hass, current_entry, ref)

    assert owner_entry.entry_id == fallback_entry.entry_id
    assert subentry.subentry_id == "fallback-model"


async def test_chat_model_for_profile_uses_owner_provider_credentials(
    hass: HomeAssistant,
) -> None:
    """Test fallback model construction uses the owning provider runtime data."""
    fallback_entry = await _loaded_model_entry(
        hass,
        title="Fallback",
        subentry_id="fallback-model",
        api_key="sk-fallback",
        base_url="https://fallback.example/v1",
        headers={"X-Test": "fallback"},
    )
    profile = model_profile(fallback_entry, "fallback-model")
    model = object()

    with patch(
        "custom_components.pydantic_ai_agent.model_profiles.openai_compatible_completions_model",
        return_value=model,
    ) as completions_model:
        result = chat_model_for_profile(hass, profile)

    assert result is model
    completions_model.assert_called_once_with(
        hass,
        api_key="sk-fallback",
        base_url="https://fallback.example/v1",
        headers={"X-Test": "fallback"},
        model_name="Fallback-model",
    )


async def test_model_profile_chain_includes_cross_provider_fallback(
    hass: HomeAssistant,
) -> None:
    """Test model chains keep local primary and resolve foreign fallbacks."""
    current_entry = await _loaded_model_entry(
        hass, title="Primary", subentry_id="primary-model"
    )
    fallback_entry = await _loaded_model_entry(
        hass, title="Fallback", subentry_id="fallback-model"
    )
    owner_subentry = SimpleNamespace(
        subentry_id="conversation-1",
        data={
            CONF_MODEL_SUBENTRY_ID: "primary-model",
            CONF_FALLBACK_MODEL_SUBENTRY_IDS: [
                model_profile_ref(fallback_entry.entry_id, "fallback-model")
            ],
        },
    )

    profiles = model_profile_chain(
        hass, current_entry, cast(ConfigSubentry, owner_subentry)
    )

    assert [profile.owner_entry_id for profile in profiles] == [
        current_entry.entry_id,
        fallback_entry.entry_id,
    ]
    assert [profile.model_name for profile in profiles] == [
        "Primary-model",
        "Fallback-model",
    ]


def test_model_settings_excludes_capability_backed_thinking() -> None:
    """Test thinking is exposed through capabilities, not ModelSettings."""
    profile = ModelProfile(
        subentry_id="model_profile_1",
        owner_entry_id="entry-1",
        title="Fast GPT",
        provider_title="Provider",
        provider_mode="openai_compatible_completions",
        model_name="gpt-test",
        model_settings={"temperature": 0.7, "thinking": "high"},
    )

    settings = model_settings(profile)
    thinking = thinking_capability(profile)

    assert settings["temperature"] == 0.7
    assert settings.get("thinking") is None
    assert thinking is not None
    assert thinking.effort == "high"


def test_thinking_capability_keeps_explicit_false() -> None:
    """Test explicit thinking=False creates a capability."""
    profile = ModelProfile(
        subentry_id="model_profile_1",
        owner_entry_id="entry-1",
        title="Fast GPT",
        provider_title="Provider",
        provider_mode="openai_compatible_completions",
        model_name="gpt-test",
        model_settings={"thinking": False},
    )

    thinking = thinking_capability(profile)

    assert thinking is not None
    assert thinking.effort is False


def test_thinking_capability_absent_when_unconfigured() -> None:
    """Test absent thinking means no Thinking capability."""
    profile = ModelProfile(
        subentry_id="model_profile_1",
        owner_entry_id="entry-1",
        title="Fast GPT",
        provider_title="Provider",
        provider_mode="openai_compatible_completions",
        model_name="gpt-test",
        model_settings={},
    )

    assert thinking_capability(profile) is None
