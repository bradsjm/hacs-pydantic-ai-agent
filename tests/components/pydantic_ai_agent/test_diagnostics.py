"""Test diagnostics for Pydantic AI Agent."""

from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers.redact import REDACTED
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_MODEL_SETTINGS,
    CONF_PROMPT,
    CONF_PROVIDER_MODE,
    DOMAIN,
    PROVIDER_OPENAI_COMPATIBLE,
    SUBENTRY_TYPE_CONVERSATION,
)
from custom_components.pydantic_ai_agent.diagnostics import (
    async_get_config_entry_diagnostics,
)


async def test_diagnostics_redacts_sensitive_config_entry_data(
    hass: HomeAssistant,
) -> None:
    """Test diagnostics redact credentials, prompts, and sensitive headers."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Local Provider",
        data={
            CONF_NAME: "Local Provider",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE,
            CONF_API_KEY: "sk-secret",
            CONF_BASE_URL: "http://localhost:11434/v1",
        },
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "data": {
                    CONF_AGENT_NAME: "Kitchen Agent",
                    CONF_MODEL: "gpt-test",
                    CONF_PROMPT: "Private system prompt",
                    CONF_LLM_HASS_API: ["assist"],
                    CONF_MODEL_SETTINGS: {
                        "max_tokens": 500,
                        "extra_headers": {"Authorization": "Bearer secret"},
                        "extra_body": {"api_key": "nested-secret"},
                    },
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
            },
        ),
        options={},
        unique_id=None,
    )
    entry.add_to_hass(hass)

    diagnostics = await async_get_config_entry_diagnostics(hass, entry)

    assert diagnostics["entry"]["data"][CONF_API_KEY] == REDACTED
    assert diagnostics["entry"]["data"][CONF_BASE_URL] == "http://localhost:11434/v1"
    subentry_data = diagnostics["subentries"][0]["data"]
    assert subentry_data[CONF_PROMPT] == REDACTED
    assert subentry_data[CONF_MODEL_SETTINGS]["max_tokens"] == 500
    assert subentry_data[CONF_MODEL_SETTINGS]["extra_headers"] == REDACTED
    assert subentry_data[CONF_MODEL_SETTINGS]["extra_body"]["api_key"] == REDACTED
    assert diagnostics["subentries"][0]["ha_tools_enabled"] is True
