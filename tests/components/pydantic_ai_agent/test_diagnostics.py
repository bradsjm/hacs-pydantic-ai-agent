"""Test diagnostics for Pydantic AI Agent."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers.redact import REDACTED
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_BASE_URL,
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
    CONF_CHAT_TEMPLATE_KWARGS,
    CONF_LOGFIRE_INCLUDE_CONTENT,
    CONF_LOGFIRE_TOKEN,
    CONF_MCP_HEADERS,
    CONF_MCP_URL,
    CONF_MODEL,
    CONF_MODEL_SETTINGS,
    CONF_PROMPT,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_MODE,
    DOMAIN,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_MCP_SERVER,
    SUBENTRY_TYPE_MODEL,
)
from custom_components.pydantic_ai_agent.diagnostics import (
    async_get_config_entry_diagnostics,
)
from custom_components.pydantic_ai_agent.logfire_support import async_configure_logfire


async def test_diagnostics_redacts_sensitive_config_entry_data(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test diagnostics redact credentials, prompts, and sensitive headers."""
    monkeypatch.setitem(
        sys.modules,
        "logfire",
        SimpleNamespace(configure=Mock()),
    )
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Local Provider",
        data={
            CONF_NAME: "Local Provider",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "sk-secret",
            CONF_BASE_URL: "http://localhost:11434/v1",
            CONF_PROVIDER_HEADERS: {"Authorization": "Bearer provider-secret"},
            CONF_LOGFIRE_TOKEN: "lf-secret",
            CONF_LOGFIRE_INCLUDE_CONTENT: True,
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
            {
                "data": {
                    CONF_NAME: "Reasoning Model",
                    CONF_MODEL: "gpt-test",
                    CONF_MODEL_SETTINGS: {
                        CONF_CHAT_TEMPLATE_KWARGS: [
                            {
                                CONF_CHAT_TEMPLATE_KWARG_KEY: "secret_arg",
                                CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ states('sensor.secret') }}",
                            }
                        ],
                    },
                },
                "subentry_type": SUBENTRY_TYPE_MODEL,
                "title": "Reasoning Model",
                "unique_id": None,
            },
            {
                "data": {
                    CONF_NAME: "Filesystem MCP",
                    CONF_MCP_URL: "https://user:pass@mcp.example.com/mcp?token=visible",
                    CONF_MCP_HEADERS: {
                        "Authorization": "Bearer mcp-secret",
                        "X-API-Key": "nested-secret",
                    },
                },
                "subentry_type": SUBENTRY_TYPE_MCP_SERVER,
                "title": "Filesystem MCP",
                "unique_id": None,
            },
        ),
        options={},
        unique_id=None,
    )
    entry.add_to_hass(hass)
    assert await async_configure_logfire(hass, entry)

    diagnostics = await async_get_config_entry_diagnostics(hass, entry)

    assert diagnostics["entry"]["data"][CONF_API_KEY] == REDACTED
    assert diagnostics["entry"]["data"][CONF_PROVIDER_HEADERS] == REDACTED
    assert diagnostics["entry"]["data"][CONF_LOGFIRE_TOKEN] == REDACTED
    assert diagnostics["entry"]["data"][CONF_BASE_URL] == "http://localhost:11434/v1"
    assert diagnostics["entry"]["logfire_enabled"] is True
    assert diagnostics["entry"]["logfire_include_content"] is True
    subentry_data = diagnostics["subentries"][0]["data"]
    assert subentry_data[CONF_PROMPT] == REDACTED
    assert subentry_data[CONF_MODEL_SETTINGS]["max_tokens"] == 500
    assert subentry_data[CONF_MODEL_SETTINGS]["extra_headers"] == REDACTED
    assert subentry_data[CONF_MODEL_SETTINGS]["extra_body"]["api_key"] == REDACTED
    assert diagnostics["subentries"][0]["ha_tools_enabled"] is True
    model_data = diagnostics["subentries"][1]["data"]
    assert model_data[CONF_MODEL_SETTINGS][CONF_CHAT_TEMPLATE_KWARGS] == REDACTED
    mcp_data = diagnostics["subentries"][2]["data"]
    assert mcp_data[CONF_MCP_URL] == REDACTED
    assert mcp_data[CONF_MCP_HEADERS] == REDACTED
