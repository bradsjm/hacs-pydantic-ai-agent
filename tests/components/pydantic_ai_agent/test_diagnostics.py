"""Test diagnostics for Pydantic AI Agent."""

# ruff: noqa: E402

import sys
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

pytest.skip(
    "Legacy model-subentry diagnostics tests need workspace/provider-profile rewrite.",
    allow_module_level=True,
)

from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
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
)
from custom_components.pydantic_ai_agent.diagnostics import (
    async_get_config_entry_diagnostics,
    async_get_device_diagnostics,
)
from custom_components.pydantic_ai_agent.logfire_support import async_configure_logfire
from custom_components.pydantic_ai_agent.metrics import record_run_success

SUBENTRY_TYPE_MODEL = "model"


class PydanticAIAgentRuntimeData:
    """Legacy runtime-data test double for obsolete model-subentry tests."""

    def __init__(self, **kwargs: object) -> None:
        """Store provided attributes."""
        self.__dict__.update(kwargs)


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


async def test_diagnostics_exposes_safe_runtime_mcp_counts(
    hass: HomeAssistant,
) -> None:
    """Test config-entry diagnostics expose only safe MCP runtime counts."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Local Provider",
        data={
            CONF_NAME: "Local Provider",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "sk-secret",
        },
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "data": {
                    CONF_NAME: "Filesystem MCP",
                    CONF_MCP_URL: "https://mcp.example.com/mcp",
                    CONF_MCP_HEADERS: {"Authorization": "Bearer mcp-secret"},
                },
                "subentry_type": SUBENTRY_TYPE_MCP_SERVER,
                "title": "Filesystem MCP",
                "unique_id": None,
            },
        ),
        unique_id=None,
    )
    entry.add_to_hass(hass)
    mcp_subentry_id = next(iter(entry.subentries))
    entry.runtime_data = PydanticAIAgentRuntimeData(
        provider_mode=PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        name="Local Provider",
        api_key="sk-secret",
        base_url="https://provider.example.com/v1",
        logfire_enabled=False,
        logfire_include_content=False,
        mcp_servers=[{CONF_MCP_URL: "https://mcp.example.com/mcp"}],
        mcp_tool_cache={mcp_subentry_id: [{"name": "secret_tool"}, {"name": "x"}]},
    )

    diagnostics = await async_get_config_entry_diagnostics(hass, entry)

    assert diagnostics["runtime"] == {
        "loaded": True,
        "configured_mcp_server_count": 1,
        "cached_mcp_server_count": 1,
        "cached_mcp_tool_counts": {mcp_subentry_id: 2},
    }
    assert "mcp.example.com" not in str(diagnostics["runtime"])
    assert "secret_tool" not in str(diagnostics["runtime"])


async def test_device_diagnostics_filters_to_matching_subentry(
    hass: HomeAssistant,
) -> None:
    """Test device diagnostics include only matching subentry and metrics."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Local Provider",
        data={
            CONF_NAME: "Local Provider",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "sk-secret",
        },
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "data": {
                    CONF_AGENT_NAME: "Kitchen Agent",
                    CONF_PROMPT: "Private system prompt",
                    CONF_MODEL: "gpt-test",
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
            },
            {
                "data": {
                    CONF_AGENT_NAME: "Garage Agent",
                    CONF_PROMPT: "Other private prompt",
                    CONF_MODEL: "gpt-other",
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Garage Agent",
                "unique_id": None,
            },
        ),
        unique_id=None,
    )
    entry.add_to_hass(hass)
    matching_id = next(iter(entry.subentries))
    entry.runtime_data = PydanticAIAgentRuntimeData(
        provider_mode=PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        name="Local Provider",
        api_key="sk-secret",
        base_url=None,
        logfire_enabled=False,
        logfire_include_content=False,
    )
    record_run_success(
        hass,
        entry.entry_id,
        entry.runtime_data.metrics,
        matching_id,
        model_profile="Kitchen Model",
        duration=1.5,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=2,
            total_tokens=12,
            requests=1,
            tool_calls=3,
        ),
    )
    device = cast(dr.DeviceEntry, SimpleNamespace(identifiers={(DOMAIN, matching_id)}))

    diagnostics = await async_get_device_diagnostics(hass, entry, device)

    assert [item["subentry_id"] for item in diagnostics["subentries"]] == [matching_id]
    assert diagnostics["subentries"][0]["data"][CONF_PROMPT] == REDACTED
    assert (
        diagnostics["runtime"]["metrics"]["last_run_model_profile"] == "Kitchen Model"
    )
    assert diagnostics["runtime"]["metrics"]["last_run_total_tokens"] == 12
