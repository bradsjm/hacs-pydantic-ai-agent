"""Test diagnostics for Pydantic AI Agent."""

import json
import sys
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.pydantic_ai_agent import (
    MCPServerRuntimeData,
    WorkspaceRuntimeData,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_BASE_URL,
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
    CONF_CHAT_TEMPLATE_KWARGS,
    CONF_DEFAULT_MODEL_PROFILE_ID,
    CONF_ENABLED,
    CONF_LOGFIRE_INCLUDE_CONTENT,
    CONF_LOGFIRE_TOKEN,
    CONF_MCP_HEADERS,
    CONF_MCP_URL,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROMPT,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_MODE,
    CONF_SKILL_CONTENT,
    CONF_SKILL_REFERENCES,
    DOMAIN,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_MCP_SERVER,
    SUBENTRY_TYPE_PROVIDER,
    SUBENTRY_TYPE_SKILL,
)
from custom_components.pydantic_ai_agent.diagnostics import (
    async_get_config_entry_diagnostics,
    async_get_device_diagnostics,
)
from custom_components.pydantic_ai_agent.logfire_support import async_configure_logfire
from custom_components.pydantic_ai_agent.metrics import record_run_success

_PROVIDER_SUBENTRY_ID = "provider-1"
_MODEL_PROFILE_ID = "profile-1"
_MODEL_PROFILE_REF = f"{_PROVIDER_SUBENTRY_ID}:{_MODEL_PROFILE_ID}"


async def test_diagnostics_returns_unredacted_bounded_config_entry_data(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test diagnostics keep owner-requested values visible."""
    monkeypatch.setitem(
        sys.modules,
        "logfire",
        SimpleNamespace(configure=Mock()),
    )
    entry = MockConfigEntry(
        version=2,
        minor_version=0,
        domain=DOMAIN,
        title="Workspace",
        data={
            CONF_NAME: "Workspace",
            CONF_LOGFIRE_TOKEN: "lf-secret",
            CONF_LOGFIRE_INCLUDE_CONTENT: True,
        },
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "data": {
                    CONF_AGENT_NAME: "Kitchen Agent",
                    CONF_PRIMARY_MODEL_REF: _MODEL_PROFILE_REF,
                    CONF_PROMPT: "Private system prompt",
                    CONF_LLM_HASS_API: ["assist"],
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
            },
            {
                "subentry_id": _PROVIDER_SUBENTRY_ID,
                "data": {
                    CONF_NAME: "Local Provider",
                    CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
                    CONF_API_KEY: "sk-secret",
                    CONF_BASE_URL: "http://localhost:11434/v1",
                    CONF_PROVIDER_HEADERS: {"Authorization": "Bearer provider-secret"},
                    CONF_PROVIDER_EXTRA_BODY: {"api_key": "provider-body-secret"},
                    CONF_MODEL_PROFILES: {
                        _MODEL_PROFILE_ID: {
                            "id": _MODEL_PROFILE_ID,
                            CONF_NAME: "Reasoning Model",
                            CONF_MODEL: "gpt-test",
                            CONF_ENABLED: True,
                            CONF_MODEL_SETTINGS: {
                                CONF_CHAT_TEMPLATE_KWARGS: [
                                    {
                                        CONF_CHAT_TEMPLATE_KWARG_KEY: "secret_arg",
                                        CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ states('sensor.secret') }}",
                                    }
                                ],
                                "max_tokens": 500,
                                "extra_headers": {"Authorization": "Bearer secret"},
                                "extra_body": {"api_key": "nested-secret"},
                            },
                        }
                    },
                    CONF_DEFAULT_MODEL_PROFILE_ID: _MODEL_PROFILE_ID,
                },
                "subentry_type": SUBENTRY_TYPE_PROVIDER,
                "title": "Local Provider",
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
            {
                "data": {
                    CONF_NAME: "Kitchen Skill",
                    CONF_SKILL_CONTENT: "Private skill body",
                    CONF_SKILL_REFERENCES: [
                        {"title": "Secret Reference", "content": "secret reference"}
                    ],
                },
                "subentry_type": SUBENTRY_TYPE_SKILL,
                "title": "Kitchen Skill",
                "unique_id": None,
            },
        ),
        options={},
        unique_id=None,
    )
    entry.add_to_hass(hass)
    assert await async_configure_logfire(hass, entry)

    diagnostics = await async_get_config_entry_diagnostics(hass, entry)

    assert diagnostics["entry"]["data"][CONF_LOGFIRE_TOKEN] == "lf-secret"
    assert diagnostics["entry"]["logfire_enabled"] is True
    assert diagnostics["entry"]["logfire_include_content"] is True
    subentry_data = diagnostics["subentries"][0]["data"]
    assert subentry_data[CONF_PROMPT] == "Private system prompt"
    assert diagnostics["subentries"][0]["ha_tools_enabled"] is True
    assert diagnostics["subentries"][0]["configuration_summary"] == {
        "subentry_type": SUBENTRY_TYPE_CONVERSATION,
        "name": "Kitchen Agent",
        CONF_PRIMARY_MODEL_REF: _MODEL_PROFILE_REF,
        "fallback_model_profile_count": 0,
        "mcp_server_count": 0,
        "skill_count": 0,
        CONF_LLM_HASS_API: ["assist"],
        "web_fetch_enabled": False,
        "virtual_workspace_enabled": False,
    }
    provider_data = diagnostics["subentries"][1]["data"]
    assert provider_data[CONF_API_KEY] == "sk-secret"
    assert provider_data[CONF_PROVIDER_HEADERS] == {
        "Authorization": "Bearer provider-secret"
    }
    assert provider_data[CONF_PROVIDER_EXTRA_BODY] == {
        "api_key": "provider-body-secret"
    }
    assert provider_data[CONF_BASE_URL] == "http://localhost:11434/v1"
    model_data = provider_data[CONF_MODEL_PROFILES][_MODEL_PROFILE_ID]
    assert model_data[CONF_MODEL_SETTINGS]["max_tokens"] == 500
    assert model_data[CONF_MODEL_SETTINGS]["extra_headers"] == {
        "Authorization": "Bearer secret"
    }
    assert model_data[CONF_MODEL_SETTINGS][CONF_CHAT_TEMPLATE_KWARGS] == [
        {
            CONF_CHAT_TEMPLATE_KWARG_KEY: "secret_arg",
            CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ states('sensor.secret') }}",
        }
    ]
    assert model_data[CONF_MODEL_SETTINGS]["extra_body"] == {
        "api_key": "nested-secret"
    }
    mcp_data = diagnostics["subentries"][2]["data"]
    assert mcp_data[CONF_MCP_URL] == (
        "https://user:pass@mcp.example.com/mcp?token=visible"
    )
    assert mcp_data[CONF_MCP_HEADERS] == {
        "Authorization": "Bearer mcp-secret",
        "X-API-Key": "nested-secret",
    }
    skill_data = diagnostics["subentries"][3]["data"]
    assert skill_data[CONF_NAME] == "Kitchen Skill"
    assert skill_data[CONF_SKILL_CONTENT] == "Private skill body"
    assert skill_data[CONF_SKILL_REFERENCES] == [
        {"title": "Secret Reference", "content": "secret reference"}
    ]


async def test_diagnostics_exposes_safe_runtime_mcp_counts(
    hass: HomeAssistant,
) -> None:
    """Test config-entry diagnostics expose only safe MCP runtime counts."""
    entry = MockConfigEntry(
        version=2,
        minor_version=0,
        domain=DOMAIN,
        title="Workspace",
        data={CONF_NAME: "Workspace"},
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "subentry_id": "mcp-server-1",
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
    entry.runtime_data = WorkspaceRuntimeData(
        workspace_name="Workspace",
        mcp_servers={
            mcp_subentry_id: MCPServerRuntimeData(
                subentry_id=mcp_subentry_id,
                name="Filesystem MCP",
                url="https://mcp.example.com/mcp",
            )
        },
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


async def test_diagnostics_bounds_large_values(hass: HomeAssistant) -> None:
    """Test diagnostics bound large values without redacting them."""
    entry = MockConfigEntry(
        version=2,
        minor_version=0,
        domain=DOMAIN,
        title="Workspace",
        data={
            CONF_NAME: "Workspace",
            "large_secret_text": f"start-{'x' * 9000}-end",
            "large_list": list(range(250)),
            "large_mapping": {f"key_{index:03d}": index for index in range(125)},
        },
        source=config_entries.SOURCE_USER,
        unique_id=None,
    )
    entry.add_to_hass(hass)

    diagnostics = await async_get_config_entry_diagnostics(hass, entry)

    bounded_text = diagnostics["entry"]["data"]["large_secret_text"]
    assert bounded_text["__diagnostics_bounded__"] == "string"
    assert bounded_text["head"].startswith("start-")
    assert bounded_text["tail"].endswith("-end")
    assert bounded_text["omitted_chars"] > 0
    bounded_list = diagnostics["entry"]["data"]["large_list"]
    assert bounded_list["__diagnostics_bounded__"] == "sequence"
    assert bounded_list["total_count"] == 250
    assert bounded_list["head"][0] == 0
    assert bounded_list["tail"][-1] == 249
    bounded_mapping = diagnostics["entry"]["data"]["large_mapping"]
    assert bounded_mapping["__diagnostics_bounded__"] == "mapping"
    assert bounded_mapping["total_count"] == 125
    assert bounded_mapping["head"]["key_000"] == 0
    assert bounded_mapping["tail"]["key_124"] == 124


async def test_device_diagnostics_filters_to_matching_subentry(
    hass: HomeAssistant,
) -> None:
    """Test device diagnostics include only matching configuration data."""
    entry = MockConfigEntry(
        version=2,
        minor_version=0,
        domain=DOMAIN,
        title="Workspace",
        data={CONF_NAME: "Workspace"},
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "subentry_id": "conversation-1",
                "data": {
                    CONF_AGENT_NAME: "Kitchen Agent",
                    CONF_PROMPT: "Private system prompt",
                    CONF_PRIMARY_MODEL_REF: _MODEL_PROFILE_REF,
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
            },
            {
                "subentry_id": "conversation-2",
                "data": {
                    CONF_AGENT_NAME: "Garage Agent",
                    CONF_PROMPT: "Other private prompt",
                    CONF_PRIMARY_MODEL_REF: _MODEL_PROFILE_REF,
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Garage Agent",
                "unique_id": None,
            },
            {
                "subentry_id": _PROVIDER_SUBENTRY_ID,
                "data": {
                    CONF_NAME: "Local Provider",
                    CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
                    CONF_API_KEY: "sk-secret",
                    CONF_MODEL_PROFILES: {
                        _MODEL_PROFILE_ID: {
                            "id": _MODEL_PROFILE_ID,
                            CONF_NAME: "Kitchen Model",
                            CONF_MODEL: "gpt-test",
                            CONF_ENABLED: True,
                        }
                    },
                    CONF_DEFAULT_MODEL_PROFILE_ID: _MODEL_PROFILE_ID,
                },
                "subentry_type": SUBENTRY_TYPE_PROVIDER,
                "title": "Local Provider",
                "unique_id": None,
            },
        ),
        unique_id=None,
    )
    entry.add_to_hass(hass)
    matching_id = next(iter(entry.subentries))
    entry.runtime_data = WorkspaceRuntimeData(workspace_name="Workspace")
    entry.runtime_data.latest_stream_traces[matching_id] = {
        "events_total": 1,
        "events": [
            {
                "event_type": "PartDeltaEvent",
                "delta": {"content_delta_chars": 12},
            }
        ],
        "debug_preview": "should not be persisted",
    }
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
    device = cast(
        dr.DeviceEntry,
        SimpleNamespace(
            identifiers={
                (DOMAIN, f"{entry.entry_id}:{SUBENTRY_TYPE_CONVERSATION}:{matching_id}")
            }
        ),
    )

    diagnostics = await async_get_device_diagnostics(hass, entry, device)

    assert [item["subentry_id"] for item in diagnostics["subentries"]] == [matching_id]
    assert diagnostics["subentries"][0]["data"][CONF_PROMPT] == (
        "Private system prompt"
    )
    assert diagnostics["runtime"] == {"loaded": True}
    assert "latest_stream_trace" not in diagnostics["runtime"]
    assert "metrics" not in diagnostics["runtime"]
    assert "debug_preview" not in json.dumps(diagnostics)
