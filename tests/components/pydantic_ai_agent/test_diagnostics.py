"""Test diagnostics for Pydantic AI Agent."""

import json
import sys
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from custom_components.pydantic_ai_agent import (
    MCPServerRuntimeData,
    WorkspaceRuntimeData,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_BASE_URL,
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
    CONF_DEFAULT_MODEL_PROFILE_ID,
    CONF_ENABLED,
    CONF_LOGFIRE_INCLUDE_CONTENT,
    CONF_LOGFIRE_TOKEN,
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_DEFERRED_LOADING,
    CONF_MCP_HEADERS,
    CONF_MCP_INCLUDE_RETURN_SCHEMA,
    CONF_MCP_SECRET_HEADER_KEYS,
    CONF_MCP_SERVER_IDS,
    CONF_MCP_TOOL_MODE,
    CONF_MCP_URL,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROMPT,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_MODE,
    CONF_PROVIDER_SECRET_HEADER_KEYS,
    CONF_SKILL_CONTENT,
    CONF_SKILL_REFERENCES,
    CONF_TEMPLATED_EXTRA_BODY,
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
from custom_components.pydantic_ai_agent.observability.logfire_support import (
    async_configure_logfire,
)
from custom_components.pydantic_ai_agent.observability.metrics import record_run_success
from homeassistant import config_entries
from homeassistant.components.diagnostics import REDACTED
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from pytest_homeassistant_custom_component.common import MockConfigEntry
from tests.components.pydantic_ai_agent.support.runtime import diagnostics_subentry

_PROVIDER_SUBENTRY_ID = "provider-1"
_MODEL_PROFILE_ID = "profile-1"
_MODEL_PROFILE_REF = f"{_PROVIDER_SUBENTRY_ID}:{_MODEL_PROFILE_ID}"


async def test_diagnostics_returns_redacted_bounded_config_entry_data(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test diagnostics redact sensitive keys and keep useful values visible."""
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
                    CONF_MCP_SERVER_IDS: ["mcp-1"],
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
                    CONF_PROVIDER_HEADERS: {
                        "Authorization": "Bearer provider-secret",
                        "X-Visible": "provider-visible",
                    },
                    CONF_PROVIDER_SECRET_HEADER_KEYS: ["Authorization"],
                    CONF_PROVIDER_EXTRA_BODY: {"api_key": "provider-body-secret"},
                    CONF_MODEL_PROFILES: {
                        _MODEL_PROFILE_ID: {
                            "id": _MODEL_PROFILE_ID,
                            CONF_NAME: "Reasoning Model",
                            CONF_MODEL: "gpt-test",
                            CONF_ENABLED: True,
                            CONF_MODEL_SETTINGS: {
                                CONF_TEMPLATED_EXTRA_BODY: [
                                    {
                                        CONF_CHAT_TEMPLATE_KWARG_KEY: (
                                            "chat_template_kwargs.secret_arg"
                                        ),
                                        CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: (
                                            "{{ states('sensor.secret') }}"
                                        ),
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
                "subentry_id": "mcp-1",
                "data": {
                    CONF_NAME: "Echo MCP",
                    CONF_MCP_URL: "https://mcp.example.com/mcp?token=secret",
                    CONF_MCP_HEADERS: {
                        "Authorization": "Bearer mcp-secret",
                        "X-Visible": "mcp-visible",
                    },
                    CONF_MCP_SECRET_HEADER_KEYS: ["Authorization"],
                    CONF_MCP_ALLOWED_TOOLS: ["echo"],
                    CONF_MCP_INCLUDE_RETURN_SCHEMA: False,
                    CONF_MCP_DEFERRED_LOADING: True,
                },
                "subentry_type": SUBENTRY_TYPE_MCP_SERVER,
                "title": "Echo MCP",
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

    assert diagnostics["entry"]["data"][CONF_LOGFIRE_TOKEN] == REDACTED
    assert diagnostics["entry"]["logfire_enabled"] is True
    assert diagnostics["entry"]["logfire_include_content"] is True
    conversation_diagnostics = diagnostics_subentry(
        diagnostics, subentry_type=SUBENTRY_TYPE_CONVERSATION
    )
    subentry_data = conversation_diagnostics["data"]
    assert subentry_data[CONF_PROMPT] == REDACTED
    assert conversation_diagnostics["ha_tools_enabled"] is True
    assert conversation_diagnostics["configuration_summary"] == {
        "subentry_type": SUBENTRY_TYPE_CONVERSATION,
        "name": "Kitchen Agent",
        CONF_PRIMARY_MODEL_REF: _MODEL_PROFILE_REF,
        "fallback_model_profile_count": 0,
        "mcp_server_count": 1,
        "skill_count": 0,
        CONF_LLM_HASS_API: ["assist"],
        "web_fetch_enabled": False,
        "virtual_workspace_enabled": False,
    }
    provider_data = diagnostics_subentry(
        diagnostics, subentry_type=SUBENTRY_TYPE_PROVIDER
    )["data"]
    assert provider_data[CONF_API_KEY] == REDACTED
    assert provider_data[CONF_PROVIDER_HEADERS] == {
        "Authorization": REDACTED,
        "X-Visible": "provider-visible",
    }
    assert provider_data[CONF_PROVIDER_EXTRA_BODY] == {"api_key": REDACTED}
    assert provider_data[CONF_BASE_URL] == "http://localhost:11434/v1"
    model_data = provider_data[CONF_MODEL_PROFILES][_MODEL_PROFILE_ID]
    assert model_data[CONF_MODEL_SETTINGS]["max_tokens"] == 500
    assert model_data[CONF_MODEL_SETTINGS]["extra_headers"] == REDACTED
    assert model_data[CONF_MODEL_SETTINGS][CONF_TEMPLATED_EXTRA_BODY] == [
        {
            CONF_CHAT_TEMPLATE_KWARG_KEY: "chat_template_kwargs.secret_arg",
            CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ states('sensor.secret') }}",
        }
    ]
    assert model_data[CONF_MODEL_SETTINGS]["extra_body"] == {"api_key": REDACTED}
    mcp_diagnostics = diagnostics_subentry(
        diagnostics, subentry_type=SUBENTRY_TYPE_MCP_SERVER
    )
    mcp_data = mcp_diagnostics["data"]
    assert mcp_data[CONF_MCP_URL] == REDACTED
    assert mcp_data[CONF_MCP_HEADERS] == {
        "Authorization": REDACTED,
        "X-Visible": "mcp-visible",
    }
    assert mcp_diagnostics["configuration_summary"] == {
        "subentry_type": SUBENTRY_TYPE_MCP_SERVER,
        "has_headers": True,
        CONF_MCP_TOOL_MODE: "specified",
        "allowed_tool_count": 1,
        CONF_MCP_INCLUDE_RETURN_SCHEMA: False,
        CONF_MCP_DEFERRED_LOADING: True,
    }
    skill_data = diagnostics_subentry(diagnostics, subentry_type=SUBENTRY_TYPE_SKILL)[
        "data"
    ]
    assert skill_data[CONF_NAME] == "Kitchen Skill"
    assert skill_data[CONF_SKILL_CONTENT] == REDACTED
    assert skill_data[CONF_SKILL_REFERENCES] == [
        {"title": "Secret Reference", "content": REDACTED}
    ]
    assert "sk-secret" not in json.dumps(diagnostics)
    assert "Private system prompt" not in json.dumps(diagnostics)


async def test_diagnostics_keeps_legacy_header_mappings_visible(
    hass: HomeAssistant,
) -> None:
    """Test header mappings without secret metadata stay visible in diagnostics."""
    entry = MockConfigEntry(
        version=2,
        minor_version=0,
        domain=DOMAIN,
        title="Workspace",
        data={CONF_NAME: "Workspace"},
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "subentry_id": _PROVIDER_SUBENTRY_ID,
                "data": {
                    CONF_NAME: "Local Provider",
                    CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
                    CONF_API_KEY: "sk-secret",
                    CONF_PROVIDER_HEADERS: {"Authorization": "Bearer legacy-provider"},
                    CONF_MODEL_PROFILES: {
                        _MODEL_PROFILE_ID: {
                            "id": _MODEL_PROFILE_ID,
                            CONF_NAME: "Reasoning Model",
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
            {
                "subentry_id": "mcp-1",
                "data": {
                    CONF_NAME: "Echo MCP",
                    CONF_MCP_URL: "https://mcp.example.com/mcp",
                    CONF_MCP_HEADERS: {"Authorization": "Bearer legacy-mcp"},
                },
                "subentry_type": SUBENTRY_TYPE_MCP_SERVER,
                "title": "Echo MCP",
                "unique_id": None,
            },
        ),
        options={},
        unique_id=None,
    )
    entry.add_to_hass(hass)

    diagnostics = await async_get_config_entry_diagnostics(hass, entry)

    provider_data = diagnostics_subentry(
        diagnostics, subentry_type=SUBENTRY_TYPE_PROVIDER
    )["data"]
    assert provider_data[CONF_PROVIDER_HEADERS] == {
        "Authorization": "Bearer legacy-provider"
    }
    mcp_data = diagnostics_subentry(
        diagnostics, subentry_type=SUBENTRY_TYPE_MCP_SERVER
    )["data"]
    assert mcp_data[CONF_MCP_HEADERS] == {"Authorization": "Bearer legacy-mcp"}


async def test_diagnostics_redacts_runtime_snapshots(hass: HomeAssistant) -> None:
    """Test runtime diagnostic snapshots use the shared redaction policy."""
    entry = MockConfigEntry(
        version=2,
        minor_version=0,
        domain=DOMAIN,
        title="Workspace",
        data={CONF_NAME: "Workspace"},
        source=config_entries.SOURCE_USER,
        unique_id=None,
    )
    entry.add_to_hass(hass)
    entry.runtime_data = WorkspaceRuntimeData(workspace_name="Workspace")
    entry.runtime_data.latest_run_diagnostics["conversation-1"] = {
        "model_settings": {
            "api_key": "runtime-secret",
            "session_token": "visible",
        }
    }
    entry.runtime_data.latest_stream_traces["conversation-1"] = {
        "headers": {"Authorization": "Bearer stream-secret"},
        "request_url": "https://provider.example.com/path?token=visible",
    }
    entry.runtime_data.mcp_servers["mcp-1"] = MCPServerRuntimeData(
        subentry_id="mcp-1",
        name="Echo MCP",
        url="https://mcp.example.com/mcp",
    )
    entry.runtime_data.mcp_tool_cache["cache-1"] = [{"name": "echo"}]

    diagnostics = await async_get_config_entry_diagnostics(hass, entry)

    runtime = diagnostics["runtime"]
    model_settings = runtime["latest_run_diagnostics"]["conversation-1"][
        "model_settings"
    ]
    assert model_settings["api_key"] == REDACTED
    assert model_settings["session_token"] == "visible"
    stream_trace = runtime["latest_stream_traces"]["conversation-1"]
    assert stream_trace["headers"] == REDACTED
    assert (
        stream_trace["request_url"] == "https://provider.example.com/path?token=visible"
    )
    assert runtime["mcp_server_count"] == 1
    assert runtime["cached_mcp_server_count"] == 1
    assert "runtime-secret" not in json.dumps(diagnostics)
    assert "stream-secret" not in json.dumps(diagnostics)


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
    assert (
        diagnostics_subentry(diagnostics, subentry_id=matching_id)["data"][CONF_PROMPT]
        == REDACTED
    )
    assert diagnostics["runtime"] == {"loaded": True}
    assert "latest_stream_trace" not in diagnostics["runtime"]
    assert "metrics" not in diagnostics["runtime"]
    assert "debug_preview" not in json.dumps(diagnostics)
