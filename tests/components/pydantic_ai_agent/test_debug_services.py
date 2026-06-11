"""Tests for read-only debug response services."""

from typing import Any, cast

import pytest
from custom_components.pydantic_ai_agent import WorkspaceRuntimeData, async_setup
from custom_components.pydantic_ai_agent.const import (
    CONF_ENABLED,
    CONF_MODEL,
    CONF_MODEL_PRICING,
    DOMAIN,
)
from custom_components.pydantic_ai_agent.debug_services import (
    SERVICE_GET_AGENT_METRICS,
    SERVICE_GET_TOOL_SOURCE_STATUS,
    SERVICE_GET_WORKSPACE_STATUS,
    SERVICE_LIST_MODEL_PROFILES,
)
from custom_components.pydantic_ai_agent.mcp import _cache_key
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from tests.components.pydantic_ai_agent.support.builders import (
    ai_task_subentry_data,
    conversation_subentry_data,
    mcp_server_runtime_data,
    mcp_server_subentry_data,
    provider_runtime_data,
    provider_subentry_data,
    skill_subentry_data,
    workspace_entry,
)


async def _call_service(
    hass: HomeAssistant, service: str, data: dict[str, Any]
) -> dict[str, Any]:
    """Call one debug response service."""
    result = await hass.services.async_call(
        DOMAIN,
        service,
        data,
        blocking=True,
        return_response=True,
    )
    return cast(dict[str, Any], result)


async def test_setup_registers_debug_response_services(hass: HomeAssistant) -> None:
    """Test async setup registers read-only debug services."""
    assert await async_setup(hass, {})

    assert hass.services.has_service(DOMAIN, SERVICE_GET_WORKSPACE_STATUS)
    assert hass.services.has_service(DOMAIN, SERVICE_LIST_MODEL_PROFILES)
    assert hass.services.has_service(DOMAIN, SERVICE_GET_AGENT_METRICS)
    assert hass.services.has_service(DOMAIN, SERVICE_GET_TOOL_SOURCE_STATUS)


async def test_debug_response_services_raise_for_unknown_config_entry(
    hass: HomeAssistant,
) -> None:
    """Test debug response services reject unknown explicit config entries."""
    assert await async_setup(hass, {})

    with pytest.raises(ServiceValidationError) as err:
        await _call_service(
            hass, SERVICE_GET_WORKSPACE_STATUS, {"config_entry_id": "missing-entry"}
        )

    assert err.value.translation_key == "config_entry_not_found"
    assert err.value.translation_placeholders == {"config_entry_id": "missing-entry"}


async def test_workspace_status_and_model_profiles_services(
    hass: HomeAssistant,
) -> None:
    """Test workspace and model profile services return compact safe data."""
    profile_ref = "provider-1:fast"
    entry = workspace_entry(
        (
            provider_subentry_data(
                subentry_id="provider-1",
                profile_id="fast",
                profile_name="Fast GPT",
                model="gpt-fast",
                model_profiles={
                    "fast": {
                        "id": "fast",
                        "name": "Fast GPT",
                        CONF_MODEL: "gpt-fast",
                        CONF_ENABLED: True,
                        CONF_MODEL_PRICING: {"input": 0.1},
                    },
                    "disabled": {
                        "id": "disabled",
                        "name": "Disabled GPT",
                        CONF_MODEL: "gpt-disabled",
                        CONF_ENABLED: False,
                    },
                },
            ),
            conversation_subentry_data(
                profile_ref,
                subentry_id="conversation-1",
                llm_hass_api=["assist-api"],
                mcp_server_ids=["mcp-1"],
                skills=["skill-1"],
            ),
            ai_task_subentry_data(profile_ref, subentry_id="ai-task-1"),
            mcp_server_subentry_data(subentry_id="mcp-1", allowed_tools=["echo"]),
            skill_subentry_data(subentry_id="skill-1", description="Useful skill"),
        )
    )
    entry.add_to_hass(hass)
    runtime_data = WorkspaceRuntimeData(
        workspace_name="Workspace",
        providers={"provider-1": provider_runtime_data(subentry_id="provider-1")},
        mcp_servers={"mcp-1": mcp_server_runtime_data(subentry_id="mcp-1")},
        logfire_enabled=True,
    )
    runtime_data.mcp_tool_cache["cache-1"] = [{"name": "echo"}]
    entry.runtime_data = runtime_data
    assert await async_setup(hass, {})

    workspace = await _call_service(
        hass, SERVICE_GET_WORKSPACE_STATUS, {"config_entry_id": entry.entry_id}
    )
    assert workspace["count"] == 1
    status = workspace["entries"][0]
    assert status["loaded"] is True
    assert status["subentry_counts"] == {
        "provider": 1,
        "conversation": 1,
        "ai_task_data": 1,
        "mcp_server": 1,
        "skill": 1,
    }
    assert status["subentries"]["providers"][0]["has_api_key"] is True
    assert "api_key" not in status["subentries"]["providers"][0]
    assert status["subentries"]["mcp_servers"][0]["allowed_tool_count"] == 1
    assert status["runtime"]["provider_count"] == 1
    assert status["runtime"]["mcp_server_count"] == 1
    assert status["runtime"]["logfire_enabled"] is True

    profiles = await _call_service(
        hass,
        SERVICE_LIST_MODEL_PROFILES,
        {"config_entry_id": entry.entry_id, "enabled_only": True},
    )
    assert profiles["count"] == 1
    assert profiles["profiles"][0]["ref"] == profile_ref
    assert profiles["profiles"][0]["pricing_present"] is True


async def test_metrics_and_tool_source_status_services(
    hass: HomeAssistant,
) -> None:
    """Test metrics and tool-source services read runtime state without side effects."""
    entry = workspace_entry(
        (
            mcp_server_subentry_data(
                subentry_id="mcp-1",
                allowed_tools=["echo"],
                headers={"Authorization": "Bearer secret"},
            ),
            skill_subentry_data(subentry_id="skill-1", content="Use concise answers."),
        )
    )
    entry.add_to_hass(hass)
    runtime_data = WorkspaceRuntimeData(workspace_name="Workspace")
    metrics = runtime_data.metrics.record_for("conversation-1")
    metrics.last_run_total_tokens = 42
    metrics.last_run_succeeded = True
    runtime_data.mcp_tool_cache[_cache_key(entry, "mcp-1")] = [
        {"name": "echo"},
        {"name": "list_files"},
    ]
    entry.runtime_data = runtime_data
    assert await async_setup(hass, {})

    metrics_response = await _call_service(
        hass,
        SERVICE_GET_AGENT_METRICS,
        {"config_entry_id": entry.entry_id, "subentry_id": "conversation-1"},
    )
    assert metrics_response["entries"][0]["records"] == [
        {
            "subentry_id": "conversation-1",
            "subentry_type": None,
            "metrics": {
                "last_run_model_profile": None,
                "last_run_input_tokens": None,
                "last_run_output_tokens": None,
                "last_run_cache_read_tokens": None,
                "last_run_total_tokens": 42,
                "last_run_input_cost": None,
                "last_run_output_cost": None,
                "last_run_cache_read_cost": None,
                "last_run_total_cost": None,
                "last_run_model_request_count": None,
                "last_run_tool_use_count": None,
                "cumulative_input_tokens": 0,
                "cumulative_output_tokens": 0,
                "cumulative_cache_read_tokens": 0,
                "cumulative_total_tokens": 0,
                "cumulative_input_cost": None,
                "cumulative_output_cost": None,
                "cumulative_cache_read_cost": None,
                "cumulative_total_cost": None,
                "last_run_duration": None,
                "last_error_type": None,
                "consecutive_failures": 0,
                "provider_healthy": None,
                "last_run_succeeded": True,
            },
        }
    ]

    tool_status = await _call_service(
        hass,
        SERVICE_GET_TOOL_SOURCE_STATUS,
        {"config_entry_id": entry.entry_id, "limit": 1},
    )
    assert tool_status["entries"][0]["mcp_servers"][0]["allowed_tool_count"] == 1
    assert tool_status["entries"][0]["mcp_servers"][0]["cached_tool_count"] == 2
    assert tool_status["entries"][0]["mcp_servers"][0]["tool_names"] == ["echo"]
    assert tool_status["entries"][0]["skills"][0]["content_length"] == 20
