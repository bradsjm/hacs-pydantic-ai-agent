"""Tests for MCP server structured reconfigure fields."""

from typing import Any
from unittest.mock import patch

from custom_components.pydantic_ai_agent.config_flows import mcp_server_flow
from custom_components.pydantic_ai_agent.const import (
    CONF_KEY_VALUE_IS_SECRET,
    CONF_KEY_VALUE_KEY,
    CONF_KEY_VALUE_VALUE,
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_DEFERRED_LOADING,
    CONF_MCP_HEADERS,
    CONF_MCP_INCLUDE_RETURN_SCHEMA,
    CONF_MCP_SECRET_HEADER_KEYS,
    CONF_MCP_URL,
    CONF_NAME,
    SUBENTRY_TYPE_MCP_SERVER,
)
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from tests.components.pydantic_ai_agent.support.builders import mcp_server_subentry_data
from tests.components.pydantic_ai_agent.support.schemas import (
    serialized_section_default,
)
from tests.components.pydantic_ai_agent.support.wizard import (
    loaded_workspace_entry,
    subentry_configure_result,
    subentry_init_result,
)


def _section_field_default(
    data_schema: Any, section_name: str, field_name: str
) -> object:
    """Return a nested section field default from a flow schema."""
    for section_key, section_value in data_schema.schema.items():
        if section_key.schema != section_name:
            continue
        for field_key in section_value.schema.schema:
            if field_key.schema == field_name:
                return field_key.default()
    raise AssertionError(f"Section field {section_name}.{field_name} not found")


async def _validate_mcp_server_success(
    _self: object, data: dict[str, Any], _current_subentry_id: str | None
) -> tuple[dict[str, Any], list[dict[str, str]], None]:
    """Return successful MCP validation for edit-flow behavior tests."""
    return dict(data), [], None


async def _start_edit_server_flow(
    hass: HomeAssistant, entry_id: str, subentry_id: str
) -> dict[str, Any]:
    """Open the MCP server edit form for a reconfigure flow."""
    result = await subentry_init_result(
        hass,
        (entry_id, SUBENTRY_TYPE_MCP_SERVER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": subentry_id,
        },
    )
    result = await subentry_configure_result(
        hass,
        result["flow_id"],
        {"next_step_id": "edit_server"},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "edit_server"
    return result


async def _submit_edit_server_and_finish(
    hass: HomeAssistant, flow_id: str, user_input: dict[str, Any]
) -> dict[str, Any]:
    """Submit edit_server, drive validation progress, and return final result."""
    result = await subentry_configure_result(hass, flow_id, user_input)
    assert result["type"] is FlowResultType.SHOW_PROGRESS
    assert result["step_id"] == "validate_mcp_server_progress"
    assert result["progress_action"] == "validate_mcp_server"
    await hass.async_block_till_done()
    result = await subentry_configure_result(hass, flow_id, None)
    if result["type"] is FlowResultType.SHOW_PROGRESS_DONE:
        assert result["next_step_id"] == "validate_mcp_server_finish"
        result = await subentry_configure_result(hass, flow_id, None)
    return result


async def test_mcp_server_edit_round_trips_headers_as_structured_rows(
    hass: HomeAssistant,
) -> None:
    """Test MCP headers reopen as selector rows and save back as stored headers."""
    subentry_data = mcp_server_subentry_data(headers={"Authorization": "Bearer old"})
    subentry_config = subentry_data["data"]
    assert isinstance(subentry_config, dict)
    subentry_config[CONF_MCP_SECRET_HEADER_KEYS] = ["Authorization"]
    entry = await loaded_workspace_entry(
        hass,
        (subentry_data,),
    )
    subentry = next(iter(entry.subentries.values()))

    result = await _start_edit_server_flow(hass, entry.entry_id, subentry.subentry_id)

    assert serialized_section_default(result["data_schema"], "advanced_mcp") == {}
    assert _section_field_default(
        result["data_schema"], "advanced_mcp", CONF_MCP_HEADERS
    ) == [
        {
            CONF_KEY_VALUE_KEY: "Authorization",
            CONF_KEY_VALUE_VALUE: "Bearer old",
            CONF_KEY_VALUE_IS_SECRET: True,
        }
    ]

    with patch.object(
        mcp_server_flow.MCPServerSubentryFlowHandler,
        "_async_validate_mcp_server",
        new=_validate_mcp_server_success,
    ):
        result = await _submit_edit_server_and_finish(
            hass,
            result["flow_id"],
            {
                CONF_NAME: "Echo MCP",
                CONF_MCP_URL: "https://mcp.example.com/mcp",
                "advanced_mcp": {
                    CONF_MCP_HEADERS: [
                        {
                            CONF_KEY_VALUE_KEY: "Authorization",
                            CONF_KEY_VALUE_VALUE: "Bearer new",
                            CONF_KEY_VALUE_IS_SECRET: True,
                        },
                        {
                            CONF_KEY_VALUE_KEY: "X-Trace",
                            CONF_KEY_VALUE_VALUE: "trace-1",
                            CONF_KEY_VALUE_IS_SECRET: False,
                        },
                    ]
                },
            },
        )

    assert result["type"] is FlowResultType.ABORT
    updated_subentry = entry.subentries[subentry.subentry_id]
    assert updated_subentry.data[CONF_MCP_HEADERS] == {
        "Authorization": "Bearer new",
        "X-Trace": "trace-1",
    }
    assert updated_subentry.data[CONF_MCP_SECRET_HEADER_KEYS] == ["Authorization"]

    reopened_result = await _start_edit_server_flow(
        hass, entry.entry_id, subentry.subentry_id
    )
    assert _section_field_default(
        reopened_result["data_schema"], "advanced_mcp", CONF_MCP_HEADERS
    ) == [
        {
            CONF_KEY_VALUE_KEY: "Authorization",
            CONF_KEY_VALUE_VALUE: "Bearer new",
            CONF_KEY_VALUE_IS_SECRET: True,
        },
        {
            CONF_KEY_VALUE_KEY: "X-Trace",
            CONF_KEY_VALUE_VALUE: "trace-1",
            CONF_KEY_VALUE_IS_SECRET: False,
        },
    ]


async def test_mcp_server_edit_clearing_headers_removes_stored_headers(
    hass: HomeAssistant,
) -> None:
    """Test clearing MCP headers during edit removes the stored header mapping."""
    entry = await loaded_workspace_entry(
        hass,
        (mcp_server_subentry_data(headers={"Authorization": "Bearer old"}),),
    )
    subentry = next(iter(entry.subentries.values()))
    result = await _start_edit_server_flow(hass, entry.entry_id, subentry.subentry_id)

    with patch.object(
        mcp_server_flow.MCPServerSubentryFlowHandler,
        "_async_validate_mcp_server",
        new=_validate_mcp_server_success,
    ):
        result = await _submit_edit_server_and_finish(
            hass,
            result["flow_id"],
            {
                CONF_NAME: "Echo MCP",
                CONF_MCP_URL: "https://mcp.example.com/mcp",
                "advanced_mcp": {CONF_MCP_HEADERS: []},
            },
        )

    assert result["type"] is FlowResultType.ABORT
    updated_subentry = entry.subentries[subentry.subentry_id]
    assert CONF_MCP_HEADERS not in updated_subentry.data
    assert CONF_MCP_SECRET_HEADER_KEYS not in updated_subentry.data


async def test_mcp_server_edit_preserves_existing_tool_allowlist(
    hass: HomeAssistant,
) -> None:
    """Test server edits keep an existing explicit MCP tool allowlist."""
    entry = await loaded_workspace_entry(
        hass,
        (mcp_server_subentry_data(allowed_tools=["echo", "fetch"]),),
    )
    subentry = next(iter(entry.subentries.values()))
    result = await _start_edit_server_flow(hass, entry.entry_id, subentry.subentry_id)

    with patch.object(
        mcp_server_flow.MCPServerSubentryFlowHandler,
        "_async_validate_mcp_server",
        new=_validate_mcp_server_success,
    ):
        result = await _submit_edit_server_and_finish(
            hass,
            result["flow_id"],
            {
                CONF_NAME: "Echo MCP",
                CONF_MCP_URL: "https://mcp.example.com/mcp",
                "advanced_mcp": {},
            },
        )

    assert result["type"] is FlowResultType.ABORT
    updated_subentry = entry.subentries[subentry.subentry_id]
    assert updated_subentry.data[CONF_MCP_ALLOWED_TOOLS] == ["echo", "fetch"]


async def test_mcp_server_edit_empty_advanced_section_preserves_settings(
    hass: HomeAssistant,
) -> None:
    """Test editing only name and URL keeps untouched advanced MCP settings."""
    entry = await loaded_workspace_entry(
        hass,
        (
            mcp_server_subentry_data(
                name="Echo MCP",
                url="https://mcp.example.com/original",
                headers={"Authorization": "Bearer old"},
                include_return_schema=False,
                deferred_loading=True,
            ),
        ),
    )
    subentry = next(iter(entry.subentries.values()))
    result = await _start_edit_server_flow(hass, entry.entry_id, subentry.subentry_id)

    with patch.object(
        mcp_server_flow.MCPServerSubentryFlowHandler,
        "_async_validate_mcp_server",
        new=_validate_mcp_server_success,
    ):
        result = await _submit_edit_server_and_finish(
            hass,
            result["flow_id"],
            {
                CONF_NAME: "Updated MCP",
                CONF_MCP_URL: "https://mcp.example.com/updated",
                "advanced_mcp": {},
            },
        )

    assert result["type"] is FlowResultType.ABORT
    updated_subentry = entry.subentries[subentry.subentry_id]
    assert updated_subentry.data[CONF_NAME] == "Updated MCP"
    assert updated_subentry.data[CONF_MCP_URL] == "https://mcp.example.com/updated"
    assert updated_subentry.data[CONF_MCP_HEADERS] == {"Authorization": "Bearer old"}
    assert updated_subentry.data[CONF_MCP_INCLUDE_RETURN_SCHEMA] is False
    assert updated_subentry.data[CONF_MCP_DEFERRED_LOADING] is True
