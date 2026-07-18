"""Tests for MCP status derived from configured subentries."""

from typing import Any

from custom_components.pydantic_ai_agent.const import (
    CONF_MCP_URL,
    SUBENTRY_TYPE_MCP_SERVER,
)
from custom_components.pydantic_ai_agent.diagnostics import _runtime_diagnostics
from custom_components.pydantic_ai_agent.observability._debug_service_responses import (
    workspace_status,
)
from custom_components.pydantic_ai_agent.runtime.types import WorkspaceRuntimeData


def test_mcp_status_uses_configured_subentries(
    make_config_entry: Any, make_subentry: Any
) -> None:
    """Configured MCP servers do not require a duplicate runtime snapshot."""
    mcp_subentry = make_subentry(
        subentry_id="mcp-1",
        title="MCP",
        subentry_type=SUBENTRY_TYPE_MCP_SERVER,
        data={CONF_MCP_URL: "https://example.test/mcp"},
    )
    entry = make_config_entry(
        entry_id="entry-1",
        subentries=(mcp_subentry,),
    )
    entry.runtime_data = WorkspaceRuntimeData(
        workspace_name="Workspace",
        mcp_tool_cache={
            "stale-mcp-1": [{"name": "stale-tool-1"}],
            "stale-mcp-2": [{"name": "stale-tool-2"}],
        },
    )

    diagnostics = _runtime_diagnostics(entry)
    status = workspace_status(entry, include_subentries=True, include_runtime=True)

    assert diagnostics["mcp_server_count"] == 1
    assert status["runtime"]["mcp_server_count"] == 1
    assert status["subentries"]["mcp_servers"][0]["runtime_loaded"] is True
