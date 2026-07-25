"""Tests for config-entry diagnostics."""

import json
from typing import Any

from custom_components.pydantic_ai_agent.const import (
    CONF_API_KEY,
    CONF_LOGFIRE_TOKEN,
    CONF_MCP_HEADERS,
    CONF_MCP_SECRET_HEADER_KEYS,
    CONF_MCP_URL,
    CONF_MODEL_PROFILES,
    CONF_PROMPT,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_MODE,
    CONF_PROVIDER_SECRET_HEADER_KEYS,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_MCP_SERVER,
    SUBENTRY_TYPE_PROVIDER,
)
from custom_components.pydantic_ai_agent.diagnostics import (
    async_get_config_entry_diagnostics,
)
from custom_components.pydantic_ai_agent.runtime.header_metadata import REDACTED
from custom_components.pydantic_ai_agent.runtime.types import WorkspaceRuntimeData
from homeassistant.core import HomeAssistant


async def test_config_entry_diagnostics_redact_secrets_and_count_mcp_subentries(
    hass: HomeAssistant, make_config_entry: Any, make_subentry: Any
) -> None:
    """Diagnostics expose configuration shape without provider or MCP secrets."""
    secrets = {
        "entry": "entry-logfire-secret",
        "provider_key": "provider-api-secret",
        "provider_header": "provider-header-secret",
        "mcp_url": "https://mcp-secret.example/token",
        "mcp_header": "mcp-header-secret",
        "prompt": "private-agent-prompt",
    }
    provider = make_subentry(
        subentry_id="provider-1",
        subentry_type=SUBENTRY_TYPE_PROVIDER,
        title="Provider",
        data={
            CONF_API_KEY: secrets["provider_key"],
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_PROVIDER_HEADERS: {"Authorization": secrets["provider_header"]},
            CONF_PROVIDER_SECRET_HEADER_KEYS: ["Authorization"],
            CONF_MODEL_PROFILES: {},
            CONF_PROMPT: secrets["prompt"],
        },
    )
    mcp = make_subentry(
        subentry_id="mcp-1",
        subentry_type=SUBENTRY_TYPE_MCP_SERVER,
        title="MCP",
        data={
            CONF_MCP_URL: secrets["mcp_url"],
            CONF_MCP_HEADERS: {"X-API-Key": secrets["mcp_header"]},
            CONF_MCP_SECRET_HEADER_KEYS: ["X-API-Key"],
        },
    )
    entry = make_config_entry(
        data={CONF_LOGFIRE_TOKEN: secrets["entry"]},
        subentries=(provider, mcp),
    )
    entry.add_to_hass(hass)
    entry.runtime_data = WorkspaceRuntimeData(workspace_name="Workspace")

    diagnostics = await async_get_config_entry_diagnostics(hass, entry)

    assert set(diagnostics) == {"entry", "subentries", "runtime"}
    assert diagnostics["runtime"]["mcp_server_count"] == 1
    assert diagnostics["entry"]["data"][CONF_LOGFIRE_TOKEN] == REDACTED
    subentry_data = {subentry["subentry_id"]: subentry["data"] for subentry in diagnostics["subentries"]}
    provider_data = subentry_data["provider-1"]
    mcp_data = subentry_data["mcp-1"]
    assert provider_data[CONF_API_KEY] == REDACTED
    assert provider_data[CONF_PROVIDER_HEADERS]["Authorization"] == REDACTED
    assert provider_data[CONF_PROMPT] == REDACTED
    assert mcp_data[CONF_MCP_URL] == REDACTED
    assert mcp_data[CONF_MCP_HEADERS]["X-API-Key"] == REDACTED
    serialized = json.dumps(diagnostics)
    assert all(secret not in serialized for secret in secrets.values())
