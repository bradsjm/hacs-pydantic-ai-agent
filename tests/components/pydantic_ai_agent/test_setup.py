"""Test setup lifecycle for Pydantic AI Agent."""

from collections.abc import Mapping
from unittest.mock import AsyncMock, call, patch

from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_NAME, Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.pydantic_ai_agent import (
    PLATFORMS,
    SERVICE_REFRESH_MCP_TOOLS,
    async_remove_entry,
    async_setup,
    async_setup_entry,
    async_unload_entry,
)
from custom_components.pydantic_ai_agent.config_flow import ProviderValidationError
from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_DEFAULT_MODEL_PROFILE_ID,
    CONF_DISCOVERED,
    CONF_ENABLED,
    CONF_MCP_URL,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_OUTPUT_MODE,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROVIDER_MODE,
    DOMAIN,
    OUTPUT_MODE_TOOL,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_MCP_SERVER,
    SUBENTRY_TYPE_PROVIDER,
)
from custom_components.pydantic_ai_agent.metrics import EVENT_MCP_TOOL_REFRESH_COMPLETED
from custom_components.pydantic_ai_agent.model_profiles import model_profile_ref
from custom_components.pydantic_ai_agent.repairs import model_validation_issue_id


def _provider_subentry(
    *,
    subentry_id: str = "provider-1",
    profile_id: str = "profile-1",
    model: str = "gpt-test",
    model_settings: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return a provider subentry with one embedded model profile."""
    profile: dict[str, object] = {
        "id": profile_id,
        CONF_NAME: "Fast GPT",
        CONF_MODEL: model,
        CONF_ENABLED: True,
        CONF_DISCOVERED: True,
    }
    if model_settings is not None:
        profile[CONF_MODEL_SETTINGS] = dict(model_settings)
    return {
        "subentry_id": subentry_id,
        "subentry_type": SUBENTRY_TYPE_PROVIDER,
        "title": "OpenAI-compatible",
        "unique_id": None,
        "data": {
            CONF_NAME: "OpenAI-compatible",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "sk-test",
            CONF_MODEL_PROFILES: {profile_id: profile},
            CONF_DEFAULT_MODEL_PROFILE_ID: profile_id,
        },
    }


def _conversation_subentry(profile_ref: str) -> dict[str, object]:
    """Return a conversation subentry using a model profile ref."""
    return {
        "subentry_id": "conversation-1",
        "subentry_type": SUBENTRY_TYPE_CONVERSATION,
        "title": "Kitchen Agent",
        "unique_id": None,
        "data": {CONF_AGENT_NAME: "Kitchen Agent", CONF_PRIMARY_MODEL_REF: profile_ref},
    }


def _ai_task_subentry(profile_ref: str) -> dict[str, object]:
    """Return an AI task subentry using a model profile ref."""
    return {
        "subentry_id": "ai-task-1",
        "subentry_type": SUBENTRY_TYPE_AI_TASK,
        "title": "Report task",
        "unique_id": None,
        "data": {
            CONF_PRIMARY_MODEL_REF: profile_ref,
            CONF_OUTPUT_MODE: OUTPUT_MODE_TOOL,
        },
    }


def _mcp_server_subentry() -> dict[str, object]:
    """Return an MCP server subentry."""
    return {
        "subentry_id": "mcp-server-1",
        "subentry_type": SUBENTRY_TYPE_MCP_SERVER,
        "title": "Filesystem MCP",
        "unique_id": None,
        "data": {
            CONF_NAME: "Filesystem MCP",
            CONF_MCP_URL: "https://mcp.example.com/mcp",
        },
    }


def _workspace_entry(
    subentries_data: tuple[dict[str, object], ...] = (),
) -> MockConfigEntry:
    """Return a workspace config entry."""
    return MockConfigEntry(
        domain=DOMAIN,
        title="Workspace",
        data={CONF_NAME: "Workspace"},
        source=config_entries.SOURCE_USER,
        subentries_data=subentries_data,
        options={},
        unique_id=None,
        version=2,
        minor_version=0,
    )


def test_platforms_include_conversation_ai_task_and_diagnostics() -> None:
    """Test setup forwards all runtime platforms."""
    assert PLATFORMS == (
        Platform.CONVERSATION,
        Platform.AI_TASK,
        Platform.SENSOR,
        Platform.BINARY_SENSOR,
    )


async def test_setup_entry_stores_workspace_runtime_data(hass: HomeAssistant) -> None:
    """Test setup stores workspace runtime data from provider subentries."""
    profile_ref = model_profile_ref("provider-1", "profile-1")
    entry = _workspace_entry(
        (
            _provider_subentry(),
            _conversation_subentry(profile_ref),
            _ai_task_subentry(profile_ref),
            _mcp_server_subentry(),
        )
    )
    entry.add_to_hass(hass)
    provider_data = entry.subentries["provider-1"].data

    with (
        patch(
            "custom_components.pydantic_ai_agent.async_probe_model",
            new_callable=AsyncMock,
        ) as probe_model,
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new_callable=AsyncMock,
        ) as forward_setups,
    ):
        assert await async_setup_entry(hass, entry)

    assert entry.runtime_data.workspace_name == "Workspace"
    assert entry.runtime_data.providers["provider-1"].api_key == "sk-test"
    assert entry.runtime_data.providers["provider-1"].name == "OpenAI-compatible"
    assert entry.runtime_data.model_profiles[profile_ref].model_name == "gpt-test"
    assert (
        entry.runtime_data.mcp_servers["mcp-server-1"].url
        == "https://mcp.example.com/mcp"
    )
    forward_setups.assert_awaited_once_with(entry, PLATFORMS)
    probe_model.assert_has_awaits(
        [
            call(hass, provider_data, "gpt-test", {}),
            call(
                hass,
                provider_data,
                "gpt-test",
                {},
                structured_output_mode=OUTPUT_MODE_TOOL,
            ),
        ]
    )
    assert probe_model.await_count == 2


async def test_setup_entry_deduplicates_model_setting_probes(
    hass: HomeAssistant,
) -> None:
    """Test setup probes each provider/model/settings/output combination once."""
    first_ref = model_profile_ref("provider-1", "first-profile")
    second_ref = model_profile_ref("provider-2", "second-profile")
    entry = _workspace_entry(
        (
            _provider_subentry(
                profile_id="first-profile",
                model="shared-model",
                model_settings={"timeout": 20.0},
            ),
            _provider_subentry(
                subentry_id="provider-2",
                profile_id="second-profile",
                model="shared-model",
                model_settings={"timeout": 20.0},
            ),
            _conversation_subentry(first_ref),
            _ai_task_subentry(second_ref),
        )
    )
    entry.add_to_hass(hass)

    with (
        patch(
            "custom_components.pydantic_ai_agent.async_probe_model",
            new_callable=AsyncMock,
        ) as probe_model,
        patch.object(
            hass.config_entries, "async_forward_entry_setups", new_callable=AsyncMock
        ),
    ):
        assert await async_setup_entry(hass, entry)

    assert probe_model.await_count == 3
    probe_model.assert_has_awaits(
        [
            call(
                hass,
                entry.subentries["provider-1"].data,
                "shared-model",
                {"timeout": 20.0},
            ),
            call(
                hass,
                entry.subentries["provider-2"].data,
                "shared-model",
                {"timeout": 20.0},
            ),
            call(
                hass,
                entry.subentries["provider-2"].data,
                "shared-model",
                {"timeout": 20.0},
                structured_output_mode=OUTPUT_MODE_TOOL,
            ),
        ]
    )


async def test_setup_entry_model_errors_create_repair_issue(
    hass: HomeAssistant,
) -> None:
    """Test setup-time model validation failures create repair issues."""
    profile_ref = model_profile_ref("provider-1", "profile-1")
    entry = _workspace_entry((_provider_subentry(),))
    entry.add_to_hass(hass)

    with (
        patch(
            "custom_components.pydantic_ai_agent.async_probe_model",
            new_callable=AsyncMock,
            side_effect=ProviderValidationError("invalid_model", "model unavailable"),
        ),
        patch.object(
            hass.config_entries, "async_forward_entry_setups", new_callable=AsyncMock
        ),
    ):
        assert await async_setup_entry(hass, entry)

    issue = ir.async_get(hass).async_get_issue(
        DOMAIN, model_validation_issue_id(entry, profile_ref, {})
    )
    assert issue is not None
    assert issue.translation_key == "model_validation_failed"
    assert issue.translation_placeholders == {
        "entry_title": "Workspace",
        "model": "gpt-test",
        "reason": "invalid_model",
        "error_message": "model unavailable",
    }


async def test_setup_entry_success_clears_model_validation_repair_issue(
    hass: HomeAssistant,
) -> None:
    """Test successful setup clears stale model validation repair issues."""
    profile_ref = model_profile_ref("provider-1", "profile-1")
    entry = _workspace_entry((_provider_subentry(),))
    entry.add_to_hass(hass)
    issue_id = model_validation_issue_id(entry, profile_ref, {})
    ir.async_create_issue(
        hass,
        DOMAIN,
        issue_id,
        is_fixable=True,
        severity=ir.IssueSeverity.ERROR,
        translation_key="model_validation_failed",
    )

    with (
        patch(
            "custom_components.pydantic_ai_agent.async_probe_model",
            new_callable=AsyncMock,
        ),
        patch.object(
            hass.config_entries, "async_forward_entry_setups", new_callable=AsyncMock
        ),
    ):
        assert await async_setup_entry(hass, entry)

    assert ir.async_get(hass).async_get_issue(DOMAIN, issue_id) is None


async def test_setup_registers_mcp_response_services(hass: HomeAssistant) -> None:
    """Test async setup registers MCP discovery response services."""
    assert await async_setup(hass, {})

    assert hass.services.has_service(DOMAIN, "list_mcp_tools")
    assert hass.services.has_service(DOMAIN, SERVICE_REFRESH_MCP_TOOLS)


async def test_refresh_mcp_tools_service_returns_discovered_tools(
    hass: HomeAssistant,
) -> None:
    """Test refresh_mcp_tools returns tools for a configured MCP server."""
    entry = _workspace_entry((_mcp_server_subentry(),))
    entry.add_to_hass(hass)
    await async_setup(hass, {})
    tools = [
        {
            "server_id": "mcp-server-1",
            "name": "list_files",
            "schema_hash": "abc123",
        }
    ]
    events: list[dict[str, object]] = []
    hass.bus.async_listen(
        f"{DOMAIN}_{EVENT_MCP_TOOL_REFRESH_COMPLETED}",
        lambda event: events.append(dict(event.data)),
    )

    with patch(
        "custom_components.pydantic_ai_agent.async_refresh_mcp_tools",
        new_callable=AsyncMock,
        return_value=tools,
    ) as refresh_tools:
        response = await hass.services.async_call(
            DOMAIN,
            SERVICE_REFRESH_MCP_TOOLS,
            {
                "config_entry_id": entry.entry_id,
                "mcp_server_id": "mcp-server-1",
            },
            blocking=True,
            return_response=True,
        )
        await hass.async_block_till_done()

    assert response == {
        "success": True,
        "servers": {"mcp-server-1": tools},
        "tools": tools,
        "errors": [],
    }
    refresh_tools.assert_awaited_once_with(hass, entry, "mcp-server-1")
    assert events == [
        {
            "config_entry_id": entry.entry_id,
            "mcp_server_id": "mcp-server-1",
            "tool_count": 1,
        }
    ]


async def test_unload_and_remove_entry_clean_entry_repair_issues(
    hass: HomeAssistant,
) -> None:
    """Test unload/remove cleanup entry-owned repair issues."""
    profile_ref = model_profile_ref("provider-1", "profile-1")
    entry = _workspace_entry((_provider_subentry(),))
    entry.add_to_hass(hass)
    logfire_issue_id = f"logfire_token_conflict_{entry.entry_id}"
    model_issue_id = model_validation_issue_id(entry, profile_ref, {})
    for issue_id, translation_key in (
        (logfire_issue_id, "logfire_token_conflict"),
        (model_issue_id, "model_validation_failed"),
    ):
        ir.async_create_issue(
            hass,
            DOMAIN,
            issue_id,
            is_fixable=True,
            severity=ir.IssueSeverity.WARNING,
            translation_key=translation_key,
        )

    with patch.object(
        hass.config_entries,
        "async_unload_platforms",
        new_callable=AsyncMock,
        return_value=True,
    ) as unload_platforms:
        assert await async_unload_entry(hass, entry)

    unload_platforms.assert_awaited_once_with(entry, PLATFORMS)
    assert ir.async_get(hass).async_get_issue(DOMAIN, logfire_issue_id) is None
    assert ir.async_get(hass).async_get_issue(DOMAIN, model_issue_id) is not None

    await async_remove_entry(hass, entry)
    assert ir.async_get(hass).async_get_issue(DOMAIN, model_issue_id) is None
