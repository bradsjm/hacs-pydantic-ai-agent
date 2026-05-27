"""Test setup lifecycle for Pydantic AI Agent."""

import asyncio
from collections.abc import Mapping
from unittest.mock import AsyncMock, call, patch

from homeassistant import config_entries
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.pydantic_ai_agent import (
    PLATFORMS,
    SERVICE_REFRESH_MCP_TOOLS,
    async_remove_entry,
    async_setup,
    async_setup_entry,
    async_unload_entry,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_LOGFIRE_INCLUDE_CONTENT,
    CONF_LOGFIRE_TOKEN,
    DOMAIN,
    OUTPUT_MODE_TOOL,
)
from custom_components.pydantic_ai_agent.logfire_support import (
    async_release_logfire,
    logfire_active_for_entry,
    logfire_include_content,
)
from custom_components.pydantic_ai_agent.metrics import EVENT_MCP_TOOL_REFRESH_COMPLETED
from custom_components.pydantic_ai_agent.model_profiles import model_profile_ref
from custom_components.pydantic_ai_agent.provider_validation import (
    ProviderValidationError,
)
from custom_components.pydantic_ai_agent.repairs import model_validation_issue_id
from tests.components.pydantic_ai_agent.support.builders import (
    ai_task_subentry_data,
    conversation_subentry_data,
    mcp_server_subentry_data,
    provider_subentry_data,
    workspace_entry,
)


def _provider_subentry(
    *,
    subentry_id: str = "provider-1",
    profile_id: str = "profile-1",
    model: str = "gpt-test",
    model_settings: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return a provider subentry with setup-specific defaults."""
    return provider_subentry_data(
        subentry_id=subentry_id,
        profile_id=profile_id,
        model=model,
        model_settings=model_settings,
        discovered=True,
    )


def _conversation_subentry(profile_ref: str) -> dict[str, object]:
    """Return a conversation subentry using setup-specific defaults."""
    return conversation_subentry_data(profile_ref, subentry_id="conversation-1")


def _ai_task_subentry(profile_ref: str) -> dict[str, object]:
    """Return an AI task subentry using setup-specific defaults."""
    return ai_task_subentry_data(
        profile_ref,
        subentry_id="ai-task-1",
        output_mode=OUTPUT_MODE_TOOL,
    )


def _mcp_server_subentry() -> dict[str, object]:
    """Return an MCP server subentry using setup-specific defaults."""
    return mcp_server_subentry_data()


def _workspace_entry(
    subentries_data: tuple[dict[str, object], ...] = (),
    data: Mapping[str, object] | None = None,
) -> MockConfigEntry:
    """Return a workspace config entry using setup-specific defaults."""
    return workspace_entry(subentries_data, data=data)


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
        ],
        any_order=True,
    )
    assert probe_model.await_count == 2


async def test_unload_releases_logfire_owner_for_new_token(
    hass: HomeAssistant,
) -> None:
    """Test unloading the Logfire owner releases token conflict bookkeeping."""
    first_entry = _workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-a"})
    first_entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent.logfire_support._configure_logfire_sync"
    ) as configure_logfire:
        with patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new_callable=AsyncMock,
        ):
            assert await async_setup_entry(hass, first_entry)

        assert first_entry.runtime_data.logfire_enabled is True
        assert logfire_active_for_entry(hass, first_entry)

        with patch.object(
            hass.config_entries,
            "async_unload_platforms",
            new_callable=AsyncMock,
            return_value=True,
        ):
            assert await async_unload_entry(hass, first_entry)

        assert not logfire_active_for_entry(hass, first_entry)

        second_entry = _workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-b"})
        second_entry.add_to_hass(hass)
        with patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new_callable=AsyncMock,
        ):
            assert await async_setup_entry(hass, second_entry)

        assert second_entry.runtime_data.logfire_enabled is True
        assert logfire_active_for_entry(hass, second_entry)
        assert configure_logfire.call_count == 2


async def test_logfire_release_waits_for_last_same_token_owner(
    hass: HomeAssistant,
) -> None:
    """Test Logfire ownership remains active while same-token entries are loaded."""
    first_entry = _workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-a"})
    second_entry = _workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-a"})
    first_entry.add_to_hass(hass)
    second_entry.add_to_hass(hass)

    with (
        patch(
            "custom_components.pydantic_ai_agent.logfire_support._configure_logfire_sync"
        ) as configure_logfire,
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new_callable=AsyncMock,
        ),
    ):
        assert await async_setup_entry(hass, first_entry)
        assert await async_setup_entry(hass, second_entry)

    assert configure_logfire.call_count == 1
    assert first_entry.runtime_data.logfire_enabled is True
    assert second_entry.runtime_data.logfire_enabled is True
    assert logfire_active_for_entry(hass, first_entry)
    assert logfire_active_for_entry(hass, second_entry)

    with patch.object(
        hass.config_entries,
        "async_unload_platforms",
        new_callable=AsyncMock,
        return_value=True,
    ):
        assert await async_unload_entry(hass, first_entry)

    assert not logfire_active_for_entry(hass, first_entry)
    assert logfire_active_for_entry(hass, second_entry)

    with patch.object(
        hass.config_entries,
        "async_unload_platforms",
        new_callable=AsyncMock,
        return_value=True,
    ):
        assert await async_unload_entry(hass, second_entry)

    assert not logfire_active_for_entry(hass, second_entry)


async def test_logfire_include_content_is_scoped_per_same_token_owner(
    hass: HomeAssistant,
) -> None:
    """Test same-token entries keep their own content-capture setting."""
    first_entry = _workspace_entry(
        data={CONF_LOGFIRE_TOKEN: "token-a", CONF_LOGFIRE_INCLUDE_CONTENT: True}
    )
    second_entry = _workspace_entry(
        data={CONF_LOGFIRE_TOKEN: "token-a", CONF_LOGFIRE_INCLUDE_CONTENT: False}
    )
    first_entry.add_to_hass(hass)
    second_entry.add_to_hass(hass)

    with (
        patch(
            "custom_components.pydantic_ai_agent.logfire_support._configure_logfire_sync"
        ),
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new_callable=AsyncMock,
        ),
    ):
        assert await async_setup_entry(hass, first_entry)
        assert await async_setup_entry(hass, second_entry)

    assert logfire_include_content(hass, first_entry) is True
    assert logfire_include_content(hass, second_entry) is False

    with patch.object(
        hass.config_entries,
        "async_unload_platforms",
        new_callable=AsyncMock,
        return_value=True,
    ):
        assert await async_unload_entry(hass, first_entry)

    assert logfire_active_for_entry(hass, second_entry)
    assert logfire_include_content(hass, second_entry) is False


async def test_setup_failure_releases_logfire_owner(
    hass: HomeAssistant,
) -> None:
    """Test failed setup does not leave stale Logfire ownership."""
    first_entry = _workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-a"})
    first_entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent.logfire_support._configure_logfire_sync"
    ) as configure_logfire:
        with (
            patch.object(
                hass.config_entries,
                "async_forward_entry_setups",
                new_callable=AsyncMock,
                side_effect=RuntimeError("platform setup failed"),
            ),
            pytest.raises(RuntimeError, match="platform setup failed"),
        ):
            await async_setup_entry(hass, first_entry)

        assert not logfire_active_for_entry(hass, first_entry)

        second_entry = _workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-b"})
        second_entry.add_to_hass(hass)
        with patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new_callable=AsyncMock,
        ):
            assert await async_setup_entry(hass, second_entry)

        assert logfire_active_for_entry(hass, second_entry)
        assert configure_logfire.call_count == 2


async def test_setup_cancellation_releases_logfire_owner(
    hass: HomeAssistant,
) -> None:
    """Test cancelled setup does not leave stale Logfire ownership."""
    first_entry = _workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-a"})
    first_entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent.logfire_support._configure_logfire_sync"
    ) as configure_logfire:
        with (
            patch.object(
                hass.config_entries,
                "async_forward_entry_setups",
                new_callable=AsyncMock,
                side_effect=asyncio.CancelledError,
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await async_setup_entry(hass, first_entry)

        assert not logfire_active_for_entry(hass, first_entry)

        second_entry = _workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-b"})
        second_entry.add_to_hass(hass)
        with patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new_callable=AsyncMock,
        ):
            assert await async_setup_entry(hass, second_entry)

        assert logfire_active_for_entry(hass, second_entry)
        assert configure_logfire.call_count == 2


async def test_setup_retries_logfire_after_conflicting_owner_releases(
    hass: HomeAssistant,
) -> None:
    """Test an in-progress conflicting entry can claim Logfire before setup ends."""
    first_entry = _workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-a"})
    second_entry = _workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-b"})
    first_entry.add_to_hass(hass)
    second_entry.add_to_hass(hass)
    second_issue_id = f"logfire_token_conflict_{second_entry.entry_id}"

    with patch(
        "custom_components.pydantic_ai_agent.logfire_support._configure_logfire_sync"
    ) as configure_logfire:
        with patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new_callable=AsyncMock,
        ):
            assert await async_setup_entry(hass, first_entry)

        async def release_first_owner(*_: object) -> None:
            await async_release_logfire(hass, first_entry)

        with patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            side_effect=release_first_owner,
        ):
            assert await async_setup_entry(hass, second_entry)

        assert not logfire_active_for_entry(hass, first_entry)
        assert logfire_active_for_entry(hass, second_entry)
        assert second_entry.runtime_data.logfire_enabled is True
        assert ir.async_get(hass).async_get_issue(DOMAIN, second_issue_id) is None
        assert configure_logfire.call_count == 2


async def test_unload_promotes_loaded_conflicting_logfire_entry(
    hass: HomeAssistant,
) -> None:
    """Test releasing the owner enables a loaded entry that had conflicted."""
    first_entry = _workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-a"})
    second_entry = _workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-b"})
    third_entry = _workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-b"})
    first_entry.add_to_hass(hass)
    second_entry.add_to_hass(hass)
    third_entry.add_to_hass(hass)
    second_issue_id = f"logfire_token_conflict_{second_entry.entry_id}"
    third_issue_id = f"logfire_token_conflict_{third_entry.entry_id}"

    with (
        patch(
            "custom_components.pydantic_ai_agent.logfire_support._configure_logfire_sync"
        ) as configure_logfire,
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new_callable=AsyncMock,
        ),
    ):
        assert await async_setup_entry(hass, first_entry)
        assert await async_setup_entry(hass, second_entry)
        assert await async_setup_entry(hass, third_entry)

        assert logfire_active_for_entry(hass, first_entry)
        assert not logfire_active_for_entry(hass, second_entry)
        assert not logfire_active_for_entry(hass, third_entry)
        assert ir.async_get(hass).async_get_issue(DOMAIN, second_issue_id) is not None
        assert ir.async_get(hass).async_get_issue(DOMAIN, third_issue_id) is not None

        second_entry.mock_state(hass, config_entries.ConfigEntryState.LOADED)
        third_entry.mock_state(hass, config_entries.ConfigEntryState.LOADED)
        with patch.object(
            hass.config_entries,
            "async_unload_platforms",
            new_callable=AsyncMock,
            return_value=True,
        ):
            assert await async_unload_entry(hass, first_entry)

        assert logfire_active_for_entry(hass, second_entry)
        assert logfire_active_for_entry(hass, third_entry)
        assert ir.async_get(hass).async_get_issue(DOMAIN, second_issue_id) is None
        assert ir.async_get(hass).async_get_issue(DOMAIN, third_issue_id) is None
        assert configure_logfire.call_count == 2


async def test_setup_entry_validates_selected_model_setting_combinations(
    hass: HomeAssistant,
) -> None:
    """Test setup probes selected model/settings/output combinations."""
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
                structured_output_mode=OUTPUT_MODE_TOOL,
            ),
        ],
        any_order=True,
    )
    assert probe_model.await_count == 2


async def test_setup_entry_model_errors_create_repair_issue(
    hass: HomeAssistant,
) -> None:
    """Test selected model validation failures create repair issues."""
    profile_ref = model_profile_ref("provider-1", "profile-1")
    entry = _workspace_entry(
        (_provider_subentry(), _conversation_subentry(profile_ref))
    )
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
    entry = _workspace_entry(
        (_provider_subentry(), _conversation_subentry(profile_ref))
    )
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
