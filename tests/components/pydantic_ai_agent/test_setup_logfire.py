"""Test Logfire lifecycle in Pydantic AI Agent setup."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from custom_components.pydantic_ai_agent import (
    PLATFORMS,
    async_setup_entry,
    async_unload_entry,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_LOGFIRE_INCLUDE_CONTENT,
    CONF_LOGFIRE_TOKEN,
    DOMAIN,
)
from custom_components.pydantic_ai_agent.observability.logfire_support import (
    async_release_logfire,
    logfire_active_for_entry,
    logfire_include_content,
)
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir
from pytest_homeassistant_custom_component.common import MockConfigEntry
from tests.components.pydantic_ai_agent.support.builders import workspace_entry


def _workspace_entry(data: dict[str, object] | None = None) -> MockConfigEntry:
    """Return a workspace config entry."""
    return workspace_entry(data=data)


async def test_unload_releases_logfire_owner_for_new_token(
    hass: HomeAssistant,
) -> None:
    """Test unloading the Logfire owner releases token conflict bookkeeping."""
    first_entry = _workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-a"})
    first_entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent.observability.logfire_support._configure_logfire_sync"
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
            "custom_components.pydantic_ai_agent.observability.logfire_support._configure_logfire_sync"
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
            "custom_components.pydantic_ai_agent.observability.logfire_support._configure_logfire_sync"
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
        "custom_components.pydantic_ai_agent.observability.logfire_support._configure_logfire_sync"
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


async def test_setup_failure_after_forward_unloads_platforms(
    hass: HomeAssistant,
) -> None:
    """Test setup rolls back forwarded platforms if a later step fails."""
    entry = _workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-a"})
    entry.add_to_hass(hass)

    with (
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new_callable=AsyncMock,
        ) as forward_entry_setups,
        patch.object(
            hass.config_entries,
            "async_unload_platforms",
            new_callable=AsyncMock,
            return_value=True,
        ) as unload_platforms,
        patch(
            "custom_components.pydantic_ai_agent.async_configure_logfire",
            new_callable=AsyncMock,
            side_effect=[True, RuntimeError("logfire reconfigure failed")],
        ) as configure_logfire,
        patch(
            "custom_components.pydantic_ai_agent.async_release_logfire",
            new_callable=AsyncMock,
        ) as release_logfire,
        pytest.raises(RuntimeError, match="logfire reconfigure failed"),
    ):
        await async_setup_entry(hass, entry)

    forward_entry_setups.assert_awaited_once_with(entry, PLATFORMS)
    unload_platforms.assert_awaited_once_with(entry, PLATFORMS)
    release_logfire.assert_awaited_once_with(hass, entry)
    assert configure_logfire.await_count == 2


async def test_setup_cancellation_releases_logfire_owner(
    hass: HomeAssistant,
) -> None:
    """Test cancelled setup does not leave stale Logfire ownership."""
    first_entry = _workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-a"})
    first_entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent.observability.logfire_support._configure_logfire_sync"
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

    with patch(
        "custom_components.pydantic_ai_agent.observability.logfire_support._configure_logfire_sync"
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
            "custom_components.pydantic_ai_agent.observability.logfire_support._configure_logfire_sync"
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
