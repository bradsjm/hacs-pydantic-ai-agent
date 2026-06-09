"""Test setup lifecycle for Pydantic AI Agent."""

import asyncio
from collections.abc import Mapping
from unittest.mock import AsyncMock, call, patch

import pytest
from custom_components.pydantic_ai_agent import (
    PLATFORMS,
    async_remove_entry,
    async_setup,
    async_setup_entry,
    async_unload_entry,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_LOGFIRE_INCLUDE_CONTENT,
    CONF_LOGFIRE_TOKEN,
    CONF_MAX_TOKENS,
    CONF_THINKING,
    CONF_TIMEOUT,
    DOMAIN,
    OUTPUT_MODE_TOOL,
)
from custom_components.pydantic_ai_agent.logfire_support import (
    async_release_logfire,
    logfire_active_for_entry,
    logfire_include_content,
)
from custom_components.pydantic_ai_agent.model_profiles import model_profile_ref
from custom_components.pydantic_ai_agent.provider_validation import (
    ProviderValidationError,
)
from custom_components.pydantic_ai_agent.repairs import (
    model_validation_issue_id,
    provider_auth_issue_id,
)
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import issue_registry as ir
from pytest_homeassistant_custom_component.common import MockConfigEntry
from tests.components.pydantic_ai_agent.support.builders import (
    ai_task_subentry_data,
    conversation_subentry_data,
    model_profile_data,
    provider_subentry_data,
    workspace_entry,
)


def _provider_subentry(
    *,
    subentry_id: str = "provider-1",
    profile_id: str = "profile-1",
    model: str = "gpt-test",
    model_settings: Mapping[str, object] | None = None,
    model_profiles: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Return a provider subentry with setup-specific defaults."""
    return provider_subentry_data(
        subentry_id=subentry_id,
        profile_id=profile_id,
        model=model,
        model_settings=model_settings,
        model_profiles=model_profiles,
        discovered=True,
    )


def _conversation_subentry(
    profile_ref: str, extra_data: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Return a conversation subentry using setup-specific defaults."""
    return conversation_subentry_data(
        profile_ref, subentry_id="conversation-1", extra_data=extra_data
    )


def _ai_task_subentry(
    profile_ref: str, extra_data: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Return an AI task subentry using setup-specific defaults."""
    return ai_task_subentry_data(
        profile_ref,
        subentry_id="ai-task-1",
        output_mode=OUTPUT_MODE_TOOL,
        extra_data=extra_data,
    )


def _workspace_entry(
    subentries_data: tuple[dict[str, object], ...] = (),
    data: Mapping[str, object] | None = None,
) -> MockConfigEntry:
    """Return a workspace config entry using setup-specific defaults."""
    return workspace_entry(subentries_data, data=data)


async def test_setup_entry_stores_workspace_runtime_data(hass: HomeAssistant) -> None:
    """Test setup stores workspace runtime data from provider subentries."""
    profile_ref = model_profile_ref("provider-1", "profile-1")
    entry = _workspace_entry(
        (
            _provider_subentry(),
            _conversation_subentry(profile_ref),
            _ai_task_subentry(profile_ref),
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
                model_settings={CONF_TIMEOUT: 99.0, CONF_MAX_TOKENS: 99},
            ),
            _provider_subentry(
                subentry_id="provider-2",
                profile_id="second-profile",
                model="shared-model",
                model_settings={CONF_TIMEOUT: 99.0, CONF_THINKING: "low"},
            ),
            _conversation_subentry(
                first_ref,
                extra_data={CONF_TIMEOUT: 20.0, CONF_MAX_TOKENS: 512},
            ),
            _ai_task_subentry(
                second_ref,
                extra_data={CONF_TIMEOUT: 30.0, CONF_THINKING: "high"},
            ),
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
                {CONF_MAX_TOKENS: 512, CONF_TIMEOUT: 20.0},
            ),
            call(
                hass,
                entry.subentries["provider-2"].data,
                "shared-model",
                {CONF_THINKING: "high", CONF_TIMEOUT: 30.0},
                structured_output_mode=OUTPUT_MODE_TOOL,
            ),
        ],
        any_order=True,
    )
    assert probe_model.await_count == 2


async def test_setup_entry_probes_distinct_subentry_run_settings(
    hass: HomeAssistant,
) -> None:
    """Test setup keeps same-model probes distinct by subentry run settings."""
    profile_ref = model_profile_ref("provider-1", "profile-1")
    entry = _workspace_entry(
        (
            _provider_subentry(model="shared-model"),
            conversation_subentry_data(
                profile_ref,
                subentry_id="conversation-1",
                extra_data={CONF_TIMEOUT: 20.0},
            ),
            conversation_subentry_data(
                profile_ref,
                subentry_id="conversation-2",
                title="Second Agent",
                agent_name="Second Agent",
                extra_data={CONF_THINKING: "high", CONF_TIMEOUT: 20.0},
            ),
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
            hass.config_entries, "async_forward_entry_setups", new_callable=AsyncMock
        ),
    ):
        assert await async_setup_entry(hass, entry)

    probe_model.assert_has_awaits(
        [
            call(hass, provider_data, "shared-model", {CONF_TIMEOUT: 20.0}),
            call(
                hass,
                provider_data,
                "shared-model",
                {CONF_THINKING: "high", CONF_TIMEOUT: 20.0},
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
    failure_key = f"conversation-1:{profile_ref}"
    entry = _workspace_entry(
        (_provider_subentry(), _conversation_subentry(profile_ref))
    )
    entry.add_to_hass(hass)
    forward_setups = AsyncMock()

    async def assert_failures_stored_before_platform_setup(
        forwarded_entry: MockConfigEntry, platforms: tuple[object, ...]
    ) -> None:
        assert forwarded_entry is entry
        assert platforms == PLATFORMS
        assert entry.runtime_data.model_validation_failures == {
            failure_key: "invalid_model"
        }

    forward_setups.side_effect = assert_failures_stored_before_platform_setup

    with (
        patch(
            "custom_components.pydantic_ai_agent.async_probe_model",
            new_callable=AsyncMock,
            side_effect=ProviderValidationError("invalid_model", "model unavailable"),
        ),
        patch.object(hass.config_entries, "async_forward_entry_setups", forward_setups),
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
    assert entry.runtime_data.model_validation_failures == {
        failure_key: "invalid_model"
    }


async def test_setup_entry_auth_errors_create_provider_auth_repair_issue(
    hass: HomeAssistant,
) -> None:
    """Test provider auth failures create provider-scoped repair issues."""
    profile_ref = model_profile_ref("provider-1", "profile-1")
    entry = _workspace_entry(
        (_provider_subentry(), _conversation_subentry(profile_ref))
    )
    entry.add_to_hass(hass)

    with (
        patch(
            "custom_components.pydantic_ai_agent.async_probe_model",
            new_callable=AsyncMock,
            side_effect=ProviderValidationError(
                "invalid_auth", "provider rejected credentials", 401
            ),
        ),
        patch.object(
            hass.config_entries, "async_forward_entry_setups", new_callable=AsyncMock
        ),
    ):
        assert await async_setup_entry(hass, entry)

    issue = ir.async_get(hass).async_get_issue(
        DOMAIN, provider_auth_issue_id(entry, "provider-1")
    )
    assert issue is not None
    assert issue.translation_key == "provider_auth_failed"
    assert issue.translation_placeholders == {
        "entry_title": "Workspace",
        "provider_title": "OpenAI-compatible",
        "reason": "invalid_auth",
        "error_message": "provider rejected credentials",
    }


async def test_setup_entry_non_auth_model_error_clears_provider_auth_issue(
    hass: HomeAssistant,
) -> None:
    """Test non-auth validation failures remove stale provider auth repairs."""
    profile_ref = model_profile_ref("provider-1", "profile-1")
    entry = _workspace_entry(
        (_provider_subentry(), _conversation_subentry(profile_ref))
    )
    entry.add_to_hass(hass)
    issue_id = provider_auth_issue_id(entry, "provider-1")
    ir.async_create_issue(
        hass,
        DOMAIN,
        issue_id,
        is_fixable=True,
        severity=ir.IssueSeverity.ERROR,
        translation_key="provider_auth_failed",
    )

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

    assert ir.async_get(hass).async_get_issue(DOMAIN, issue_id) is None


async def test_setup_entry_keeps_provider_auth_issue_until_all_provider_probes_pass(
    hass: HomeAssistant,
) -> None:
    """Test provider auth repairs are not cleared by another profile probe."""
    auth_profile_ref = model_profile_ref("provider-1", "auth-profile")
    working_profile_ref = model_profile_ref("provider-1", "working-profile")
    entry = _workspace_entry(
        (
            _provider_subentry(
                model_profiles={
                    "auth-profile": model_profile_data(
                        profile_id="auth-profile", model="auth-model"
                    ),
                    "working-profile": model_profile_data(
                        profile_id="working-profile", model="working-model"
                    ),
                },
            ),
            conversation_subentry_data(
                auth_profile_ref,
                subentry_id="auth-conversation",
            ),
            conversation_subentry_data(
                working_profile_ref,
                subentry_id="working-conversation",
                title="Working Agent",
                agent_name="Working Agent",
            ),
        )
    )
    entry.add_to_hass(hass)

    async def probe_side_effect(
        _hass: HomeAssistant,
        _provider_data: Mapping[str, object],
        model: str,
        _settings: Mapping[str, object],
    ) -> None:
        if model == "auth-model":
            raise ProviderValidationError("invalid_auth", "provider rejected", 401)

    with (
        patch(
            "custom_components.pydantic_ai_agent.async_probe_model",
            new_callable=AsyncMock,
            side_effect=probe_side_effect,
        ),
        patch.object(
            hass.config_entries, "async_forward_entry_setups", new_callable=AsyncMock
        ),
    ):
        assert await async_setup_entry(hass, entry)

    issue = ir.async_get(hass).async_get_issue(
        DOMAIN, provider_auth_issue_id(entry, "provider-1")
    )
    assert issue is not None
    assert issue.translation_key == "provider_auth_failed"


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
    provider_issue_id = provider_auth_issue_id(entry, "provider-1")
    ir.async_create_issue(
        hass,
        DOMAIN,
        provider_issue_id,
        is_fixable=True,
        severity=ir.IssueSeverity.ERROR,
        translation_key="provider_auth_failed",
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
    assert ir.async_get(hass).async_get_issue(DOMAIN, provider_issue_id) is None


async def test_response_services_raise_for_unknown_config_entry(
    hass: HomeAssistant,
) -> None:
    """Test response services raise translated service errors for bad entries."""
    await async_setup(hass, {})

    with pytest.raises(ServiceValidationError) as err:
        await hass.services.async_call(
            DOMAIN,
            "get_agent_run_diagnostics",
            {"config_entry_id": "missing-entry", "subentry_id": "missing-subentry"},
            blocking=True,
            return_response=True,
        )

    assert err.value.translation_key == "config_entry_not_found"
    assert err.value.translation_placeholders == {"config_entry_id": "missing-entry"}


async def test_response_services_raise_for_unknown_agent_subentry(
    hass: HomeAssistant,
) -> None:
    """Test run diagnostics raises a service error for missing subentries."""
    entry = _workspace_entry(())
    entry.add_to_hass(hass)
    await async_setup(hass, {})

    with pytest.raises(ServiceValidationError) as err:
        await hass.services.async_call(
            DOMAIN,
            "get_agent_run_diagnostics",
            {"config_entry_id": entry.entry_id, "subentry_id": "missing-subentry"},
            blocking=True,
            return_response=True,
        )

    assert err.value.translation_key == "subentry_not_found"
    assert err.value.translation_placeholders == {
        "config_entry_id": entry.entry_id,
        "subentry_id": "missing-subentry",
    }


async def test_setup_entry_removes_stale_subentry_registry_entries(
    hass: HomeAssistant,
) -> None:
    """Test setup removes orphaned entities and empty devices for deleted subentries."""
    stale_subentry_id = "deleted-conversation"
    entry = _workspace_entry(
        (
            conversation_subentry_data(
                "provider-1:profile-1", subentry_id=stale_subentry_id
            ),
        )
    )
    entry.add_to_hass(hass)
    entity_registry = er.async_get(hass)
    device_registry = dr.async_get(hass)
    stale_entity = entity_registry.async_get_or_create(
        "conversation",
        DOMAIN,
        f"{DOMAIN}_{entry.entry_id}_conversation_{stale_subentry_id}",
        config_entry=entry,
        config_subentry_id=stale_subentry_id,
        suggested_object_id="deleted_conversation",
    )
    stale_device = device_registry.async_get_or_create(
        config_entry_id=entry.entry_id,
        config_subentry_id=stale_subentry_id,
        identifiers={(DOMAIN, f"{entry.entry_id}:conversation:{stale_subentry_id}")},
        name="Deleted Conversation Configuration",
    )
    hass.config_entries.async_remove_subentry(entry, stale_subentry_id)

    with patch.object(
        hass.config_entries, "async_forward_entry_setups", new_callable=AsyncMock
    ):
        assert await async_setup_entry(hass, entry)

    assert entity_registry.async_get(stale_entity.entity_id) is None
    assert device_registry.async_get(stale_device.id) is None


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
