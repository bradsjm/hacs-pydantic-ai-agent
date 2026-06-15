"""Test setup repair issue creation/cleanup and response services."""

from unittest.mock import AsyncMock, patch

import pytest
from custom_components.pydantic_ai_agent import (
    async_setup,
    async_setup_entry,
)
from custom_components.pydantic_ai_agent.const import (
    DOMAIN,
)
from custom_components.pydantic_ai_agent.repair_issues import (
    provider_auth_issue_id,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import issue_registry as ir
from pytest_homeassistant_custom_component.common import MockConfigEntry
from tests.components.pydantic_ai_agent.support.builders import (
    conversation_subentry_data,
    provider_subentry_data,
    workspace_entry,
)


def _provider_subentry(
    *,
    subentry_id: str = "provider-1",
    profile_id: str = "profile-1",
    model: str = "gpt-test",
    model_settings: dict[str, object] | None = None,
    model_profiles: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    return provider_subentry_data(
        subentry_id=subentry_id,
        profile_id=profile_id,
        model=model,
        model_settings=model_settings,
        model_profiles=model_profiles,
        discovered=True,
    )


def _conversation_subentry(
    profile_ref: str, extra_data: dict[str, object] | None = None
) -> dict[str, object]:
    return conversation_subentry_data(
        profile_ref, subentry_id="conversation-1", extra_data=extra_data
    )


def _workspace_entry(
    subentries_data: tuple[dict[str, object], ...] = (),
    data: dict[str, object] | None = None,
) -> MockConfigEntry:
    return workspace_entry(subentries_data, data=data)


def _legacy_model_validation_issue_id(entry: MockConfigEntry, suffix: str) -> str:
    """Return a legacy setup-time model validation issue id."""
    return f"model_validation_{entry.entry_id}_{suffix}"


async def test_setup_entry_removes_legacy_model_validation_repair_issue(
    hass: HomeAssistant,
) -> None:
    """Test setup always clears legacy probe-driven model validation issues."""
    profile_ref = "provider-1:profile-1"
    entry = _workspace_entry(
        (_provider_subentry(), _conversation_subentry(profile_ref))
    )
    entry.add_to_hass(hass)
    issue_id = _legacy_model_validation_issue_id(entry, "conversation-1")
    ir.async_create_issue(
        hass,
        DOMAIN,
        issue_id,
        is_fixable=False,
        severity=ir.IssueSeverity.ERROR,
        translation_key="model_validation_failed",
    )

    with patch.object(
        hass.config_entries, "async_forward_entry_setups", new_callable=AsyncMock
    ):
        assert await async_setup_entry(hass, entry)

    assert ir.async_get(hass).async_get_issue(DOMAIN, issue_id) is None


async def test_setup_entry_preserves_provider_auth_issue_for_current_provider(
    hass: HomeAssistant,
) -> None:
    """Test setup no longer clears auth issues for still-configured providers."""
    profile_ref = "provider-1:profile-1"
    entry = _workspace_entry(
        (_provider_subentry(), _conversation_subentry(profile_ref))
    )
    entry.add_to_hass(hass)
    issue_id = provider_auth_issue_id(entry, "provider-1")
    ir.async_create_issue(
        hass,
        DOMAIN,
        issue_id,
        is_fixable=False,
        severity=ir.IssueSeverity.ERROR,
        translation_key="provider_auth_failed",
    )

    with patch.object(
        hass.config_entries, "async_forward_entry_setups", new_callable=AsyncMock
    ):
        assert await async_setup_entry(hass, entry)

    issue = ir.async_get(hass).async_get_issue(DOMAIN, issue_id)
    assert issue is not None
    assert issue.translation_key == "provider_auth_failed"


async def test_setup_entry_removes_stale_provider_auth_issue_for_removed_provider(
    hass: HomeAssistant,
) -> None:
    """Test setup removes auth issues for provider subentries that no longer exist."""
    profile_ref = "provider-2:profile-1"
    entry = _workspace_entry(
        (
            _provider_subentry(subentry_id="provider-2"),
            _conversation_subentry(profile_ref),
        )
    )
    entry.add_to_hass(hass)
    stale_issue_id = provider_auth_issue_id(entry, "provider-1")
    ir.async_create_issue(
        hass,
        DOMAIN,
        stale_issue_id,
        is_fixable=False,
        severity=ir.IssueSeverity.ERROR,
        translation_key="provider_auth_failed",
    )

    with patch.object(
        hass.config_entries, "async_forward_entry_setups", new_callable=AsyncMock
    ):
        assert await async_setup_entry(hass, entry)

    assert ir.async_get(hass).async_get_issue(DOMAIN, stale_issue_id) is None


async def test_setup_entry_removes_stale_subentry_registry_entries(
    hass: HomeAssistant,
) -> None:
    """Test setup removes orphaned entities and empty devices for deleted subentries."""
    from homeassistant.helpers import device_registry as dr
    from homeassistant.helpers import entity_registry as er

    stale_id = "deleted-conversation"
    entry = _workspace_entry(
        (conversation_subentry_data("provider-1:profile-1", subentry_id=stale_id),)
    )
    entry.add_to_hass(hass)
    ereg = er.async_get(hass)
    dreg = dr.async_get(hass)
    entity = ereg.async_get_or_create(
        "conversation",
        DOMAIN,
        f"{DOMAIN}_{entry.entry_id}_conversation_{stale_id}",
        config_entry=entry,
        config_subentry_id=stale_id,
        suggested_object_id="deleted_conversation",
    )
    device = dreg.async_get_or_create(
        config_entry_id=entry.entry_id,
        config_subentry_id=stale_id,
        identifiers={(DOMAIN, f"{entry.entry_id}:conversation:{stale_id}")},
        name="Deleted",
    )
    hass.config_entries.async_remove_subentry(entry, stale_id)

    with patch.object(
        hass.config_entries, "async_forward_entry_setups", new_callable=AsyncMock
    ):
        assert await async_setup_entry(hass, entry)

    assert ereg.async_get(entity.entity_id) is None
    assert dreg.async_get(device.id) is None


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
