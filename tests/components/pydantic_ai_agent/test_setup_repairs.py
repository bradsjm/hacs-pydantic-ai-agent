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
from custom_components.pydantic_ai_agent.model_profiles import model_profile_ref
from custom_components.pydantic_ai_agent.provider_validation import (
    ProviderValidationError,
)
from custom_components.pydantic_ai_agent.repair_issues import (
    model_validation_issue_id,
    provider_auth_issue_id,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import issue_registry as ir
from pytest_homeassistant_custom_component.common import MockConfigEntry
from tests.components.pydantic_ai_agent.support.builders import (
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

    with (
        patch(
            "custom_components.pydantic_ai_agent._model_validation.async_probe_model",
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
    assert issue.is_fixable is False
    assert issue.translation_key == "model_validation_failed"
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
            "custom_components.pydantic_ai_agent._model_validation.async_probe_model",
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
    assert issue.is_fixable is False
    assert issue.translation_key == "provider_auth_failed"


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
        is_fixable=False,
        severity=ir.IssueSeverity.ERROR,
        translation_key="provider_auth_failed",
    )

    with (
        patch(
            "custom_components.pydantic_ai_agent._model_validation.async_probe_model",
            side_effect=ProviderValidationError("invalid_model", "model unavailable"),
        ),
        patch.object(
            hass.config_entries, "async_forward_entry_setups", new_callable=AsyncMock
        ),
    ):
        assert await async_setup_entry(hass, entry)

    assert ir.async_get(hass).async_get_issue(DOMAIN, issue_id) is None


async def test_setup_entry_success_clears_model_validation_repair_issue(
    hass: HomeAssistant,
) -> None:
    """Test successful setup clears stale model validation repair issues."""
    profile_ref = model_profile_ref("provider-1", "profile-1")
    entry = _workspace_entry(
        (_provider_subentry(), _conversation_subentry(profile_ref))
    )
    entry.add_to_hass(hass)
    ir.async_create_issue(
        hass,
        DOMAIN,
        model_validation_issue_id(entry, profile_ref, {}),
        is_fixable=False,
        severity=ir.IssueSeverity.ERROR,
        translation_key="model_validation_failed",
    )
    ir.async_create_issue(
        hass,
        DOMAIN,
        provider_auth_issue_id(entry, "provider-1"),
        is_fixable=False,
        severity=ir.IssueSeverity.ERROR,
        translation_key="provider_auth_failed",
    )

    with (
        patch(
            "custom_components.pydantic_ai_agent._model_validation.async_probe_model",
            new_callable=AsyncMock,
        ),
        patch.object(
            hass.config_entries, "async_forward_entry_setups", new_callable=AsyncMock
        ),
    ):
        assert await async_setup_entry(hass, entry)

    assert (
        ir.async_get(hass).async_get_issue(
            DOMAIN, model_validation_issue_id(entry, profile_ref, {})
        )
        is None
    )


async def test_setup_entry_keeps_provider_auth_issue_until_all_provider_probes_pass(
    hass: HomeAssistant,
) -> None:
    """Test provider auth repairs are not cleared by another profile probe."""
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
                "provider-1:auth-profile", subentry_id="auth-conversation"
            ),
            conversation_subentry_data(
                "provider-1:working-profile",
                subentry_id="working-conversation",
                title="Working Agent",
                agent_name="Working Agent",
            ),
        )
    )
    entry.add_to_hass(hass)

    async def probe(hass, provider_data, model, settings):
        if model == "auth-model":
            raise ProviderValidationError("invalid_auth", "provider rejected", 401)

    with (
        patch(
            "custom_components.pydantic_ai_agent._model_validation.async_probe_model",
            side_effect=probe,
        ),
        patch.object(
            hass.config_entries, "async_forward_entry_setups", new_callable=AsyncMock
        ),
    ):
        assert await async_setup_entry(hass, entry)

    assert (
        ir.async_get(hass).async_get_issue(
            DOMAIN, provider_auth_issue_id(entry, "provider-1")
        )
        is not None
    )


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
