"""Tests for integration repair issue lifecycle."""

from typing import Any

from custom_components.pydantic_ai_agent.const import DOMAIN
from custom_components.pydantic_ai_agent.models.provider_validation import (
    ProviderValidationError,
)
from custom_components.pydantic_ai_agent.repair_issues import (
    LOGFIRE_TOKEN_CONFLICT_ISSUE_ID,
    async_create_logfire_token_conflict_issue,
    async_create_provider_auth_issue,
    async_delete_logfire_token_conflict_issue,
    async_delete_provider_auth_issue,
    provider_auth_issue_id,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir


def test_provider_auth_issue_create_delete_round_trip(
    hass: HomeAssistant, make_config_entry: Any
) -> None:
    """Provider authentication failures are visible until validation recovers."""
    entry = make_config_entry(entry_id="workspace-1")
    issue_id = provider_auth_issue_id(entry, "provider-1")

    async_create_provider_auth_issue(
        hass,
        entry,
        "provider-1",
        "Provider",
        ProviderValidationError("invalid_auth", "Authentication failed", 401),
    )

    registry = ir.async_get(hass)
    issue = registry.async_get_issue(DOMAIN, issue_id)
    assert issue is not None
    assert issue.translation_key == "provider_auth_failed"
    assert issue.severity is ir.IssueSeverity.ERROR

    async_delete_provider_auth_issue(hass, entry, "provider-1")
    assert registry.async_get_issue(DOMAIN, issue_id) is None


def test_logfire_conflict_issue_create_delete_round_trip(
    hass: HomeAssistant, make_config_entry: Any
) -> None:
    """A Logfire token conflict is visible only while the conflict exists."""
    entry = make_config_entry(entry_id="workspace-1")
    issue_id = f"{LOGFIRE_TOKEN_CONFLICT_ISSUE_ID}_{entry.entry_id}"

    async_create_logfire_token_conflict_issue(hass, entry)

    registry = ir.async_get(hass)
    issue = registry.async_get_issue(DOMAIN, issue_id)
    assert issue is not None
    assert issue.translation_key == "logfire_token_conflict"
    assert issue.severity is ir.IssueSeverity.WARNING

    async_delete_logfire_token_conflict_issue(hass, entry)
    assert registry.async_get_issue(DOMAIN, issue_id) is None
