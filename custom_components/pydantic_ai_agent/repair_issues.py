"""Repair issue helpers for Pydantic AI Agent."""

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import issue_registry as ir

from .const import DOMAIN
from .models.provider_validation import ProviderValidationError

MODEL_VALIDATION_ISSUE_PREFIX = "model_validation"
PROVIDER_AUTH_ISSUE_PREFIX = "provider_auth"
LOGFIRE_TOKEN_CONFLICT_ISSUE_ID = "logfire_token_conflict"


def provider_auth_issue_id(entry: ConfigEntry, provider_subentry_id: str) -> str:
    """Return a stable issue ID for one provider credential failure."""
    return f"{PROVIDER_AUTH_ISSUE_PREFIX}_{entry.entry_id}_{provider_subentry_id}"


@callback
def async_create_provider_auth_issue(
    hass: HomeAssistant,
    entry: ConfigEntry,
    provider_subentry_id: str,
    provider_title: str,
    err: ProviderValidationError,
) -> None:
    """Create an actionable repair issue for provider credential failures."""
    ir.async_create_issue(
        hass,
        DOMAIN,
        provider_auth_issue_id(entry, provider_subentry_id),
        data={
            "entry_id": entry.entry_id,
            "provider_subentry_id": provider_subentry_id,
            "reason": err.reason,
            "status_code": err.status_code,
        },
        is_fixable=False,
        is_persistent=False,
        severity=ir.IssueSeverity.ERROR,
        translation_key="provider_auth_failed",
        translation_placeholders={
            "entry_title": entry.title,
            "provider_title": provider_title,
            "reason": err.reason,
            "error_message": err.message,
        },
    )


@callback
def async_delete_provider_auth_issue(
    hass: HomeAssistant,
    entry: ConfigEntry,
    provider_subentry_id: str,
) -> None:
    """Delete one provider credential repair issue after successful validation."""
    ir.async_delete_issue(
        hass,
        DOMAIN,
        provider_auth_issue_id(entry, provider_subentry_id),
    )


@callback
def async_create_logfire_token_conflict_issue(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> None:
    """Create a repair issue when an entry cannot use its Logfire token."""
    ir.async_create_issue(
        hass,
        DOMAIN,
        f"{LOGFIRE_TOKEN_CONFLICT_ISSUE_ID}_{entry.entry_id}",
        data={"entry_id": entry.entry_id},
        is_fixable=False,
        is_persistent=False,
        severity=ir.IssueSeverity.WARNING,
        translation_key=LOGFIRE_TOKEN_CONFLICT_ISSUE_ID,
        translation_placeholders={"entry_title": entry.title},
    )


@callback
def async_delete_logfire_token_conflict_issue(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> None:
    """Delete the Logfire token conflict repair issue for an entry."""
    ir.async_delete_issue(
        hass, DOMAIN, f"{LOGFIRE_TOKEN_CONFLICT_ISSUE_ID}_{entry.entry_id}"
    )


@callback
def async_delete_entry_repair_issues(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Delete repair issues owned by a permanently removed entry."""
    async_delete_logfire_token_conflict_issue(hass, entry)
    prefixes = (
        f"{MODEL_VALIDATION_ISSUE_PREFIX}_{entry.entry_id}_",
        f"{PROVIDER_AUTH_ISSUE_PREFIX}_{entry.entry_id}_",
    )
    issue_registry = ir.async_get(hass)
    issue_ids = [
        issue_id
        for domain, issue_id in issue_registry.issues
        if domain == DOMAIN and issue_id.startswith(prefixes)
    ]
    for issue_id in issue_ids:
        ir.async_delete_issue(hass, DOMAIN, issue_id)


@callback
def async_delete_model_validation_issues(
    hass: HomeAssistant, entry: ConfigEntry
) -> None:
    """Delete legacy model-validation repair issues for an entry."""
    prefix = f"{MODEL_VALIDATION_ISSUE_PREFIX}_{entry.entry_id}_"
    issue_registry = ir.async_get(hass)
    issue_ids = [
        issue_id
        for domain, issue_id in issue_registry.issues
        if domain == DOMAIN and issue_id.startswith(prefix)
    ]
    for issue_id in issue_ids:
        ir.async_delete_issue(hass, DOMAIN, issue_id)


@callback
def async_delete_stale_provider_auth_issues(
    hass: HomeAssistant, entry: ConfigEntry, current_provider_subentry_ids: set[str]
) -> None:
    """Delete provider auth repair issues for removed provider subentries."""
    prefix = f"{PROVIDER_AUTH_ISSUE_PREFIX}_{entry.entry_id}_"
    issue_registry = ir.async_get(hass)
    issue_ids = [
        issue_id
        for domain, issue_id in issue_registry.issues
        if domain == DOMAIN
        and issue_id.startswith(prefix)
        and issue_id.removeprefix(prefix) not in current_provider_subentry_ids
    ]
    for issue_id in issue_ids:
        ir.async_delete_issue(hass, DOMAIN, issue_id)
