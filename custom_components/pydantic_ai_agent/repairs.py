"""Repair issue helpers for Pydantic AI Agent."""

from hashlib import sha1
import json
from collections.abc import Mapping
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import issue_registry as ir

from .const import DOMAIN
from .provider_validation import ProviderValidationError

MODEL_VALIDATION_ISSUE_PREFIX = "model_validation"
PROVIDER_AUTH_ISSUE_PREFIX = "provider_auth"
LOGFIRE_TOKEN_CONFLICT_ISSUE_ID = "logfire_token_conflict"


def provider_auth_issue_id(entry: ConfigEntry, provider_subentry_id: str) -> str:
    """Return a stable issue ID for one provider credential failure."""
    return f"{PROVIDER_AUTH_ISSUE_PREFIX}_{entry.entry_id}_{provider_subentry_id}"


def provider_validation_is_auth_failure(err: ProviderValidationError) -> bool:
    """Return if a provider validation failure requires credential repair."""
    return err.reason in {"invalid_auth", "permission_denied"} or err.status_code in {
        401,
        403,
    }


def model_validation_issue_id(
    entry: ConfigEntry, model_subentry_id: str, model_settings: Mapping[str, Any]
) -> str:
    """Return a stable issue ID for one entry/model validation failure."""
    issue_key = json.dumps(
        {"model_subentry_id": model_subentry_id, "model_settings": model_settings},
        sort_keys=True,
        separators=(",", ":"),
    )
    issue_digest = sha1(issue_key.encode()).hexdigest()[:12]
    return f"{MODEL_VALIDATION_ISSUE_PREFIX}_{entry.entry_id}_{issue_digest}"


@callback
def async_create_model_validation_issue(
    hass: HomeAssistant,
    entry: ConfigEntry,
    model_subentry_id: str,
    model: str,
    model_settings: Mapping[str, Any],
    err: ProviderValidationError,
) -> None:
    """Create an actionable repair issue for a configured model failure."""
    ir.async_create_issue(
        hass,
        DOMAIN,
        model_validation_issue_id(entry, model_subentry_id, model_settings),
        data={
            "entry_id": entry.entry_id,
            "model": model,
            "reason": err.reason,
            "status_code": err.status_code,
        },
        is_fixable=True,
        is_persistent=False,
        severity=ir.IssueSeverity.ERROR,
        translation_key="model_validation_failed",
        translation_placeholders={
            "entry_title": entry.title,
            "model": model,
            "reason": err.reason,
            "error_message": err.message,
        },
    )


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
        is_fixable=True,
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
def async_delete_model_validation_issue(
    hass: HomeAssistant,
    entry: ConfigEntry,
    model_subentry_id: str,
    model: str,
    model_settings: Mapping[str, Any],
) -> None:
    """Delete one model validation repair issue for a successful validation."""
    ir.async_delete_issue(
        hass,
        DOMAIN,
        model_validation_issue_id(entry, model_subentry_id, model_settings),
    )


@callback
def async_delete_stale_model_validation_issues(
    hass: HomeAssistant, entry: ConfigEntry, current_issue_ids: set[str]
) -> None:
    """Delete model validation repair issues for removed model/settings pairs."""
    prefix = f"{MODEL_VALIDATION_ISSUE_PREFIX}_{entry.entry_id}_"
    issue_registry = ir.async_get(hass)
    issue_ids = [
        issue_id
        for domain, issue_id in issue_registry.issues
        if domain == DOMAIN
        and issue_id.startswith(prefix)
        and issue_id not in current_issue_ids
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
