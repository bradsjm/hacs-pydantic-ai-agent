"""Repair issue helpers for Pydantic AI Agent."""

from hashlib import sha1
import json
from collections.abc import Mapping
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import issue_registry as ir

from .config_flow import ProviderValidationError
from .const import DOMAIN

MODEL_VALIDATION_ISSUE_PREFIX = "model_validation"
LOGFIRE_TOKEN_CONFLICT_ISSUE_ID = "logfire_token_conflict"


def model_validation_issue_id(
    entry: ConfigEntry, model: str, model_settings: Mapping[str, Any]
) -> str:
    """Return a stable issue ID for one entry/model validation failure."""
    issue_key = json.dumps(
        {"model": model, "model_settings": model_settings},
        sort_keys=True,
        separators=(",", ":"),
    )
    issue_digest = sha1(issue_key.encode()).hexdigest()[:12]
    return f"{MODEL_VALIDATION_ISSUE_PREFIX}_{entry.entry_id}_{issue_digest}"


@callback
def async_create_model_validation_issue(
    hass: HomeAssistant,
    entry: ConfigEntry,
    model: str,
    model_settings: Mapping[str, Any],
    err: ProviderValidationError,
) -> None:
    """Create an actionable repair issue for a configured model failure."""
    ir.async_create_issue(
        hass,
        DOMAIN,
        model_validation_issue_id(entry, model, model_settings),
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
def async_delete_model_validation_issue(
    hass: HomeAssistant,
    entry: ConfigEntry,
    model: str,
    model_settings: Mapping[str, Any],
) -> None:
    """Delete one model validation repair issue for a successful validation."""
    ir.async_delete_issue(
        hass, DOMAIN, model_validation_issue_id(entry, model, model_settings)
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
