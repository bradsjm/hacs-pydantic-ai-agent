"""Validation and logging helpers for config-flow profile forms."""

import logging
from collections.abc import Mapping
from typing import Any

from homeassistant.components.todo.const import DOMAIN as TODO_DOMAIN
from homeassistant.core import HomeAssistant

from ..const import CONF_TODO_LIST_ENTITY_ID
from ..provider_validation import ProviderValidationError
from ._constants import _TODO_WORKSPACE_REQUIRED_FEATURES

_LOGGER = logging.getLogger(__name__)


def _provider_validation_placeholders(
    err: ProviderValidationError,
) -> dict[str, str]:
    """Return translation placeholders for provider validation errors."""
    placeholders = {"error_message": err.message}
    if err.status_code is not None:
        placeholders["status_code"] = str(err.status_code)
    return placeholders


def _log_provider_validation_failure(
    *, step: str, model_name: str, err: ProviderValidationError
) -> None:
    """Log provider validation failures without request details or credentials."""
    if err.status_code == 429:
        _LOGGER.warning(
            'Provider validation rate limited during %s for model "%s": '
            "reason=%s status_code=%s",
            step,
            model_name,
            err.reason,
            err.status_code,
        )
        return

    _LOGGER.warning(
        'Provider validation failed during %s for model "%s": reason=%s status_code=%s',
        step,
        model_name,
        err.reason,
        err.status_code,
    )


def _selected_todo_workspace_error(
    hass: HomeAssistant, data: Mapping[str, Any]
) -> str | None:
    """Return a form error for an invalid todo workspace entity."""
    entity_id = data.get(CONF_TODO_LIST_ENTITY_ID)
    if not entity_id:
        return None
    if not isinstance(entity_id, str) or not entity_id.startswith(f"{TODO_DOMAIN}."):
        return "todo_list_not_found"
    state = hass.states.get(entity_id)
    if state is None:
        return "todo_list_not_found"
    supported_features = state.attributes.get("supported_features", 0)
    if not isinstance(supported_features, int):
        return "todo_list_unsupported"
    if (
        supported_features & _TODO_WORKSPACE_REQUIRED_FEATURES
        != _TODO_WORKSPACE_REQUIRED_FEATURES
    ):
        return "todo_list_unsupported"
    return None
