"""Context-management schema helpers for config flows."""

from typing import Any

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)

from ..const import (
    CONF_CONTEXT_MANAGEMENT_MODE,
    CONF_CONTEXT_SUMMARIZATION_MODEL_REF,
    CONTEXT_MANAGEMENT_CONTEXT_MANAGER,
    CONTEXT_MANAGEMENT_MODES,
)
from ..models.model_profiles import configured_model_profile_exists
from ._profile_selection import _model_profile_select_options


def _context_management_schema(
    options: dict[str, Any] | None,
    entry: ConfigEntry | None,
    *,
    default_mode: str,
) -> vol.Schema:
    """Return context-management settings schema."""
    options = dict(options or {})
    mode = options.get(CONF_CONTEXT_MANAGEMENT_MODE, default_mode)
    if mode not in CONTEXT_MANAGEMENT_MODES:
        mode = default_mode
    model_options = [SelectOptionDict(label="Current active model", value="")]
    model_options.extend(_model_profile_select_options(entry))
    return vol.Schema(
        {
            vol.Required(CONF_CONTEXT_MANAGEMENT_MODE, default=mode): SelectSelector(
                SelectSelectorConfig(
                    options=list(CONTEXT_MANAGEMENT_MODES),
                    mode=SelectSelectorMode.DROPDOWN,
                    translation_key=CONF_CONTEXT_MANAGEMENT_MODE,
                )
            ),
            vol.Optional(
                CONF_CONTEXT_SUMMARIZATION_MODEL_REF,
                default=options.get(CONF_CONTEXT_SUMMARIZATION_MODEL_REF, ""),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=model_options,
                    mode=SelectSelectorMode.DROPDOWN,
                    translation_key=CONF_CONTEXT_SUMMARIZATION_MODEL_REF,
                )
            ),
        }
    )


def _normalise_context_management_settings(
    data: dict[str, Any], entry: ConfigEntry | None, *, default_mode: str
) -> None:
    """Normalize context-management settings stored on subentries."""
    mode = data.get(CONF_CONTEXT_MANAGEMENT_MODE, default_mode)
    if mode not in CONTEXT_MANAGEMENT_MODES:
        mode = default_mode
    data[CONF_CONTEXT_MANAGEMENT_MODE] = mode
    summary_ref = data.get(CONF_CONTEXT_SUMMARIZATION_MODEL_REF)
    if (
        mode != CONTEXT_MANAGEMENT_CONTEXT_MANAGER
        or not isinstance(summary_ref, str)
        or not summary_ref
        or entry is None
        or not configured_model_profile_exists(entry, summary_ref)
    ):
        data.pop(CONF_CONTEXT_SUMMARIZATION_MODEL_REF, None)
