"""Workspace config-flow helpers."""

from collections.abc import Mapping
from typing import Any

from homeassistant.const import CONF_NAME
from homeassistant.data_entry_flow import section
from homeassistant.helpers.selector import (
    BooleanSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
from homeassistant.helpers.typing import VolDictType
import voluptuous as vol

from ..const import (
    CONF_LOGFIRE_INCLUDE_CONTENT,
    CONF_LOGFIRE_TOKEN,
    DEFAULT_WORKSPACE_NAME,
)
from .helpers import _flatten_section_data

_SECTION_LOGFIRE = "logfire"


def _base_schema(user_input: dict[str, Any] | None = None) -> vol.Schema:
    """Return the workspace schema."""
    data = _flatten_section_data(user_input or {}, (_SECTION_LOGFIRE,))
    schema: VolDictType = {
        vol.Required(
            CONF_NAME,
            default=data.get(CONF_NAME, DEFAULT_WORKSPACE_NAME),
        ): TextSelector(TextSelectorConfig()),
    }
    schema[vol.Optional(_SECTION_LOGFIRE, default={})] = section(
        vol.Schema(
            {
                vol.Optional(CONF_LOGFIRE_TOKEN): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
                vol.Optional(
                    CONF_LOGFIRE_INCLUDE_CONTENT,
                    default=bool(data.get(CONF_LOGFIRE_INCLUDE_CONTENT, False)),
                ): BooleanSelector(),
            }
        ),
        {"collapsed": True},
    )
    return vol.Schema(schema)


def _provider_form_suggested_values(data: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return workspace form suggested values."""
    return dict(data or {})


def _normalise_workspace_data(user_input: Mapping[str, Any]) -> dict[str, Any]:
    """Return normalized workspace data for storage."""
    data = _flatten_section_data(user_input, (_SECTION_LOGFIRE,))
    token = data.get(CONF_LOGFIRE_TOKEN)
    if isinstance(token, str):
        token = token.strip()
    if token:
        data[CONF_LOGFIRE_TOKEN] = token
        data[CONF_LOGFIRE_INCLUDE_CONTENT] = bool(data.get(CONF_LOGFIRE_INCLUDE_CONTENT, False))
    else:
        data.pop(CONF_LOGFIRE_TOKEN, None)
        data.pop(CONF_LOGFIRE_INCLUDE_CONTENT, None)
    return data
