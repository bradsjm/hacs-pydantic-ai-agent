"""Templated chat-template keyword argument helpers."""

from collections.abc import Mapping
import json
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers.template import Template

from .const import (
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
    CONF_CHAT_TEMPLATE_KWARGS,
)


def render_chat_template_kwargs(
    hass: HomeAssistant, configured: object
) -> dict[str, Any]:
    """Render configured chat-template keyword arguments for one model request."""
    if configured in (None, ""):
        return {}
    if not isinstance(configured, list):
        raise HomeAssistantError("Configured chat template arguments are invalid")

    rendered: dict[str, Any] = {}
    for item in configured:
        if not isinstance(item, Mapping):
            raise HomeAssistantError("Configured chat template arguments are invalid")
        key = str(item.get(CONF_CHAT_TEMPLATE_KWARG_KEY, "")).strip()
        value_template = item.get(CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE)
        if not key or not isinstance(value_template, str):
            raise HomeAssistantError("Configured chat template arguments are invalid")
        try:
            value = Template(value_template, hass).async_render(parse_result=True)
            json.dumps(value)
        except (TemplateError, TypeError, ValueError) as err:
            raise HomeAssistantError(
                f'Failed to render chat template argument "{key}"'
            ) from err
        rendered[key] = value
    return rendered


def reject_chat_template_kwargs_in_extra_body(extra_body: object) -> None:
    """Reject the legacy/raw placement of chat_template_kwargs in extra_body."""
    if isinstance(extra_body, Mapping) and CONF_CHAT_TEMPLATE_KWARGS in extra_body:
        raise HomeAssistantError(
            "chat_template_kwargs must use the dedicated model setting"
        )
