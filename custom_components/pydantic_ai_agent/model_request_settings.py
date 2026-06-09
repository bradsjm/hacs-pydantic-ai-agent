"""Request-time model settings augmentation helpers."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from pydantic_ai.settings import ModelSettings

from .chat_template_kwargs import (
    reject_chat_template_kwargs_in_extra_body,
    render_chat_template_kwargs,
)
from .const import (
    CONF_CHAT_TEMPLATE_KWARGS,
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
)
from .model_profiles import ModelProfile, provider_extra_body

if TYPE_CHECKING:
    from . import PydanticAIAgentConfigEntry


def _model_settings_with_chat_template_kwargs(
    hass: HomeAssistant, profile: ModelProfile, settings: ModelSettings
) -> ModelSettings:
    """Return request settings with rendered chat-template kwargs injected."""
    rendered_kwargs = render_chat_template_kwargs(
        hass, profile.model_settings.get(CONF_CHAT_TEMPLATE_KWARGS)
    )
    if not rendered_kwargs:
        reject_chat_template_kwargs_in_extra_body(settings.get("extra_body"))
        return settings

    request_settings = dict(settings)
    extra_body = request_settings.get("extra_body")
    reject_chat_template_kwargs_in_extra_body(extra_body)
    request_extra_body = dict(extra_body) if isinstance(extra_body, Mapping) else {}
    request_extra_body[CONF_CHAT_TEMPLATE_KWARGS] = rendered_kwargs
    request_settings["extra_body"] = request_extra_body
    return ModelSettings(**cast(Any, request_settings))


def _model_settings_with_provider_extra_body(
    entry: PydanticAIAgentConfigEntry, profile: ModelProfile, settings: ModelSettings
) -> ModelSettings:
    """Return request settings with provider-level extra body merged."""
    extra_body = provider_extra_body(entry, profile)
    if not extra_body:
        return settings
    if profile.provider_mode not in {
        PROVIDER_ANTHROPIC,
        PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
    }:
        raise HomeAssistantError(
            "Provider extra body is only supported by OpenAI-compatible"
            " and Anthropic provider modes"
        )
    reject_chat_template_kwargs_in_extra_body(extra_body)
    request_settings = dict(settings)
    request_settings["extra_body"] = extra_body
    return ModelSettings(**cast(Any, request_settings))
