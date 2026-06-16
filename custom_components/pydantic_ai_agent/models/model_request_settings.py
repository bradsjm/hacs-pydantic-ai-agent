"""Request-time model settings augmentation helpers."""

from typing import TYPE_CHECKING, Any, cast

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from pydantic_ai.settings import ModelSettings

from ..const import (
    CONF_TEMPLATED_EXTRA_BODY,
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
)
from .model_profiles import ModelProfile, provider_extra_body
from .templated_extra_body import merge_extra_body, render_templated_extra_body

if TYPE_CHECKING:
    from ..runtime.types import PydanticAIAgentConfigEntry


def _model_settings_with_templated_extra_body(
    hass: HomeAssistant, profile: ModelProfile, settings: ModelSettings
) -> ModelSettings:
    """Return request settings with rendered templated extra body merged."""
    rendered_extra_body = render_templated_extra_body(
        hass, profile.model_settings.get(CONF_TEMPLATED_EXTRA_BODY)
    )
    if not rendered_extra_body:
        return settings

    request_settings = dict(settings)
    request_settings["extra_body"] = merge_extra_body(
        request_settings.get("extra_body"), rendered_extra_body
    )
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
    request_settings = dict(settings)
    request_settings["extra_body"] = merge_extra_body(
        request_settings.get("extra_body"), extra_body
    )
    return ModelSettings(**cast(Any, request_settings))
