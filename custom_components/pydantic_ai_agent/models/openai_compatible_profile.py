"""Persisted OpenAI-compatible model profile helpers."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.settings import ThinkingLevel

from ..const import (
    CONF_STRUCTURED_OUTPUT_SUPPORT,
    CONF_SUPPORTS_TOOLS,
    CONF_THINKING_SUPPORT,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
)

type StructuredOutputSupportMode = Literal["none", "json_object", "json_schema"]
OPENAI_COMPATIBLE_STRUCTURED_OUTPUT_SUPPORT_OPTIONS: tuple[StructuredOutputSupportMode, ...] = (
    "none",
    "json_object",
    "json_schema",
)


@dataclass(frozen=True, slots=True)
class PersistedOpenAICompatibleProfile:
    """Persisted capability settings for one OpenAI-compatible model profile."""

    thinking_support: bool
    structured_output_support: StructuredOutputSupportMode
    supports_tools: bool

    @classmethod
    def from_mapping(cls, profile_data: Mapping[str, Any]) -> PersistedOpenAICompatibleProfile:
        """Return parsed persisted OpenAI-compatible capability settings."""
        thinking_support = profile_data[CONF_THINKING_SUPPORT]
        if not isinstance(thinking_support, bool):
            raise ValueError(f"Invalid {CONF_THINKING_SUPPORT!r} value")

        structured_output_support = profile_data[CONF_STRUCTURED_OUTPUT_SUPPORT]
        if structured_output_support not in OPENAI_COMPATIBLE_STRUCTURED_OUTPUT_SUPPORT_OPTIONS:
            raise ValueError(f"Invalid {CONF_STRUCTURED_OUTPUT_SUPPORT!r} value")

        supports_tools = profile_data[CONF_SUPPORTS_TOOLS]
        if not isinstance(supports_tools, bool):
            raise ValueError(f"Invalid {CONF_SUPPORTS_TOOLS!r} value")

        return cls(
            thinking_support=thinking_support,
            structured_output_support=structured_output_support,
            supports_tools=supports_tools,
        )

    def supports_thinking(self) -> bool:
        """Return if the persisted profile supports configurable thinking."""
        return self.thinking_support

    def can_disable_thinking(self) -> bool:
        """Return if persisted thinking support can be disabled."""
        return self.thinking_support

    def effective_thinking_setting(self, thinking: object) -> ThinkingLevel | None:
        """Return thinking only when supported by this persisted profile."""
        if thinking in (None, "none", False) or not self.supports_thinking():
            return None
        return thinking  # type: ignore[return-value]

    def as_model_profile(self) -> OpenAIModelProfile:
        """Return an OpenAIModelProfile synthesized from persisted settings."""
        supports_json_object_output = self.structured_output_support in {
            "json_object",
            "json_schema",
        }
        return OpenAIModelProfile(
            supports_thinking=self.supports_thinking(),
            thinking_always_enabled=False,
            supports_tools=self.supports_tools,
            supports_json_schema_output=self.structured_output_support == "json_schema",
            supports_json_object_output=supports_json_object_output,
            supported_native_tools=frozenset(),
            openai_supports_strict_tool_definition=True,
            openai_system_prompt_role="system",
        )


def is_openai_compatible_provider_mode(provider_mode: str) -> bool:
    """Return if the provider mode uses the OpenAI-compatible adapter."""
    return provider_mode in {
        PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
    }


def default_openai_compatible_profile_data() -> dict[str, object]:
    """Return default persisted capability values for custom/discovered models."""
    return {
        CONF_THINKING_SUPPORT: False,
        CONF_STRUCTURED_OUTPUT_SUPPORT: "none",
        CONF_SUPPORTS_TOOLS: True,
    }
