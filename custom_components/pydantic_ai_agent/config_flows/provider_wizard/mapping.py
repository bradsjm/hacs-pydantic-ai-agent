"""Provider catalog to integration provider-mode mapping."""

from ...const import (
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE_GEMINI,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
)
from .const import OPENAI_COMPATIBLE_NPM, OPENROUTER_NPM


def supported_drivers_for_provider(provider_id: str, npm: object) -> tuple[str, ...]:
    """Return supported integration provider modes for one catalog provider."""
    provider_id = provider_id.strip().lower()
    if provider_id == "anthropic":
        return (PROVIDER_ANTHROPIC,)
    if provider_id == "google":
        return (PROVIDER_GOOGLE_GEMINI,)
    if provider_id == "openai":
        return (
            PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
        )
    if npm in {OPENAI_COMPATIBLE_NPM, OPENROUTER_NPM}:
        return (PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,)
    return ()
