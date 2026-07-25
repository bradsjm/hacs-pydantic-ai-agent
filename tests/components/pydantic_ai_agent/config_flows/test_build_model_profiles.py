"""Tests for provider wizard model-profile builders."""

from custom_components.pydantic_ai_agent.config_flows.provider_wizard.flow import (
    build_model_profiles,
)
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.types import (
    CatalogModelOption,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_ENABLED,
    CONF_MODEL,
    CONF_SUPPORTS_IMAGES,
)


def _catalog_model(*, model_id: str, input_modalities: tuple[str, ...]) -> CatalogModelOption:
    """Return a minimal catalog model option for the given modalities."""
    return CatalogModelOption(
        id=model_id,
        name=model_id,
        provider_id="provider-1",
        family=None,
        tool_call=True,
        structured_output=None,
        reasoning=False,
        attachment="image" in input_modalities,
        input_modalities=input_modalities,
        text_output=True,
        context_limit=0,
        output_limit=0,
        status=None,
    )


def test_build_model_profiles_persists_supports_images_from_modalities() -> None:
    """Image support is derived from input modalities for each profile."""
    profiles = build_model_profiles(
        (
            _catalog_model(model_id="vision", input_modalities=("text", "image")),
            _catalog_model(model_id="text-only", input_modalities=("text",)),
        ),
        profile_id_factory=iter(("vision-id", "text-id")).__next__,
    )

    assert profiles["vision-id"][CONF_MODEL] == "vision"
    assert profiles["vision-id"][CONF_ENABLED] is True
    assert profiles["text-id"][CONF_MODEL] == "text-only"
    assert profiles["text-id"][CONF_ENABLED] is True
    assert profiles["vision-id"][CONF_SUPPORTS_IMAGES] is True
    assert profiles["text-id"][CONF_SUPPORTS_IMAGES] is False
