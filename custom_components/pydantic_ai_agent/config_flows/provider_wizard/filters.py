"""Model filters for the provider setup wizard."""

from dataclasses import dataclass

from .types import CatalogModelOption


@dataclass(frozen=True, slots=True)
class ModelFilterOptions:
    """Model filter options for catalog model selection."""

    hide_without_tool_call: bool = True
    hide_without_structured_output: bool = True
    hide_deprecated: bool = True
    hide_non_text_output: bool = True
    family: str | None = None


def model_matches_filters(
    model: CatalogModelOption, filters: ModelFilterOptions | None = None
) -> bool:
    """Return if a model should be shown for the selected filters."""
    filters = filters or ModelFilterOptions()
    if filters.family and model.family != filters.family:
        return False
    if filters.hide_non_text_output and not model.text_output:
        return False
    if filters.hide_deprecated and model.status == "deprecated":
        return False
    if filters.hide_without_tool_call and not model.tool_call:
        return False
    if filters.hide_without_structured_output and model.structured_output is False:
        return False
    return True


def filtered_models(
    models: tuple[CatalogModelOption, ...], filters: ModelFilterOptions | None = None
) -> tuple[CatalogModelOption, ...]:
    """Return models matching filters in display order."""
    return tuple(
        sorted(
            (model for model in models if model_matches_filters(model, filters)),
            key=lambda model: (model.name.casefold(), model.id.casefold()),
        )
    )
