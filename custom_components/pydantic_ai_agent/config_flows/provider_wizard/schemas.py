"""Home Assistant schemas for the provider setup wizard."""

import voluptuous as vol
from homeassistant.helpers.selector import (
    BooleanSelector,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)

from .const import (
    CONF_DRIVER,
    CONF_FAMILY,
    CONF_INCLUDE_DEPRECATED,
    CONF_INCLUDE_NON_TEXT_OUTPUT,
    CONF_INCLUDE_WITHOUT_STRUCTURED_OUTPUT,
    CONF_INCLUDE_WITHOUT_TOOL_CALL,
    CONF_PROVIDER_ID,
    CONF_SELECTED_MODEL_IDS,
    CONF_SETUP_METHOD,
    CUSTOM_PROVIDER_ID,
    DEFAULT_MODEL_FILTER_THRESHOLD,
    MODE_LABELS,
    SETUP_METHOD_CUSTOM,
    SETUP_METHOD_GUIDED,
)
from .filters import ModelFilterOptions, filtered_models
from .types import CatalogModelOption, CatalogProviderOption, CompactCatalog


def setup_method_schema() -> vol.Schema:
    """Return the guided/custom setup method schema."""
    return vol.Schema(
        {
            vol.Required(CONF_SETUP_METHOD, default=SETUP_METHOD_GUIDED): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(label="Guided setup", value=SETUP_METHOD_GUIDED),
                        SelectOptionDict(label="Custom provider", value=SETUP_METHOD_CUSTOM),
                    ],
                    mode=SelectSelectorMode.LIST,
                )
            )
        }
    )


def provider_selection_schema(catalog: CompactCatalog) -> vol.Schema:
    """Return a provider selection schema."""
    return vol.Schema(
        {
            vol.Required(CONF_PROVIDER_ID): SelectSelector(
                SelectSelectorConfig(
                    options=provider_options(catalog),
                    mode=SelectSelectorMode.DROPDOWN,
                )
            )
        }
    )


def driver_selection_schema(provider: CatalogProviderOption) -> vol.Schema:
    """Return an API mode selection schema."""
    return vol.Schema(
        {
            vol.Required(CONF_DRIVER, default=provider.supported_drivers[0]): SelectSelector(
                SelectSelectorConfig(
                    options=driver_options(provider),
                    mode=SelectSelectorMode.LIST,
                )
            )
        }
    )


def model_filter_schema(
    provider: CatalogProviderOption,
    filters: ModelFilterOptions | None = None,
) -> vol.Schema:
    """Return model filter controls for a large provider."""
    filters = filters or ModelFilterOptions()
    family_options = [SelectOptionDict(label="All families", value="")]
    family_options.extend(
        SelectOptionDict(label=family, value=family) for family in provider.families
    )
    return vol.Schema(
        {
            vol.Optional(CONF_FAMILY, default=filters.family or ""): SelectSelector(
                SelectSelectorConfig(options=family_options, mode=SelectSelectorMode.DROPDOWN)
            ),
            vol.Optional(
                CONF_INCLUDE_WITHOUT_TOOL_CALL,
                default=filters.include_without_tool_call,
            ): BooleanSelector(),
            vol.Optional(
                CONF_INCLUDE_WITHOUT_STRUCTURED_OUTPUT,
                default=filters.include_without_structured_output,
            ): BooleanSelector(),
            vol.Optional(
                CONF_INCLUDE_DEPRECATED,
                default=filters.include_deprecated,
            ): BooleanSelector(),
            vol.Optional(
                CONF_INCLUDE_NON_TEXT_OUTPUT,
                default=filters.include_non_text_output,
            ): BooleanSelector(),
        }
    )


def model_selection_schema(
    models: tuple[CatalogModelOption, ...],
    selected_model_ids: tuple[str, ...] = (),
) -> vol.Schema:
    """Return a model multi-select schema."""
    default = list(selected_model_ids or default_selected_model_ids(models))
    return vol.Schema(
        {
            vol.Required(CONF_SELECTED_MODEL_IDS, default=default): SelectSelector(
                SelectSelectorConfig(
                    options=model_options(models),
                    mode=SelectSelectorMode.DROPDOWN,
                    multiple=True,
                )
            )
        }
    )


def provider_options(catalog: CompactCatalog) -> list[SelectOptionDict]:
    """Return provider selector options including custom setup."""
    options = [
        SelectOptionDict(label=provider.name, value=provider.id)
        for provider in catalog.sorted_providers()
    ]
    options.append(SelectOptionDict(label="Custom provider", value=CUSTOM_PROVIDER_ID))
    return options


def driver_options(provider: CatalogProviderOption) -> list[SelectOptionDict]:
    """Return API mode selector options for a provider."""
    return [
        SelectOptionDict(label=MODE_LABELS[driver], value=driver)
        for driver in provider.supported_drivers
    ]


def model_options(models: tuple[CatalogModelOption, ...]) -> list[SelectOptionDict]:
    """Return model selector options."""
    return [
        SelectOptionDict(label=_model_label(model), value=model.id)
        for model in sorted(models, key=lambda model: (model.name.casefold(), model.id))
    ]


def filters_from_user_input(user_input: dict[str, object]) -> ModelFilterOptions:
    """Return model filters from user input."""
    family = user_input.get(CONF_FAMILY)
    return ModelFilterOptions(
        include_without_tool_call=bool(user_input.get(CONF_INCLUDE_WITHOUT_TOOL_CALL)),
        include_without_structured_output=bool(
            user_input.get(CONF_INCLUDE_WITHOUT_STRUCTURED_OUTPUT)
        ),
        include_deprecated=bool(user_input.get(CONF_INCLUDE_DEPRECATED)),
        include_non_text_output=bool(user_input.get(CONF_INCLUDE_NON_TEXT_OUTPUT)),
        family=family if isinstance(family, str) and family else None,
    )


def needs_model_filter_step(
    models: tuple[CatalogModelOption, ...], threshold: int = DEFAULT_MODEL_FILTER_THRESHOLD
) -> bool:
    """Return if model filtering should be shown before model selection."""
    return len(filtered_models(models)) >= threshold


def default_selected_model_ids(models: tuple[CatalogModelOption, ...]) -> tuple[str, ...]:
    """Return model IDs that can be auto-selected."""
    return (models[0].id,) if len(models) == 1 else ()


def _model_label(model: CatalogModelOption) -> str:
    """Return a compact model label with useful capability hints."""
    badges: list[str] = []
    if model.reasoning:
        badges.append("reasoning")
    if model.attachment:
        badges.append("attachments")
    if model.context_limit:
        badges.append(f"{model.context_limit:,} context")
    return f"{model.name} ({', '.join(badges)})" if badges else model.name
