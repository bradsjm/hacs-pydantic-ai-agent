"""Home Assistant schemas for the provider setup wizard."""

from collections.abc import Iterable

import voluptuous as vol
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.helpers.selector import (
    BooleanSelector,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from ...const import (
    CONF_BASE_URL,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
)
from .const import (
    CATALOG_RETRY_PROVIDER_ID,
    CONF_DRIVER,
    CONF_FAMILY,
    CONF_INCLUDE_DEPRECATED,
    CONF_INCLUDE_NON_TEXT_OUTPUT,
    CONF_INCLUDE_WITHOUT_STRUCTURED_OUTPUT,
    CONF_INCLUDE_WITHOUT_TOOL_CALL,
    CONF_PROVIDER_ID,
    CONF_SELECTED_MODEL_IDS,
    CUSTOM_PROVIDER_ID,
    DEFAULT_MODEL_FILTER_THRESHOLD,
    MODE_LABELS,
)
from .filters import ModelFilterOptions, filtered_models
from .types import CatalogModelOption, CatalogProviderOption, CompactCatalog


_PROVIDER_EXTRA_BODY_MODES = {
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
}


def provider_selection_schema(
    catalog: CompactCatalog, *, include_retry: bool = False
) -> vol.Schema:
    """Return a provider selection schema."""
    return vol.Schema(
        {
            vol.Required(CONF_PROVIDER_ID): SelectSelector(
                SelectSelectorConfig(
                    options=provider_options(catalog, include_retry=include_retry),
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


def connection_schema(
    provider: CatalogProviderOption, driver: str, options: dict[str, object]
) -> vol.Schema:
    """Return guided provider connection details schema."""
    schema = {
        vol.Required(CONF_NAME, default=options.get(CONF_NAME, provider.name)): TextSelector(
            TextSelectorConfig()
        ),
        vol.Required(CONF_API_KEY, default=options.get(CONF_API_KEY, "")): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
        vol.Optional(
            CONF_BASE_URL,
            default=options.get(CONF_BASE_URL, provider.default_base_url or ""),
        ): TextSelector(TextSelectorConfig()),
        vol.Optional(
            CONF_PROVIDER_HEADERS,
            default=options.get(CONF_PROVIDER_HEADERS, ""),
        ): TextSelector(TextSelectorConfig(multiline=True)),
    }
    if driver in _PROVIDER_EXTRA_BODY_MODES:
        schema[
            vol.Optional(
                CONF_PROVIDER_EXTRA_BODY,
                default=options.get(CONF_PROVIDER_EXTRA_BODY, ""),
            )
        ] = TextSelector(TextSelectorConfig(multiline=True))
    return vol.Schema(schema)


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


def provider_options(
    catalog: CompactCatalog, *, include_retry: bool = False
) -> list[SelectOptionDict]:
    """Return provider selector options including custom setup."""
    providers = catalog.sorted_providers()
    duplicate_names = _duplicate_values(provider.name for provider in providers)
    options = []
    if include_retry:
        options.append(
            SelectOptionDict(label="Try loading catalog again", value=CATALOG_RETRY_PROVIDER_ID)
        )
    for provider in providers:
        label = provider.name
        if label in duplicate_names:
            label = f"{label} ({provider.id})"
        options.append(SelectOptionDict(label=label, value=provider.id))
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
    sorted_models = sorted(models, key=lambda model: (model.name.casefold(), model.id))
    labels_by_id = {model.id: _model_label(model) for model in sorted_models}
    duplicate_labels = _duplicate_values(labels_by_id.values())
    return [
        SelectOptionDict(
            label=_disambiguated_model_label(model, labels_by_id[model.id], duplicate_labels),
            value=model.id,
        )
        for model in sorted_models
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


def _disambiguated_model_label(
    model: CatalogModelOption, label: str, duplicate_labels: set[str]
) -> str:
    """Return a model label with a stable ID hint when needed."""
    if label not in duplicate_labels:
        return label
    return f"{label} - {model.id}"


def _duplicate_values(values: Iterable[str]) -> set[str]:
    """Return values that occur more than once."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)
    return duplicates
