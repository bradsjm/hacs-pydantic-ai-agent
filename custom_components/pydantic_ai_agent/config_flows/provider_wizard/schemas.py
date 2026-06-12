"""Home Assistant schemas for the provider setup wizard."""

from collections.abc import Iterable

import voluptuous as vol
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.data_entry_flow import section
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
from homeassistant.helpers.typing import VolDictType

from ...const import (
    CONF_BASE_URL,
    CONF_KEY_VALUE_JSON_VALUE,
    CONF_KEY_VALUE_VALUE,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
)
from .._key_value_rows import _format_key_value_json_rows
from .._provider_data import _format_http_headers
from ..helpers import _key_value_rows_selector
from .const import (
    CATALOG_RETRY_PROVIDER_ID,
    CONF_DRIVER,
    CONF_FAMILY,
    CONF_HIDE_DEPRECATED,
    CONF_HIDE_NON_TEXT_OUTPUT,
    CONF_HIDE_WITHOUT_STRUCTURED_OUTPUT,
    CONF_HIDE_WITHOUT_TOOL_CALL,
    CONF_PROVIDER_ID,
    CONF_SELECTED_MODEL_IDS,
    CUSTOM_PROVIDER_ID,
    DEFAULT_MODEL_FILTER_THRESHOLD,
    MODE_LABELS,
    SECTION_ADVANCED_FILTERS,
)
from .filters import ModelFilterOptions, filtered_models
from .types import CatalogModelOption, CatalogProviderOption, CompactCatalog

_PROVIDER_EXTRA_BODY_MODES = {
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
}
_SECTION_ADVANCED_OPTIONS = "advanced_options"


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
            vol.Required(
                CONF_DRIVER, default=provider.supported_drivers[0]
            ): SelectSelector(
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
    data = _flatten_section_data(options, (_SECTION_ADVANCED_OPTIONS,))
    schema: VolDictType = {
        vol.Required(
            CONF_NAME, default=data.get(CONF_NAME, provider.name)
        ): TextSelector(TextSelectorConfig()),
        vol.Required(CONF_API_KEY, default=data.get(CONF_API_KEY, "")): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
        vol.Optional(
            CONF_BASE_URL,
            default=data.get(CONF_BASE_URL, provider.default_base_url or ""),
        ): TextSelector(TextSelectorConfig()),
    }
    advanced_schema: VolDictType = {
        vol.Optional(
            CONF_PROVIDER_HEADERS,
            default=_format_http_headers(data.get(CONF_PROVIDER_HEADERS)),
        ): _key_value_rows_selector(
            CONF_KEY_VALUE_VALUE,
            {"text": None},
            key_label="header name",
            value_label="header value",
            translation_key=CONF_PROVIDER_HEADERS,
        ),
    }
    if driver in _PROVIDER_EXTRA_BODY_MODES:
        advanced_schema[
            vol.Optional(
                CONF_PROVIDER_EXTRA_BODY,
                default=_format_key_value_json_rows(data.get(CONF_PROVIDER_EXTRA_BODY)),
            )
        ] = _key_value_rows_selector(
            CONF_KEY_VALUE_JSON_VALUE,
            {"template": None},
            key_label="parameter name",
            value_label="value",
            translation_key=CONF_PROVIDER_EXTRA_BODY,
        )
    schema[vol.Optional(_SECTION_ADVANCED_OPTIONS, default={})] = section(
        vol.Schema(advanced_schema), {"collapsed": True}
    )
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
                SelectSelectorConfig(
                    options=family_options, mode=SelectSelectorMode.DROPDOWN
                )
            ),
            vol.Optional(SECTION_ADVANCED_FILTERS, default={}): section(
                vol.Schema(
                    {
                        vol.Optional(
                            CONF_HIDE_WITHOUT_TOOL_CALL,
                            default=filters.hide_without_tool_call,
                        ): BooleanSelector(),
                        vol.Optional(
                            CONF_HIDE_WITHOUT_STRUCTURED_OUTPUT,
                            default=filters.hide_without_structured_output,
                        ): BooleanSelector(),
                        vol.Optional(
                            CONF_HIDE_DEPRECATED,
                            default=filters.hide_deprecated,
                        ): BooleanSelector(),
                        vol.Optional(
                            CONF_HIDE_NON_TEXT_OUTPUT,
                            default=filters.hide_non_text_output,
                        ): BooleanSelector(),
                    }
                ),
                {"collapsed": False},
            ),
        }
    )


def model_selection_schema(
    models: tuple[CatalogModelOption, ...],
    selected_model_ids: tuple[str, ...] = (),
    *,
    allow_custom_value: bool = False,
) -> vol.Schema:
    """Return a model multi-select schema."""
    default = list(selected_model_ids or default_selected_model_ids(models))
    schema: VolDictType = {
        vol.Required(CONF_SELECTED_MODEL_IDS, default=default): SelectSelector(
            SelectSelectorConfig(
                options=model_options(models),
                custom_value=allow_custom_value,
                mode=SelectSelectorMode.DROPDOWN,
                multiple=True,
            )
        )
    }
    return vol.Schema(schema)


def provider_options(
    catalog: CompactCatalog, *, include_retry: bool = False
) -> list[SelectOptionDict]:
    """Return provider selector options including custom setup."""
    providers = catalog.sorted_providers()
    duplicate_names = _duplicate_values(provider.name for provider in providers)
    options = []
    if include_retry:
        options.append(
            SelectOptionDict(
                label="Try loading catalog again", value=CATALOG_RETRY_PROVIDER_ID
            )
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
            label=_disambiguated_model_label(
                model, labels_by_id[model.id], duplicate_labels
            ),
            value=model.id,
        )
        for model in sorted_models
    ]


def filters_from_user_input(user_input: dict[str, object]) -> ModelFilterOptions:
    """Return model filters from user input."""
    data = _flatten_section_data(user_input, (SECTION_ADVANCED_FILTERS,))
    family = data.get(CONF_FAMILY)
    return ModelFilterOptions(
        hide_without_tool_call=bool(data.get(CONF_HIDE_WITHOUT_TOOL_CALL, True)),
        hide_without_structured_output=bool(
            data.get(CONF_HIDE_WITHOUT_STRUCTURED_OUTPUT, True)
        ),
        hide_deprecated=bool(data.get(CONF_HIDE_DEPRECATED, True)),
        hide_non_text_output=bool(data.get(CONF_HIDE_NON_TEXT_OUTPUT, True)),
        family=family if isinstance(family, str) and family else None,
    )


def needs_model_filter_step(
    models: tuple[CatalogModelOption, ...],
    threshold: int = DEFAULT_MODEL_FILTER_THRESHOLD,
) -> bool:
    """Return if model filtering should be shown before model selection."""
    return len(filtered_models(models)) >= threshold


def default_selected_model_ids(
    models: tuple[CatalogModelOption, ...],
) -> tuple[str, ...]:
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
        badges.append(f"{_format_context_limit(model.context_limit)} context")
    return f"{model.name} ({', '.join(badges)})" if badges else model.name


def _format_context_limit(context_limit: int) -> str:
    """Return a compact context limit label."""
    if context_limit < 1000:
        return str(context_limit)
    return f"{round(context_limit / 1000):,}K"


def _flatten_section_data(
    data: dict[str, object], section_keys: Iterable[str]
) -> dict[str, object]:
    """Return form data with HA section namespaces flattened."""
    flattened = dict(data)
    for key in section_keys:
        value = flattened.pop(key, None)
        if isinstance(value, dict):
            flattened.update(value)
        elif value is not None:
            flattened[key] = value
    return flattened


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
