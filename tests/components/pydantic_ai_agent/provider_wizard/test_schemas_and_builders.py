"""Tests for provider wizard schemas and provider data builders."""

from typing import cast

import voluptuous as vol
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.const import (
    CONF_FAMILY,
    CONF_HIDE_DEPRECATED,
    CONF_HIDE_NON_TEXT_OUTPUT,
    CONF_HIDE_WITHOUT_STRUCTURED_OUTPUT,
    CONF_HIDE_WITHOUT_TOOL_CALL,
    CUSTOM_PROVIDER_ID,
    SECTION_ADVANCED_FILTERS,
)
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.filters import (
    ModelFilterOptions,
)
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.flow import (
    build_model_profiles,
    build_provider_data,
    selected_models_by_id,
)
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.schemas import (
    connection_schema,
    default_selected_model_ids,
    filters_from_user_input,
    needs_model_filter_step,
    provider_options,
)
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.types import (
    CatalogModelOption,
    CatalogProviderOption,
    CompactCatalog,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_BASE_URL,
    CONF_DISCOVERED,
    CONF_ENABLED,
    CONF_KEY_VALUE_JSON_VALUE,
    CONF_KEY_VALUE_KEY,
    CONF_KEY_VALUE_VALUE,
    CONF_MODEL,
    CONF_MODEL_PRICING,
    CONF_MODEL_PROFILES,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_METADATA,
    CONF_PROVIDER_MODE,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE_GEMINI,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
)
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.helpers.selector import ObjectSelector, TextSelector


def test_provider_options_include_supported_providers_and_custom() -> None:
    """Test provider options are sorted and include custom provider."""
    catalog = CompactCatalog(
        providers={
            "deepseek": _provider("deepseek", name="DeepSeek"),
            "openai": _provider("openai", name="OpenAI"),
        },
        models_by_provider={},
    )

    assert [option["value"] for option in provider_options(catalog)] == [
        "deepseek",
        "openai",
        CUSTOM_PROVIDER_ID,
    ]


def test_connection_schema_uses_password_api_key_selector() -> None:
    """Test guided setup does not expose API keys as plain text."""
    schema = connection_schema(
        _provider("openai"), PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS, {}
    )
    selector = cast(TextSelector, _selector_for_schema_key(schema, CONF_API_KEY))

    assert isinstance(selector, TextSelector)
    assert selector.config["type"] == "password"


def test_connection_schema_hides_extra_body_for_google() -> None:
    """Test guided Google setup does not show an unsupported field."""
    schema = connection_schema(_provider("google"), PROVIDER_GOOGLE_GEMINI, {})

    assert _schema_has_key(schema, "advanced_options")
    assert not _schema_has_key(schema, CONF_PROVIDER_EXTRA_BODY, nested=True)


def test_connection_schema_shows_extra_body_for_supported_modes() -> None:
    """Test guided setup shows extra body only for supported provider modes."""
    assert _schema_has_key(
        connection_schema(
            _provider("openai"), PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS, {}
        ),
        CONF_PROVIDER_EXTRA_BODY,
        nested=True,
    )
    assert _schema_has_key(
        connection_schema(
            _provider("openai"), PROVIDER_OPENAI_COMPATIBLE_RESPONSES, {}
        ),
        CONF_PROVIDER_EXTRA_BODY,
        nested=True,
    )
    assert _schema_has_key(
        connection_schema(_provider("anthropic"), PROVIDER_ANTHROPIC, {}),
        CONF_PROVIDER_EXTRA_BODY,
        nested=True,
    )


def test_connection_schema_uses_structured_row_selectors() -> None:
    schema = connection_schema(
        _provider("openai"),
        PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        {
            CONF_PROVIDER_HEADERS: {"Authorization": "Bearer token"},
            CONF_PROVIDER_EXTRA_BODY: {"service_tier": "flex"},
        },
    )

    header_selector = _nested_selector_for_schema_key(schema, CONF_PROVIDER_HEADERS)
    extra_body_selector = _nested_selector_for_schema_key(
        schema, CONF_PROVIDER_EXTRA_BODY
    )

    assert isinstance(header_selector, ObjectSelector)
    assert isinstance(extra_body_selector, ObjectSelector)
    assert _nested_default_for_schema_key(schema, CONF_PROVIDER_HEADERS) == [
        {CONF_KEY_VALUE_KEY: "Authorization", CONF_KEY_VALUE_VALUE: "Bearer token"}
    ]
    assert _nested_default_for_schema_key(schema, CONF_PROVIDER_EXTRA_BODY) == [
        {CONF_KEY_VALUE_KEY: "service_tier", CONF_KEY_VALUE_JSON_VALUE: '"flex"'}
    ]


def test_filters_from_user_input_parses_flags() -> None:
    """Test model filter form input parses to filter options."""
    filters = filters_from_user_input(
        {
            CONF_FAMILY: "gpt",
            SECTION_ADVANCED_FILTERS: {
                CONF_HIDE_WITHOUT_TOOL_CALL: False,
                CONF_HIDE_WITHOUT_STRUCTURED_OUTPUT: False,
                CONF_HIDE_DEPRECATED: False,
                CONF_HIDE_NON_TEXT_OUTPUT: False,
            },
        }
    )

    assert filters == ModelFilterOptions(
        hide_without_tool_call=False,
        hide_without_structured_output=False,
        hide_deprecated=False,
        hide_non_text_output=False,
        family="gpt",
    )


def test_model_filter_step_threshold_uses_default_filtered_models() -> None:
    """Test large eligible provider catalogs require a filter step."""
    models = tuple(_model(f"model-{index}") for index in range(100))

    assert needs_model_filter_step(models) is True
    assert needs_model_filter_step(models[:99]) is False


def test_default_selected_model_ids_only_auto_selects_single_model() -> None:
    """Test only one available model is auto-selected."""
    first = _model("first")
    second = _model("second")

    assert default_selected_model_ids((first,)) == ("first",)
    assert default_selected_model_ids((first, second)) == ()


def test_build_model_profiles_enables_selected_models() -> None:
    """Test guided selected models create enabled profiles."""
    profiles = build_model_profiles(
        (_model("gpt-4.1-mini", name="GPT 4.1 Mini"),),
        profile_id_factory=lambda: "p1",
    )

    assert profiles == {
        "p1": {
            "id": "p1",
            CONF_NAME: "GPT 4.1 Mini",
            CONF_MODEL: "gpt-4.1-mini",
            CONF_ENABLED: True,
            CONF_DISCOVERED: True,
        }
    }


def test_build_model_profiles_seeds_catalog_pricing() -> None:
    """Test guided selected models store catalog USD-per-million pricing."""
    profiles = build_model_profiles(
        (
            _model(
                "gpt-4.1-mini",
                input_price=0.4,
                output_price=1.6,
                cache_read_price=0.1,
            ),
        ),
        profile_id_factory=lambda: "p1",
    )

    assert profiles["p1"][CONF_MODEL_PRICING] == {
        "input": 0.4,
        "output": 1.6,
        "cache_read": 0.1,
    }


def test_build_provider_data_uses_runtime_provider_schema() -> None:
    """Test guided provider data matches the runtime provider subentry shape."""
    provider = _provider(
        "deepseek", name="DeepSeek", base_url="https://api.deepseek.com"
    )
    data = build_provider_data(
        provider,
        provider_mode=PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        api_key=" sk-test ",
        selected_models=(_model("deepseek-chat"),),
        provider_headers={"X-Test": "value"},
        profile_id_factory=lambda: "profile-1",
    )

    assert data[CONF_NAME] == "DeepSeek"
    assert data[CONF_PROVIDER_MODE] == PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS
    assert data[CONF_API_KEY] == "sk-test"
    assert data[CONF_BASE_URL] == "https://api.deepseek.com"
    assert data[CONF_PROVIDER_HEADERS] == {"X-Test": "value"}
    assert data[CONF_PROVIDER_METADATA] == {"catalog_provider_id": "deepseek"}
    profiles = cast(dict[str, dict[str, object]], data[CONF_MODEL_PROFILES])
    assert profiles["profile-1"][CONF_ENABLED] is True


def test_selected_models_by_id_preserves_catalog_order() -> None:
    """Test selected model IDs resolve in catalog order."""
    first = _model("first")
    second = _model("second")

    assert selected_models_by_id((first, second), ["second", "first"]) == (
        first,
        second,
    )
    assert selected_models_by_id((first, second), "first") == ()


def _provider(
    provider_id: str,
    *,
    name: str = "Provider",
    base_url: str | None = None,
    drivers: tuple[str, ...] = (PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,),
) -> CatalogProviderOption:
    """Return a catalog provider option fixture."""
    return CatalogProviderOption(
        id=provider_id,
        name=name,
        doc_url="https://models.dev/providers/provider",
        api_key_hints=("PROVIDER_API_KEY",),
        default_base_url=base_url,
        supported_drivers=drivers,
        model_count=1,
        families=("gpt",),
    )


def _selector_for_schema_key(schema: vol.Schema, key: str) -> object:
    """Return a selector for a voluptuous schema key."""
    for schema_key, selector in schema.schema.items():
        if getattr(schema_key, "schema", None) == key:
            return selector
    raise AssertionError(f"Schema key {key} not found")


def _nested_selector_for_schema_key(schema: vol.Schema, key: str) -> object:
    """Return a nested selector for a voluptuous schema key."""
    for selector_schema in _nested_schema_dicts(schema):
        for nested_key, nested_selector in selector_schema.items():
            if getattr(nested_key, "schema", None) == key:
                return nested_selector
    raise AssertionError(f"Nested schema key {key} not found")


def _nested_default_for_schema_key(schema: vol.Schema, key: str) -> object:
    """Return a nested field default from a voluptuous schema."""
    for selector_schema in _nested_schema_dicts(schema):
        for nested_key in selector_schema:
            if getattr(nested_key, "schema", None) == key:
                return nested_key.default()
    raise AssertionError(f"Nested schema key {key} not found")


def _nested_schema_dicts(schema: vol.Schema) -> list[dict[object, object]]:
    """Return nested schema dictionaries from section wrappers."""
    nested_schemas: list[dict[object, object]] = []
    for _schema_key, selector in schema.schema.items():
        selector_schema = getattr(selector, "schema", None)
        has_nested_schema = hasattr(selector_schema, "schema")
        if isinstance(selector_schema, vol.Schema) or has_nested_schema:
            selector_schema = selector_schema.schema
        if isinstance(selector_schema, dict):
            nested_schemas.append(selector_schema)
    return nested_schemas


def _schema_has_key(schema: vol.Schema, key: str, *, nested: bool = False) -> bool:
    """Return if a voluptuous schema contains a key."""
    if isinstance(schema, vol.Schema):
        schema = schema.schema
    if not isinstance(schema, dict):
        if nested and hasattr(schema, "schema"):
            return _schema_has_key(schema.schema, key, nested=nested)
        return False
    schema_dict = schema
    for schema_key, selector in schema_dict.items():
        if getattr(schema_key, "schema", None) == key:
            return True
        if nested and _schema_has_key(selector, key, nested=True):
            return True
    return False


def _model(
    model_id: str,
    *,
    name: str | None = None,
    reasoning: bool = False,
    attachment: bool = False,
    context_limit: int = 0,
    input_price: float | None = None,
    output_price: float | None = None,
    cache_read_price: float | None = None,
) -> CatalogModelOption:
    """Return a catalog model option fixture."""
    return CatalogModelOption(
        id=model_id,
        name=name or model_id.replace("-", " ").title(),
        provider_id="provider",
        family="gpt",
        tool_call=True,
        structured_output=True,
        reasoning=reasoning,
        attachment=attachment,
        text_output=True,
        context_limit=context_limit,
        output_limit=0,
        status=None,
        input_price=input_price,
        output_price=output_price,
        cache_read_price=cache_read_price,
    )
