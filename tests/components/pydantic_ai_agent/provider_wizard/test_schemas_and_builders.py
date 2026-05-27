"""Tests for provider wizard schemas and provider data builders."""

from typing import cast

import voluptuous as vol
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.helpers.selector import TextSelector

from custom_components.pydantic_ai_agent.config_flows.provider_wizard.const import (
    CONF_FAMILY,
    CONF_HIDE_DEPRECATED,
    CONF_HIDE_NON_TEXT_OUTPUT,
    CONF_HIDE_WITHOUT_STRUCTURED_OUTPUT,
    CONF_HIDE_WITHOUT_TOOL_CALL,
    SECTION_ADVANCED_FILTERS,
    CUSTOM_PROVIDER_ID,
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
    driver_options,
    filters_from_user_input,
    model_options,
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


def test_provider_options_include_supported_providers_and_custom() -> None:
    """Test provider options are sorted and include custom provider."""
    catalog = CompactCatalog(
        providers={
            "deepseek": _provider("deepseek", name="DeepSeek"),
            "openai": _provider("openai", name="OpenAI"),
        },
        models_by_provider={},
    )

    assert provider_options(catalog) == [
        {"label": "DeepSeek", "value": "deepseek"},
        {"label": "OpenAI", "value": "openai"},
        {"label": "Custom provider", "value": CUSTOM_PROVIDER_ID},
    ]


def test_provider_options_disambiguate_duplicate_names() -> None:
    """Test duplicate provider names include stable provider IDs."""
    catalog = CompactCatalog(
        providers={
            "stepfun": _provider("stepfun", name="StepFun"),
            "stepfun-ai": _provider("stepfun-ai", name="StepFun"),
        },
        models_by_provider={},
    )

    assert provider_options(catalog)[:2] == [
        {"label": "StepFun (stepfun)", "value": "stepfun"},
        {"label": "StepFun (stepfun-ai)", "value": "stepfun-ai"},
    ]


def test_driver_options_use_user_facing_labels() -> None:
    """Test driver options hide internal SDK details."""
    provider = _provider(
        "openai",
        drivers=(
            PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
        ),
    )

    assert driver_options(provider) == [
        {"label": "Chat Completions", "value": PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS},
        {"label": "Responses", "value": PROVIDER_OPENAI_COMPATIBLE_RESPONSES},
    ]


def test_connection_schema_uses_password_api_key_selector() -> None:
    """Test guided setup does not expose API keys as plain text."""
    schema = connection_schema(
        _provider("openai"), PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS, {}
    )
    selector = _selector_for_schema_key(schema, CONF_API_KEY)

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


def test_model_options_include_capability_badges() -> None:
    """Test model labels include useful compact hints."""
    model = _model(
        "gpt-4.1-mini",
        name="GPT 4.1 Mini",
        reasoning=True,
        attachment=True,
        context_limit=128000,
    )

    assert model_options((model,)) == [
        {
            "label": "GPT 4.1 Mini (reasoning, attachments, 128K context)",
            "value": "gpt-4.1-mini",
        }
    ]


def test_model_options_disambiguate_duplicate_labels() -> None:
    """Test duplicate model labels include stable model IDs."""
    models = (
        _model("gpt-4.1-mini", name="OpenAI GPT-4.1 Mini", context_limit=1047576),
        _model(
            "gpt-4.1-mini-2025-04-14",
            name="OpenAI GPT-4.1 Mini",
            context_limit=1047576,
        ),
    )

    assert model_options(models) == [
        {
            "label": "OpenAI GPT-4.1 Mini (1,048K context) - gpt-4.1-mini",
            "value": "gpt-4.1-mini",
        },
        {
            "label": "OpenAI GPT-4.1 Mini (1,048K context) - gpt-4.1-mini-2025-04-14",
            "value": "gpt-4.1-mini-2025-04-14",
        },
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
