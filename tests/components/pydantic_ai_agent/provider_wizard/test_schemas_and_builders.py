"""Tests for provider wizard schemas and provider data builders."""

from typing import cast

import voluptuous as vol
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.helpers.selector import TextSelector

from custom_components.pydantic_ai_agent.config_flows.provider_wizard.const import (
    CONF_FAMILY,
    CONF_SETUP_METHOD,
    CUSTOM_PROVIDER_ID,
    SETUP_METHOD_GUIDED,
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
    setup_method_schema,
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
    CONF_MODEL_PROFILES,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_MODE,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE_GEMINI,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
)


def test_setup_method_schema_defaults_to_guided() -> None:
    """Test setup method schema defaults to guided setup."""
    schema = setup_method_schema()

    assert schema({})[CONF_SETUP_METHOD] == SETUP_METHOD_GUIDED


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

    assert not _schema_has_key(schema, CONF_PROVIDER_EXTRA_BODY)


def test_connection_schema_shows_extra_body_for_supported_modes() -> None:
    """Test guided setup shows extra body only for supported provider modes."""
    assert _schema_has_key(
        connection_schema(_provider("openai"), PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS, {}),
        CONF_PROVIDER_EXTRA_BODY,
    )
    assert _schema_has_key(
        connection_schema(_provider("openai"), PROVIDER_OPENAI_COMPATIBLE_RESPONSES, {}),
        CONF_PROVIDER_EXTRA_BODY,
    )
    assert _schema_has_key(
        connection_schema(_provider("anthropic"), PROVIDER_ANTHROPIC, {}),
        CONF_PROVIDER_EXTRA_BODY,
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
            "label": "GPT 4.1 Mini (reasoning, attachments, 128,000 context)",
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
            "label": "OpenAI GPT-4.1 Mini (1,047,576 context) - gpt-4.1-mini",
            "value": "gpt-4.1-mini",
        },
        {
            "label": "OpenAI GPT-4.1 Mini (1,047,576 context) - gpt-4.1-mini-2025-04-14",
            "value": "gpt-4.1-mini-2025-04-14",
        },
    ]


def test_filters_from_user_input_parses_flags() -> None:
    """Test model filter form input parses to filter options."""
    filters = filters_from_user_input(
        {
            CONF_FAMILY: "gpt",
            "include_without_tool_call": True,
            "include_without_structured_output": True,
            "include_deprecated": True,
            "include_non_text_output": True,
        }
    )

    assert filters == ModelFilterOptions(
        include_without_tool_call=True,
        include_without_structured_output=True,
        include_deprecated=True,
        include_non_text_output=True,
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


def test_build_provider_data_uses_runtime_provider_schema() -> None:
    """Test guided provider data matches the runtime provider subentry shape."""
    provider = _provider("deepseek", name="DeepSeek", base_url="https://api.deepseek.com")
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


def _schema_has_key(schema: vol.Schema, key: str) -> bool:
    """Return if a voluptuous schema contains a key."""
    return any(getattr(schema_key, "schema", None) == key for schema_key in schema.schema)


def _model(
    model_id: str,
    *,
    name: str | None = None,
    reasoning: bool = False,
    attachment: bool = False,
    context_limit: int = 0,
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
    )
