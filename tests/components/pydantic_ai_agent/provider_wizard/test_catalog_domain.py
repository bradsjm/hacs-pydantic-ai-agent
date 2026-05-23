"""Tests for provider wizard catalog mapping, filtering, and normalization."""

import pytest

from custom_components.pydantic_ai_agent.config_flows.provider_wizard.filters import (
    ModelFilterOptions,
    filtered_models,
)
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.mapping import (
    supported_drivers_for_provider,
)
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.normalize import (
    normalize_catalog,
)
from custom_components.pydantic_ai_agent.config_flows.provider_wizard.types import (
    CatalogModelOption,
)
from custom_components.pydantic_ai_agent.const import (
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE_GEMINI,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
)


@pytest.mark.parametrize(
    ("provider_id", "npm", "expected"),
    [
        ("anthropic", "@ai-sdk/anthropic", (PROVIDER_ANTHROPIC,)),
        ("google", "@ai-sdk/google", (PROVIDER_GOOGLE_GEMINI,)),
        (
            "openai",
            "@ai-sdk/openai",
            (
                PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
                PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
            ),
        ),
        (
            "deepseek",
            "@ai-sdk/openai-compatible",
            (PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,),
        ),
        (
            "openrouter",
            "@openrouter/ai-sdk-provider",
            (PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,),
        ),
        ("unsupported", "@ai-sdk/unsupported", ()),
    ],
)
def test_supported_drivers_for_provider(
    provider_id: str, npm: str, expected: tuple[str, ...]
) -> None:
    """Test catalog providers map to supported integration modes."""
    assert supported_drivers_for_provider(provider_id, npm) == expected


def test_filtered_models_default_hides_non_recommended_models() -> None:
    """Test default filters hide known unsupported models."""
    models = (
        _model("good", name="Good"),
        _model("image", text_output=False),
        _model("deprecated", status="deprecated"),
        _model("no-tools", tool_call=False),
        _model("no-structured", structured_output=False),
        _model("unknown-structured", structured_output=None),
    )

    assert [model.id for model in filtered_models(models)] == [
        "good",
        "unknown-structured",
    ]


def test_filtered_models_advanced_flags_include_additional_models() -> None:
    """Test advanced filters can include models hidden by defaults."""
    models = (
        _model("good", name="Good"),
        _model("deprecated", status="deprecated"),
        _model("no-tools", tool_call=False),
        _model("no-structured", structured_output=False),
        _model("image", text_output=False),
    )

    filtered = filtered_models(
        models,
        ModelFilterOptions(
            include_without_tool_call=True,
            include_without_structured_output=True,
            include_deprecated=True,
            include_non_text_output=True,
        ),
    )

    assert {model.id for model in filtered} == {
        "deprecated",
        "good",
        "image",
        "no-structured",
        "no-tools",
    }


def test_filtered_models_can_filter_by_family() -> None:
    """Test family filters narrow model choices."""
    models = (_model("claude", family="claude"), _model("gpt", family="gpt"))

    assert [model.id for model in filtered_models(models, ModelFilterOptions(family="gpt"))] == [
        "gpt"
    ]


def test_normalize_catalog_keeps_supported_compact_data() -> None:
    """Test models.dev payloads normalize to compact supported options."""
    catalog = normalize_catalog(
        {
            "openai": {
                "id": "openai",
                "name": "OpenAI",
                "npm": "@ai-sdk/openai",
                "env": ["OPENAI_API_KEY"],
                "api": None,
                "doc": "https://models.dev/providers/openai",
                "models": {
                    "gpt-4.1-mini": {
                        "id": "gpt-4.1-mini",
                        "name": "GPT 4.1 Mini",
                        "family": "gpt-4.1",
                        "tool_call": True,
                        "structured_output": True,
                        "reasoning": False,
                        "attachment": True,
                        "modalities": {"output": ["text"]},
                        "limit": {"context": 128000, "output": 16384},
                    }
                },
            },
            "unsupported": {
                "id": "unsupported",
                "name": "Unsupported",
                "npm": "@ai-sdk/unsupported",
                "env": [],
                "api": "https://unsupported.example.com/v1",
                "doc": "https://example.com",
                "models": {
                    "model": {
                        "name": "Model",
                        "tool_call": True,
                        "structured_output": True,
                        "modalities": {"output": ["text"]},
                    }
                },
            },
        }
    )

    assert set(catalog.providers) == {"openai"}
    provider = catalog.providers["openai"]
    assert provider.name == "OpenAI"
    assert provider.api_key_hints == ("OPENAI_API_KEY",)
    assert provider.default_base_url is None
    assert provider.supported_drivers == (
        PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
    )
    assert provider.families == ("gpt-4.1",)
    assert provider.model_count == 1

    model = catalog.models_for_provider("openai")[0]
    assert model.id == "gpt-4.1-mini"
    assert model.name == "GPT 4.1 Mini"
    assert model.text_output is True
    assert model.context_limit == 128000
    assert model.output_limit == 16384


def test_normalize_catalog_uses_openai_compatible_base_url() -> None:
    """Test compatible gateway catalog API URLs are compacted."""
    catalog = normalize_catalog(
        {
            "deepseek": {
                "id": "deepseek",
                "name": "DeepSeek",
                "npm": "@ai-sdk/openai-compatible",
                "env": ["DEEPSEEK_API_KEY"],
                "api": "https://api.deepseek.com/",
                "doc": "https://models.dev/providers/deepseek",
                "models": {
                    "deepseek-chat": {
                        "name": "DeepSeek Chat",
                        "family": "deepseek",
                        "tool_call": True,
                        "structured_output": True,
                        "reasoning": False,
                        "attachment": False,
                        "modalities": {"output": ["text"]},
                        "limit": {"context": 64000, "output": 8000},
                    }
                },
            }
        }
    )

    assert catalog.providers["deepseek"].default_base_url == "https://api.deepseek.com"


def _model(
    model_id: str,
    *,
    name: str | None = None,
    family: str | None = None,
    text_output: bool = True,
    tool_call: bool = True,
    structured_output: bool | None = True,
    status: str | None = None,
) -> CatalogModelOption:
    """Return a compact model option for filtering tests."""
    return CatalogModelOption(
        id=model_id,
        name=name or model_id,
        provider_id="provider",
        family=family,
        tool_call=tool_call,
        structured_output=structured_output,
        reasoning=False,
        attachment=False,
        text_output=text_output,
        context_limit=0,
        output_limit=0,
        status=status,
    )
