"""Tests for templated extra-body path and merge helpers."""

from custom_components.pydantic_ai_agent.const import (
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
)
from custom_components.pydantic_ai_agent.models.templated_extra_body import (
    merge_extra_body,
    render_templated_extra_body,
    validate_templated_extra_body_paths,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
import pytest


def _field(key: str, template: str = "1") -> dict[str, str]:
    return {
        CONF_CHAT_TEMPLATE_KWARG_KEY: key,
        CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: template,
    }


def test_validate_templated_extra_body_paths_accepts_nested_and_legacy_keys() -> None:
    """Dotted keys and legacy chat_template_kwargs keys build valid paths."""
    validate_templated_extra_body_paths(
        [_field("metadata.user"), _field("chat_template_kwargs.temperature")]
    )


@pytest.mark.parametrize(
    "configured",
    [
        [_field("metadata"), _field("metadata.user")],
        [_field("metadata.user"), _field("metadata.user")],
        [_field("chat_template_kwargs.")],
        [_field("metadata..user")],
    ],
)
def test_validate_templated_extra_body_paths_rejects_conflicts(
    configured: list[dict[str, str]],
) -> None:
    """Scalar/mapping conflicts and invalid dotted paths raise HA errors."""
    with pytest.raises(HomeAssistantError):
        validate_templated_extra_body_paths(configured)


def test_merge_extra_body_deep_merges_and_replaces_lists_without_mutation() -> None:
    """Nested mappings are merged, lists replaced, and base is copied."""
    base = {
        "metadata": {"labels": ["old"], "keep": True},
        "unchanged": {"nested": 1},
    }
    overlay = {"metadata": {"labels": ["new"], "added": 2}}

    merged = merge_extra_body(base, overlay)

    assert merged == {
        "metadata": {"labels": ["new"], "keep": True, "added": 2},
        "unchanged": {"nested": 1},
    }
    assert base == {
        "metadata": {"labels": ["old"], "keep": True},
        "unchanged": {"nested": 1},
    }


@pytest.mark.parametrize(
    ("base", "overlay"),
    [
        ({"metadata": "scalar"}, {"metadata": {"user": "alice"}}),
        ({"metadata": {"user": "alice"}}, {"metadata": "scalar"}),
    ],
)
def test_merge_extra_body_rejects_mapping_scalar_conflicts(
    base: dict[str, object], overlay: dict[str, object]
) -> None:
    """Existing mapping/scalar shape conflicts are surfaced as HA errors."""
    with pytest.raises(HomeAssistantError):
        merge_extra_body(base, overlay)


async def test_render_templated_extra_body_renders_nested_values(
    hass: HomeAssistant,
) -> None:
    """Configured HA templates render into nested request body mappings."""
    rendered = render_templated_extra_body(
        hass,
        [
            _field("metadata.count", "{{ 1 + 1 }}"),
            _field("chat_template_kwargs.mode", "json"),
        ],
    )

    assert rendered == {
        "metadata": {"count": 2},
        "chat_template_kwargs": {"mode": "json"},
    }
