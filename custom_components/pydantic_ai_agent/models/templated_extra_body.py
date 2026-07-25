"""Templated request extra-body helpers."""

from collections.abc import Mapping
import json

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers.template import Template

from ..const import (
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
)

_LEGACY_CHAT_TEMPLATE_KWARGS_PREFIX = "chat_template_kwargs."


def render_templated_extra_body(hass: HomeAssistant, configured: object) -> dict[str, object]:
    """Render configured templated extra-body fields for one model request."""
    if configured in (None, ""):
        return {}
    if not isinstance(configured, list):
        raise HomeAssistantError("Configured templated extra body is invalid")

    rendered: dict[str, object] = {}
    for item in configured:
        if not isinstance(item, Mapping):
            raise HomeAssistantError("Configured templated extra body is invalid")
        key = str(item.get(CONF_CHAT_TEMPLATE_KWARG_KEY, "")).strip()
        value_template = item.get(CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE)
        if not key or not isinstance(value_template, str):
            raise HomeAssistantError("Configured templated extra body is invalid")
        try:
            value = Template(value_template, hass).async_render(parse_result=True)
            json.dumps(value)
        except TemplateError as err:
            raise HomeAssistantError(f'Failed to render templated extra body field "{key}"') from err
        except (TypeError, ValueError) as err:
            raise HomeAssistantError(f'Failed to render templated extra body field "{key}"') from err
        try:
            _set_nested_value(rendered, _path_segments(key), value)
        except ValueError as err:
            raise HomeAssistantError(str(err)) from err
    return rendered


def validate_templated_extra_body_paths(configured: object) -> None:
    """Validate templated extra-body key paths without rendering templates."""
    if configured in (None, ""):
        return
    if not isinstance(configured, list):
        raise HomeAssistantError("Configured templated extra body is invalid")
    tree: dict[str, object] = {}
    for item in configured:
        if not isinstance(item, Mapping):
            raise HomeAssistantError("Configured templated extra body is invalid")
        key = str(item.get(CONF_CHAT_TEMPLATE_KWARG_KEY, "")).strip()
        if not key:
            raise HomeAssistantError("Configured templated extra body is invalid")
        try:
            _set_nested_value(tree, _path_segments(key), object())
        except ValueError as err:
            raise HomeAssistantError(str(err)) from err


def merge_extra_body(base: object, overlay: Mapping[str, object]) -> dict[str, object]:
    """Return a deep merge of request extra-body mappings."""
    base_mapping = base if isinstance(base, Mapping) else {}
    try:
        return _deep_merge(base_mapping, overlay)
    except ValueError as err:
        raise HomeAssistantError(str(err)) from err


def _path_segments(key: str) -> tuple[str, ...]:
    """Return validated dotted-path segments for one templated field key."""
    if key.startswith(_LEGACY_CHAT_TEMPLATE_KWARGS_PREFIX):
        remainder = key.removeprefix(_LEGACY_CHAT_TEMPLATE_KWARGS_PREFIX).strip()
        if not remainder:
            raise ValueError(f'Conflicting templated extra body path "{key}"')
        return ("chat_template_kwargs", remainder)
    segments = tuple(segment.strip() for segment in key.split("."))
    if not segments or any(not segment for segment in segments):
        raise ValueError(f'Conflicting templated extra body path "{key}"')
    return segments


def _set_nested_value(target: dict[str, object], path: tuple[str, ...], value: object) -> None:
    """Set one dotted-path value and reject conflicting mapping/scalar paths."""
    current = target
    for segment in path[:-1]:
        existing = current.get(segment)
        if existing is None:
            child: dict[str, object] = {}
            current[segment] = child
            current = child
            continue
        if not isinstance(existing, dict):
            raise ValueError(f'Conflicting templated extra body path "{".".join(path)}"')
        current = existing

    leaf = path[-1]
    if leaf in current:
        raise ValueError(f'Conflicting templated extra body path "{".".join(path)}"')
    current[leaf] = value


def _deep_merge(base: Mapping[str, object], overlay: Mapping[str, object]) -> dict[str, object]:
    """Return a recursive merge of mapping values."""
    merged = {key: _clone_json_value(value) for key, value in base.items()}
    for key, value in overlay.items():
        has_key = key in merged
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
            continue
        if has_key and isinstance(existing, dict) != isinstance(value, Mapping):
            raise ValueError(f'Conflicting templated extra body path "{key}"')
        merged[key] = _clone_json_value(value)
    return merged


def _clone_json_value(value: object) -> object:
    """Clone nested JSON-like values without mutating caller-owned data."""
    if isinstance(value, Mapping):
        return {key: _clone_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_json_value(item) for item in value]
    return value


def _validate_path_against_base(path: tuple[str, ...], base: Mapping[str, object]) -> None:
    """Reject dotted paths that descend into non-mapping base extra-body values."""
    current: object = base
    for segment in path[:-1]:
        if not isinstance(current, Mapping) or segment not in current:
            return
        current = current[segment]
        if not isinstance(current, Mapping):
            raise HomeAssistantError(f'Conflicting templated extra body path "{".".join(path)}"')
