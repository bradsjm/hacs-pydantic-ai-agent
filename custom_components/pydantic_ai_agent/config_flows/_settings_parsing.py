"""Model settings and pricing parsing helpers for config flows.

This module must not import from any other config_flows module.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers.template import Template

from ..const import (
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
    CONF_MODEL_PRICING,
    CONF_MODEL_SETTINGS,
)
from ..templated_extra_body import validate_templated_extra_body_paths
from ._constants import (
    _ADVANCED_MODEL_SETTING_KEYS,
    _MAIN_MODEL_SETTING_KEYS,
    _MODEL_PRICING_CACHE_READ,
    _MODEL_PRICING_INPUT,
    _MODEL_PRICING_OUTPUT,
    _MODEL_SETTING_EXTRA_BODY,
    _MODEL_SETTING_FREQUENCY_PENALTY,
    _MODEL_SETTING_MAX_ITERATIONS,
    _MODEL_SETTING_MAX_TOKENS,
    _MODEL_SETTING_PARALLEL_TOOL_CALLS,
    _MODEL_SETTING_PRESENCE_PENALTY,
    _MODEL_SETTING_SEED,
    _MODEL_SETTING_TEMPERATURE,
    _MODEL_SETTING_TEMPLATED_EXTRA_BODY,
    _MODEL_SETTING_THINKING,
    _MODEL_SETTING_TIMEOUT,
    _MODEL_SETTING_TOP_P,
    _REMOVED_MODEL_SETTING_KEYS,
    _RUN_SETTING_KEYS,
    _THINKING_OPTIONS,
)
from ._key_value_rows import _parse_key_value_json_rows


class RunSettingsValidationError(ValueError):
    """Error raised when conversation/task run settings are invalid."""

    def __init__(self, errors: dict[str, str]) -> None:
        """Initialize the error with Home Assistant form error keys."""
        super().__init__("invalid_run_settings")
        self.errors = errors


def _format_key_value_json_setting(value: object) -> str:
    """Return a key/value JSON setting as one ``key: value`` line each."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        return ""
    return "\n".join(
        f"{key}: {json.dumps(value[key], sort_keys=True)}" for key in sorted(value)
    )


def _format_templated_extra_body(value: object) -> list[dict[str, str]]:
    """Return stored templated extra-body rows in selector-compatible shape."""
    if not isinstance(value, list):
        return []
    formatted: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        key = item.get(CONF_CHAT_TEMPLATE_KWARG_KEY)
        value_template = item.get(CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE)
        if isinstance(key, str) and isinstance(value_template, str):
            formatted.append(
                {
                    CONF_CHAT_TEMPLATE_KWARG_KEY: key,
                    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: value_template,
                }
            )
    return formatted


def _format_thinking_value(model_settings: Mapping[str, Any]) -> str:
    """Return the selector value for the configured thinking setting."""
    value = model_settings.get(_MODEL_SETTING_THINKING)
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return value
    return ""


def _is_blank(value: object) -> bool:
    """Return if a submitted optional field should be treated as unset."""
    return value is None or (isinstance(value, str) and not value.strip())


def _parse_float_setting(value: object) -> float:
    """Return a float model setting from user input."""
    if isinstance(value, bool):
        raise ValueError
    if not isinstance(value, (int, float, str)):
        raise ValueError
    return float(value)


def _parse_positive_float_setting(value: object) -> float:
    """Return a positive float model setting from user input."""
    parsed = _parse_float_setting(value)
    if parsed <= 0:
        raise ValueError
    return parsed


def _parse_non_negative_float_setting(value: object) -> float:
    """Return a non-negative float setting from user input."""
    parsed = _parse_float_setting(value)
    if parsed < 0:
        raise ValueError
    return parsed


def _parse_int_setting(value: object) -> int:
    """Return an integer model setting from user input."""
    if isinstance(value, bool):
        raise ValueError
    if not isinstance(value, (int, float, str)):
        raise ValueError
    parsed = int(value)
    if float(value) != parsed:
        raise ValueError
    return parsed


def _parse_positive_int_setting(value: object) -> int:
    """Return a positive integer model setting from user input."""
    parsed = _parse_int_setting(value)
    if parsed <= 0:
        raise ValueError
    return parsed


def _parse_non_negative_int_setting(value: object) -> int:
    """Return a non-negative integer model setting from user input."""
    parsed = _parse_int_setting(value)
    if parsed < 0:
        raise ValueError
    return parsed


def _parse_key_value_json_setting(value: object) -> dict[str, Any]:
    """Return a key/value JSON model setting from user input."""
    if isinstance(value, list):
        parsed = _parse_key_value_json_rows(value)
        return parsed
    if not isinstance(value, str):
        raise ValueError("invalid_key_value")
    parsed: dict[str, Any] = {}
    for line in value.splitlines():
        line = line.strip()
        if not line:
            continue
        key, separator, item = line.partition(":")
        key = key.strip()
        if not separator or not key:
            raise ValueError("invalid_key_value")
        if key in parsed:
            raise ValueError("duplicate_key")
        try:
            parsed[key] = json.loads(item.strip())
        except json.JSONDecodeError as err:
            raise ValueError("invalid_json") from err
    return parsed


def _parse_templated_extra_body(
    hass: HomeAssistant, value: object
) -> list[dict[str, str]]:
    """Return configured templated extra-body rows from selector input."""
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError("invalid_chat_template_kwargs")
    parsed: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value:
        parsed_row = _parse_templated_extra_body_row(hass, item, seen)
        if parsed_row is None:
            continue
        key, value_template = parsed_row
        seen.add(key)
        parsed.append(
            {
                CONF_CHAT_TEMPLATE_KWARG_KEY: key,
                CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: value_template,
            }
        )
    try:
        validate_templated_extra_body_paths(parsed)
    except HomeAssistantError as err:
        raise ValueError("templated_extra_body_path_conflict") from err
    return parsed


def _parse_templated_extra_body_row(
    hass: HomeAssistant, item: object, seen: set[str]
) -> tuple[str, str] | None:
    """Return one parsed templated extra-body row."""
    if not isinstance(item, Mapping):
        raise ValueError("invalid_chat_template_kwargs")
    key = str(item.get(CONF_CHAT_TEMPLATE_KWARG_KEY, "")).strip()
    value_template = item.get(CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE)
    if not key and not value_template:
        return None
    if not key:
        raise ValueError("invalid_chat_template_key")
    if key in seen:
        raise ValueError("duplicate_key")
    if not isinstance(value_template, str) or not value_template.strip():
        raise ValueError("invalid_chat_template")
    try:
        Template(value_template, hass).ensure_valid()
        rendered = Template(value_template, hass).async_render(parse_result=True)
        json.dumps(rendered)
    except TemplateError as err:
        raise ValueError("invalid_chat_template") from err
    except (TypeError, ValueError) as err:
        raise ValueError("invalid_chat_template") from err
    return key, value_template


def _parse_thinking_setting(value: object) -> bool | str:
    """Return a Pydantic AI thinking setting."""
    if not isinstance(value, str):
        raise ValueError
    parsed = value.strip()
    if parsed == "true":
        return True
    if parsed == "false":
        return False
    if parsed not in _THINKING_OPTIONS:
        raise ValueError
    return parsed


def _parse_model_settings(
    hass: HomeAssistant, user_input: Mapping[str, Any], setting_keys: set[str]
) -> tuple[dict[str, Any], dict[str, str], set[str]]:
    """Return parsed model settings, field errors, and explicitly cleared keys."""
    settings: dict[str, Any] = {}
    errors: dict[str, str] = {}
    cleared: set[str] = set()
    for key in setting_keys:
        if key not in user_input:
            continue
        value = user_input[key]
        if _is_blank(value):
            cleared.add(key)
            continue
        try:
            parsed_value, clear_key = _parse_model_setting_value(hass, key, value)
        except ValueError as err:
            errors[key] = _model_setting_error(key, str(err))
        else:
            if clear_key:
                cleared.add(key)
            elif parsed_value is not None:
                settings[key] = parsed_value
    return settings, errors, cleared


def _parse_model_setting_value(
    hass: HomeAssistant, key: str, value: object
) -> tuple[object | None, bool]:
    """Return one parsed model setting and whether it should be cleared."""
    if key in {_MODEL_SETTING_MAX_TOKENS, _MODEL_SETTING_MAX_ITERATIONS}:
        return _parse_positive_int_setting(value), False
    if key == _MODEL_SETTING_SEED:
        return _parse_non_negative_int_setting(value), False
    if key == _MODEL_SETTING_TIMEOUT:
        return _parse_positive_float_setting(value), False
    if key in {
        _MODEL_SETTING_TEMPERATURE,
        _MODEL_SETTING_TOP_P,
        _MODEL_SETTING_PRESENCE_PENALTY,
        _MODEL_SETTING_FREQUENCY_PENALTY,
    }:
        return _parse_float_setting(value), False
    if key == _MODEL_SETTING_PARALLEL_TOOL_CALLS:
        if not isinstance(value, bool):
            raise ValueError
        return value, False
    if key == _MODEL_SETTING_EXTRA_BODY:
        return _parse_key_value_json_setting(value), False
    if key == _MODEL_SETTING_TEMPLATED_EXTRA_BODY:
        parsed = _parse_templated_extra_body(hass, value)
        return parsed or None, not parsed
    if key == _MODEL_SETTING_THINKING:
        return _parse_thinking_setting(value), False
    return None, False


def _normalise_run_settings(data: dict[str, Any]) -> None:
    """Normalize conversation/task run settings stored directly on subentries."""
    errors: dict[str, str] = {}
    for key in (_MODEL_SETTING_MAX_TOKENS, _MODEL_SETTING_THINKING):
        if _is_blank(data.get(key)):
            data.pop(key, None)
    for key, parser in (
        (_MODEL_SETTING_MAX_TOKENS, _parse_positive_int_setting),
        (_MODEL_SETTING_MAX_ITERATIONS, _parse_positive_int_setting),
        (_MODEL_SETTING_TIMEOUT, _parse_positive_float_setting),
        (_MODEL_SETTING_THINKING, _parse_thinking_setting),
    ):
        _normalise_run_setting(data, key, parser, errors)
    if errors:
        raise RunSettingsValidationError(errors)


def _normalise_run_setting(
    data: dict[str, Any],
    key: str,
    parser: Callable[[object], object],
    errors: dict[str, str],
) -> None:
    """Parse one stored run setting in place and collect validation errors."""
    if key not in data:
        return
    try:
        data[key] = parser(data[key])
    except ValueError as err:
        errors[key] = _model_setting_error(key, str(err))


def _parse_model_pricing(
    user_input: Mapping[str, Any], pricing_keys: set[str]
) -> tuple[dict[str, float], dict[str, str], set[str]]:
    """Return parsed pricing, field errors, and explicitly cleared pricing keys."""
    pricing: dict[str, float] = {}
    errors: dict[str, str] = {}
    cleared: set[str] = set()
    for field_key in pricing_keys:
        if field_key not in user_input:
            continue
        pricing_key = _pricing_storage_key(field_key)
        value = user_input[field_key]
        if _is_blank(value):
            cleared.add(pricing_key)
            continue
        try:
            pricing[pricing_key] = _parse_non_negative_float_setting(value)
        except ValueError:
            errors[field_key] = "non_negative_number"
    return pricing, errors, cleared


def _pricing_storage_key(field_key: str) -> str:
    """Return stored pricing key for one form field."""
    return {
        _MODEL_PRICING_INPUT: "input",
        _MODEL_PRICING_OUTPUT: "output",
        _MODEL_PRICING_CACHE_READ: "cache_read",
    }[field_key]


def _model_setting_error(key: str, detail: str) -> str:
    """Return a translation key for a model setting validation error."""
    if detail in {
        "duplicate_key",
        "invalid_chat_template",
        "invalid_chat_template_key",
        "invalid_chat_template_kwargs",
        "invalid_json",
        "invalid_key_value",
        "templated_extra_body_path_conflict",
    }:
        return detail
    if key in {
        _MODEL_SETTING_MAX_TOKENS,
        _MODEL_SETTING_MAX_ITERATIONS,
        _MODEL_SETTING_SEED,
    }:
        return "invalid_integer"
    if key == _MODEL_SETTING_TIMEOUT:
        return "positive_number"
    return "invalid_number"


def _model_settings_from_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return existing model settings from subentry options."""
    model_settings = options.get(CONF_MODEL_SETTINGS)
    if isinstance(model_settings, Mapping):
        return {
            key: value
            for key, value in model_settings.items()
            if key not in _REMOVED_MODEL_SETTING_KEYS
        }
    return {}


def _merge_model_settings(
    existing: Mapping[str, Any],
    parsed: Mapping[str, Any],
    cleared: set[str],
) -> dict[str, Any]:
    """Return model settings with parsed values applied and cleared keys removed."""
    merged = dict(existing)
    for key in cleared:
        merged.pop(key, None)
    merged.update(parsed)
    return merged


def _merge_model_pricing(
    existing: Mapping[str, Any], parsed: Mapping[str, float], cleared: set[str]
) -> dict[str, float]:
    """Return pricing with parsed values applied and cleared keys removed."""
    merged = _model_pricing_from_options({CONF_MODEL_PRICING: existing})
    for key in cleared:
        merged.pop(key, None)
    merged.update(parsed)
    return merged


def _store_model_settings(
    data: dict[str, Any], model_settings: Mapping[str, Any]
) -> None:
    """Store model settings only when at least one setting is configured."""
    if model_settings:
        data[CONF_MODEL_SETTINGS] = dict(model_settings)
    else:
        data.pop(CONF_MODEL_SETTINGS, None)


def _store_model_pricing(
    data: dict[str, Any], model_pricing: Mapping[str, float]
) -> None:
    """Store profile pricing, including an empty mapping after explicit clears."""
    data[CONF_MODEL_PRICING] = dict(model_pricing)


def _model_pricing_from_options(options: Mapping[str, Any]) -> dict[str, float]:
    """Return valid stored model pricing from subentry options."""
    pricing = options.get(CONF_MODEL_PRICING)
    if not isinstance(pricing, Mapping):
        return {}
    parsed: dict[str, float] = {}
    for key in ("input", "output", "cache_read"):
        value = pricing.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        price = float(value)
        if price >= 0:
            parsed[key] = price
    return parsed


def _model_profile_data_from_user_input(
    user_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Return model profile data excluding form-only setting fields."""
    return {
        key: value
        for key, value in user_input.items()
        if key
        not in _MAIN_MODEL_SETTING_KEYS
        | _ADVANCED_MODEL_SETTING_KEYS
        | _RUN_SETTING_KEYS
        | {_MODEL_PRICING_INPUT, _MODEL_PRICING_OUTPUT, _MODEL_PRICING_CACHE_READ}
    }
