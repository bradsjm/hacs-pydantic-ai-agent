"""Model setting normalization helpers."""

import json
from collections.abc import Mapping, Set
from typing import Any, Final

from ..const import (
    CONF_MAX_ITERATIONS,
    CONF_MAX_TOKENS,
    CONF_TEMPLATED_EXTRA_BODY,
    CONF_THINKING,
    CONF_TIMEOUT,
    CONF_TOOL_RETRIES,
)

MODEL_SETTING_EXTRA_BODY: Final = "extra_body"

RUN_SETTING_KEYS: Final[frozenset[str]] = frozenset(
    {
        CONF_MAX_TOKENS,
        CONF_MAX_ITERATIONS,
        CONF_TIMEOUT,
        CONF_THINKING,
        CONF_TOOL_RETRIES,
    }
)
REMOVED_PROFILE_MODEL_SETTING_KEYS: Final[frozenset[str]] = (
    RUN_SETTING_KEYS | frozenset({MODEL_SETTING_EXTRA_BODY})
)
RUNTIME_STRIPPED_MODEL_SETTING_KEYS: Final[frozenset[str]] = (
    RUN_SETTING_KEYS | frozenset({CONF_TEMPLATED_EXTRA_BODY, MODEL_SETTING_EXTRA_BODY})
)


def strip_model_settings(
    settings: Mapping[str, Any] | None, keys: Set[str]
) -> dict[str, Any]:
    """Return model settings with integration-owned keys removed."""
    stripped = dict(settings or {})
    for key in keys:
        stripped.pop(key, None)
    return stripped


def normalise_persisted_model_settings(settings: Mapping[str, Any] | None) -> str:
    """Return stable persisted model settings for comparison."""
    provider_settings = strip_model_settings(
        settings, REMOVED_PROFILE_MODEL_SETTING_KEYS
    )
    return json.dumps(provider_settings, sort_keys=True, separators=(",", ":"))


def normalise_applied_model_settings(settings: Mapping[str, Any]) -> str:
    """Return stable model settings after run settings are applied."""
    return json.dumps(dict(settings), sort_keys=True, separators=(",", ":"))


def profile_model_settings(settings: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return persisted profile settings without run-owned keys."""
    return strip_model_settings(settings, RUN_SETTING_KEYS)


def runtime_model_settings_data(
    profile_settings: Mapping[str, Any] | None, run_settings: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Return request model settings before default timeout is applied."""
    settings = strip_model_settings(
        profile_settings, RUNTIME_STRIPPED_MODEL_SETTING_KEYS
    )
    if run_settings is not None:
        if CONF_MAX_TOKENS in run_settings:
            settings[CONF_MAX_TOKENS] = run_settings[CONF_MAX_TOKENS]
        if CONF_TIMEOUT in run_settings:
            settings[CONF_TIMEOUT] = run_settings[CONF_TIMEOUT]
    return settings
