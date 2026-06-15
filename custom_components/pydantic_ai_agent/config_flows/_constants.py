"""Shared constants for config-flow modules.

This module must not import from any other config_flows module.
"""

from __future__ import annotations

import re
from datetime import timedelta

from homeassistant.components.todo.const import TodoListEntityFeature

from ..const import (
    CONF_MAX_ITERATIONS,
    CONF_MAX_TOKENS,
    CONF_TEMPLATED_EXTRA_BODY,
    CONF_THINKING,
    CONF_TIMEOUT,
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
)
from ..model_settings import (
    MODEL_SETTING_EXTRA_BODY,
    REMOVED_PROFILE_MODEL_SETTING_KEYS,
    RUN_SETTING_KEYS,
)

_HTTP_HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")

_HTTP_STATUS_LABELS: dict[int, str] = {
    400: "invalid request",
    401: "authentication issue",
    402: "payment issue",
    403: "permission issue",
    404: "model not found",
    408: "timeout",
    409: "conflict",
    422: "validation issue",
    429: "rate limit",
    504: "timeout",
}

_MODEL_SETTING_MAX_TOKENS = CONF_MAX_TOKENS
_MODEL_SETTING_MAX_ITERATIONS = CONF_MAX_ITERATIONS
_MODEL_SETTING_TEMPERATURE = "temperature"
_MODEL_SETTING_TOP_P = "top_p"
_MODEL_SETTING_TIMEOUT = CONF_TIMEOUT
_MODEL_SETTING_PARALLEL_TOOL_CALLS = "parallel_tool_calls"
_MODEL_SETTING_SEED = "seed"
_MODEL_SETTING_PRESENCE_PENALTY = "presence_penalty"
_MODEL_SETTING_FREQUENCY_PENALTY = "frequency_penalty"
_MODEL_SETTING_THINKING = CONF_THINKING
_MODEL_SETTING_EXTRA_BODY = MODEL_SETTING_EXTRA_BODY
_MODEL_SETTING_TEMPLATED_EXTRA_BODY = CONF_TEMPLATED_EXTRA_BODY
_MODEL_PRICING_INPUT = "model_pricing_input"
_MODEL_PRICING_OUTPUT = "model_pricing_output"
_MODEL_PRICING_CACHE_READ = "model_pricing_cache_read"

_MODEL_LIST_CACHE_TTL = timedelta(minutes=10)

_BASE_URL_ENDPOINT_SUFFIXES: set[tuple[str, ...]] = {
    ("audio", "speech"),
    ("audio", "transcriptions"),
    ("audio", "translations"),
    ("batches",),
    ("chat", "completions"),
    ("completions",),
    ("embeddings",),
    ("files",),
    ("fine_tuning", "jobs"),
    ("images", "edits"),
    ("images", "generations"),
    ("images", "variations"),
    ("messages",),
    ("models",),
    ("moderations",),
    ("responses",),
    ("threads",),
}

_BASE_URL_ENDPOINT_PATH_ENDINGS: tuple[str, ...] = (
    ":generatecontent",
    ":streamgeneratecontent",
)

_PROVIDER_EXTRA_BODY_MODES: set[str] = {
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
}

_MAX_METADATA_REPR_LENGTH = 1000

_MAIN_MODEL_SETTING_KEYS: set[str] = {
    _MODEL_SETTING_TEMPERATURE,
}

_ADVANCED_MODEL_SETTING_KEYS: set[str] = {
    _MODEL_SETTING_TOP_P,
    _MODEL_SETTING_PARALLEL_TOOL_CALLS,
    _MODEL_SETTING_SEED,
    _MODEL_SETTING_PRESENCE_PENALTY,
    _MODEL_SETTING_FREQUENCY_PENALTY,
    _MODEL_SETTING_TEMPLATED_EXTRA_BODY,
}

_RUN_SETTING_KEYS = RUN_SETTING_KEYS

_REMOVED_MODEL_SETTING_KEYS: set[str] = {
    "extra_headers",
    *REMOVED_PROFILE_MODEL_SETTING_KEYS,
}

_THINKING_OPTIONS: tuple[str, ...] = (
    "",
    "true",
    "false",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)

_CONF_MODEL_PROFILE_ID = "model_profile_id"
_SECTION_ADVANCED_MODEL_SETTINGS = "advanced_model_settings"
_SECTION_OPENAI_COMPATIBLE_CAPABILITIES = "openai_compatible_capabilities"
_SECTION_ADVANCED_OPTIONS = "advanced_options"
_SECTION_EXTERNAL_TOOLS = "external_tools"
_SECTION_FALLBACK_MODELS = "fallback_models"
_SECTION_HASS_CONTROL = "hass_control"
_SECTION_MODEL_PRICING = "model_pricing"
_SECTION_CUSTOMIZE_MODEL_LIST = "customize_model_list"
_SECTION_RUN_SETTINGS = "run_settings"
_SECTION_SKILLS = "skill_settings"

_TODO_WORKSPACE_REQUIRED_FEATURES = (
    TodoListEntityFeature.CREATE_TODO_ITEM
    | TodoListEntityFeature.DELETE_TODO_ITEM
    | TodoListEntityFeature.UPDATE_TODO_ITEM
    | TodoListEntityFeature.SET_DESCRIPTION_ON_ITEM
)
