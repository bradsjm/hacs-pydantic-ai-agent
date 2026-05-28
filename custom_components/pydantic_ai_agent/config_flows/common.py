"""Shared config-flow helpers for Pydantic AI Agent."""

# ruff: noqa: F401

from __future__ import annotations

import errno
import json
import logging
import socket
import ssl
from collections.abc import AsyncIterable, Iterable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx
import voluptuous as vol
from homeassistant.components.todo import DOMAIN as TODO_DOMAIN
from homeassistant.components.todo import TodoListEntityFeature
from homeassistant.config_entries import (
    SOURCE_USER,
    ConfigEntry,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentry,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import section
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    BooleanSelector,
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    ObjectSelector,
    ObjectSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
from homeassistant.helpers.template import Template
from homeassistant.helpers.typing import VolDictType
from homeassistant.util import dt as dt_util
from pydantic_ai import (
    ModelRequest,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
)
from pydantic_ai.direct import model_request_stream
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import ModelResponseStreamEvent
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings

from .._redaction import redact_data
from ..chat_template_kwargs import (
    reject_chat_template_kwargs_in_extra_body,
    render_chat_template_kwargs,
)
from ..const import (
    CONF_AGENT_NAME,
    CONF_AI_TASK_NAME,
    CONF_BASE_URL,
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
    CONF_CHAT_TEMPLATE_KWARGS,
    CONF_CUSTOM_MODEL_NAMES,
    CONF_DISCOVERED,
    CONF_DISCOVERED_MODELS,
    CONF_DISCOVERED_MODELS_AT,
    CONF_DISCOVERED_MODELS_CACHE_KEY,
    CONF_ENABLED,
    CONF_FALLBACK_MODEL_REFS,
    CONF_MAX_ITERATIONS,
    CONF_MAX_TOKENS,
    CONF_MCP_SERVER_IDS,
    CONF_MODEL,
    CONF_MODEL_PRICING,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_OUTPUT_MODE,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROMPT,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_METADATA,
    CONF_PROVIDER_MODE,
    CONF_SKILLS,
    CONF_THINKING,
    CONF_TIMEOUT,
    CONF_TODO_LIST_ENTITY_ID,
    CONF_VIRTUAL_WORKSPACE_ENABLED,
    CONF_WEB_FETCH_ENABLED,
    DEFAULT_AGENT_NAME,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_OUTPUT_MODE,
    DEFAULT_SERVICE_NAME,
    DEFAULT_TIMEOUT,
    DOMAIN,
    OUTPUT_MODE_NATIVE,
    OUTPUT_MODE_PROMPTED,
    OUTPUT_MODE_TOOL,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE_GEMINI,
    PROVIDER_MODES,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
    STRUCTURED_OUTPUT_MODES,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_MCP_SERVER,
    SUBENTRY_TYPE_PROVIDER,
    SUBENTRY_TYPE_SKILL,
    default_conversation_options,
)
from ..mcp import (
    parse_mcp_headers,
)
from ..model_profiles import (
    configured_model_profile_exists,
    model_profile_display_name,
    model_profile_ref,
    parse_model_profile_ref,
    provider_model_profiles,
    provider_subentries,
)
from ..provider import (
    anthropic_model,
    google_gemini_model,
    list_anthropic_model_names,
    list_google_gemini_model_names,
    normalise_base_url,
    openai_compatible_client_from_config,
    openai_compatible_completions_model_from_config,
    openai_compatible_responses_model_from_config,
)
from ..provider_validation import (
    ProviderValidationError,
    _format_api_error,
    _map_http_error,
    async_list_provider_model_names,
    async_probe_model,
)
from ..model_settings import (
    MODEL_SETTING_EXTRA_BODY,
    REMOVED_PROFILE_MODEL_SETTING_KEYS,
    RUN_SETTING_KEYS,
)
from .helpers import _flatten_section_data, _section_schema_key
from .skill_helpers import (
    _append_skill_schema_fields,
    _normalise_skill_selection,
)
from ..structured_output import (
    structured_model_request_parameters,
    structured_output_name,
)
from ..structured_output import (
    structured_output_mode as normalise_structured_output_mode,
)

_LOGGER = logging.getLogger(__name__)

_HTTP_STATUS_LABELS = {
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
_MODEL_SETTING_CHAT_TEMPLATE_KWARGS = CONF_CHAT_TEMPLATE_KWARGS
_MODEL_PRICING_INPUT = "model_pricing_input"
_MODEL_PRICING_OUTPUT = "model_pricing_output"
_MODEL_PRICING_CACHE_READ = "model_pricing_cache_read"
_MODEL_LIST_CACHE_TTL = timedelta(minutes=10)
_BASE_URL_ENDPOINT_SUFFIXES = {
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
_BASE_URL_ENDPOINT_PATH_ENDINGS = (":generatecontent", ":streamgeneratecontent")
_PROVIDER_EXTRA_BODY_MODES = {
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
}
_MAX_METADATA_REPR_LENGTH = 1000
_SENSITIVE_METADATA_KEYS = {
    "access_token",
    "api_key",
    "authorization",
    "cookie",
    "headers",
    "password",
    "request_headers",
    "response_headers",
    "secret",
    "token",
    "x-api-key",
}

_MAIN_MODEL_SETTING_KEYS = {
    _MODEL_SETTING_TEMPERATURE,
}
_ADVANCED_MODEL_SETTING_KEYS = {
    _MODEL_SETTING_TOP_P,
    _MODEL_SETTING_PARALLEL_TOOL_CALLS,
    _MODEL_SETTING_SEED,
    _MODEL_SETTING_PRESENCE_PENALTY,
    _MODEL_SETTING_FREQUENCY_PENALTY,
    _MODEL_SETTING_CHAT_TEMPLATE_KWARGS,
}
_RUN_SETTING_KEYS = RUN_SETTING_KEYS
_REMOVED_MODEL_SETTING_KEYS = {
    "extra_headers",
    *REMOVED_PROFILE_MODEL_SETTING_KEYS,
}
_THINKING_OPTIONS = ("", "true", "false", "minimal", "low", "medium", "high", "xhigh")
_OUTPUT_MODE_OPTIONS = tuple(
    SelectOptionDict(value=value, label=value) for value in STRUCTURED_OUTPUT_MODES
)
class RunSettingsValidationError(ValueError):
    """Error raised when conversation/task run settings are invalid."""

    def __init__(self, errors: dict[str, str]) -> None:
        """Initialize the error with Home Assistant form error keys."""
        super().__init__("invalid_run_settings")
        self.errors = errors


_CONF_MODEL_PROFILE_ID = "model_profile_id"
_SECTION_ADVANCED_MODEL_SETTINGS = "advanced_model_settings"
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
_STRUCTURED_PROBE_OUTPUT_NAME = structured_output_name(
    "probe_response", "probe_response"
)
_STRUCTURED_PROBE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def _format_http_headers(headers: object) -> str:
    """Return HTTP headers as one ``Header-Name: value`` line each."""
    if headers is None:
        return ""
    if isinstance(headers, str):
        return headers
    if not isinstance(headers, Mapping):
        return ""
    return "\n".join(f"{name}: {headers[name]}" for name in sorted(headers))


def _agent_form_suggested_values(
    options: Mapping[str, Any], hass: HomeAssistant | None = None
) -> dict[str, Any]:
    """Return per-agent suggested values matching the sectioned form schema."""
    suggested_values = dict(options)
    suggested_values[_SECTION_RUN_SETTINGS] = {
        key: options[key] for key in _RUN_SETTING_KEYS if key in options
    }
    if CONF_LLM_HASS_API in options:
        llm_hass_api = options[CONF_LLM_HASS_API]
        if hass is not None:
            valid_api_ids = {api.id for api in llm.async_get_apis(hass)}
            llm_hass_api = [api for api in llm_hass_api if api in valid_api_ids]
        suggested_values[_SECTION_HASS_CONTROL] = {CONF_LLM_HASS_API: llm_hass_api}
    return suggested_values


def _parse_provider_headers(value: object) -> dict[str, str]:
    """Return provider HTTP headers from form input."""
    try:
        return parse_mcp_headers(value)
    except vol.Invalid as err:
        raise ProviderValidationError(
            "invalid_provider_headers",
            "Enter HTTP headers one per line using 'Header-Name: value'.",
        ) from err


def _provider_connection_schema(options: Mapping[str, Any] | None = None) -> vol.Schema:
    """Return the provider connection form schema."""
    data = _flatten_section_data(options or {}, (_SECTION_ADVANCED_OPTIONS,))
    provider_mode = data.get(CONF_PROVIDER_MODE, PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS)
    schema: VolDictType = {
        vol.Required(
            CONF_NAME,
            default=data.get(CONF_NAME, DEFAULT_SERVICE_NAME),
        ): TextSelector(TextSelectorConfig()),
        vol.Required(CONF_PROVIDER_MODE, default=provider_mode): SelectSelector(
            SelectSelectorConfig(
                options=list(PROVIDER_MODES),
                mode=SelectSelectorMode.DROPDOWN,
                translation_key=CONF_PROVIDER_MODE,
            )
        ),
        vol.Required(
            CONF_API_KEY,
            default=data.get(CONF_API_KEY, ""),
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
        vol.Optional(
            CONF_BASE_URL,
            default=data.get(CONF_BASE_URL, ""),
        ): TextSelector(TextSelectorConfig()),
    }
    schema[vol.Optional(_SECTION_ADVANCED_OPTIONS, default={})] = section(
        vol.Schema(
            {
                vol.Optional(
                    CONF_PROVIDER_HEADERS,
                    default=_format_http_headers(data.get(CONF_PROVIDER_HEADERS)),
                ): TextSelector(TextSelectorConfig(multiline=True)),
                vol.Optional(
                    CONF_PROVIDER_EXTRA_BODY,
                    default=_format_key_value_json_setting(
                        data.get(CONF_PROVIDER_EXTRA_BODY)
                    ),
                ): TextSelector(TextSelectorConfig(multiline=True)),
            }
        ),
        {"collapsed": True},
    )
    return vol.Schema(schema)


def _provider_custom_model_names(options: Mapping[str, Any]) -> list[str]:
    """Return configured custom model names for one provider form state."""
    custom_model_names = options.get(CONF_CUSTOM_MODEL_NAMES)
    if isinstance(custom_model_names, str):
        return _parse_custom_model_names(custom_model_names)
    if not isinstance(custom_model_names, list):
        return []
    seen: set[str] = set()
    names: list[str] = []
    for model_name in custom_model_names:
        if not isinstance(model_name, str):
            continue
        model_name = model_name.strip()
        if not model_name or model_name in seen:
            continue
        seen.add(model_name)
        names.append(model_name)
    return names


def _format_custom_model_names(options: Mapping[str, Any]) -> str:
    """Return custom model names as multiline text for the form."""
    return "\n".join(_provider_custom_model_names(options))


def _parse_custom_model_names(value: object) -> list[str]:
    """Return deduplicated custom model names from multiline form input."""
    if not isinstance(value, str):
        return []
    seen: set[str] = set()
    models: list[str] = []
    for line in value.splitlines():
        model_name = line.strip()
        if not model_name or model_name in seen:
            continue
        seen.add(model_name)
        models.append(model_name)
    return models


def _provider_schema(
    options: Mapping[str, Any] | None = None,
) -> vol.Schema:
    """Return the provider subentry schema."""
    options = dict(options or {})
    return _provider_connection_schema(options)


def _normalise_base_url(data: Mapping[str, Any]) -> str | None:
    """Return a normalized base URL if one is configured."""
    return normalise_base_url(data.get(CONF_BASE_URL))


def _base_url_endpoint_suffix(base_url: str | None) -> str | None:
    """Return a forbidden endpoint suffix if the base URL points at one."""
    if base_url is None:
        return None
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/").lower()
    for ending in _BASE_URL_ENDPOINT_PATH_ENDINGS:
        if path.endswith(ending):
            return ending.lstrip(":")
    segments = tuple(segment for segment in parsed.path.split("/") if segment)
    lowered = tuple(segment.lower() for segment in segments)
    for suffix in _BASE_URL_ENDPOINT_SUFFIXES:
        if len(lowered) >= len(suffix) and lowered[-len(suffix) :] == suffix:
            return "/".join(suffix)
    return None


def _validate_base_url(data: Mapping[str, Any]) -> None:
    """Reject endpoint URLs that the client appends itself."""
    if suffix := _base_url_endpoint_suffix(data.get(CONF_BASE_URL)):
        raise ProviderValidationError(
            "invalid_base_url_endpoint",
            (
                "Enter the provider API base URL, not an endpoint URL. "
                f"Remove the trailing /{suffix}."
            ),
        )


def _provider_extra_body_supported(data: Mapping[str, Any]) -> bool:
    """Return if the provider mode consumes provider-level extra body."""
    return data.get(CONF_PROVIDER_MODE) in _PROVIDER_EXTRA_BODY_MODES


def _dedupe_data(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return the provider fields that identify one connection."""
    dedupe = {
        CONF_PROVIDER_MODE: data[CONF_PROVIDER_MODE],
        CONF_API_KEY: data[CONF_API_KEY],
    }
    if base_url := data.get(CONF_BASE_URL):
        dedupe[CONF_BASE_URL] = base_url
    if headers := data.get(CONF_PROVIDER_HEADERS):
        dedupe[CONF_PROVIDER_HEADERS] = headers
    if provider_extra_body := data.get(CONF_PROVIDER_EXTRA_BODY):
        dedupe[CONF_PROVIDER_EXTRA_BODY] = provider_extra_body
    return dedupe


def _provider_model_cache_key(data: Mapping[str, Any]) -> str:
    """Return a stable cache key for provider model discovery."""
    api_key = data.get(CONF_API_KEY)
    headers = data.get(CONF_PROVIDER_HEADERS)
    return json.dumps(
        {
            CONF_PROVIDER_MODE: data.get(CONF_PROVIDER_MODE),
            CONF_API_KEY: sha256(str(api_key or "").encode()).hexdigest(),
            CONF_BASE_URL: data.get(CONF_BASE_URL),
            CONF_PROVIDER_HEADERS: dict(headers)
            if isinstance(headers, Mapping)
            else {},
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _cached_provider_model_names(data: Mapping[str, Any]) -> list[str] | None:
    """Return cached provider model names when the persisted cache is fresh."""
    if data.get(CONF_DISCOVERED_MODELS_CACHE_KEY) != _provider_model_cache_key(data):
        return None
    discovered_at = data.get(CONF_DISCOVERED_MODELS_AT)
    if not isinstance(discovered_at, str):
        return None
    parsed_at = dt_util.parse_datetime(discovered_at)
    if parsed_at is None or dt_util.utcnow() - parsed_at > _MODEL_LIST_CACHE_TTL:
        return None
    model_names = data.get(CONF_DISCOVERED_MODELS)
    if not isinstance(model_names, list):
        return None
    parsed_names = [name for name in model_names if isinstance(name, str) and name]
    return parsed_names or None


def _store_provider_model_cache(data: dict[str, Any], model_names: list[str]) -> None:
    """Store a successful provider model discovery response on provider data."""
    if not model_names:
        return
    data[CONF_DISCOVERED_MODELS] = sorted(set(model_names))
    data[CONF_DISCOVERED_MODELS_AT] = dt_util.utcnow().isoformat()
    data[CONF_DISCOVERED_MODELS_CACHE_KEY] = _provider_model_cache_key(data)


def _clear_provider_model_cache(data: dict[str, Any]) -> None:
    """Remove provider model discovery cache fields."""
    data.pop(CONF_DISCOVERED_MODELS, None)
    data.pop(CONF_DISCOVERED_MODELS_AT, None)
    data.pop(CONF_DISCOVERED_MODELS_CACHE_KEY, None)


def _referenced_provider_profile_ids(
    entry: ConfigEntry, provider_subentry_id: str
) -> set[str]:
    """Return model profile IDs referenced by conversation or AI task subentries."""
    referenced: set[str] = set()
    for subentry in entry.subentries.values():
        if subentry.subentry_type not in {
            SUBENTRY_TYPE_CONVERSATION,
            SUBENTRY_TYPE_AI_TASK,
        }:
            continue
        for profile_ref in _selected_model_profile_refs(subentry.data):
            try:
                ref_provider_subentry_id, profile_id = parse_model_profile_ref(
                    profile_ref
                )
            except HomeAssistantError:
                continue
            if ref_provider_subentry_id == provider_subentry_id:
                referenced.add(profile_id)
    return referenced


def _provider_validation_placeholders(
    err: ProviderValidationError,
) -> dict[str, str]:
    """Return translation placeholders for provider validation errors."""
    placeholders = {"error_message": err.message}
    if err.status_code is not None:
        placeholders["status_code"] = str(err.status_code)
    return placeholders


def _log_provider_validation_failure(
    *, step: str, model_name: str, err: ProviderValidationError
) -> None:
    """Log provider validation failures without request details or credentials."""
    if err.status_code == 429:
        _LOGGER.warning(
            'Provider validation rate limited during %s for model "%s": '
            "reason=%s status_code=%s",
            step,
            model_name,
            err.reason,
            err.status_code,
        )
        return

    _LOGGER.warning(
        'Provider validation failed during %s for model "%s": reason=%s status_code=%s',
        step,
        model_name,
        err.reason,
        err.status_code,
    )


def _normalise_provider_data(user_input: Mapping[str, Any]) -> dict[str, Any]:
    """Return normalized provider data for storage and validation."""
    data = _flatten_section_data(
        user_input, (_SECTION_ADVANCED_OPTIONS, _SECTION_CUSTOMIZE_MODEL_LIST)
    )
    data[CONF_NAME] = str(data[CONF_NAME]).strip() or DEFAULT_SERVICE_NAME
    data[CONF_BASE_URL] = _normalise_base_url(data)
    headers = _parse_provider_headers(data.get(CONF_PROVIDER_HEADERS))
    if headers:
        data[CONF_PROVIDER_HEADERS] = headers
    else:
        data.pop(CONF_PROVIDER_HEADERS, None)
    try:
        provider_extra_body = _parse_key_value_json_setting(
            data.get(CONF_PROVIDER_EXTRA_BODY, "")
        )
    except ValueError as err:
        raise ProviderValidationError(
            _model_setting_error(_MODEL_SETTING_EXTRA_BODY, str(err)),
            "Enter provider extra body fields one per line using 'key: JSON value'.",
        ) from err
    if provider_extra_body:
        try:
            reject_chat_template_kwargs_in_extra_body(provider_extra_body)
        except HomeAssistantError as err:
            raise ProviderValidationError(
                "chat_template_kwargs_conflict", str(err)
            ) from err
        data[CONF_PROVIDER_EXTRA_BODY] = provider_extra_body
    else:
        data.pop(CONF_PROVIDER_EXTRA_BODY, None)
    if not data[CONF_BASE_URL]:
        data.pop(CONF_BASE_URL, None)
    api_key = data.get(CONF_API_KEY)
    data[CONF_API_KEY] = str(api_key or "").strip()
    data[CONF_CUSTOM_MODEL_NAMES] = _provider_custom_model_names(data)
    return data


def _normalise_provider_model_profiles(
    existing_profiles: Mapping[str, Any],
    model_names: list[str],
    discovered_model_names: Iterable[str],
    *,
    model_labels: Mapping[str, str] | None = None,
    keep_profile_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return provider-owned profile storage synced to provider model names."""
    model_labels = model_labels or {}
    discovered_set = set(discovered_model_names)
    model_set = set(model_names)
    keep_profile_ids = keep_profile_ids or set()
    existing_by_model: dict[str, tuple[str, dict[str, Any]]] = {}
    kept_profiles: dict[str, dict[str, Any]] = {}
    for profile_id, profile in existing_profiles.items():
        if not isinstance(profile_id, str) or not isinstance(profile, Mapping):
            continue
        model_name = profile.get(CONF_MODEL)
        if not isinstance(model_name, str) or not model_name.strip():
            continue
        profile = dict(profile)
        if model_name in model_set:
            existing_by_model.setdefault(model_name, (profile_id, profile))
            continue
        if profile_id in keep_profile_ids:
            model_settings = profile.get(CONF_MODEL_SETTINGS)
            if isinstance(model_settings, Mapping):
                profile[CONF_MODEL_SETTINGS] = _model_settings_from_options(profile)
            else:
                profile.pop(CONF_MODEL_SETTINGS, None)
            profile[CONF_ENABLED] = bool(profile.get(CONF_ENABLED, False))
            kept_profiles[profile_id] = profile

    profiles: dict[str, dict[str, Any]] = dict(kept_profiles)
    for model_name in model_names:
        existing_profile = existing_by_model.get(model_name)
        if existing_profile is None:
            profile_id = uuid4().hex
            profile = {
                "id": profile_id,
                CONF_NAME: model_labels.get(model_name, model_name),
                CONF_MODEL: model_name,
                CONF_ENABLED: False,
                CONF_DISCOVERED: model_name in discovered_set,
            }
        else:
            profile_id, profile = existing_profile
            profile = dict(profile)
        profile["id"] = profile_id
        profile_name = str(profile.get(CONF_NAME) or "").strip()
        if not profile_name or profile_name == model_name:
            profile_name = model_labels.get(model_name, model_name)
        profile[CONF_NAME] = profile_name
        profile[CONF_MODEL] = model_name
        profile[CONF_ENABLED] = bool(profile.get(CONF_ENABLED, False))
        profile[CONF_DISCOVERED] = model_name in discovered_set
        model_settings = profile.get(CONF_MODEL_SETTINGS)
        if isinstance(model_settings, Mapping):
            profile[CONF_MODEL_SETTINGS] = _model_settings_from_options(profile)
        else:
            profile.pop(CONF_MODEL_SETTINGS, None)
        profiles[profile_id] = profile
    return profiles


def _provider_model_profiles_for_discovery_mode(
    existing_profiles: Mapping[str, Any], *, keep_profile_ids: set[str]
) -> dict[str, dict[str, Any]]:
    """Return existing profiles that remain valid before discovery refresh."""
    profiles: dict[str, dict[str, Any]] = {}
    for profile_id, profile in existing_profiles.items():
        if not isinstance(profile_id, str) or not isinstance(profile, Mapping):
            continue
        if (
            not bool(profile.get(CONF_DISCOVERED, False))
            and profile_id not in keep_profile_ids
        ):
            continue
        model_name = profile.get(CONF_MODEL)
        if not isinstance(model_name, str) or not model_name.strip():
            continue
        profile = dict(profile)
        profile["id"] = profile_id
        profile[CONF_MODEL] = model_name
        profile[CONF_ENABLED] = bool(profile.get(CONF_ENABLED, False))
        model_settings = profile.get(CONF_MODEL_SETTINGS)
        if isinstance(model_settings, Mapping):
            profile[CONF_MODEL_SETTINGS] = _model_settings_from_options(profile)
        else:
            profile.pop(CONF_MODEL_SETTINGS, None)
        profiles[profile_id] = profile
    return profiles


def _provider_profile_options(
    data: Mapping[str, Any],
    model_ids: set[str] | None = None,
    *,
    enabled_only: bool = False,
) -> list[SelectOptionDict]:
    """Return provider model profiles as select options."""
    options: list[SelectOptionDict] = []
    profiles = data.get(CONF_MODEL_PROFILES)
    if not isinstance(profiles, Mapping):
        return []
    for profile_id, profile in profiles.items():
        if not isinstance(profile_id, str) or not isinstance(profile, Mapping):
            continue
        model_name = profile.get(CONF_MODEL)
        if not isinstance(model_name, str) or not model_name.strip():
            continue
        enabled = bool(profile.get(CONF_ENABLED, False))
        if enabled_only and not enabled:
            continue
        if model_ids is not None and model_name not in model_ids:
            continue
        label = model_profile_display_name(profile)
        if not enabled:
            label = f"{label} (disabled)"
        options.append(SelectOptionDict(label=label, value=profile_id))
    return options


def _provider_profile_dependents(entry: ConfigEntry, profile_ref: str) -> list[str]:
    """Return conversation and AI task titles that reference one profile."""
    dependents: list[str] = []
    for subentry in entry.subentries.values():
        if subentry.subentry_type not in {
            SUBENTRY_TYPE_CONVERSATION,
            SUBENTRY_TYPE_AI_TASK,
        }:
            continue
        refs = _selected_model_profile_refs(subentry.data)
        if profile_ref in refs:
            dependents.append(subentry.title)
    return dependents


def _provider_profile_selector_schema(
    data: Mapping[str, Any],
    model_ids: set[str] | None = None,
    *,
    enabled_only: bool = False,
) -> vol.Schema:
    """Return a selector schema for existing provider-owned profiles."""
    return vol.Schema(
        {
            vol.Required(_CONF_MODEL_PROFILE_ID): SelectSelector(
                SelectSelectorConfig(
                    options=_provider_profile_options(
                        data, model_ids, enabled_only=enabled_only
                    ),
                    mode=SelectSelectorMode.DROPDOWN,
                )
            )
        }
    )


def _model_profile_edit_schema(
    profile: Mapping[str, Any],
) -> vol.Schema:
    """Return the provider-owned model profile edit schema."""
    options: dict[str, Any] = {
        CONF_NAME: model_profile_display_name(profile),
        CONF_MODEL_PRICING: profile.get(CONF_MODEL_PRICING, {}),
        CONF_MODEL_SETTINGS: profile.get(CONF_MODEL_SETTINGS, {}),
    }
    schema: VolDictType = {
        vol.Required(CONF_NAME, default=options[CONF_NAME]): TextSelector(
            TextSelectorConfig()
        ),
    }
    if not bool(profile.get(CONF_DISCOVERED, False)):
        schema[vol.Required(CONF_MODEL, default=profile.get(CONF_MODEL, ""))] = (
            TextSelector(TextSelectorConfig())
        )
    schema[
        vol.Optional(
            _MODEL_SETTING_TEMPERATURE,
            description={
                "suggested_value": options[CONF_MODEL_SETTINGS].get(
                    _MODEL_SETTING_TEMPERATURE
                )
                if isinstance(options[CONF_MODEL_SETTINGS], Mapping)
                else None
            },
        )
    ] = NumberSelector(NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=0.1))
    schema[vol.Optional(_SECTION_ADVANCED_MODEL_SETTINGS, default={})] = section(
        _model_settings_schema(options), {"collapsed": True}
    )
    schema[vol.Optional(_SECTION_MODEL_PRICING, default={})] = section(
        _model_pricing_schema(options), {"collapsed": True}
    )
    return vol.Schema(schema)


def _validate_provider_data(hass: HomeAssistant, data: Mapping[str, Any]) -> None:
    """Validate provider data that does not require a model."""
    del hass
    if data.get(CONF_PROVIDER_MODE) not in PROVIDER_MODES:
        raise ProviderValidationError(
            "invalid_provider_config",
            f"Unsupported provider mode: {data.get(CONF_PROVIDER_MODE)!r}.",
        )
    _validate_base_url(data)
    if data.get(CONF_PROVIDER_EXTRA_BODY) and not _provider_extra_body_supported(data):
        raise ProviderValidationError(
            "provider_extra_body_unsupported",
            "Extra body is only supported by OpenAI-compatible and Anthropic provider modes.",
        )


def _provider_data_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return if two provider configurations identify the same connection."""
    return _dedupe_data(left) == _dedupe_data(right)


def _mcp_server_select_options(entry: ConfigEntry | None) -> list[SelectOptionDict]:
    """Return configured MCP servers as select options."""
    if entry is None:
        return []
    return [
        SelectOptionDict(label=subentry.title, value=subentry.subentry_id)
        for subentry in entry.subentries.values()
        if subentry.subentry_type == SUBENTRY_TYPE_MCP_SERVER
    ]


def _model_profile_select_options(entry: ConfigEntry | None) -> list[SelectOptionDict]:
    """Return enabled workspace model profiles as select options."""
    if entry is None:
        return []
    options: list[SelectOptionDict] = []
    for provider_subentry in provider_subentries(entry):
        for profile_id, profile in provider_model_profiles(provider_subentry).items():
            if not bool(profile.get(CONF_ENABLED, False)):
                continue
            model_name = profile.get(CONF_MODEL)
            if not isinstance(model_name, str) or not model_name.strip():
                continue
            label = model_profile_display_name(profile)
            options.append(
                SelectOptionDict(
                    label=f"{provider_subentry.title} / {label}",
                    value=model_profile_ref(provider_subentry.subentry_id, profile_id),
                )
            )
    return options


def _normalise_fallback_model_refs(
    raw_refs: object,
) -> list[str]:
    """Return canonical workspace-local fallback refs, preserving order."""
    if isinstance(raw_refs, str) or not isinstance(raw_refs, list):
        return []
    refs: list[str] = []
    for raw_ref in raw_refs:
        if not isinstance(raw_ref, str) or not raw_ref:
            continue
        try:
            provider_subentry_id, profile_id = parse_model_profile_ref(raw_ref)
        except HomeAssistantError:
            continue
        refs.append(model_profile_ref(provider_subentry_id, profile_id))
    return refs


def _fallback_model_profile_select_options(
    hass: HomeAssistant, entry: ConfigEntry | None, selected_refs: object = None
) -> list[SelectOptionDict]:
    """Return workspace-local fallback profile options."""
    del hass
    options = _model_profile_select_options(entry)
    configured_refs = {str(option["value"]) for option in options if "value" in option}
    for ref in _normalise_fallback_model_refs(selected_refs):
        if ref not in configured_refs:
            options.append(SelectOptionDict(label=f"Unavailable / {ref}", value=ref))
    return options


def _selected_model_profile_refs(data: Mapping[str, Any]) -> list[str]:
    """Return selected primary plus ordered fallback profile refs."""
    primary_ref = data.get(CONF_PRIMARY_MODEL_REF)
    if not isinstance(primary_ref, str) or not primary_ref:
        return []
    fallback_refs = data.get(CONF_FALLBACK_MODEL_REFS, [])
    if isinstance(fallback_refs, str) or not isinstance(fallback_refs, list):
        fallback_refs = []
    return [primary_ref, *[item for item in fallback_refs if isinstance(item, str)]]


def _selected_model_profile_error(
    hass: HomeAssistant, entry: ConfigEntry, data: Mapping[str, Any]
) -> str | None:
    """Return a form error for missing or invalid model profile selections."""
    del hass
    primary_ref = data.get(CONF_PRIMARY_MODEL_REF)
    if not isinstance(primary_ref, str) or not primary_ref:
        return "model_profile_required"
    if not configured_model_profile_exists(entry, primary_ref):
        return "model_profile_not_found"
    fallback_refs = _normalise_fallback_model_refs(
        data.get(CONF_FALLBACK_MODEL_REFS, [])
    )
    if primary_ref in fallback_refs:
        return "primary_model_in_fallbacks"
    if len(fallback_refs) != len(set(fallback_refs)):
        return "duplicate_fallback_model"
    for profile_ref in fallback_refs:
        if not configured_model_profile_exists(entry, profile_ref):
            return "model_profile_not_found"
    return None


def _selected_todo_workspace_error(
    hass: HomeAssistant, data: Mapping[str, Any]
) -> str | None:
    """Return a form error for an invalid todo workspace entity."""
    entity_id = data.get(CONF_TODO_LIST_ENTITY_ID)
    if not entity_id:
        return None
    if not isinstance(entity_id, str) or not entity_id.startswith(f"{TODO_DOMAIN}."):
        return "todo_list_not_found"
    state = hass.states.get(entity_id)
    if state is None:
        return "todo_list_not_found"
    supported_features = state.attributes.get("supported_features", 0)
    if not isinstance(supported_features, int):
        return "todo_list_unsupported"
    if (
        supported_features & _TODO_WORKSPACE_REQUIRED_FEATURES
        != _TODO_WORKSPACE_REQUIRED_FEATURES
    ):
        return "todo_list_unsupported"
    return None


def _conversation_schema(
    hass: HomeAssistant,
    options: Mapping[str, Any] | None = None,
    entry: ConfigEntry | None = None,
) -> vol.Schema:
    """Return the conversation subentry schema, pruning unavailable HA APIs."""
    options = dict(options or {})
    model_options = _model_profile_select_options(entry)
    fallback_model_options = _fallback_model_profile_select_options(
        hass, entry, options.get(CONF_FALLBACK_MODEL_REFS, [])
    )
    hass_apis: list[SelectOptionDict] = []
    valid_api_ids: set[str] = set()
    for api in llm.async_get_apis(hass):
        hass_apis.append(SelectOptionDict(label=api.name, value=api.id))
        valid_api_ids.add(api.id)

    if CONF_LLM_HASS_API in options:
        options[CONF_LLM_HASS_API] = [
            api for api in options[CONF_LLM_HASS_API] if api in valid_api_ids
        ]
    schema: VolDictType = {
        vol.Required(
            CONF_AGENT_NAME,
            default=options.get(CONF_AGENT_NAME, DEFAULT_AGENT_NAME),
        ): str,
        vol.Optional(
            CONF_PROMPT,
            description={"suggested_value": options.get(CONF_PROMPT, "")},
        ): TemplateSelector(),
    }
    if model_options:
        schema[
            vol.Required(
                CONF_PRIMARY_MODEL_REF,
                default=options.get(CONF_PRIMARY_MODEL_REF, ""),
            )
        ] = SelectSelector(
            SelectSelectorConfig(
                options=model_options,
                mode=SelectSelectorMode.DROPDOWN,
                translation_key=CONF_PRIMARY_MODEL_REF,
            )
        )
        fallback_schema: VolDictType = {}
        fallback_schema[
            vol.Optional(
                CONF_FALLBACK_MODEL_REFS,
                default=_normalise_fallback_model_refs(
                    options.get(CONF_FALLBACK_MODEL_REFS, [])
                ),
            )
        ] = SelectSelector(
            SelectSelectorConfig(
                options=fallback_model_options,
                multiple=True,
                translation_key=CONF_FALLBACK_MODEL_REFS,
            )
        )
        schema[_section_schema_key(_SECTION_FALLBACK_MODELS, fallback_schema)] = (
            section(vol.Schema(fallback_schema), {"collapsed": True})
        )
    run_settings_schema = _run_settings_schema(options, default_max_iterations=10)
    schema[_section_schema_key(_SECTION_RUN_SETTINGS, run_settings_schema.schema)] = (
        section(run_settings_schema, {"collapsed": True})
    )
    api_schema_key = vol.Optional(CONF_LLM_HASS_API)
    if CONF_LLM_HASS_API in options:
        api_schema_key = vol.Optional(
            CONF_LLM_HASS_API,
            default=options[CONF_LLM_HASS_API],
        )
    hass_control_schema: VolDictType = {}
    hass_control_schema[api_schema_key] = SelectSelector(
        SelectSelectorConfig(options=hass_apis, multiple=True)
    )
    schema[_section_schema_key(_SECTION_HASS_CONTROL, hass_control_schema)] = section(
        vol.Schema(hass_control_schema), {"collapsed": True}
    )
    external_tools_schema: VolDictType = {}
    mcp_servers = _mcp_server_select_options(entry)
    if mcp_servers:
        mcp_schema_key = vol.Optional(CONF_MCP_SERVER_IDS)
        if CONF_MCP_SERVER_IDS in options:
            configured_servers = {
                option["value"] for option in mcp_servers if "value" in option
            }
            mcp_schema_key = vol.Optional(
                CONF_MCP_SERVER_IDS,
                default=[
                    server_id
                    for server_id in options[CONF_MCP_SERVER_IDS]
                    if server_id in configured_servers
                ],
            )
        external_tools_schema[mcp_schema_key] = SelectSelector(
            SelectSelectorConfig(options=mcp_servers, multiple=True)
        )
    external_tools_schema[
        vol.Optional(
            CONF_WEB_FETCH_ENABLED,
            default=bool(options.get(CONF_WEB_FETCH_ENABLED, False)),
        )
    ] = BooleanSelector()
    external_tools_schema[
        vol.Optional(
            CONF_VIRTUAL_WORKSPACE_ENABLED,
            default=options.get(CONF_VIRTUAL_WORKSPACE_ENABLED) is True,
        )
    ] = BooleanSelector()
    schema[_section_schema_key(_SECTION_EXTERNAL_TOOLS, external_tools_schema)] = (
        section(vol.Schema(external_tools_schema), {"collapsed": True})
    )
    skills_schema: VolDictType = {}
    _append_skill_schema_fields(skills_schema, options, entry)
    schema[_section_schema_key(_SECTION_SKILLS, skills_schema)] = section(
        vol.Schema(skills_schema, extra=vol.REMOVE_EXTRA), {"collapsed": True}
    )
    return vol.Schema(schema)


def _model_profile_schema(
    options: Mapping[str, Any] | None = None,
    model_names: Iterable[str] | None = None,
) -> vol.Schema:
    """Return the model profile subentry schema."""
    options = dict(options or {})
    model_settings = options.get(CONF_MODEL_SETTINGS, {})
    if not isinstance(model_settings, Mapping):
        model_settings = {}
    model_schema_key = vol.Required(
        CONF_MODEL,
        default=options.get(CONF_MODEL, ""),
    )
    if model_names:
        model_options = sorted(set(model_names))
        if existing_model := options.get(CONF_MODEL):
            if isinstance(existing_model, str) and existing_model not in model_options:
                model_options.insert(0, existing_model)
        default_model = options.get(CONF_MODEL, model_options[0])
        model_schema_key = vol.Required(CONF_MODEL, default=default_model)
        model_selector = SelectSelector(
            SelectSelectorConfig(
                options=model_options,
                mode=SelectSelectorMode.DROPDOWN,
            )
        )
    else:
        model_selector = TextSelector(TextSelectorConfig())
    return vol.Schema(
        {
            model_schema_key: model_selector,
            vol.Required(
                CONF_NAME,
                default=options.get(CONF_NAME, options.get(CONF_MODEL, "")),
            ): TextSelector(TextSelectorConfig()),
            vol.Optional(
                _MODEL_SETTING_TEMPERATURE,
                description={
                    "suggested_value": model_settings.get(_MODEL_SETTING_TEMPERATURE)
                },
            ): NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=0.1)
            ),
            vol.Optional(_SECTION_ADVANCED_MODEL_SETTINGS, default={}): section(
                _model_settings_schema(options), {"collapsed": True}
            ),
        }
    )


def _model_settings_schema(options: Mapping[str, Any] | None = None) -> vol.Schema:
    """Return the advanced model settings schema."""
    options = dict(options or {})
    model_settings = options.get(CONF_MODEL_SETTINGS, {})
    if not isinstance(model_settings, Mapping):
        model_settings = {}
    parallel_tool_calls_key = vol.Optional(_MODEL_SETTING_PARALLEL_TOOL_CALLS)
    if _MODEL_SETTING_PARALLEL_TOOL_CALLS in model_settings:
        parallel_tool_calls_key = vol.Optional(
            _MODEL_SETTING_PARALLEL_TOOL_CALLS,
            default=model_settings[_MODEL_SETTING_PARALLEL_TOOL_CALLS],
        )
    return vol.Schema(
        {
            parallel_tool_calls_key: BooleanSelector(),
            vol.Optional(
                _MODEL_SETTING_TOP_P,
                description={
                    "suggested_value": model_settings.get(_MODEL_SETTING_TOP_P)
                },
            ): NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=0.1)
            ),
            vol.Optional(
                _MODEL_SETTING_SEED,
                description={
                    "suggested_value": model_settings.get(_MODEL_SETTING_SEED)
                },
            ): NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=1)
            ),
            vol.Optional(
                _MODEL_SETTING_PRESENCE_PENALTY,
                description={
                    "suggested_value": model_settings.get(
                        _MODEL_SETTING_PRESENCE_PENALTY
                    )
                },
            ): NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=0.1)
            ),
            vol.Optional(
                _MODEL_SETTING_FREQUENCY_PENALTY,
                description={
                    "suggested_value": model_settings.get(
                        _MODEL_SETTING_FREQUENCY_PENALTY
                    )
                },
            ): NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=0.1)
            ),
            vol.Optional(
                _MODEL_SETTING_CHAT_TEMPLATE_KWARGS,
                default=_format_chat_template_kwargs(
                    model_settings.get(_MODEL_SETTING_CHAT_TEMPLATE_KWARGS)
                ),
            ): ObjectSelector(
                ObjectSelectorConfig(
                    multiple=True,
                    fields={
                        CONF_CHAT_TEMPLATE_KWARG_KEY: {
                            "selector": {"text": None},
                            "required": True,
                        },
                        CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: {
                            "selector": {"template": None},
                            "required": True,
                        },
                    },
                )
            ),
        }
    )


def _run_settings_schema(
    options: Mapping[str, Any] | None = None,
    *,
    default_max_iterations: int,
) -> vol.Schema:
    """Return per-conversation/task run settings schema."""
    options = dict(options or {})
    return vol.Schema(
        {
            vol.Optional(
                _MODEL_SETTING_MAX_TOKENS,
                description={
                    "suggested_value": options.get(_MODEL_SETTING_MAX_TOKENS)
                },
            ): NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=1)
            ),
            vol.Required(
                _MODEL_SETTING_MAX_ITERATIONS,
                default=options.get(
                    _MODEL_SETTING_MAX_ITERATIONS, default_max_iterations
                ),
            ): NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=1)
            ),
            vol.Required(
                _MODEL_SETTING_TIMEOUT,
                default=options.get(_MODEL_SETTING_TIMEOUT, DEFAULT_TIMEOUT),
            ): NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=0.1)
            ),
            vol.Optional(
                _MODEL_SETTING_THINKING,
                default=_format_thinking_value(options),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=list(_THINKING_OPTIONS),
                    mode=SelectSelectorMode.DROPDOWN,
                    translation_key=_MODEL_SETTING_THINKING,
                )
            ),
        }
    )


def _model_pricing_schema(options: Mapping[str, Any] | None = None) -> vol.Schema:
    """Return the model pricing schema."""
    options = dict(options or {})
    pricing = options.get(CONF_MODEL_PRICING, {})
    if not isinstance(pricing, Mapping):
        pricing = {}
    return vol.Schema(
        {
            vol.Optional(
                _MODEL_PRICING_INPUT,
                description={"suggested_value": pricing.get("input")},
            ): NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step="any")
            ),
            vol.Optional(
                _MODEL_PRICING_OUTPUT,
                description={"suggested_value": pricing.get("output")},
            ): NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step="any")
            ),
            vol.Optional(
                _MODEL_PRICING_CACHE_READ,
                description={"suggested_value": pricing.get("cache_read")},
            ): NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step="any")
            ),
        }
    )


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


def _format_chat_template_kwargs(value: object) -> list[dict[str, str]]:
    """Return stored chat template kwargs in selector-compatible shape."""
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
    try:
        reject_chat_template_kwargs_in_extra_body(parsed)
    except HomeAssistantError as err:
        raise ValueError("chat_template_kwargs_conflict") from err
    return parsed


def _parse_chat_template_kwargs(
    hass: HomeAssistant, value: object
) -> list[dict[str, str]]:
    """Return configured chat template kwargs from selector input."""
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError("invalid_chat_template_kwargs")
    parsed: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("invalid_chat_template_kwargs")
        key = str(item.get(CONF_CHAT_TEMPLATE_KWARG_KEY, "")).strip()
        value_template = item.get(CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE)
        if not key and not value_template:
            continue
        if not key:
            raise ValueError("invalid_chat_template_key")
        if key in seen:
            raise ValueError("duplicate_key")
        if not isinstance(value_template, str) or not value_template.strip():
            raise ValueError("invalid_chat_template")
        try:
            Template(value_template, hass).ensure_valid()
        except TemplateError as err:
            raise ValueError("invalid_chat_template") from err
        seen.add(key)
        parsed.append(
            {
                CONF_CHAT_TEMPLATE_KWARG_KEY: key,
                CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: value_template,
            }
        )
    return parsed


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
            # Blank fields mean "unset" in the HA form and must delete any
            # previously stored Pydantic AI model setting.
            cleared.add(key)
            continue
        try:
            if key in {_MODEL_SETTING_MAX_TOKENS, _MODEL_SETTING_MAX_ITERATIONS}:
                settings[key] = _parse_positive_int_setting(value)
            elif key == _MODEL_SETTING_SEED:
                settings[key] = _parse_non_negative_int_setting(value)
            elif key == _MODEL_SETTING_TIMEOUT:
                settings[key] = _parse_positive_float_setting(value)
            elif key in {
                _MODEL_SETTING_TEMPERATURE,
                _MODEL_SETTING_TOP_P,
                _MODEL_SETTING_PRESENCE_PENALTY,
                _MODEL_SETTING_FREQUENCY_PENALTY,
            }:
                settings[key] = _parse_float_setting(value)
            elif key == _MODEL_SETTING_PARALLEL_TOOL_CALLS:
                if not isinstance(value, bool):
                    raise ValueError
                settings[key] = value
            elif key == _MODEL_SETTING_EXTRA_BODY:
                settings[key] = _parse_key_value_json_setting(value)
            elif key == _MODEL_SETTING_CHAT_TEMPLATE_KWARGS:
                if parsed := _parse_chat_template_kwargs(hass, value):
                    settings[key] = parsed
                else:
                    cleared.add(key)
            elif key == _MODEL_SETTING_THINKING:
                settings[key] = _parse_thinking_setting(value)
        except ValueError as err:
            errors[key] = _model_setting_error(key, str(err))
    return settings, errors, cleared


def _normalise_run_settings(data: dict[str, Any]) -> None:
    """Normalize conversation/task run settings stored directly on subentries."""
    errors: dict[str, str] = {}
    for key in (_MODEL_SETTING_MAX_TOKENS, _MODEL_SETTING_THINKING):
        if _is_blank(data.get(key)):
            data.pop(key, None)
    for key in (_MODEL_SETTING_MAX_TOKENS, _MODEL_SETTING_MAX_ITERATIONS):
        if key in data:
            try:
                data[key] = _parse_positive_int_setting(data[key])
            except ValueError as err:
                errors[key] = _model_setting_error(key, str(err))
    if _MODEL_SETTING_TIMEOUT in data:
        try:
            data[_MODEL_SETTING_TIMEOUT] = _parse_positive_float_setting(
                data[_MODEL_SETTING_TIMEOUT]
            )
        except ValueError as err:
            errors[_MODEL_SETTING_TIMEOUT] = _model_setting_error(
                _MODEL_SETTING_TIMEOUT, str(err)
            )
    if _MODEL_SETTING_THINKING in data:
        try:
            data[_MODEL_SETTING_THINKING] = _parse_thinking_setting(
                data[_MODEL_SETTING_THINKING]
            )
        except ValueError as err:
            errors[_MODEL_SETTING_THINKING] = _model_setting_error(
                _MODEL_SETTING_THINKING, str(err)
            )
    if errors:
        raise RunSettingsValidationError(errors)


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
        "chat_template_kwargs_conflict",
        "duplicate_key",
        "invalid_chat_template",
        "invalid_chat_template_key",
        "invalid_chat_template_kwargs",
        "invalid_json",
        "invalid_key_value",
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


def _conversation_data_from_user_input(
    user_input: Mapping[str, Any],
    options: Mapping[str, Any],
) -> dict[str, Any]:
    """Return conversation fields with model profile references."""
    user_input = _flatten_section_data(
        user_input,
        (
            _SECTION_EXTERNAL_TOOLS,
            _SECTION_FALLBACK_MODELS,
            _SECTION_HASS_CONTROL,
            _SECTION_RUN_SETTINGS,
            _SECTION_SKILLS,
        ),
    )
    data = {key: value for key, value in user_input.items()}
    if not data.get(CONF_LLM_HASS_API):
        data.pop(CONF_LLM_HASS_API, None)
    if not data.get(CONF_MCP_SERVER_IDS):
        data.pop(CONF_MCP_SERVER_IDS, None)
    if not data.get(CONF_WEB_FETCH_ENABLED):
        data.pop(CONF_WEB_FETCH_ENABLED, None)
    if data.get(CONF_VIRTUAL_WORKSPACE_ENABLED) is not True:
        data.pop(CONF_VIRTUAL_WORKSPACE_ENABLED, None)
    if not data.get(CONF_FALLBACK_MODEL_REFS):
        data.pop(CONF_FALLBACK_MODEL_REFS, None)
    if CONF_SKILLS not in user_input and options.get(CONF_SKILLS):
        data[CONF_SKILLS] = options[CONF_SKILLS]
    _normalise_run_settings(data)
    _normalise_skill_selection(data)
    return data


def _merge_model_settings(
    existing: Mapping[str, Any],
    parsed: Mapping[str, Any],
    cleared: set[str],
) -> dict[str, Any]:
    """Return model settings with parsed values applied and cleared keys removed."""
    merged = dict(existing)
    # Subentry setup spans basic and advanced forms, so each save patches the
    # existing model settings instead of replacing the whole mapping blindly.
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


def _store_model_pricing(data: dict[str, Any], model_pricing: Mapping[str, float]) -> None:
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


def _ai_task_data_schema(
    hass: HomeAssistant,
    options: Mapping[str, Any] | None = None,
    entry: ConfigEntry | None = None,
) -> vol.Schema:
    """Return the AI task data subentry schema."""
    options = dict(options or {})
    model_options = _model_profile_select_options(entry)
    fallback_model_options = _fallback_model_profile_select_options(
        hass, entry, options.get(CONF_FALLBACK_MODEL_REFS, [])
    )
    hass_apis: list[SelectOptionDict] = []
    valid_api_ids: set[str] = set()
    for api in llm.async_get_apis(hass):
        hass_apis.append(SelectOptionDict(label=api.name, value=api.id))
        valid_api_ids.add(api.id)

    if CONF_LLM_HASS_API in options:
        options[CONF_LLM_HASS_API] = [
            api for api in options[CONF_LLM_HASS_API] if api in valid_api_ids
        ]
    schema: VolDictType = {
        vol.Required(
            CONF_AI_TASK_NAME,
            default=options.get(CONF_AI_TASK_NAME, DEFAULT_AI_TASK_NAME),
        ): TextSelector(TextSelectorConfig()),
    }
    if model_options:
        schema[
            vol.Required(
                CONF_PRIMARY_MODEL_REF,
                default=options.get(CONF_PRIMARY_MODEL_REF, ""),
            )
        ] = SelectSelector(
            SelectSelectorConfig(
                options=model_options,
                mode=SelectSelectorMode.DROPDOWN,
                translation_key=CONF_PRIMARY_MODEL_REF,
            )
        )
        fallback_schema: VolDictType = {}
        fallback_schema[
            vol.Optional(
                CONF_FALLBACK_MODEL_REFS,
                default=_normalise_fallback_model_refs(
                    options.get(CONF_FALLBACK_MODEL_REFS, [])
                ),
            )
        ] = SelectSelector(
            SelectSelectorConfig(
                options=fallback_model_options,
                multiple=True,
                translation_key=CONF_FALLBACK_MODEL_REFS,
            )
        )
        schema[_section_schema_key(_SECTION_FALLBACK_MODELS, fallback_schema)] = (
            section(vol.Schema(fallback_schema), {"collapsed": True})
        )
    run_settings_schema = _run_settings_schema(options, default_max_iterations=30)
    schema[_section_schema_key(_SECTION_RUN_SETTINGS, run_settings_schema.schema)] = (
        section(run_settings_schema, {"collapsed": True})
    )
    api_schema_key = vol.Optional(CONF_LLM_HASS_API)
    if CONF_LLM_HASS_API in options:
        api_schema_key = vol.Optional(
            CONF_LLM_HASS_API,
            default=options[CONF_LLM_HASS_API],
        )
    hass_control_schema: VolDictType = {}
    hass_control_schema[api_schema_key] = SelectSelector(
        SelectSelectorConfig(options=hass_apis, multiple=True)
    )
    schema[_section_schema_key(_SECTION_HASS_CONTROL, hass_control_schema)] = section(
        vol.Schema(hass_control_schema), {"collapsed": True}
    )
    external_tools_schema: VolDictType = {}
    mcp_servers = _mcp_server_select_options(entry)
    if mcp_servers:
        mcp_schema_key = vol.Optional(CONF_MCP_SERVER_IDS)
        if CONF_MCP_SERVER_IDS in options:
            configured_servers = {
                option["value"] for option in mcp_servers if "value" in option
            }
            mcp_schema_key = vol.Optional(
                CONF_MCP_SERVER_IDS,
                default=[
                    server_id
                    for server_id in options[CONF_MCP_SERVER_IDS]
                    if server_id in configured_servers
                ],
            )
        external_tools_schema[mcp_schema_key] = SelectSelector(
            SelectSelectorConfig(options=mcp_servers, multiple=True)
        )
    todo_schema_key = vol.Optional(CONF_TODO_LIST_ENTITY_ID)
    if CONF_TODO_LIST_ENTITY_ID in options:
        todo_schema_key = vol.Optional(
            CONF_TODO_LIST_ENTITY_ID,
            default=options[CONF_TODO_LIST_ENTITY_ID],
        )
    external_tools_schema[todo_schema_key] = EntitySelector(
        EntitySelectorConfig(domain=TODO_DOMAIN)
    )
    external_tools_schema[
        vol.Optional(
            CONF_WEB_FETCH_ENABLED,
            default=bool(options.get(CONF_WEB_FETCH_ENABLED, False)),
        )
    ] = BooleanSelector()
    external_tools_schema[
        vol.Optional(
            CONF_VIRTUAL_WORKSPACE_ENABLED,
            default=options.get(CONF_VIRTUAL_WORKSPACE_ENABLED) is True,
        )
    ] = BooleanSelector()
    schema[
        vol.Required(
            CONF_OUTPUT_MODE,
            default=normalise_structured_output_mode(
                options.get(CONF_OUTPUT_MODE, DEFAULT_OUTPUT_MODE)
            ),
        )
    ] = SelectSelector(
        SelectSelectorConfig(
            options=list(_OUTPUT_MODE_OPTIONS),
            mode=SelectSelectorMode.DROPDOWN,
            translation_key=CONF_OUTPUT_MODE,
        )
    )
    schema[_section_schema_key(_SECTION_EXTERNAL_TOOLS, external_tools_schema)] = (
        section(vol.Schema(external_tools_schema), {"collapsed": True})
    )
    skills_schema: VolDictType = {}
    _append_skill_schema_fields(skills_schema, options, entry)
    schema[_section_schema_key(_SECTION_SKILLS, skills_schema)] = section(
        vol.Schema(skills_schema, extra=vol.REMOVE_EXTRA), {"collapsed": True}
    )
    return vol.Schema(schema)


def _ai_task_data_from_user_input(
    user_input: Mapping[str, Any],
    options: Mapping[str, Any],
) -> dict[str, Any]:
    """Return AI task subentry data with a selected structured output mode."""
    user_input = _flatten_section_data(
        user_input,
        (
            _SECTION_EXTERNAL_TOOLS,
            _SECTION_FALLBACK_MODELS,
            _SECTION_HASS_CONTROL,
            _SECTION_RUN_SETTINGS,
            _SECTION_SKILLS,
        ),
    )
    data = dict(user_input)
    data.setdefault(
        CONF_OUTPUT_MODE,
        normalise_structured_output_mode(options.get(CONF_OUTPUT_MODE)),
    )
    if not data.get(CONF_MCP_SERVER_IDS):
        data.pop(CONF_MCP_SERVER_IDS, None)
    if not data.get(CONF_WEB_FETCH_ENABLED):
        data.pop(CONF_WEB_FETCH_ENABLED, None)
    if data.get(CONF_VIRTUAL_WORKSPACE_ENABLED) is not True:
        data.pop(CONF_VIRTUAL_WORKSPACE_ENABLED, None)
    if not data.get(CONF_FALLBACK_MODEL_REFS):
        data.pop(CONF_FALLBACK_MODEL_REFS, None)
    if not data.get(CONF_TODO_LIST_ENTITY_ID):
        data.pop(CONF_TODO_LIST_ENTITY_ID, None)
    if not data.get(CONF_LLM_HASS_API):
        data.pop(CONF_LLM_HASS_API, None)
    if CONF_SKILLS not in user_input and options.get(CONF_SKILLS):
        data[CONF_SKILLS] = options[CONF_SKILLS]
    _normalise_run_settings(data)
    _normalise_skill_selection(data)
    return data
