"""Provider data normalization and validation helpers for config flows."""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from typing import Any
from urllib.parse import urlparse

import voluptuous as vol
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import section
from homeassistant.helpers.selector import (
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
from homeassistant.helpers.typing import VolDictType
from homeassistant.util import dt as dt_util

from ..const import (
    CONF_BASE_URL,
    CONF_CUSTOM_MODEL_NAMES,
    CONF_DISCOVERED_MODELS,
    CONF_DISCOVERED_MODELS_AT,
    CONF_DISCOVERED_MODELS_CACHE_KEY,
    CONF_KEY_VALUE_JSON_VALUE,
    CONF_KEY_VALUE_VALUE,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_MODE,
    CONF_PROVIDER_SECRET_HEADER_KEYS,
    DEFAULT_SERVICE_NAME,
    PROVIDER_MODES,
)
from ..models.provider import normalise_base_url
from ..models.provider_validation import ProviderValidationError
from ..runtime.header_metadata import (
    format_header_rows,
    normalize_secret_header_keys,
    parse_header_rows,
)
from ._constants import (
    _BASE_URL_ENDPOINT_PATH_ENDINGS,
    _BASE_URL_ENDPOINT_SUFFIXES,
    _HTTP_HEADER_NAME_PATTERN,
    _MODEL_LIST_CACHE_TTL,
    _MODEL_SETTING_EXTRA_BODY,
    _PROVIDER_EXTRA_BODY_MODES,
    _SECTION_ADVANCED_OPTIONS,
    _SECTION_CUSTOMIZE_MODEL_LIST,
)
from ._key_value_rows import _format_key_value_json_rows
from ._settings_parsing import (
    _model_setting_error,
    _parse_key_value_json_setting,
)
from .helpers import _flatten_section_data, _key_value_rows_selector


def _format_http_headers(
    headers: object, secret_header_keys: object = ()
) -> list[dict[str, str | bool]]:
    """Return HTTP headers in selector-compatible row shape."""
    return format_header_rows(headers, secret_header_keys)


def _parse_provider_headers(value: object) -> tuple[dict[str, str], list[str]]:
    """Return provider HTTP headers from form input."""
    try:
        return _parse_http_headers(value)
    except vol.Invalid as err:
        raise ProviderValidationError(
            "invalid_provider_headers",
            "Add valid HTTP header rows using a header name and value.",
        ) from err


def _parse_http_headers(value: object) -> tuple[dict[str, str], list[str]]:
    """Return HTTP headers from selector rows or an existing mapping."""
    try:
        headers, secret_header_keys = parse_header_rows(value)
    except ValueError as err:
        raise vol.Invalid(str(err)) from err
    if not all(_HTTP_HEADER_NAME_PATTERN.fullmatch(key) for key in headers):
        raise vol.Invalid("invalid_headers")
    return headers, normalize_secret_header_keys(headers, secret_header_keys)


def _provider_connection_schema(options: Mapping[str, Any] | None = None) -> vol.Schema:
    """Return the provider connection form schema."""
    data = _flatten_section_data(options or {}, (_SECTION_ADVANCED_OPTIONS,))
    provider_mode = data.get(CONF_PROVIDER_MODE, PROVIDER_MODES[0])
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
                    default=_format_http_headers(
                        data.get(CONF_PROVIDER_HEADERS),
                        data.get(CONF_PROVIDER_SECRET_HEADER_KEYS),
                    ),
                ): _key_value_rows_selector(
                    CONF_KEY_VALUE_VALUE,
                    {"text": None},
                    key_label="header name",
                    value_label="header value",
                    include_secret_toggle=True,
                    secret_default=False,
                    translation_key=CONF_PROVIDER_HEADERS,
                ),
                vol.Optional(
                    CONF_PROVIDER_EXTRA_BODY,
                    default=_format_key_value_json_rows(
                        data.get(CONF_PROVIDER_EXTRA_BODY)
                    ),
                ): _key_value_rows_selector(
                    CONF_KEY_VALUE_JSON_VALUE,
                    {"template": None},
                    key_label="parameter name",
                    value_label="value",
                    translation_key=CONF_PROVIDER_EXTRA_BODY,
                ),
            }
        ),
        {"collapsed": True},
    )
    return vol.Schema(schema)


def _provider_schema(
    options: Mapping[str, Any] | None = None,
) -> vol.Schema:
    """Return the provider subentry schema."""
    options = dict(options or {})
    return _provider_connection_schema(options)


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
    raw_headers = dict(headers) if isinstance(headers, Mapping) else {}
    headers_key: dict[str, str] | str = {}
    if raw_headers:
        headers_key = sha256(
            json.dumps(
                raw_headers,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    return json.dumps(
        {
            CONF_PROVIDER_MODE: data.get(CONF_PROVIDER_MODE),
            CONF_API_KEY: sha256(str(api_key or "").encode()).hexdigest(),
            CONF_BASE_URL: data.get(CONF_BASE_URL),
            CONF_PROVIDER_HEADERS: headers_key,
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


def _normalise_provider_data(user_input: Mapping[str, Any]) -> dict[str, Any]:
    """Return normalized provider data for storage and validation."""
    data = _flatten_section_data(
        user_input, (_SECTION_ADVANCED_OPTIONS, _SECTION_CUSTOMIZE_MODEL_LIST)
    )
    data[CONF_NAME] = str(data[CONF_NAME]).strip() or DEFAULT_SERVICE_NAME
    data[CONF_BASE_URL] = _normalise_base_url(data.get(CONF_BASE_URL))
    headers, secret_header_keys = _parse_provider_headers(
        data.get(CONF_PROVIDER_HEADERS)
    )
    if headers:
        data[CONF_PROVIDER_HEADERS] = headers
        data[CONF_PROVIDER_SECRET_HEADER_KEYS] = secret_header_keys
    else:
        data.pop(CONF_PROVIDER_HEADERS, None)
        data.pop(CONF_PROVIDER_SECRET_HEADER_KEYS, None)
    try:
        provider_extra_body = _parse_key_value_json_setting(
            data.get(CONF_PROVIDER_EXTRA_BODY, "")
        )
    except ValueError as err:
        raise ProviderValidationError(
            _model_setting_error(_MODEL_SETTING_EXTRA_BODY, str(err)),
            "Add provider extra body rows using a key and JSON value.",
        ) from err
    if provider_extra_body:
        data[CONF_PROVIDER_EXTRA_BODY] = provider_extra_body
    else:
        data.pop(CONF_PROVIDER_EXTRA_BODY, None)
    if not data[CONF_BASE_URL]:
        data.pop(CONF_BASE_URL, None)
    api_key = data.get(CONF_API_KEY)
    data[CONF_API_KEY] = str(api_key or "").strip()
    data[CONF_CUSTOM_MODEL_NAMES] = _provider_custom_model_names(data)
    return data


def _normalise_base_url(url: str | None) -> str | None:
    """Return a normalized base URL if one is configured."""
    if not url:
        return None
    return normalise_base_url(url)


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
            "Extra body is only supported by OpenAI-compatible and Anthropic "
            "provider modes.",
        )


def _provider_data_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return if two provider configurations identify the same connection."""
    return _dedupe_data(left) == _dedupe_data(right)
