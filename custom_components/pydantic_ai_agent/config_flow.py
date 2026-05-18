"""Config flow for Pydantic AI Agent."""

from __future__ import annotations

from collections.abc import AsyncIterable, Iterable, Mapping
from dataclasses import dataclass
import errno
import json
import logging
from pathlib import PurePosixPath
import socket
import ssl
from typing import Any
from urllib.parse import parse_qsl, urlparse

import httpx
from pydantic_ai import (
    ModelRequest,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
)
from pydantic_ai.messages import ModelResponseStreamEvent
from pydantic_ai.direct import model_request_stream
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
import voluptuous as vol

from homeassistant.config_entries import (
    SOURCE_USER,
    ConfigEntry,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    BooleanSelector,
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

from ._redaction import redact_data
from .const import (
    CONF_AGENT_NAME,
    CONF_AI_TASK_NAME,
    CONF_BASE_URL,
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
    CONF_CHAT_TEMPLATE_KWARGS,
    CONF_CONFIGURE_ADVANCED_MODEL_SETTINGS,
    CONF_ENABLE_SKILLS,
    CONF_ENABLE_SKILL_SCRIPT_EXECUTION,
    CONF_FALLBACK_MODEL_SUBENTRY_IDS,
    CONF_LOGFIRE_INCLUDE_CONTENT,
    CONF_LOGFIRE_TOKEN,
    CONF_MAX_ITERATIONS,
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_DEFERRED_LOADING,
    CONF_MCP_HEADERS,
    CONF_MCP_INCLUDE_RETURN_SCHEMA,
    CONF_MCP_SERVER_IDS,
    CONF_MCP_URL,
    CONF_MODEL,
    CONF_MODEL_SETTINGS,
    CONF_MODEL_SUBENTRY_ID,
    CONF_OUTPUT_MODE,
    CONF_PROMPT,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_MODE,
    CONF_SKILLS,
    CONF_SKILLS_FOLDER,
    CONF_WEB_FETCH_ENABLED,
    DEFAULT_AGENT_NAME,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_OUTPUT_MODE,
    DEFAULT_SERVICE_NAME,
    DEFAULT_SKILLS_FOLDER,
    DEFAULT_TIMEOUT,
    DOMAIN,
    OUTPUT_MODE_NATIVE,
    OUTPUT_MODE_PROMPTED,
    OUTPUT_MODE_TOOL,
    PROVIDER_MODES,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE_GEMINI,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
    STRUCTURED_OUTPUT_MODES,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_MCP_SERVER,
    SUBENTRY_TYPE_MODEL,
    default_conversation_options,
)
from .chat_template_kwargs import (
    reject_chat_template_kwargs_in_extra_body,
    render_chat_template_kwargs,
)
from .mcp import (
    MCPValidationError,
    async_discover_mcp_tools_from_config,
    async_validate_mcp_url,
    normalise_mcp_url,
    parse_allowed_tools,
    parse_mcp_headers,
)
from .provider import (
    anthropic_model,
    google_gemini_model,
    list_anthropic_model_names,
    list_google_gemini_model_names,
    normalise_base_url,
    openai_compatible_completions_model_from_config,
    openai_compatible_client_from_config,
    openai_compatible_responses_model_from_config,
)
from .skills import (
    AvailableSkill,
    async_available_skills,
    selected_available_skill_names,
    skills_folder_path,
)
from .structured_output import (
    structured_model_request_parameters,
    structured_output_mode as normalise_structured_output_mode,
    structured_output_name,
)

_LOGGER = logging.getLogger(__name__)

_MCP_TOOL_DESCRIPTION_LABEL_MAX_LENGTH = 80

_RECONFIGURABLE_MODEL_VALIDATION_REASONS = {
    "invalid_model",
    "invalid_provider_config",
    "model_does_not_support_streaming",
    "permission_denied",
}

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

_MODEL_SETTING_MAX_TOKENS = "max_tokens"
_MODEL_SETTING_MAX_ITERATIONS = CONF_MAX_ITERATIONS
_MODEL_SETTING_TEMPERATURE = "temperature"
_MODEL_SETTING_TOP_P = "top_p"
_MODEL_SETTING_TIMEOUT = "timeout"
_MODEL_SETTING_PARALLEL_TOOL_CALLS = "parallel_tool_calls"
_MODEL_SETTING_SEED = "seed"
_MODEL_SETTING_PRESENCE_PENALTY = "presence_penalty"
_MODEL_SETTING_FREQUENCY_PENALTY = "frequency_penalty"
_MODEL_SETTING_THINKING = "thinking"
_MODEL_SETTING_EXTRA_BODY = "extra_body"
_MODEL_SETTING_CHAT_TEMPLATE_KWARGS = CONF_CHAT_TEMPLATE_KWARGS
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
    _MODEL_SETTING_THINKING,
}
_ADVANCED_MODEL_SETTING_KEYS = {
    _MODEL_SETTING_MAX_TOKENS,
    _MODEL_SETTING_MAX_ITERATIONS,
    _MODEL_SETTING_TOP_P,
    _MODEL_SETTING_TIMEOUT,
    _MODEL_SETTING_PARALLEL_TOOL_CALLS,
    _MODEL_SETTING_SEED,
    _MODEL_SETTING_PRESENCE_PENALTY,
    _MODEL_SETTING_FREQUENCY_PENALTY,
    _MODEL_SETTING_EXTRA_BODY,
    _MODEL_SETTING_CHAT_TEMPLATE_KWARGS,
}
_REMOVED_MODEL_SETTING_KEYS = {"extra_headers"}
_THINKING_OPTIONS = ("", "true", "false", "minimal", "low", "medium", "high", "xhigh")
_OUTPUT_MODE_OPTIONS = tuple(
    SelectOptionDict(value=value, label=value) for value in STRUCTURED_OUTPUT_MODES
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


@dataclass(slots=True)
class ProviderValidationError(Exception):
    """Provider validation failed with a translation-ready reason."""

    reason: str
    message: str
    status_code: int | None = None


def _format_http_headers(headers: object) -> str:
    """Return HTTP headers as one ``Header-Name: value`` line each."""
    if headers is None:
        return ""
    if isinstance(headers, str):
        return headers
    if not isinstance(headers, Mapping):
        return ""
    return "\n".join(f"{name}: {headers[name]}" for name in sorted(headers))


def _parse_provider_headers(value: object) -> dict[str, str]:
    """Return provider HTTP headers from form input."""
    try:
        return parse_mcp_headers(value)
    except vol.Invalid as err:
        raise ProviderValidationError(
            "invalid_provider_headers",
            "Enter HTTP headers one per line using 'Header-Name: value'.",
        ) from err


def _base_schema(user_input: dict[str, Any] | None = None) -> vol.Schema:
    """Return the provider connection schema."""
    data = user_input or {}
    provider_mode = data.get(CONF_PROVIDER_MODE, PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS)
    schema: VolDictType = {
        vol.Required(CONF_NAME, default=DEFAULT_SERVICE_NAME): str,
        vol.Required(CONF_PROVIDER_MODE, default=provider_mode): SelectSelector(
            SelectSelectorConfig(
                options=list(PROVIDER_MODES),
                mode=SelectSelectorMode.DROPDOWN,
                translation_key=CONF_PROVIDER_MODE,
            )
        ),
        vol.Required(CONF_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
    }
    schema[vol.Optional(CONF_BASE_URL)] = str
    schema[
        vol.Optional(
            CONF_PROVIDER_HEADERS,
            default=_format_http_headers(data.get(CONF_PROVIDER_HEADERS)),
        )
    ] = TextSelector(TextSelectorConfig(multiline=True))
    schema[vol.Optional(CONF_LOGFIRE_TOKEN)] = TextSelector(
        TextSelectorConfig(type=TextSelectorType.PASSWORD)
    )
    schema[
        vol.Optional(
            CONF_LOGFIRE_INCLUDE_CONTENT,
            default=bool(data.get(CONF_LOGFIRE_INCLUDE_CONTENT, False)),
        )
    ] = BooleanSelector()
    return vol.Schema(schema)


def _provider_form_suggested_values(data: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return provider form suggested values with text-only header fields."""
    values = dict(data or {})
    if CONF_PROVIDER_HEADERS in values:
        values[CONF_PROVIDER_HEADERS] = _format_http_headers(
            values[CONF_PROVIDER_HEADERS]
        )
    return values


def _normalise_base_url(data: Mapping[str, Any]) -> str | None:
    """Return a normalized base URL if one is configured."""
    return normalise_base_url(data.get(CONF_BASE_URL))


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
    return dedupe


async def _validate_configured_models_for_provider_update(
    hass: HomeAssistant,
    entry: ConfigEntry,
    data: Mapping[str, Any],
    step: str,
    skip_reconfigurable_model_errors: bool,
) -> tuple[dict[str, str], dict[str, str]]:
    """Probe existing subentry models against new provider settings."""
    for model, model_settings, output_mode in _configured_model_probes(entry):
        try:
            if output_mode is None:
                await async_probe_model(hass, data, model, model_settings)
            else:
                await async_probe_model(
                    hass,
                    data,
                    model,
                    model_settings,
                    structured_output_mode=output_mode,
                )
        except ProviderValidationError as err:
            _log_provider_validation_failure(step=step, model_name=model, err=err)
            if (
                skip_reconfigurable_model_errors
                and err.reason in _RECONFIGURABLE_MODEL_VALIDATION_REASONS
            ):
                continue
            return {"base": err.reason}, _provider_validation_placeholders(err)
        except Exception:
            _LOGGER.exception("Unexpected exception validating provider")
            return {"base": "unknown"}, {}
    return {}, {}


def _configured_model_probes(
    entry: ConfigEntry,
) -> list[tuple[str, dict[str, Any], str | None]]:
    """Return unique model profile probes required by configured subentries."""
    model_profiles = {
        subentry.subentry_id: subentry
        for subentry in entry.subentries.values()
        if subentry.subentry_type == SUBENTRY_TYPE_MODEL
    }
    probes: list[tuple[str, dict[str, Any], str | None]] = []
    seen: set[tuple[str, str, str | None]] = set()

    def add_probe(profile_id: str, output_mode: str | None) -> None:
        profile = model_profiles.get(profile_id)
        if profile is None:
            return
        model = profile.data.get(CONF_MODEL)
        if not isinstance(model, str) or not model:
            return
        settings = profile.data.get(CONF_MODEL_SETTINGS)
        model_settings = dict(settings) if isinstance(settings, Mapping) else {}
        provider_settings = dict(model_settings)
        provider_settings.pop(_MODEL_SETTING_MAX_ITERATIONS, None)
        dedupe_key = (
            model,
            json.dumps(provider_settings, sort_keys=True, separators=(",", ":")),
            output_mode,
        )
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        probes.append((model, model_settings, output_mode))

    for profile_id in model_profiles:
        add_probe(profile_id, None)

    for subentry in entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_AI_TASK:
            continue
        output_mode = normalise_structured_output_mode(
            subentry.data.get(CONF_OUTPUT_MODE)
        )
        profile_ids = _selected_model_profile_ids(subentry.data)
        if not profile_ids:
            _LOGGER.warning(
                "Skipping legacy AI task subentry without model profile during provider validation: %s",
                subentry.subentry_id,
            )
            continue
        for profile_id in profile_ids:
            add_probe(profile_id, output_mode)
    return probes


def _provider_validation_placeholders(
    err: ProviderValidationError,
) -> dict[str, str]:
    """Return translation placeholders for provider validation errors."""
    placeholders = {"error_message": err.message}
    if err.status_code is not None:
        placeholders["status_code"] = str(err.status_code)
    return placeholders


def _mcp_validation_placeholders(err: MCPValidationError) -> dict[str, str]:
    """Return translation placeholders for MCP validation errors."""
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


def _format_metadata(metadata: object) -> str:
    """Return redacted, bounded provider metadata for config-flow display."""
    redacted = redact_data(metadata, _SENSITIVE_METADATA_KEYS)
    formatted = repr(redacted)
    if len(formatted) > _MAX_METADATA_REPR_LENGTH:
        return f"{formatted[:_MAX_METADATA_REPR_LENGTH]}..."
    return formatted


def _status_label(status_code: int) -> str:
    """Return a user-facing category for an HTTP status code."""
    if label := _HTTP_STATUS_LABELS.get(status_code):
        return label
    if 500 <= status_code <= 599:
        return "provider server issue"
    return "HTTP error"


def _format_http_error(err: ModelHTTPError) -> str:
    """Return a compact user-facing message for a provider HTTP error."""
    message = (
        f"The provider returned error {err.status_code} ({_status_label(err.status_code)}) "
        f'for model "{err.model_name}".'
    )
    if isinstance(err.body, Mapping) and (metadata := err.body.get("metadata")):
        message = f"{message} Metadata: {_format_metadata(metadata)}."
    return message


def _iter_exception_chain(err: BaseException) -> list[BaseException]:
    """Return an exception and its causes/contexts without looping forever."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = err
    while current is not None and id(current) not in seen and len(chain) < 8:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def _format_connection_error(err: BaseException) -> str | None:
    """Return a well-defined connection error message if one can be identified."""
    for item in _iter_exception_chain(err):
        if isinstance(item, TimeoutError | httpx.TimeoutException):
            return "Request timed out."
        if isinstance(item, socket.gaierror):
            return "Host not found."
        if isinstance(item, ssl.SSLError):
            return "TLS error."
        if isinstance(item, OSError):
            if item.errno == errno.ECONNREFUSED:
                return "Connection refused."
            if item.errno in (errno.ENETUNREACH, errno.EHOSTUNREACH):
                return "Network unreachable."
        if isinstance(item, httpx.ConnectError):
            return "Connection failed."
    return None


def _format_api_error(err: ModelAPIError) -> ProviderValidationError:
    """Map a non-HTTP model API error to a config-flow validation error."""
    if connection_message := _format_connection_error(err):
        reason = (
            "timeout"
            if connection_message == "Request timed out."
            else "cannot_connect"
        )
        return ProviderValidationError(reason, connection_message)
    return ProviderValidationError(
        "provider_error",
        f'The provider returned an API error for model "{err.model_name}".',
    )


def _map_http_error(err: ModelHTTPError) -> ProviderValidationError:
    """Map a model HTTP error to a config-flow validation error."""
    status_code = err.status_code
    if status_code == 401:
        reason = "invalid_auth"
    elif status_code == 403:
        reason = "permission_denied"
    elif status_code == 404:
        reason = "invalid_model"
    elif status_code in (408, 504):
        reason = "timeout"
    elif status_code == 429:
        reason = "rate_limited"
    elif status_code == 400:
        # OpenAI-compatible providers often report unknown models as 400 instead
        # of 404, so both statuses drive the same reconfigure path.
        reason = "invalid_model"
    else:
        reason = "provider_error"
    return ProviderValidationError(reason, _format_http_error(err), status_code)


def _map_structured_http_error(
    err: ModelHTTPError, output_mode: str
) -> ProviderValidationError:
    """Map structured-output probe HTTP errors to capability errors."""
    if err.status_code == 400:
        return ProviderValidationError(
            "unsupported_output_mode",
            (
                f'Model "{err.model_name}" rejected structured output mode '
                f'"{output_mode}". Try a different structured output mode or a '
                "model/provider that supports this mode."
            ),
            err.status_code,
        )
    return _map_http_error(err)


def _openai_compatible_model(
    hass: HomeAssistant, data: Mapping[str, Any], model_name: str
) -> Any:
    """Build a Pydantic AI model for validation."""
    provider_mode = data[CONF_PROVIDER_MODE]
    try:
        if provider_mode == PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS:
            return openai_compatible_completions_model_from_config(
                hass, data, model_name
            )
        if provider_mode == PROVIDER_OPENAI_COMPATIBLE_RESPONSES:
            return openai_compatible_responses_model_from_config(hass, data, model_name)
        kwargs = {
            "api_key": data[CONF_API_KEY],
            "base_url": _normalise_base_url(data),
            "headers": dict(data.get(CONF_PROVIDER_HEADERS, {})),
            "model_name": model_name,
        }
        if provider_mode == PROVIDER_ANTHROPIC:
            return anthropic_model(hass, **kwargs)
        if provider_mode == PROVIDER_GOOGLE_GEMINI:
            return google_gemini_model(hass, **kwargs)
    except ValueError as err:
        raise ProviderValidationError("invalid_model", str(err)) from err
    raise ProviderValidationError(
        "invalid_provider_config", f"Unsupported provider mode: {provider_mode!r}."
    )


async def async_list_provider_model_names(
    hass: HomeAssistant, data: Mapping[str, Any]
) -> list[str]:
    """Return model names advertised by the configured provider."""
    provider_mode = data[CONF_PROVIDER_MODE]
    if provider_mode == PROVIDER_ANTHROPIC:
        return await list_anthropic_model_names(hass, data, timeout=DEFAULT_TIMEOUT)
    if provider_mode == PROVIDER_GOOGLE_GEMINI:
        return await list_google_gemini_model_names(hass, data, timeout=DEFAULT_TIMEOUT)
    client = openai_compatible_client_from_config(hass, data)
    return await client.models.list(timeout=DEFAULT_TIMEOUT)


def _structured_probe_request_parameters(
    output_mode: str,
) -> ModelRequestParameters:
    """Return request parameters for a structured-output capability probe."""
    return structured_model_request_parameters(
        function_tools=[],
        output_mode=output_mode,
        output_name=_STRUCTURED_PROBE_OUTPUT_NAME,
        json_schema=_STRUCTURED_PROBE_SCHEMA,
    )


async def async_probe_model(
    hass: HomeAssistant,
    data: Mapping[str, Any],
    model_name: str,
    model_settings: Mapping[str, Any] | None = None,
    *,
    structured_output_mode: str | None = None,
) -> None:
    """Probe model access with the same streaming path used at runtime."""
    try:
        settings = dict(model_settings or {})
        settings.pop(_MODEL_SETTING_MAX_ITERATIONS, None)
        chat_template_kwargs = settings.pop(_MODEL_SETTING_CHAT_TEMPLATE_KWARGS, None)
        reject_chat_template_kwargs_in_extra_body(
            settings.get(_MODEL_SETTING_EXTRA_BODY)
        )
        if rendered_kwargs := render_chat_template_kwargs(hass, chat_template_kwargs):
            extra_body = dict(settings.get(_MODEL_SETTING_EXTRA_BODY) or {})
            extra_body[CONF_CHAT_TEMPLATE_KWARGS] = rendered_kwargs
            settings[_MODEL_SETTING_EXTRA_BODY] = extra_body
        settings.setdefault(_MODEL_SETTING_TIMEOUT, DEFAULT_TIMEOUT)
        model = _openai_compatible_model(hass, data, model_name)
        model_request_parameters = None
        if structured_output_mode is not None:
            output_mode = normalise_structured_output_mode(structured_output_mode)
            model_request_parameters = _structured_probe_request_parameters(output_mode)
        messages = [
            ModelRequest.user_text_prompt(
                (
                    'Reply with exactly "OK". No explanation.'
                    if structured_output_mode is None
                    else 'Return structured data where "ok" is true.'
                ),
                instructions=(
                    "Reply only with OK."
                    if structured_output_mode is None
                    else "Return only data matching the requested schema."
                ),
            )
        ]
        model_settings_obj = ModelSettings(**settings)
        async with model_request_stream(
            model,
            messages,
            model_settings=model_settings_obj,
            model_request_parameters=model_request_parameters,
        ) as stream:
            if structured_output_mode is not None:
                await _validate_structured_probe_stream(
                    stream,
                    normalise_structured_output_mode(structured_output_mode),
                )
                return
            saw_event = False
            async for _event in stream:
                saw_event = True
            if not saw_event:
                raise ProviderValidationError(
                    "provider_error",
                    "The provider returned an empty streamed response.",
                )
            return
    except ModelHTTPError as err:
        if structured_output_mode is not None:
            raise _map_structured_http_error(
                err, normalise_structured_output_mode(structured_output_mode)
            ) from err
        raise _map_http_error(err) from err
    except ModelAPIError as err:
        raise _format_api_error(err) from err
    except NotImplementedError as err:
        raise ProviderValidationError(
            "model_does_not_support_streaming", str(err)
        ) from err
    except UnexpectedModelBehavior as err:
        raise ProviderValidationError("provider_error", str(err)) from err
    except (TimeoutError, httpx.TimeoutException) as err:
        raise ProviderValidationError("timeout", "Request timed out.") from err
    except (ImportError, UserError) as err:
        raise ProviderValidationError("invalid_provider_config", str(err)) from err
    except HomeAssistantError as err:
        raise ProviderValidationError("invalid_provider_config", str(err)) from err


async def _validate_structured_probe_stream(
    stream: AsyncIterable[ModelResponseStreamEvent],
    output_mode: str,
) -> None:
    """Validate that structured-output probing returns schema data."""
    text_parts: list[str] = []
    output_tool_data: object | None = None
    async for event in stream:
        if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
            text_parts.append(event.part.content)
        elif isinstance(event, PartDeltaEvent) and isinstance(
            event.delta, TextPartDelta
        ):
            text_parts.append(event.delta.content_delta)
        elif (
            output_mode == OUTPUT_MODE_TOOL
            and isinstance(event, PartEndEvent)
            and isinstance(event.part, ToolCallPart)
            and event.part.tool_name == _STRUCTURED_PROBE_OUTPUT_NAME
        ):
            output_tool_data = event.part.args
        elif (
            isinstance(event, PartEndEvent)
            and isinstance(event.part, TextPart)
            and not text_parts
        ):
            text_parts.append(event.part.content)

    if output_tool_data is not None:
        data = _structured_probe_data_from_tool_args(output_tool_data)
    elif text_parts:
        try:
            data = json.loads("".join(text_parts))
        except json.JSONDecodeError as err:
            raise ProviderValidationError(
                "invalid_provider_config",
                _invalid_structured_output_message(output_mode),
            ) from err
    else:
        raise ProviderValidationError(
            "provider_error", "The provider returned an empty structured response."
        )

    if not isinstance(data, Mapping) or data.get("ok") is not True:
        raise ProviderValidationError(
            "invalid_provider_config",
            "The provider returned structured output that did not match the schema.",
        )


def _structured_probe_data_from_tool_args(args: object) -> object:
    """Return parsed output-tool arguments for a structured probe."""
    if isinstance(args, Mapping):
        return args
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError as err:
            raise ProviderValidationError(
                "invalid_provider_config",
                _invalid_structured_output_message(OUTPUT_MODE_TOOL),
            ) from err
    return None


def _invalid_structured_output_message(output_mode: str) -> str:
    """Return a validation error for malformed structured output."""
    if output_mode == OUTPUT_MODE_NATIVE:
        return "The provider did not return valid native structured output."
    if output_mode == OUTPUT_MODE_PROMPTED:
        return "The provider did not return valid prompted structured output."
    return "The provider did not return valid tool structured output."


def _normalise_provider_data(user_input: Mapping[str, Any]) -> dict[str, Any]:
    """Return normalized provider data for storage and validation."""
    data = dict(user_input)
    data.pop(CONF_SKILLS_FOLDER, None)
    data.pop(CONF_ENABLE_SKILL_SCRIPT_EXECUTION, None)
    data.pop(CONF_ENABLE_SKILLS, None)
    data.pop(CONF_SKILLS, None)
    data[CONF_BASE_URL] = _normalise_base_url(data)
    headers = _parse_provider_headers(data.get(CONF_PROVIDER_HEADERS))
    if headers:
        data[CONF_PROVIDER_HEADERS] = headers
    else:
        data.pop(CONF_PROVIDER_HEADERS, None)
    if not data[CONF_BASE_URL]:
        data.pop(CONF_BASE_URL, None)
    token = data.get(CONF_LOGFIRE_TOKEN)
    if isinstance(token, str):
        token = token.strip()
    if token:
        data[CONF_LOGFIRE_TOKEN] = token
        data[CONF_LOGFIRE_INCLUDE_CONTENT] = bool(
            data.get(CONF_LOGFIRE_INCLUDE_CONTENT, False)
        )
    else:
        data.pop(CONF_LOGFIRE_TOKEN, None)
        data.pop(CONF_LOGFIRE_INCLUDE_CONTENT, None)
    return data


def _normalise_skills_folder(folder: object) -> str:
    """Return a canonical skills folder path for storage."""
    configured = str(folder or DEFAULT_SKILLS_FOLDER).strip() or DEFAULT_SKILLS_FOLDER
    path = PurePosixPath(configured)
    if not path.is_absolute() and path.parts and path.parts[0] == "skills":
        return f"/config/{path.as_posix()}".rstrip("/")
    return configured.rstrip("/")


def _validate_provider_data(hass: HomeAssistant, data: Mapping[str, Any]) -> None:
    """Validate provider data that does not require a model."""
    if data.get(CONF_PROVIDER_MODE) not in PROVIDER_MODES:
        raise ProviderValidationError(
            "invalid_provider_config",
            f"Unsupported provider mode: {data.get(CONF_PROVIDER_MODE)!r}.",
        )


def _validate_skills_folder(hass: HomeAssistant, folder: object) -> None:
    """Validate that skills folders stay inside Home Assistant config."""
    try:
        skills_folder_path(hass, folder)
    except ValueError as err:
        raise ProviderValidationError(
            "invalid_skills_folder",
            "Skills folder must be /config/skills or one of its subfolders.",
        ) from err


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
    """Return configured model profiles as select options."""
    if entry is None:
        return []
    return [
        SelectOptionDict(label=subentry.title, value=subentry.subentry_id)
        for subentry in entry.subentries.values()
        if subentry.subentry_type == SUBENTRY_TYPE_MODEL
    ]


def _selected_model_profile_ids(data: Mapping[str, Any]) -> list[str]:
    """Return selected primary plus ordered fallback profile IDs."""
    primary_id = data.get(CONF_MODEL_SUBENTRY_ID)
    if not isinstance(primary_id, str) or not primary_id:
        return []
    fallback_ids = data.get(CONF_FALLBACK_MODEL_SUBENTRY_IDS, [])
    if isinstance(fallback_ids, str) or not isinstance(fallback_ids, list):
        fallback_ids = []
    return [primary_id, *[item for item in fallback_ids if isinstance(item, str)]]


def _selected_model_profile_error(
    entry: ConfigEntry, data: Mapping[str, Any]
) -> str | None:
    """Return a form error for missing or invalid model profile selections."""
    primary_id = data.get(CONF_MODEL_SUBENTRY_ID)
    if not isinstance(primary_id, str) or not primary_id:
        return "model_profile_required"
    fallback_ids = data.get(CONF_FALLBACK_MODEL_SUBENTRY_IDS, [])
    if isinstance(fallback_ids, str) or not isinstance(fallback_ids, list):
        fallback_ids = []
    if primary_id in fallback_ids:
        return "primary_model_in_fallbacks"
    if len(fallback_ids) != len(set(fallback_ids)):
        return "duplicate_fallback_model"
    for profile_id in [primary_id, *fallback_ids]:
        subentry = entry.subentries.get(profile_id)
        if subentry is None or subentry.subentry_type != SUBENTRY_TYPE_MODEL:
            return "model_profile_not_found"
    return None


def _skill_select_options(
    available_skills: list[AvailableSkill],
) -> list[SelectOptionDict]:
    """Return discovered skills as select options."""
    return [
        SelectOptionDict(label=skill.name, value=skill.name)
        for skill in available_skills
    ]


def _hidden_configured_skill_names(
    configured: object, available_skills: list[AvailableSkill]
) -> list[str]:
    """Return configured skill names that are not currently selectable."""
    available_names = {skill.name for skill in available_skills}
    if not configured:
        return []
    if isinstance(configured, str):
        return [] if configured in available_names else [configured]
    if not isinstance(configured, Iterable):
        return []
    return [name for name in configured if name not in available_names]


def _merge_submitted_skills_with_hidden(
    user_input: Mapping[str, Any],
    options: Mapping[str, Any],
    available_skills: list[AvailableSkill],
) -> list[str]:
    """Return submitted selectable skills plus hidden existing selections."""
    submitted = user_input.get(CONF_SKILLS, [])
    if isinstance(submitted, str):
        selected = [submitted]
    else:
        selected = list(submitted or [])
    if not selected:
        return []
    return selected + _hidden_configured_skill_names(
        options.get(CONF_SKILLS), available_skills
    )


def _normalise_skill_settings(data: dict[str, Any]) -> None:
    """Normalize and prune per-agent skill settings in place."""
    enable_skills = bool(data.get(CONF_ENABLE_SKILLS, False))
    data[CONF_SKILLS_FOLDER] = _normalise_skills_folder(data.get(CONF_SKILLS_FOLDER))
    data[CONF_ENABLE_SKILL_SCRIPT_EXECUTION] = bool(
        data.get(CONF_ENABLE_SKILL_SCRIPT_EXECUTION, False)
    )
    if not enable_skills:
        data.pop(CONF_ENABLE_SKILLS, None)
        data.pop(CONF_SKILLS, None)
        data.pop(CONF_SKILLS_FOLDER, None)
        data.pop(CONF_ENABLE_SKILL_SCRIPT_EXECUTION, None)
        return
    data[CONF_ENABLE_SKILLS] = True
    if data[CONF_SKILLS_FOLDER] == DEFAULT_SKILLS_FOLDER:
        data.pop(CONF_SKILLS_FOLDER, None)
    if not data[CONF_ENABLE_SKILL_SCRIPT_EXECUTION]:
        data.pop(CONF_ENABLE_SKILL_SCRIPT_EXECUTION, None)


def _skill_source(data: Mapping[str, Any]) -> tuple[bool, str, bool]:
    """Return the fields that determine selectable skills for one agent."""
    return (
        bool(data.get(CONF_ENABLE_SKILLS, False)),
        _normalise_skills_folder(data.get(CONF_SKILLS_FOLDER)),
        bool(data.get(CONF_ENABLE_SKILL_SCRIPT_EXECUTION, False)),
    )


def _append_skill_schema_fields(
    schema: VolDictType,
    options: Mapping[str, Any],
    available_skills: list[AvailableSkill] | None,
) -> None:
    """Append per-agent skill controls to a subentry form schema."""
    enable_skills = bool(options.get(CONF_ENABLE_SKILLS, False))
    schema[
        vol.Optional(
            CONF_ENABLE_SKILLS,
            default=enable_skills,
        )
    ] = BooleanSelector()
    schema[
        vol.Required(
            CONF_SKILLS_FOLDER,
            default=options.get(CONF_SKILLS_FOLDER, DEFAULT_SKILLS_FOLDER),
        )
    ] = TextSelector(TextSelectorConfig())
    schema[
        vol.Optional(
            CONF_ENABLE_SKILL_SCRIPT_EXECUTION,
            default=bool(options.get(CONF_ENABLE_SKILL_SCRIPT_EXECUTION, False)),
        )
    ] = BooleanSelector()
    skill_options = _skill_select_options(available_skills or [])
    if not skill_options:
        return
    skills_schema_key = vol.Optional(CONF_SKILLS)
    if CONF_SKILLS in options:
        selected_skill_names = selected_available_skill_names(
            options[CONF_SKILLS], available_skills or []
        )
        skills_schema_key = vol.Optional(
            CONF_SKILLS,
            default=selected_skill_names,
        )
    schema[skills_schema_key] = SelectSelector(
        SelectSelectorConfig(options=skill_options, multiple=True)
    )


def _selected_mcp_server_error(
    entry: ConfigEntry, data: Mapping[str, Any]
) -> str | None:
    """Return a form error for selected MCP servers that cannot run."""
    for server_id in data.get(CONF_MCP_SERVER_IDS, []):
        subentry = entry.subentries.get(server_id)
        if subentry is None or subentry.subentry_type != SUBENTRY_TYPE_MCP_SERVER:
            return "mcp_server_not_found"
        if not parse_allowed_tools(subentry.data.get(CONF_MCP_ALLOWED_TOOLS)):
            return "mcp_tools_not_allowlisted"
    return None


def _conversation_schema(
    hass: HomeAssistant,
    options: Mapping[str, Any] | None = None,
    entry: ConfigEntry | None = None,
    available_skills: list[AvailableSkill] | None = None,
) -> vol.Schema:
    """Return the conversation subentry schema, pruning unavailable HA APIs."""
    options = dict(options or {})
    model_options = _model_profile_select_options(entry)
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
        configured_profiles = {option["value"] for option in model_options}
        schema[
            vol.Required(
                CONF_MODEL_SUBENTRY_ID,
                default=options.get(CONF_MODEL_SUBENTRY_ID, ""),
            )
        ] = SelectSelector(
            SelectSelectorConfig(
                options=model_options,
                mode=SelectSelectorMode.DROPDOWN,
                translation_key=CONF_MODEL_SUBENTRY_ID,
            )
        )
        schema[
            vol.Optional(
                CONF_FALLBACK_MODEL_SUBENTRY_IDS,
                default=[
                    profile_id
                    for profile_id in options.get(CONF_FALLBACK_MODEL_SUBENTRY_IDS, [])
                    if profile_id in configured_profiles
                ],
            )
        ] = SelectSelector(
            SelectSelectorConfig(
                options=model_options,
                multiple=True,
                translation_key=CONF_FALLBACK_MODEL_SUBENTRY_IDS,
            )
        )
    api_schema_key = vol.Optional(CONF_LLM_HASS_API)
    if CONF_LLM_HASS_API in options:
        api_schema_key = vol.Optional(
            CONF_LLM_HASS_API,
            default=options[CONF_LLM_HASS_API],
        )
    schema[api_schema_key] = SelectSelector(
        SelectSelectorConfig(options=hass_apis, multiple=True)
    )
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
        schema[mcp_schema_key] = SelectSelector(
            SelectSelectorConfig(options=mcp_servers, multiple=True)
        )
    schema[
        vol.Optional(
            CONF_WEB_FETCH_ENABLED,
            default=bool(options.get(CONF_WEB_FETCH_ENABLED, False)),
        )
    ] = BooleanSelector()
    _append_skill_schema_fields(schema, options, available_skills)
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
                translation_key=CONF_MODEL,
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
            ): TextSelector(
                TextSelectorConfig()
            ),
            vol.Optional(
                _MODEL_SETTING_TEMPERATURE,
                description={
                    "suggested_value": model_settings.get(_MODEL_SETTING_TEMPERATURE)
                },
            ): NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=0.1)
            ),
            vol.Optional(
                _MODEL_SETTING_THINKING,
                description={"suggested_value": _format_thinking_value(model_settings)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=list(_THINKING_OPTIONS),
                    mode=SelectSelectorMode.DROPDOWN,
                    translation_key=_MODEL_SETTING_THINKING,
                )
            ),
            vol.Optional(
                CONF_CONFIGURE_ADVANCED_MODEL_SETTINGS,
                default=False,
            ): BooleanSelector(),
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
                _MODEL_SETTING_MAX_TOKENS,
                description={
                    "suggested_value": model_settings.get(_MODEL_SETTING_MAX_TOKENS)
                },
            ): NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=1)
            ),
            vol.Optional(
                _MODEL_SETTING_MAX_ITERATIONS,
                description={
                    "suggested_value": model_settings.get(_MODEL_SETTING_MAX_ITERATIONS)
                },
            ): NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=1)
            ),
            vol.Optional(
                _MODEL_SETTING_TOP_P,
                description={
                    "suggested_value": model_settings.get(_MODEL_SETTING_TOP_P)
                },
            ): NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=0.1)
            ),
            vol.Optional(
                _MODEL_SETTING_TIMEOUT,
                description={
                    "suggested_value": model_settings.get(_MODEL_SETTING_TIMEOUT)
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
                _MODEL_SETTING_EXTRA_BODY,
                description={
                    "suggested_value": _format_key_value_json_setting(
                        model_settings.get(_MODEL_SETTING_EXTRA_BODY)
                    )
                },
            ): TextSelector(TextSelectorConfig(multiline=True)),
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
    *,
    available_skills: list[AvailableSkill] | None = None,
) -> dict[str, Any]:
    """Return conversation fields with model profile references."""
    data = {
        key: value
        for key, value in user_input.items()
        if key not in {CONF_CONFIGURE_ADVANCED_MODEL_SETTINGS}
    }
    if not data.get(CONF_LLM_HASS_API):
        data.pop(CONF_LLM_HASS_API, None)
    if not data.get(CONF_MCP_SERVER_IDS):
        data.pop(CONF_MCP_SERVER_IDS, None)
    if not data.get(CONF_WEB_FETCH_ENABLED):
        data.pop(CONF_WEB_FETCH_ENABLED, None)
    if not data.get(CONF_FALLBACK_MODEL_SUBENTRY_IDS):
        data.pop(CONF_FALLBACK_MODEL_SUBENTRY_IDS, None)
    if data.get(CONF_ENABLE_SKILLS) and CONF_SKILLS in user_input and available_skills:
        data[CONF_SKILLS] = _merge_submitted_skills_with_hidden(
            user_input, options, available_skills
        )
    elif data.get(CONF_ENABLE_SKILLS) and CONF_SKILLS not in user_input and options.get(
        CONF_SKILLS
    ):
        data[CONF_SKILLS] = options[CONF_SKILLS]
    if not data.get(CONF_SKILLS):
        data.pop(CONF_SKILLS, None)
    _normalise_skill_settings(data)
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


def _store_model_settings(
    data: dict[str, Any], model_settings: Mapping[str, Any]
) -> None:
    """Store model settings only when at least one setting is configured."""
    if model_settings:
        data[CONF_MODEL_SETTINGS] = dict(model_settings)
    else:
        data.pop(CONF_MODEL_SETTINGS, None)


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
        | {CONF_CONFIGURE_ADVANCED_MODEL_SETTINGS}
    }


def _ai_task_data_schema(
    options: Mapping[str, Any] | None = None,
    entry: ConfigEntry | None = None,
    available_skills: list[AvailableSkill] | None = None,
) -> vol.Schema:
    """Return the AI task data subentry schema."""
    options = dict(options or {})
    model_options = _model_profile_select_options(entry)
    schema: VolDictType = {
        vol.Required(
            CONF_AI_TASK_NAME,
            default=options.get(CONF_AI_TASK_NAME, DEFAULT_AI_TASK_NAME),
        ): TextSelector(TextSelectorConfig()),
    }
    if model_options:
        configured_profiles = {option["value"] for option in model_options}
        schema[
            vol.Required(
                CONF_MODEL_SUBENTRY_ID,
                default=options.get(CONF_MODEL_SUBENTRY_ID, ""),
            )
        ] = SelectSelector(
            SelectSelectorConfig(
                options=model_options,
                mode=SelectSelectorMode.DROPDOWN,
                translation_key=CONF_MODEL_SUBENTRY_ID,
            )
        )
        schema[
            vol.Optional(
                CONF_FALLBACK_MODEL_SUBENTRY_IDS,
                default=[
                    profile_id
                    for profile_id in options.get(CONF_FALLBACK_MODEL_SUBENTRY_IDS, [])
                    if profile_id in configured_profiles
                ],
            )
        ] = SelectSelector(
            SelectSelectorConfig(
                options=model_options,
                multiple=True,
                translation_key=CONF_FALLBACK_MODEL_SUBENTRY_IDS,
            )
        )
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
        schema[mcp_schema_key] = SelectSelector(
            SelectSelectorConfig(options=mcp_servers, multiple=True)
        )
    schema[
        vol.Optional(
            CONF_WEB_FETCH_ENABLED,
            default=bool(options.get(CONF_WEB_FETCH_ENABLED, False)),
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
    _append_skill_schema_fields(schema, options, available_skills)
    return vol.Schema(schema)


def _ai_task_data_from_user_input(
    user_input: Mapping[str, Any],
    options: Mapping[str, Any],
    *,
    available_skills: list[AvailableSkill] | None = None,
) -> dict[str, Any]:
    """Return AI task subentry data with a selected structured output mode."""
    data = dict(user_input)
    data.setdefault(
        CONF_OUTPUT_MODE,
        normalise_structured_output_mode(options.get(CONF_OUTPUT_MODE)),
    )
    if not data.get(CONF_MCP_SERVER_IDS):
        data.pop(CONF_MCP_SERVER_IDS, None)
    if not data.get(CONF_WEB_FETCH_ENABLED):
        data.pop(CONF_WEB_FETCH_ENABLED, None)
    if not data.get(CONF_FALLBACK_MODEL_SUBENTRY_IDS):
        data.pop(CONF_FALLBACK_MODEL_SUBENTRY_IDS, None)
    if data.get(CONF_ENABLE_SKILLS) and CONF_SKILLS in user_input and available_skills:
        data[CONF_SKILLS] = _merge_submitted_skills_with_hidden(
            user_input, options, available_skills
        )
    elif data.get(CONF_ENABLE_SKILLS) and CONF_SKILLS not in user_input and options.get(
        CONF_SKILLS
    ):
        data[CONF_SKILLS] = options[CONF_SKILLS]
    if not data.get(CONF_SKILLS):
        data.pop(CONF_SKILLS, None)
    _normalise_skill_settings(data)
    return data


def _mcp_server_schema(options: Mapping[str, Any] | None = None) -> vol.Schema:
    """Return the remote MCP server subentry schema."""
    options = dict(options or {})
    return vol.Schema(
        {
            vol.Required(CONF_NAME, default=options.get(CONF_NAME, "")): TextSelector(
                TextSelectorConfig()
            ),
            vol.Required(
                CONF_MCP_URL,
                default=options.get(CONF_MCP_URL, ""),
            ): TextSelector(TextSelectorConfig()),
            vol.Optional(
                CONF_MCP_HEADERS,
                default=_format_mcp_headers(options.get(CONF_MCP_HEADERS)),
            ): TextSelector(TextSelectorConfig(multiline=True)),
            vol.Optional(
                CONF_MCP_INCLUDE_RETURN_SCHEMA,
                default=options.get(CONF_MCP_INCLUDE_RETURN_SCHEMA, True),
            ): BooleanSelector(),
            vol.Optional(
                CONF_MCP_DEFERRED_LOADING,
                default=options.get(CONF_MCP_DEFERRED_LOADING, False),
            ): BooleanSelector(),
        }
    )


def _format_mcp_headers(headers: object) -> str:
    """Return headers as one HTTP header per line for the config form."""
    if headers is None:
        return ""
    if isinstance(headers, str):
        return headers
    if not isinstance(headers, Mapping):
        return ""
    return "\n".join(f"{name}: {headers[name]}" for name in sorted(headers))


def _truncate_mcp_tool_description(description: str) -> str:
    """Return a compact single-line MCP tool description for selector labels."""
    description = " ".join(description.split())
    if len(description) <= _MCP_TOOL_DESCRIPTION_LABEL_MAX_LENGTH:
        return description
    return f"{description[: _MCP_TOOL_DESCRIPTION_LABEL_MAX_LENGTH - 3].rstrip()}..."


def _mcp_tool_options(
    tools: Iterable[Mapping[str, Any]],
    extra_tool_names: Iterable[str] = (),
) -> list[SelectOptionDict]:
    """Return sorted MCP tool selector options from discovered metadata."""
    options_by_name: dict[str, SelectOptionDict] = {}
    for tool in tools:
        name = str(tool.get("name", "")).strip()
        if not name or name in options_by_name:
            continue
        description = _truncate_mcp_tool_description(
            str(tool.get("description", "")).strip()
        )
        label = f"{name} ({description})" if description else name
        options_by_name[name] = SelectOptionDict(label=label, value=name)
    for name in extra_tool_names:
        if name and name not in options_by_name:
            options_by_name[name] = SelectOptionDict(label=name, value=name)
    return [options_by_name[name] for name in sorted(options_by_name)]


def _mcp_tools_schema(
    tool_options: list[SelectOptionDict], default_tool_names: list[str]
) -> vol.Schema:
    """Return the MCP discovered tools selection schema."""
    return vol.Schema(
        {
            vol.Required(
                CONF_MCP_ALLOWED_TOOLS,
                default=default_tool_names,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=tool_options,
                    multiple=True,
                )
            )
        }
    )


def _mcp_server_data_from_user_input(user_input: Mapping[str, Any]) -> dict[str, Any]:
    """Return normalized remote MCP server subentry data."""
    data: dict[str, Any] = {
        CONF_NAME: str(user_input[CONF_NAME]).strip(),
        CONF_MCP_URL: normalise_mcp_url(user_input[CONF_MCP_URL]),
        CONF_MCP_INCLUDE_RETURN_SCHEMA: bool(
            user_input.get(CONF_MCP_INCLUDE_RETURN_SCHEMA, True)
        ),
        CONF_MCP_DEFERRED_LOADING: bool(
            user_input.get(CONF_MCP_DEFERRED_LOADING, False)
        ),
    }
    headers = parse_mcp_headers(user_input.get(CONF_MCP_HEADERS))
    if headers:
        data[CONF_MCP_HEADERS] = headers
    allowed_tools = parse_allowed_tools(user_input.get(CONF_MCP_ALLOWED_TOOLS))
    if allowed_tools:
        data[CONF_MCP_ALLOWED_TOOLS] = allowed_tools
    return data


def _mcp_url_already_configured(
    entry: ConfigEntry,
    url: str,
    current_subentry_id: str | None = None,
) -> bool:
    """Return if another MCP server subentry already uses this URL."""
    url_identity = _mcp_url_identity(url)
    for subentry in entry.subentries.values():
        if subentry.subentry_id == current_subentry_id:
            continue
        if subentry.subentry_type != SUBENTRY_TYPE_MCP_SERVER:
            continue
        try:
            existing_identity = _mcp_url_identity(subentry.data.get(CONF_MCP_URL))
        except MCPValidationError:
            _LOGGER.warning(
                "Ignoring invalid stored MCP URL while checking duplicates for subentry %s",
                subentry.subentry_id,
            )
            continue
        if existing_identity == url_identity:
            return True
    return False


def _mcp_url_identity(
    url: object,
) -> tuple[str, str, int, str, tuple[tuple[str, str], ...]]:
    """Return a canonical identity for duplicate MCP URL checks."""
    parsed = urlparse(normalise_mcp_url(url))
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return (
        parsed.scheme,
        (parsed.hostname or "").lower().rstrip("."),
        port,
        parsed.path or "/",
        tuple(sorted(parse_qsl(parsed.query, keep_blank_values=True))),
    )


class PydanticAIAgentConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Pydantic AI Agent."""

    VERSION = 1
    MINOR_VERSION = 2

    def _async_update_provider_and_abort(
        self, entry: ConfigEntry, data: dict[str, Any]
    ) -> ConfigFlowResult:
        """Update a provider entry using the active reload mechanism."""
        if entry.update_listeners:
            return self.async_update_and_abort(entry, data=data)
        # Auth-failed entries may not have completed setup far enough to register
        # the reload listener, so explicitly schedule setup after updating them.
        return self.async_update_reload_and_abort(entry, data=data)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Store provider credentials; model access is validated by subentries."""
        errors: dict[str, str] = {}
        description_placeholders: dict[str, str] = {}

        if user_input is not None:
            try:
                data = _normalise_provider_data(user_input)
                _validate_provider_data(self.hass, data)
            except ProviderValidationError as err:
                errors["base"] = err.reason
                description_placeholders = _provider_validation_placeholders(err)
            else:
                for current_entry in self._async_current_entries():
                    if _provider_data_matches(current_entry.data, data):
                        return self.async_abort(reason="already_configured")
                return self.async_create_entry(title=data[CONF_NAME], data=data)

        return self.async_show_form(
            step_id="user",
            data_schema=self.add_suggested_values_to_schema(
                _base_schema(user_input), _provider_form_suggested_values(user_input)
            ),
            errors=errors,
            description_placeholders=description_placeholders,
        )

    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """Perform reauth after an authentication error."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Confirm reauth and collect provider credentials."""
        if user_input is None:
            entry_data = dict(self._get_reauth_entry().data)
            return self.async_show_form(
                step_id="reauth_confirm",
                data_schema=self.add_suggested_values_to_schema(
                    _base_schema(entry_data),
                    _provider_form_suggested_values(entry_data),
                ),
            )
        entry = self._get_reauth_entry()
        try:
            data = _normalise_provider_data(user_input)
            _validate_provider_data(self.hass, data)
        except ProviderValidationError as err:
            return self.async_show_form(
                step_id="reauth_confirm",
                data_schema=self.add_suggested_values_to_schema(
                    _base_schema(user_input),
                    _provider_form_suggested_values(user_input),
                ),
                errors={"base": err.reason},
                description_placeholders=_provider_validation_placeholders(err),
            )

        for current_entry in self._async_current_entries():
            if current_entry.entry_id == entry.entry_id:
                continue
            if _provider_data_matches(current_entry.data, data):
                return self.async_abort(reason="already_configured")
        (
            errors,
            description_placeholders,
        ) = await _validate_configured_models_for_provider_update(
            self.hass,
            entry,
            data,
            "reauth",
            skip_reconfigurable_model_errors=True,
        )
        if errors:
            return self.async_show_form(
                step_id="reauth_confirm",
                data_schema=self.add_suggested_values_to_schema(
                    _base_schema(data), _provider_form_suggested_values(data)
                ),
                errors=errors,
                description_placeholders=description_placeholders,
            )
        return self._async_update_provider_and_abort(entry, data)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Reconfigure provider credentials and connection settings."""
        entry = self._get_reconfigure_entry()
        if user_input is None:
            entry_data = dict(entry.data)
            return self.async_show_form(
                step_id="reconfigure",
                data_schema=self.add_suggested_values_to_schema(
                    _base_schema(entry_data),
                    _provider_form_suggested_values(entry_data),
                ),
            )

        try:
            data = _normalise_provider_data(user_input)
            _validate_provider_data(self.hass, data)
        except ProviderValidationError as err:
            return self.async_show_form(
                step_id="reconfigure",
                data_schema=self.add_suggested_values_to_schema(
                    _base_schema(user_input),
                    _provider_form_suggested_values(user_input),
                ),
                errors={"base": err.reason},
                description_placeholders=_provider_validation_placeholders(err),
            )

        for current_entry in self._async_current_entries():
            if current_entry.entry_id == entry.entry_id:
                continue
            if _provider_data_matches(current_entry.data, data):
                return self.async_abort(reason="already_configured")

        (
            errors,
            description_placeholders,
        ) = await _validate_configured_models_for_provider_update(
            self.hass,
            entry,
            data,
            "provider reconfigure",
            skip_reconfigurable_model_errors=False,
        )
        if errors:
            return self.async_show_form(
                step_id="reconfigure",
                data_schema=self.add_suggested_values_to_schema(
                    _base_schema(data), _provider_form_suggested_values(data)
                ),
                errors=errors,
                description_placeholders=description_placeholders,
            )

        return self._async_update_provider_and_abort(entry, data)

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {
            SUBENTRY_TYPE_MODEL: ModelSubentryFlowHandler,
            SUBENTRY_TYPE_CONVERSATION: ConversationSubentryFlowHandler,
            SUBENTRY_TYPE_AI_TASK: AITaskDataSubentryFlowHandler,
            SUBENTRY_TYPE_MCP_SERVER: MCPServerSubentryFlowHandler,
        }


class ModelSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing provider-owned model profiles."""

    _model_discovery_failed: bool
    _model_names: list[str] | None
    _options: dict[str, Any]
    _pending_data: dict[str, Any]
    _pending_model_settings: dict[str, Any]

    @property
    def _is_new(self) -> bool:
        """Return if this flow creates a new subentry."""
        return self.source == SOURCE_USER

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add a model profile subentry."""
        self._model_discovery_failed = False
        self._model_names = None
        self._options = {}
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure a model profile subentry."""
        self._model_discovery_failed = False
        self._model_names = None
        self._options = self._get_reconfigure_subentry().data.copy()
        return await self.async_step_init(user_input)

    async def _async_model_names(self) -> list[str] | None:
        """Return discovered provider model names for this flow."""
        if self._model_names is not None or self._model_discovery_failed:
            return self._model_names
        try:
            model_names = await async_list_provider_model_names(
                self.hass, self._get_entry().data
            )
        except Exception:
            _LOGGER.warning("Unable to list provider models for model profile form")
            self._model_discovery_failed = True
            return None
        if not model_names:
            self._model_discovery_failed = True
            return None
        self._model_names = model_names
        return self._model_names

    async def _async_show_model_profile_form(
        self,
        *,
        options: Mapping[str, Any],
        errors: dict[str, str] | None = None,
        description_placeholders: dict[str, str] | None = None,
        show_model_discovery_error: bool = False,
    ) -> SubentryFlowResult:
        """Show the model profile form with discovered model options when available."""
        model_names = await self._async_model_names()
        form_options = dict(options)
        if self._is_new and model_names and not form_options.get(CONF_MODEL):
            default_model = sorted(set(model_names))[0]
            form_options[CONF_MODEL] = default_model
            form_options.setdefault(CONF_NAME, default_model)
        form_errors = dict(errors or {})
        if (
            show_model_discovery_error
            and self._model_discovery_failed
            and not form_errors
        ):
            form_errors["base"] = "model_list_unavailable"
        return self.async_show_form(
            step_id="init",
            data_schema=_model_profile_schema(form_options, model_names),
            errors=form_errors,
            description_placeholders=description_placeholders,
        )

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage model profile options."""
        entry = self._get_entry()
        if entry.state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if user_input is not None:
            main_settings, errors, cleared = _parse_model_settings(
                self.hass, user_input, _MAIN_MODEL_SETTING_KEYS
            )
            data = _model_profile_data_from_user_input(user_input)
            model_names = await self._async_model_names() if self._is_new else None
            if model_names:
                default_model = sorted(set(model_names))[0]
                name = str(data.get(CONF_NAME, "")).strip()
                if not name or name == default_model:
                    data[CONF_NAME] = data[CONF_MODEL]
            existing_settings = _model_settings_from_options(self._options)
            model_settings = _merge_model_settings(
                existing_settings, main_settings, cleared
            )
            if errors:
                return await self._async_show_model_profile_form(
                    options=self._options
                    | data
                    | {CONF_MODEL_SETTINGS: model_settings},
                    errors=errors,
                )
            if user_input.get(CONF_CONFIGURE_ADVANCED_MODEL_SETTINGS):
                self._pending_data = data
                self._pending_model_settings = model_settings
                return self.async_show_form(
                    step_id="model_settings",
                    data_schema=_model_settings_schema(
                        self._options | {CONF_MODEL_SETTINGS: model_settings}
                    ),
                )
            return await self._async_finish_model_profile(data, model_settings)

        return await self._async_show_model_profile_form(
            options=self._options,
            show_model_discovery_error=True,
        )

    async def async_step_model_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage advanced model profile settings."""
        if user_input is None:
            return self.async_show_form(
                step_id="model_settings",
                data_schema=_model_settings_schema(
                    self._options | {CONF_MODEL_SETTINGS: self._pending_model_settings}
                ),
            )
        advanced_settings, errors, cleared = _parse_model_settings(
            self.hass, user_input, _ADVANCED_MODEL_SETTING_KEYS
        )
        cleared.update(_ADVANCED_MODEL_SETTING_KEYS - user_input.keys())
        model_settings = _merge_model_settings(
            self._pending_model_settings, advanced_settings, cleared
        )
        if errors:
            return self.async_show_form(
                step_id="model_settings",
                data_schema=_model_settings_schema(
                    self._options | {CONF_MODEL_SETTINGS: model_settings}
                ),
                errors=errors,
            )
        return await self._async_finish_model_profile(
            self._pending_data, model_settings, error_step="model_settings"
        )

    async def _async_finish_model_profile(
        self,
        data: dict[str, Any],
        model_settings: Mapping[str, Any],
        error_step: str = "init",
    ) -> SubentryFlowResult:
        """Probe the model profile, then create or update the subentry."""
        entry = self._get_entry()
        _store_model_settings(data, model_settings)
        try:
            await async_probe_model(
                self.hass,
                entry.data,
                data[CONF_MODEL],
                data.get(CONF_MODEL_SETTINGS, {}),
            )
        except ProviderValidationError as err:
            _log_provider_validation_failure(
                step="model profile subentry", model_name=data[CONF_MODEL], err=err
            )
            if error_step != "model_settings":
                return await self._async_show_model_profile_form(
                    options=self._options | data,
                    errors={"base": err.reason},
                    description_placeholders=_provider_validation_placeholders(err),
                )
            schema = _model_settings_schema(
                self._options | {CONF_MODEL_SETTINGS: model_settings}
            )
            return self.async_show_form(
                step_id=error_step,
                data_schema=schema,
                errors={"base": err.reason},
                description_placeholders=_provider_validation_placeholders(err),
            )
        except Exception:
            _LOGGER.exception("Unexpected exception validating model profile")
            if error_step != "model_settings":
                return await self._async_show_model_profile_form(
                    options=self._options | data,
                    errors={"base": "unknown"},
                )
            schema = _model_settings_schema(
                self._options | {CONF_MODEL_SETTINGS: model_settings}
            )
            return self.async_show_form(
                step_id=error_step,
                data_schema=schema,
                errors={"base": "unknown"},
            )
        if self._is_new:
            return self.async_create_entry(title=data[CONF_NAME], data=data)
        return self.async_update_and_abort(
            entry,
            self._get_reconfigure_subentry(),
            title=data[CONF_NAME],
            data=data,
        )


class ConversationSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing conversation subentries."""

    _options: dict[str, Any]
    _pending_conversation_data: dict[str, Any]

    @property
    def _is_new(self) -> bool:
        """Return if this flow creates a new subentry."""
        return self.source == SOURCE_USER

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add a conversation subentry."""
        self._options = default_conversation_options()
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure a conversation subentry."""
        self._options = self._get_reconfigure_subentry().data.copy()
        return await self.async_step_init(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage conversation options."""
        entry = self._get_entry()
        if entry.state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")
        if not _model_profile_select_options(entry):
            return self.async_abort(reason="no_models_configured")
        available_skills = await async_available_skills(self.hass, self._options)

        if user_input is not None:
            try:
                _validate_skills_folder(
                    self.hass,
                    user_input.get(CONF_SKILLS_FOLDER, DEFAULT_SKILLS_FOLDER),
                )
            except ProviderValidationError as err:
                return self.async_show_form(
                    step_id="init",
                    data_schema=_conversation_schema(
                        self.hass,
                        self._options | user_input,
                        entry,
                        available_skills,
                    ),
                    errors={"base": err.reason},
                    description_placeholders=_provider_validation_placeholders(err),
                )
            if _skill_source(user_input) != _skill_source(self._options):
                refreshed_options = dict(user_input)
                refreshed_options[CONF_SKILLS_FOLDER] = _normalise_skills_folder(
                    refreshed_options.get(CONF_SKILLS_FOLDER)
                )
                refreshed_options.pop(CONF_SKILLS, None)
                self._options = refreshed_options
                refreshed_skills = await async_available_skills(
                    self.hass, refreshed_options
                )
                return self.async_show_form(
                    step_id="init",
                    data_schema=_conversation_schema(
                        self.hass,
                        refreshed_options,
                        entry,
                        refreshed_skills,
                    ),
                    errors={"base": "skills_refreshed"},
                )
            available_skills = await async_available_skills(self.hass, user_input)
            data = _conversation_data_from_user_input(
                user_input,
                self._options,
                available_skills=available_skills,
            )
            if model_error := _selected_model_profile_error(entry, data):
                return self.async_show_form(
                    step_id="init",
                    data_schema=_conversation_schema(
                        self.hass,
                        self._options | data,
                        entry,
                        available_skills,
                    ),
                    errors={CONF_MODEL_SUBENTRY_ID: model_error},
                )
            if mcp_error := _selected_mcp_server_error(entry, data):
                return self.async_show_form(
                    step_id="init",
                    data_schema=_conversation_schema(
                        self.hass,
                        self._options | data,
                        entry,
                        available_skills,
                    ),
                    errors={CONF_MCP_SERVER_IDS: mcp_error},
                )
            return self._async_finish_conversation_options(data)

        return self.async_show_form(
            step_id="init",
            data_schema=_conversation_schema(
                self.hass, self._options, entry, available_skills
            ),
        )

    def _async_finish_conversation_options(
        self,
        data: dict[str, Any],
    ) -> SubentryFlowResult:
        """Create or update the conversation subentry."""
        entry = self._get_entry()
        if self._is_new:
            return self.async_create_entry(
                title=data[CONF_AGENT_NAME],
                data=data,
            )
        return self.async_update_and_abort(
            entry,
            self._get_reconfigure_subentry(),
            title=data[CONF_AGENT_NAME],
            data=data,
        )


class AITaskDataSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing AI task data subentries."""

    _options: dict[str, Any]

    @property
    def _is_new(self) -> bool:
        """Return if this flow creates a new subentry."""
        return self.source == SOURCE_USER

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add an AI task data subentry."""
        self._options = {}
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure an AI task data subentry."""
        subentry = self._get_reconfigure_subentry()
        self._options = subentry.data.copy()
        self._options.setdefault(CONF_AI_TASK_NAME, subentry.title)
        return await self.async_step_init(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage AI task model options."""
        entry = self._get_entry()
        if entry.state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")
        if not _model_profile_select_options(entry):
            return self.async_abort(reason="no_models_configured")
        available_skills = await async_available_skills(self.hass, self._options)

        if user_input is not None:
            try:
                _validate_skills_folder(
                    self.hass,
                    user_input.get(CONF_SKILLS_FOLDER, DEFAULT_SKILLS_FOLDER),
                )
            except ProviderValidationError as err:
                return self.async_show_form(
                    step_id="init",
                    data_schema=_ai_task_data_schema(
                        self._options | user_input, entry, available_skills
                    ),
                    errors={"base": err.reason},
                    description_placeholders=_provider_validation_placeholders(err),
                )
            if _skill_source(user_input) != _skill_source(self._options):
                refreshed_options = dict(user_input)
                refreshed_options[CONF_SKILLS_FOLDER] = _normalise_skills_folder(
                    refreshed_options.get(CONF_SKILLS_FOLDER)
                )
                refreshed_options.pop(CONF_SKILLS, None)
                self._options = refreshed_options
                refreshed_skills = await async_available_skills(
                    self.hass, refreshed_options
                )
                return self.async_show_form(
                    step_id="init",
                    data_schema=_ai_task_data_schema(
                        refreshed_options, entry, refreshed_skills
                    ),
                    errors={"base": "skills_refreshed"},
                )
            available_skills = await async_available_skills(self.hass, user_input)
            data = _ai_task_data_from_user_input(
                user_input,
                self._options,
                available_skills=available_skills,
            )
            if model_error := _selected_model_profile_error(entry, data):
                return self.async_show_form(
                    step_id="init",
                    data_schema=_ai_task_data_schema(
                        self._options | data, entry, available_skills
                    ),
                    errors={CONF_MODEL_SUBENTRY_ID: model_error},
                )
            if mcp_error := _selected_mcp_server_error(entry, data):
                return self.async_show_form(
                    step_id="init",
                    data_schema=_ai_task_data_schema(
                        self._options | data, entry, available_skills
                    ),
                    errors={CONF_MCP_SERVER_IDS: mcp_error},
                )
            return await self._async_finish_ai_task_options(data)

        return self.async_show_form(
            step_id="init",
            data_schema=_ai_task_data_schema(self._options, entry, available_skills),
        )

    async def _async_finish_ai_task_options(
        self,
        data: dict[str, Any],
    ) -> SubentryFlowResult:
        """Probe the selected AI task model, then create or update the subentry."""
        entry = self._get_entry()
        available_skills = await async_available_skills(self.hass, data)
        if mcp_error := _selected_mcp_server_error(entry, data):
            return self.async_show_form(
                step_id="init",
                data_schema=_ai_task_data_schema(
                    self._options | data, entry, available_skills
                ),
                errors={CONF_MCP_SERVER_IDS: mcp_error},
            )
        current_model = ""
        try:
            for profile_id in _selected_model_profile_ids(data):
                profile = entry.subentries[profile_id]
                settings = profile.data.get(CONF_MODEL_SETTINGS)
                current_model = profile.data[CONF_MODEL]
                await async_probe_model(
                    self.hass,
                    entry.data,
                    current_model,
                    dict(settings) if isinstance(settings, Mapping) else {},
                    structured_output_mode=data[CONF_OUTPUT_MODE],
                )
        except ProviderValidationError as err:
            _log_provider_validation_failure(
                step="AI task subentry", model_name=current_model, err=err
            )
            return self.async_show_form(
                step_id="init",
                data_schema=_ai_task_data_schema(
                    self._options | data, entry, available_skills
                ),
                errors={"base": err.reason},
                description_placeholders=_provider_validation_placeholders(err),
            )
        except Exception:
            _LOGGER.exception("Unexpected exception validating AI task model")
            return self.async_show_form(
                step_id="init",
                data_schema=_ai_task_data_schema(
                    self._options | data, entry, available_skills
                ),
                errors={"base": "unknown"},
            )
        if self._is_new:
            return self.async_create_entry(title=data[CONF_AI_TASK_NAME], data=data)
        return self.async_update_and_abort(
            entry,
            self._get_reconfigure_subentry(),
            title=data[CONF_AI_TASK_NAME],
            data=data,
        )


class MCPServerSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing remote MCP server subentries."""

    _options: dict[str, Any]
    _pending_data: dict[str, Any]
    _tool_options: list[SelectOptionDict]

    @property
    def _is_new(self) -> bool:
        """Return if this flow creates a new subentry."""
        return self.source == SOURCE_USER

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add an MCP server subentry."""
        self._options = {}
        self._pending_data = {}
        self._tool_options = []
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure an MCP server subentry."""
        self._options = self._get_reconfigure_subentry().data.copy()
        self._pending_data = {}
        self._tool_options = []
        return await self.async_step_init(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage remote MCP server options."""
        entry = self._get_entry()
        if entry.state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if user_input is None:
            return self.async_show_form(
                step_id="init",
                data_schema=_mcp_server_schema(self._options),
            )

        errors: dict[str, str] = {}
        description_placeholders: dict[str, str] = {}
        try:
            data = _mcp_server_data_from_user_input(user_input)
        except MCPValidationError as err:
            errors[CONF_MCP_URL] = err.reason
            description_placeholders = _mcp_validation_placeholders(err)
            data = dict(user_input)
        except vol.Invalid as err:
            reason = str(err) or "invalid_mcp_headers"
            if reason == "invalid_mcp_tools":
                errors[CONF_MCP_ALLOWED_TOOLS] = reason
            else:
                errors[CONF_MCP_HEADERS] = "invalid_mcp_headers"
            data = dict(user_input)
        else:
            form_data = (
                self._options
                | data
                | {CONF_MCP_HEADERS: user_input.get(CONF_MCP_HEADERS, "")}
            )
            try:
                data[CONF_MCP_URL] = await async_validate_mcp_url(
                    self.hass, data[CONF_MCP_URL]
                )
            except MCPValidationError as err:
                errors[CONF_MCP_URL] = err.reason
                return self.async_show_form(
                    step_id="init",
                    data_schema=_mcp_server_schema(form_data),
                    errors=errors,
                    description_placeholders=_mcp_validation_placeholders(err),
                )
            current_subentry_id = None
            if not self._is_new:
                current_subentry_id = self._get_reconfigure_subentry().subentry_id
            if _mcp_url_already_configured(
                entry, data[CONF_MCP_URL], current_subentry_id
            ):
                return self.async_abort(reason="already_configured")
            try:
                tools = await async_discover_mcp_tools_from_config(
                    self.hass,
                    data,
                    server_id=current_subentry_id or data[CONF_NAME],
                    apply_allowlist=False,
                )
                existing_allowed_tools = parse_allowed_tools(
                    self._options.get(CONF_MCP_ALLOWED_TOOLS)
                )
                self._tool_options = _mcp_tool_options(tools, existing_allowed_tools)
                if not self._tool_options:
                    raise MCPValidationError(
                        "no_mcp_tools",
                        "The MCP server did not expose any tools.",
                    )
            except MCPValidationError as err:
                target = CONF_MCP_URL if err.reason == "invalid_mcp_url" else "base"
                errors[target] = err.reason
                return self.async_show_form(
                    step_id="init",
                    data_schema=_mcp_server_schema(form_data),
                    errors=errors,
                    description_placeholders=_mcp_validation_placeholders(err),
                )
            self._pending_data = data
            return await self.async_step_tools()

        return self.async_show_form(
            step_id="init",
            data_schema=_mcp_server_schema(self._options | data),
            errors=errors,
            description_placeholders=description_placeholders,
        )

    async def async_step_tools(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Select the discovered MCP tools to allow."""
        if not self._pending_data or not self._tool_options:
            return await self.async_step_init()

        tool_names = [option["value"] for option in self._tool_options]
        default_tool_names = (
            tool_names
            if self._is_new
            else [
                tool_name
                for tool_name in parse_allowed_tools(
                    self._options.get(CONF_MCP_ALLOWED_TOOLS)
                )
                if tool_name in tool_names
            ]
        )

        if user_input is None:
            return self.async_show_form(
                step_id="tools",
                data_schema=_mcp_tools_schema(self._tool_options, default_tool_names),
            )

        allowed_tools = [
            tool_name
            for tool_name in parse_allowed_tools(user_input.get(CONF_MCP_ALLOWED_TOOLS))
            if tool_name in tool_names
        ]
        if not allowed_tools:
            return self.async_show_form(
                step_id="tools",
                data_schema=_mcp_tools_schema(self._tool_options, default_tool_names),
                errors={CONF_MCP_ALLOWED_TOOLS: "mcp_tools_not_allowlisted"},
            )

        data = {**self._pending_data, CONF_MCP_ALLOWED_TOOLS: allowed_tools}
        if self._is_new:
            return self.async_create_entry(title=data[CONF_NAME], data=data)
        return self.async_update_and_abort(
            self._get_entry(),
            self._get_reconfigure_subentry(),
            title=data[CONF_NAME],
            data=data,
        )
