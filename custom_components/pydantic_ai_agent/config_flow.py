"""Config flow for Pydantic AI Agent."""

from __future__ import annotations

from collections.abc import AsyncIterable, Mapping
from dataclasses import dataclass
import errno
import json
import logging
import socket
import ssl
from typing import Any

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
from homeassistant.helpers import llm
from homeassistant.helpers.redact import async_redact_data
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
from homeassistant.helpers.typing import VolDictType

from .const import (
    CONF_AGENT_NAME,
    CONF_BASE_URL,
    CONF_CONFIGURE_ADVANCED_MODEL_SETTINGS,
    CONF_MODEL,
    CONF_MODEL_SETTINGS,
    CONF_OUTPUT_MODE,
    CONF_PROMPT,
    CONF_PROVIDER_MODE,
    DEFAULT_AGENT_NAME,
    DEFAULT_CONVERSATION_OPTIONS,
    DEFAULT_OUTPUT_MODE,
    DEFAULT_SERVICE_NAME,
    DEFAULT_TIMEOUT,
    DOMAIN,
    OUTPUT_MODE_NATIVE,
    OUTPUT_MODE_PROMPTED,
    OUTPUT_MODE_TOOL,
    PROVIDER_MODES,
    PROVIDER_OPENAI,
    PROVIDER_OPENAI_COMPATIBLE,
    STRUCTURED_OUTPUT_MODES,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
)
from .provider import normalise_base_url, openai_chat_model_from_config
from .structured_output import (
    structured_model_request_parameters,
    structured_output_mode as normalise_structured_output_mode,
    structured_output_name,
)

_LOGGER = logging.getLogger(__name__)

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
_MODEL_SETTING_TEMPERATURE = "temperature"
_MODEL_SETTING_TOP_P = "top_p"
_MODEL_SETTING_TIMEOUT = "timeout"
_MODEL_SETTING_PARALLEL_TOOL_CALLS = "parallel_tool_calls"
_MODEL_SETTING_SEED = "seed"
_MODEL_SETTING_PRESENCE_PENALTY = "presence_penalty"
_MODEL_SETTING_FREQUENCY_PENALTY = "frequency_penalty"
_MODEL_SETTING_EXTRA_HEADERS = "extra_headers"
_MODEL_SETTING_THINKING = "thinking"
_MODEL_SETTING_EXTRA_BODY = "extra_body"
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
    _MODEL_SETTING_TOP_P,
    _MODEL_SETTING_TIMEOUT,
    _MODEL_SETTING_PARALLEL_TOOL_CALLS,
    _MODEL_SETTING_SEED,
    _MODEL_SETTING_PRESENCE_PENALTY,
    _MODEL_SETTING_FREQUENCY_PENALTY,
    _MODEL_SETTING_EXTRA_HEADERS,
    _MODEL_SETTING_EXTRA_BODY,
}
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


def _base_schema(user_input: dict[str, Any] | None = None) -> vol.Schema:
    """Return the provider connection schema."""
    provider_mode = (user_input or {}).get(CONF_PROVIDER_MODE, PROVIDER_OPENAI)
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
    return vol.Schema(schema)


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
    return dedupe


async def _validate_configured_models_for_provider_update(
    hass: HomeAssistant,
    entry: ConfigEntry,
    data: Mapping[str, Any],
    step: str,
    skip_reconfigurable_model_errors: bool,
) -> tuple[dict[str, str], dict[str, str]]:
    """Probe existing subentry models against new provider settings."""
    seen: set[tuple[str, str, str | None]] = set()
    for subentry in entry.subentries.values():
        if subentry.subentry_type not in (
            SUBENTRY_TYPE_CONVERSATION,
            SUBENTRY_TYPE_AI_TASK,
        ):
            continue
        if not (model := subentry.data.get(CONF_MODEL)):
            continue
        if subentry.subentry_type == SUBENTRY_TYPE_CONVERSATION:
            settings = subentry.data.get(CONF_MODEL_SETTINGS)
            model_settings = dict(settings) if isinstance(settings, Mapping) else {}
            output_mode = None
        else:
            model_settings = {}
            output_mode = normalise_structured_output_mode(
                subentry.data.get(CONF_OUTPUT_MODE)
            )
        dedupe_key = (
            model,
            json.dumps(model_settings, sort_keys=True, separators=(",", ":")),
            output_mode,
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
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


def _metadata_redaction_keys(metadata: object) -> set[object]:
    """Return metadata keys that should be redacted, preserving original casing."""
    keys: set[object] = set()
    if isinstance(metadata, Mapping):
        for key, value in metadata.items():
            if str(key).lower() in _SENSITIVE_METADATA_KEYS:
                keys.add(key)
            keys.update(_metadata_redaction_keys(value))
    elif isinstance(metadata, list):
        for item in metadata:
            keys.update(_metadata_redaction_keys(item))
    return keys


def _format_metadata(metadata: object) -> str:
    """Return redacted, bounded provider metadata for config-flow display."""
    redaction_keys = _metadata_redaction_keys(metadata)
    redacted = (
        async_redact_data(metadata, redaction_keys) if redaction_keys else metadata
    )
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
        if isinstance(item, TimeoutError):
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

        # Provider/network clients do not expose every connection failure as a
        # stable typed exception, so keep the user-facing flow error specific
        # when only the exception text carries the condition.
        name = type(item).__name__.lower()
        text = str(item).lower()
        if "timeout" in name or "timed out" in text or "timeout" in text:
            return "Request timed out."
        if "ssl" in name or "tls" in text or "certificate" in text:
            return "TLS error."
        if "name or service not known" in text or "nodename nor servname" in text:
            return "Host not found."
        if "connection refused" in text:
            return "Connection refused."
        if "network is unreachable" in text or "no route to host" in text:
            return "Network unreachable."
        if "connect" in name or "network" in name or "connection" in text:
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


def _openai_chat_model(
    hass: HomeAssistant, data: Mapping[str, Any], model_name: str
) -> Any:
    """Build a Pydantic AI OpenAI chat model for provider validation."""
    return openai_chat_model_from_config(hass, data, model_name)


def _structured_probe_request_parameters(
    output_mode: str,
) -> ModelRequestParameters:
    """Return request parameters for a structured-output capability probe."""
    return structured_model_request_parameters(
        function_tools=[],
        output_mode=output_mode,
        output_name=_STRUCTURED_PROBE_OUTPUT_NAME,
        json_schema=_STRUCTURED_PROBE_SCHEMA,
        strict=True,
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
    provider_mode = data[CONF_PROVIDER_MODE]
    base_url = _normalise_base_url(data)
    if provider_mode == PROVIDER_OPENAI_COMPATIBLE and not base_url:
        raise ProviderValidationError(
            "invalid_base_url", "OpenAI-compatible providers require a base URL."
        )

    try:
        settings = dict(model_settings or {})
        settings.setdefault(_MODEL_SETTING_TIMEOUT, DEFAULT_TIMEOUT)
        model = _openai_chat_model(hass, data, model_name)
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
        async with model_request_stream(
            model,
            messages,
            model_settings=ModelSettings(**settings),
            model_request_parameters=model_request_parameters,
        ) as stream:
            if structured_output_mode is not None:
                await _validate_structured_probe_stream(
                    stream,
                    normalise_structured_output_mode(structured_output_mode),
                )
                return
            async for _event in stream:
                return
        raise ProviderValidationError(
            "provider_error", "The provider returned an empty streamed response."
        )
    except ModelHTTPError as err:
        raise _map_http_error(err) from err
    except ModelAPIError as err:
        raise _format_api_error(err) from err
    except NotImplementedError as err:
        raise ProviderValidationError(
            "model_does_not_support_streaming", str(err)
        ) from err
    except UnexpectedModelBehavior as err:
        raise ProviderValidationError("provider_error", str(err)) from err
    except TimeoutError as err:
        raise ProviderValidationError("timeout", "Request timed out.") from err
    except (ImportError, UserError) as err:
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
    data[CONF_BASE_URL] = _normalise_base_url(data)
    if (
        data[CONF_PROVIDER_MODE] != PROVIDER_OPENAI_COMPATIBLE
        or not data[CONF_BASE_URL]
    ):
        data.pop(CONF_BASE_URL, None)
    return data


def _validate_provider_data(data: Mapping[str, Any]) -> None:
    """Validate provider data that does not require a model."""
    if data[CONF_PROVIDER_MODE] == PROVIDER_OPENAI_COMPATIBLE and not data.get(
        CONF_BASE_URL
    ):
        raise ProviderValidationError(
            "invalid_base_url", "OpenAI-compatible providers require a base URL."
        )


def _provider_data_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return if two provider configurations identify the same connection."""
    return _dedupe_data(left) == _dedupe_data(right)


def _conversation_schema(
    hass: HomeAssistant, options: Mapping[str, Any] | None = None
) -> vol.Schema:
    """Return the conversation subentry schema, pruning unavailable HA APIs."""
    options = dict(options or {})
    model_settings = options.get(CONF_MODEL_SETTINGS, {})
    if not isinstance(model_settings, Mapping):
        model_settings = {}
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
        vol.Required(
            CONF_MODEL,
            default=options.get(CONF_MODEL, ""),
        ): TextSelector(TextSelectorConfig()),
        vol.Optional(
            CONF_PROMPT,
            description={"suggested_value": options.get(CONF_PROMPT, "")},
        ): TemplateSelector(),
        vol.Optional(
            _MODEL_SETTING_TEMPERATURE,
            description={
                "suggested_value": model_settings.get(_MODEL_SETTING_TEMPERATURE)
            },
        ): NumberSelector(NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=0.1)),
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
    api_schema_key = vol.Optional(CONF_LLM_HASS_API)
    if CONF_LLM_HASS_API in options:
        api_schema_key = vol.Optional(
            CONF_LLM_HASS_API,
            default=options[CONF_LLM_HASS_API],
        )
    schema[api_schema_key] = SelectSelector(
        SelectSelectorConfig(options=hass_apis, multiple=True)
    )
    return vol.Schema(schema)


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
            vol.Optional(
                _MODEL_SETTING_MAX_TOKENS,
                description={
                    "suggested_value": model_settings.get(_MODEL_SETTING_MAX_TOKENS)
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
            parallel_tool_calls_key: BooleanSelector(),
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
                _MODEL_SETTING_EXTRA_HEADERS,
                description={
                    "suggested_value": _format_json_setting(
                        model_settings.get(_MODEL_SETTING_EXTRA_HEADERS)
                    )
                },
            ): TextSelector(TextSelectorConfig(multiline=True)),
            vol.Optional(
                _MODEL_SETTING_EXTRA_BODY,
                description={
                    "suggested_value": _format_json_setting(
                        model_settings.get(_MODEL_SETTING_EXTRA_BODY)
                    )
                },
            ): TextSelector(TextSelectorConfig(multiline=True)),
        }
    )


def _format_json_setting(value: object) -> str:
    """Return a JSON string for a configured object setting."""
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True)


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


def _parse_json_object_setting(value: object) -> dict[str, Any]:
    """Return a JSON object model setting from user input."""
    if not isinstance(value, str):
        raise ValueError("invalid_json")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as err:
        raise ValueError("invalid_json") from err
    if not isinstance(parsed, dict):
        raise ValueError("invalid_object")
    return parsed


def _parse_extra_headers_setting(value: object) -> dict[str, str]:
    """Return extra headers with string keys and string values."""
    parsed = _parse_json_object_setting(value)
    if not all(
        isinstance(key, str) and isinstance(item, str) for key, item in parsed.items()
    ):
        raise ValueError("invalid_headers")
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
    user_input: Mapping[str, Any], setting_keys: set[str]
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
            if key == _MODEL_SETTING_MAX_TOKENS:
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
            elif key == _MODEL_SETTING_EXTRA_HEADERS:
                settings[key] = _parse_extra_headers_setting(value)
            elif key == _MODEL_SETTING_EXTRA_BODY:
                settings[key] = _parse_json_object_setting(value)
            elif key == _MODEL_SETTING_THINKING:
                settings[key] = _parse_thinking_setting(value)
        except ValueError as err:
            errors[key] = _model_setting_error(key, str(err))
    return settings, errors, cleared


def _model_setting_error(key: str, detail: str) -> str:
    """Return a translation key for a model setting validation error."""
    if detail in {"invalid_json", "invalid_object", "invalid_headers"}:
        return detail
    if key in {_MODEL_SETTING_MAX_TOKENS, _MODEL_SETTING_SEED}:
        return "invalid_integer"
    if key == _MODEL_SETTING_TIMEOUT:
        return "positive_number"
    return "invalid_number"


def _model_settings_from_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return existing model settings from subentry options."""
    model_settings = options.get(CONF_MODEL_SETTINGS)
    if isinstance(model_settings, Mapping):
        return dict(model_settings)
    return {}


def _conversation_data_from_user_input(user_input: Mapping[str, Any]) -> dict[str, Any]:
    """Return conversation fields, leaving model settings for separate storage."""
    data = {
        key: value
        for key, value in user_input.items()
        if key
        not in _MAIN_MODEL_SETTING_KEYS
        | _ADVANCED_MODEL_SETTING_KEYS
        | {CONF_CONFIGURE_ADVANCED_MODEL_SETTINGS}
    }
    if not data.get(CONF_LLM_HASS_API):
        data.pop(CONF_LLM_HASS_API, None)
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


def _ai_task_data_schema(options: Mapping[str, Any] | None = None) -> vol.Schema:
    """Return the AI task data subentry schema."""
    options = dict(options or {})
    return vol.Schema(
        {
            vol.Required(
                CONF_MODEL,
                default=options.get(CONF_MODEL, ""),
            ): TextSelector(TextSelectorConfig()),
            vol.Optional(
                CONF_CONFIGURE_ADVANCED_MODEL_SETTINGS,
                default=False,
            ): BooleanSelector(),
        }
    )


def _ai_task_output_mode_schema(
    options: Mapping[str, Any] | None = None,
) -> vol.Schema:
    """Return the AI task advanced structured output schema."""
    options = dict(options or {})
    return vol.Schema(
        {
            vol.Required(
                CONF_OUTPUT_MODE,
                default=normalise_structured_output_mode(
                    options.get(CONF_OUTPUT_MODE, DEFAULT_OUTPUT_MODE)
                ),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=list(_OUTPUT_MODE_OPTIONS),
                    mode=SelectSelectorMode.DROPDOWN,
                    translation_key=CONF_OUTPUT_MODE,
                )
            ),
        }
    )


def _ai_task_data_from_user_input(
    user_input: Mapping[str, Any], options: Mapping[str, Any]
) -> dict[str, Any]:
    """Return AI task subentry data with a selected structured output mode."""
    data = {
        key: value
        for key, value in user_input.items()
        if key != CONF_CONFIGURE_ADVANCED_MODEL_SETTINGS
    }
    data.setdefault(
        CONF_OUTPUT_MODE,
        normalise_structured_output_mode(options.get(CONF_OUTPUT_MODE)),
    )
    return data


class PydanticAIAgentConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Pydantic AI Agent."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Store provider credentials; model access is validated by subentries."""
        errors: dict[str, str] = {}
        description_placeholders: dict[str, str] = {}

        if user_input is not None:
            data = _normalise_provider_data(user_input)
            try:
                _validate_provider_data(data)
            except ProviderValidationError as err:
                errors["base"] = err.reason
                description_placeholders = _provider_validation_placeholders(err)
            else:
                self._async_abort_entries_match(_dedupe_data(data))
                return self.async_create_entry(title=data[CONF_NAME], data=data)

        return self.async_show_form(
            step_id="user",
            data_schema=self.add_suggested_values_to_schema(
                _base_schema(user_input), user_input
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
                    _base_schema(entry_data), entry_data
                ),
            )
        data = _normalise_provider_data(user_input)
        try:
            _validate_provider_data(data)
        except ProviderValidationError as err:
            return self.async_show_form(
                step_id="reauth_confirm",
                data_schema=self.add_suggested_values_to_schema(
                    _base_schema(data), data
                ),
                errors={"base": err.reason},
                description_placeholders=_provider_validation_placeholders(err),
            )

        self._async_abort_entries_match(_dedupe_data(data))
        entry = self._get_reauth_entry()
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
                    _base_schema(data), data
                ),
                errors=errors,
                description_placeholders=description_placeholders,
            )
        return self.async_update_reload_and_abort(
            entry,
            data=data,
        )

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
                    _base_schema(entry_data), entry_data
                ),
            )

        data = _normalise_provider_data(user_input)
        try:
            _validate_provider_data(data)
        except ProviderValidationError as err:
            return self.async_show_form(
                step_id="reconfigure",
                data_schema=self.add_suggested_values_to_schema(
                    _base_schema(data), data
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
                    _base_schema(data), data
                ),
                errors=errors,
                description_placeholders=description_placeholders,
            )

        return self.async_update_reload_and_abort(entry, data=data)

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {
            SUBENTRY_TYPE_CONVERSATION: ConversationSubentryFlowHandler,
            SUBENTRY_TYPE_AI_TASK: AITaskDataSubentryFlowHandler,
        }


class ConversationSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing conversation subentries."""

    _options: dict[str, Any]
    _pending_conversation_data: dict[str, Any]
    _pending_model_settings: dict[str, Any]

    @property
    def _is_new(self) -> bool:
        """Return if this flow creates a new subentry."""
        return self.source == SOURCE_USER

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add a conversation subentry."""
        self._options = DEFAULT_CONVERSATION_OPTIONS.copy()
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

        if user_input is not None:
            main_settings, errors, cleared = _parse_model_settings(
                user_input, _MAIN_MODEL_SETTING_KEYS
            )
            data = _conversation_data_from_user_input(user_input)
            existing_settings = _model_settings_from_options(self._options)
            model_settings = _merge_model_settings(
                existing_settings, main_settings, cleared
            )
            if errors:
                form_options = data | {CONF_MODEL_SETTINGS: model_settings}
                return self.async_show_form(
                    step_id="init",
                    data_schema=_conversation_schema(
                        self.hass, self._options | form_options
                    ),
                    errors=errors,
                )
            if user_input.get(CONF_CONFIGURE_ADVANCED_MODEL_SETTINGS):
                self._pending_conversation_data = data
                self._pending_model_settings = model_settings
                return self.async_show_form(
                    step_id="model_settings",
                    data_schema=_model_settings_schema(
                        self._options | {CONF_MODEL_SETTINGS: model_settings}
                    ),
                )
            return await self._async_finish_conversation_options(data, model_settings)

        return self.async_show_form(
            step_id="init",
            data_schema=_conversation_schema(self.hass, self._options),
        )

    async def async_step_model_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage advanced model settings."""
        if user_input is None:
            return self.async_show_form(
                step_id="model_settings",
                data_schema=_model_settings_schema(
                    self._options | {CONF_MODEL_SETTINGS: self._pending_model_settings}
                ),
            )
        advanced_settings, errors, cleared = _parse_model_settings(
            user_input, _ADVANCED_MODEL_SETTING_KEYS
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
        return await self._async_finish_conversation_options(
            self._pending_conversation_data, model_settings, error_step="model_settings"
        )

    async def _async_finish_conversation_options(
        self,
        data: dict[str, Any],
        model_settings: Mapping[str, Any],
        error_step: str = "init",
    ) -> SubentryFlowResult:
        """Probe the selected chat model, then create or update the subentry."""
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
                step="conversation subentry", model_name=data[CONF_MODEL], err=err
            )
            if error_step == "model_settings":
                return self.async_show_form(
                    step_id="model_settings",
                    data_schema=_model_settings_schema(
                        self._options | {CONF_MODEL_SETTINGS: model_settings}
                    ),
                    errors={"base": err.reason},
                    description_placeholders=_provider_validation_placeholders(err),
                )
            return self.async_show_form(
                step_id="init",
                data_schema=_conversation_schema(self.hass, self._options | data),
                errors={"base": err.reason},
                description_placeholders=_provider_validation_placeholders(err),
            )
        except Exception:
            _LOGGER.exception("Unexpected exception validating conversation model")
            if error_step == "model_settings":
                return self.async_show_form(
                    step_id="model_settings",
                    data_schema=_model_settings_schema(
                        self._options | {CONF_MODEL_SETTINGS: model_settings}
                    ),
                    errors={"base": "unknown"},
                )
            return self.async_show_form(
                step_id="init",
                data_schema=_conversation_schema(self.hass, self._options | data),
                errors={"base": "unknown"},
            )
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
    _pending_ai_task_data: dict[str, Any]

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
        self._options = self._get_reconfigure_subentry().data.copy()
        return await self.async_step_init(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage AI task model options."""
        entry = self._get_entry()
        if entry.state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if user_input is not None:
            data = _ai_task_data_from_user_input(user_input, self._options)
            if user_input.get(CONF_CONFIGURE_ADVANCED_MODEL_SETTINGS):
                self._pending_ai_task_data = data
                return self.async_show_form(
                    step_id="output_mode",
                    data_schema=_ai_task_output_mode_schema(self._options | data),
                )
            return await self._async_finish_ai_task_options(data)

        return self.async_show_form(
            step_id="init",
            data_schema=_ai_task_data_schema(self._options),
        )

    async def async_step_output_mode(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage advanced structured output settings."""
        if user_input is None:
            return self.async_show_form(
                step_id="output_mode",
                data_schema=_ai_task_output_mode_schema(
                    self._options | self._pending_ai_task_data
                ),
            )
        data = self._pending_ai_task_data | {
            CONF_OUTPUT_MODE: normalise_structured_output_mode(
                user_input[CONF_OUTPUT_MODE]
            )
        }
        return await self._async_finish_ai_task_options(
            data,
            error_step="output_mode",
        )

    async def _async_finish_ai_task_options(
        self,
        data: dict[str, Any],
        error_step: str = "init",
    ) -> SubentryFlowResult:
        """Probe the selected AI task model, then create or update the subentry."""
        entry = self._get_entry()
        try:
            await async_probe_model(
                self.hass,
                entry.data,
                data[CONF_MODEL],
                structured_output_mode=data[CONF_OUTPUT_MODE],
            )
        except ProviderValidationError as err:
            _log_provider_validation_failure(
                step="AI task subentry", model_name=data[CONF_MODEL], err=err
            )
            if error_step == "output_mode":
                return self.async_show_form(
                    step_id="output_mode",
                    data_schema=_ai_task_output_mode_schema(self._options | data),
                    errors={"base": err.reason},
                    description_placeholders=_provider_validation_placeholders(err),
                )
            return self.async_show_form(
                step_id="init",
                data_schema=_ai_task_data_schema(self._options | data),
                errors={"base": err.reason},
                description_placeholders=_provider_validation_placeholders(err),
            )
        except Exception:
            _LOGGER.exception("Unexpected exception validating AI task model")
            if error_step == "output_mode":
                return self.async_show_form(
                    step_id="output_mode",
                    data_schema=_ai_task_output_mode_schema(self._options | data),
                    errors={"base": "unknown"},
                )
            return self.async_show_form(
                step_id="init",
                data_schema=_ai_task_data_schema(self._options | data),
                errors={"base": "unknown"},
            )
        if self._is_new:
            return self.async_create_entry(title=data[CONF_MODEL], data=data)
        return self.async_update_and_abort(
            entry,
            self._get_reconfigure_subentry(),
            title=data[CONF_MODEL],
            data=data,
        )
