"""Provider validation helpers for Pydantic AI Agent."""

import json
from collections.abc import AsyncIterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import httpx
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
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

from ._redaction import redact_data
from .chat_template_kwargs import (
    reject_chat_template_kwargs_in_extra_body,
    render_chat_template_kwargs,
)
from .const import (
    CONF_BASE_URL,
    CONF_CHAT_TEMPLATE_KWARGS,
    CONF_THINKING,
    CONF_TIMEOUT,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_MODE,
    DEFAULT_TIMEOUT,
    OUTPUT_MODE_NATIVE,
    OUTPUT_MODE_PROMPTED,
    OUTPUT_MODE_TOOL,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE_GEMINI,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
)
from .error_classification import connection_failure_message
from .model_settings import (
    MODEL_SETTING_EXTRA_BODY,
    PROBE_STRIPPED_MODEL_SETTING_KEYS,
    strip_model_settings,
)
from .provider import (
    anthropic_model,
    google_gemini_model,
    list_anthropic_model_names,
    list_google_gemini_model_names,
    normalise_base_url,
    openai_compatible_client_from_config,
    openai_compatible_completions_model_from_config,
    openai_compatible_responses_model_from_config,
)
from .structured_output import (
    structured_model_request_parameters,
    structured_output_mode as normalise_structured_output_mode,
    structured_output_name,
)

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
_MODEL_SETTING_TIMEOUT = CONF_TIMEOUT
_MODEL_SETTING_THINKING = CONF_THINKING
_MODEL_SETTING_EXTRA_BODY = MODEL_SETTING_EXTRA_BODY
_MODEL_SETTING_CHAT_TEMPLATE_KWARGS = CONF_CHAT_TEMPLATE_KWARGS
_PROVIDER_EXTRA_BODY_MODES = {
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
}
_MAX_METADATA_REPR_LENGTH = 1000
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


def provider_extra_body_supported(data: Mapping[str, Any]) -> bool:
    """Return if the provider mode consumes provider-level extra body."""
    return data.get(CONF_PROVIDER_MODE) in _PROVIDER_EXTRA_BODY_MODES


def _format_metadata(metadata: object) -> str:
    """Return redacted, bounded provider metadata for config-flow display."""
    redacted = redact_data(metadata)
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


def _format_connection_error(err: BaseException) -> str | None:
    """Return a well-defined connection error message if one can be identified."""
    return connection_failure_message(err)


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
            "base_url": normalise_base_url(data.get(CONF_BASE_URL)),
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
        settings = strip_model_settings(
            model_settings, PROBE_STRIPPED_MODEL_SETTING_KEYS
        )
        thinking = (
            None
            if model_settings is None
            else model_settings.get(_MODEL_SETTING_THINKING)
        )
        provider_extra_body = data.get(CONF_PROVIDER_EXTRA_BODY)
        if isinstance(provider_extra_body, Mapping) and provider_extra_body:
            if not provider_extra_body_supported(data):
                raise ProviderValidationError(
                    "provider_extra_body_unsupported",
                    "Extra body is only supported by OpenAI-compatible and Anthropic provider modes.",
                )
            reject_chat_template_kwargs_in_extra_body(provider_extra_body)
            settings[_MODEL_SETTING_EXTRA_BODY] = dict(provider_extra_body)
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
        if thinking is not None:
            model_request_parameters = (
                ModelRequestParameters(thinking=thinking)
                if model_request_parameters is None
                else replace(model_request_parameters, thinking=thinking)
            )
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
