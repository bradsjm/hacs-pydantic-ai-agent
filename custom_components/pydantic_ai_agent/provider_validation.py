"""Provider validation helpers for Pydantic AI Agent."""

import json
from collections.abc import AsyncIterable, Mapping
from dataclasses import replace
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
from pydantic_ai.direct import model_request, model_request_stream
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import ModelResponseStreamEvent
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings, ThinkingLevel

from ._provider_validation_errors import (
    ProviderValidationError,
    format_api_error,
    map_http_error,
    map_structured_http_error,
)
from .const import (
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_MODE,
    CONF_TEMPLATED_EXTRA_BODY,
    CONF_THINKING,
    CONF_TIMEOUT,
    DEFAULT_TIMEOUT,
    OUTPUT_MODE_NATIVE,
    OUTPUT_MODE_PROMPTED,
    OUTPUT_MODE_TOOL,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE_GEMINI,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
)
from .model_settings import (
    MODEL_SETTING_EXTRA_BODY,
    PROBE_STRIPPED_MODEL_SETTING_KEYS,
    strip_model_settings,
)
from .openai_compatible_profile import is_openai_compatible_provider_mode
from .provider import (
    anthropic_model,
    effective_thinking_setting,
    google_gemini_model,
    list_anthropic_model_names,
    list_google_gemini_model_names,
    normalise_base_url,
    openai_compatible_client_from_config,
    openai_compatible_completions_model_from_config,
    openai_compatible_effective_thinking_setting,
    openai_compatible_model_profile,
    openai_compatible_responses_model_from_config,
)
from .structured_output import (
    structured_model_request_parameters,
    structured_output_name,
)
from .structured_output import (
    structured_output_mode as normalise_structured_output_mode,
)
from .templated_extra_body import merge_extra_body, render_templated_extra_body

_format_api_error = format_api_error
_map_http_error = map_http_error
_map_structured_http_error = map_structured_http_error

_MODEL_SETTING_TIMEOUT = CONF_TIMEOUT
_MODEL_SETTING_THINKING = CONF_THINKING
_MODEL_SETTING_EXTRA_BODY = MODEL_SETTING_EXTRA_BODY
_MODEL_SETTING_TEMPLATED_EXTRA_BODY = CONF_TEMPLATED_EXTRA_BODY
_PROVIDER_EXTRA_BODY_MODES = {
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
}
_STRUCTURED_PROBE_OUTPUT_NAME = structured_output_name(
    "probe_response", "probe_response"
)
_STRUCTURED_PROBE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def provider_extra_body_supported(data: Mapping[str, Any]) -> bool:
    """Return if the provider mode consumes provider-level extra body."""
    return data.get(CONF_PROVIDER_MODE) in _PROVIDER_EXTRA_BODY_MODES


def _openai_compatible_model(
    hass: HomeAssistant,
    data: Mapping[str, Any],
    model_name: str,
    *,
    profile_id: str | None = None,
) -> Model:
    """Build a Pydantic AI model for validation."""
    provider_mode = data[CONF_PROVIDER_MODE]
    if is_openai_compatible_provider_mode(provider_mode):
        try:
            profile = openai_compatible_model_profile(
                _persisted_openai_compatible_profile_data(data, model_name, profile_id)
            )
        except (KeyError, ValueError) as err:
            raise ProviderValidationError("invalid_provider_config", str(err)) from err
        if provider_mode == PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS:
            return openai_compatible_completions_model_from_config(
                hass,
                data,
                model_name,
                profile=profile,
            )
        return openai_compatible_responses_model_from_config(
            hass,
            data,
            model_name,
            profile=profile,
        )
    try:
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


def _prepare_probe_settings(
    hass: HomeAssistant,
    data: Mapping[str, Any],
    model_settings: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Prepare model settings dict for a probe request."""
    settings = strip_model_settings(model_settings, PROBE_STRIPPED_MODEL_SETTING_KEYS)
    provider_extra_body = data.get(CONF_PROVIDER_EXTRA_BODY)
    if isinstance(provider_extra_body, Mapping) and provider_extra_body:
        if not provider_extra_body_supported(data):
            raise ProviderValidationError(
                "provider_extra_body_unsupported",
                "Extra body is only supported by OpenAI-compatible"
                " and Anthropic provider modes.",
            )
        settings[_MODEL_SETTING_EXTRA_BODY] = dict(provider_extra_body)
    templated_extra_body = settings.pop(_MODEL_SETTING_TEMPLATED_EXTRA_BODY, None)
    if rendered_extra_body := render_templated_extra_body(hass, templated_extra_body):
        settings[_MODEL_SETTING_EXTRA_BODY] = merge_extra_body(
            settings.get(_MODEL_SETTING_EXTRA_BODY), rendered_extra_body
        )
    settings.setdefault(_MODEL_SETTING_TIMEOUT, DEFAULT_TIMEOUT)
    return settings


def _probe_messages(
    structured_output_mode: str | None,
) -> list[ModelRequest]:
    """Return probe messages for the given structured output mode."""
    return [
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


async def _run_probe_stream(
    model: Model,
    settings: dict[str, Any],
    model_request_parameters: ModelRequestParameters | None,
    structured_output_mode: str | None,
) -> None:
    """Run a probe request stream and validate the response."""
    model_settings_obj = ModelSettings(**settings)
    async with model_request_stream(
        model,
        _probe_messages(structured_output_mode),
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


async def _run_probe(
    model: Model,
    settings: dict[str, Any],
    model_request_parameters: ModelRequestParameters | None,
) -> None:
    """Run a probe request without streaming and validate the response."""
    response = await model_request(
        model,
        _probe_messages(None),
        model_settings=ModelSettings(**settings),
        model_request_parameters=model_request_parameters,
    )
    if not any(
        isinstance(part, TextPart) and part.content
        for part in getattr(response, "parts", ())
    ):
        raise ProviderValidationError(
            "provider_error",
            "The provider returned an empty response.",
        )


def _build_probe_request_parameters(
    structured_output_mode: str | None,
    thinking: ThinkingLevel | None,
) -> ModelRequestParameters | None:
    """Build model request parameters for a probe request."""
    params: ModelRequestParameters | None = None
    if structured_output_mode is not None:
        output_mode = normalise_structured_output_mode(structured_output_mode)
        params = _structured_probe_request_parameters(output_mode)
    if thinking is not None:
        params = (
            ModelRequestParameters(thinking=thinking)
            if params is None
            else replace(params, thinking=thinking)
        )
    return params


def _probe_thinking(
    data: Mapping[str, Any],
    model_name: str,
    thinking: ThinkingLevel | None,
    *,
    profile_id: str | None = None,
) -> ThinkingLevel | None:
    """Return probe thinking only when the effective runtime profile supports it."""
    try:
        provider_mode = data[CONF_PROVIDER_MODE]
        if is_openai_compatible_provider_mode(provider_mode):
            return openai_compatible_effective_thinking_setting(
                _persisted_openai_compatible_profile_data(
                    data, model_name, profile_id
                ),
                thinking,
            )
        return effective_thinking_setting(
            provider_mode, model_name, thinking
        )
    except (KeyError, ValueError) as err:
        if is_openai_compatible_provider_mode(data[CONF_PROVIDER_MODE]):
            raise ProviderValidationError("invalid_provider_config", str(err)) from err
        return None


def _persisted_openai_compatible_profile_data(
    data: Mapping[str, Any],
    model_name: str,
    profile_id: str | None = None,
) -> Mapping[str, Any]:
    """Return persisted OpenAI-compatible profile data for one model probe."""
    profiles = data.get(CONF_MODEL_PROFILES)
    if not isinstance(profiles, Mapping):
        raise ProviderValidationError(
            "invalid_provider_config",
            "OpenAI-compatible model profiles require persisted capability settings.",
        )
    if profile_id is None:
        raise ProviderValidationError(
            "invalid_provider_config",
            "OpenAI-compatible model probes require a persisted profile id.",
        )
    profile = profiles.get(profile_id)
    if isinstance(profile, Mapping):
        configured_model_name = profile.get(CONF_MODEL)
        if configured_model_name == model_name:
            return profile
    raise ProviderValidationError(
        "invalid_provider_config",
        "OpenAI-compatible model profile capabilities were not found.",
    )


async def async_probe_model(
    hass: HomeAssistant,
    data: Mapping[str, Any],
    model_name: str,
    model_settings: Mapping[str, Any] | None = None,
    *,
    profile_id: str | None = None,
    structured_output_mode: str | None = None,
    stream: bool = True,
) -> None:
    """Probe model access with the same streaming path used at runtime."""
    try:
        settings = _prepare_probe_settings(hass, data, model_settings)
        thinking = (
            None
            if model_settings is None
            else model_settings.get(_MODEL_SETTING_THINKING)
        )
        thinking = _probe_thinking(
            data,
            model_name,
            thinking,
            profile_id=profile_id,
        )
        model = _openai_compatible_model(
            hass,
            data,
            model_name,
            profile_id=profile_id,
        )
        model_request_parameters = _build_probe_request_parameters(
            structured_output_mode, thinking
        )
        if structured_output_mode is not None or stream:
            await _run_probe_stream(
                model, settings, model_request_parameters, structured_output_mode
            )
        else:
            await _run_probe(model, settings, model_request_parameters)
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
