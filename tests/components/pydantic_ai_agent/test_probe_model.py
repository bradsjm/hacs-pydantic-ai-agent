"""Test provider model probing for Pydantic AI Agent."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from custom_components.pydantic_ai_agent.const import (
    CONF_BASE_URL,
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
    CONF_CHAT_TEMPLATE_KWARGS,
    CONF_MAX_ITERATIONS,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_MODE,
    CONF_THINKING,
    OUTPUT_MODE_NATIVE,
    OUTPUT_MODE_PROMPTED,
    OUTPUT_MODE_TOOL,
    PROVIDER_GOOGLE_GEMINI,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
)
from custom_components.pydantic_ai_agent.provider_validation import (
    ProviderValidationError,
    async_probe_model,
)
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant
from pydantic_ai import PartEndEvent, PartStartEvent, TextPart, ToolCallPart
from pydantic_ai.exceptions import ModelHTTPError


class _SingleEventStream:
    """Async stream with one validation event."""

    def __init__(self) -> None:
        """Initialize the stream."""
        self._yielded = False
        self.events_yielded = 0

    def __aiter__(self) -> _SingleEventStream:
        """Return the async iterator."""
        return self

    async def __anext__(self) -> object:
        """Return one event, then stop."""
        if self._yielded:
            raise StopAsyncIteration
        self._yielded = True
        self.events_yielded += 1
        return object()


class _StructuredTextStream:
    """Async stream with text structured-output events."""

    def __init__(self, content: str = '{"ok":true}') -> None:
        """Initialize the stream."""
        self._events = iter(
            (
                PartStartEvent(index=0, part=TextPart(content=content)),
                PartEndEvent(index=0, part=TextPart(content=content)),
            )
        )

    def __aiter__(self) -> _StructuredTextStream:
        """Return the async iterator."""
        return self

    async def __anext__(self) -> object:
        """Return the next stream event."""
        try:
            return next(self._events)
        except StopIteration as err:
            raise StopAsyncIteration from err


class _StructuredToolStream:
    """Async stream with an output-tool event."""

    def __init__(self) -> None:
        """Initialize the stream."""
        self._events = iter(
            (
                PartEndEvent(
                    index=0,
                    part=ToolCallPart(
                        tool_name="pydantic_ai_agent_output_probe_response",
                        args={"ok": True},
                        tool_call_id="tool-1",
                    ),
                ),
            )
        )

    def __aiter__(self) -> _StructuredToolStream:
        """Return the async iterator."""
        return self

    async def __anext__(self) -> object:
        """Return the next stream event."""
        try:
            return next(self._events)
        except StopIteration as err:
            raise StopAsyncIteration from err


class _FailingStreamContext:
    """Async context manager that fails before streaming starts."""

    async def __aenter__(self) -> object:
        """Raise the streaming failure."""
        raise NotImplementedError("Streamed requests not supported")

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        """Do not suppress exceptions."""
        return False


class _HTTPErrorStreamContext:
    """Async context manager that fails with a provider HTTP error."""

    def __init__(self, status_code: int = 429) -> None:
        """Initialize the HTTP error status code."""
        self._status_code = status_code

    async def __aenter__(self) -> object:
        """Raise a provider HTTP error."""
        raise ModelHTTPError(
            status_code=self._status_code, model_name="gpt-test", body=None
        )

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        """Do not suppress exceptions."""
        return False


def _provider_data(
    provider_mode: str = PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
) -> dict[str, object]:
    """Return provider data for model probes."""
    return {
        CONF_NAME: "Hosted OpenAI",
        CONF_PROVIDER_MODE: provider_mode,
        CONF_API_KEY: "sk-test",
    }


async def test_probe_model_uses_streaming(hass: HomeAssistant) -> None:
    """Test provider validation completes a streaming response."""
    model = object()
    stream_events = _SingleEventStream()

    @asynccontextmanager
    async def stream(*_: object, **__: object) -> AsyncGenerator[_SingleEventStream]:
        yield stream_events

    with (
        patch(
            "custom_components.pydantic_ai_agent.provider_validation._openai_compatible_model",
            return_value=model,
        ),
        patch(
            "custom_components.pydantic_ai_agent.provider_validation.model_request_stream",
            side_effect=stream,
        ) as model_request_stream,
    ):
        await async_probe_model(hass, _provider_data(), "gpt-test")

    model_request_stream.assert_called_once()
    assert model_request_stream.call_args.args[0] is model
    assert model_request_stream.call_args.kwargs["model_settings"] == {"timeout": 10.0}
    assert stream_events.events_yielded == 1


async def test_probe_model_streaming_not_supported_reported(
    hass: HomeAssistant,
) -> None:
    """Test non-streaming models are reported explicitly."""
    with (
        patch(
            "custom_components.pydantic_ai_agent.provider_validation._openai_compatible_model"
        ),
        patch(
            "custom_components.pydantic_ai_agent.provider_validation.model_request_stream",
            return_value=_FailingStreamContext(),
        ),
        pytest.raises(ProviderValidationError) as exc_info,
    ):
        await async_probe_model(hass, _provider_data(), "gpt-test")

    assert exc_info.value.reason == "model_does_not_support_streaming"
    assert exc_info.value.message == "Streamed requests not supported"


async def test_probe_model_maps_raw_httpx_timeout(hass: HomeAssistant) -> None:
    """Test raw SDK/httpx timeouts are reported as validation timeouts."""
    with (
        patch(
            "custom_components.pydantic_ai_agent.provider_validation._openai_compatible_model"
        ),
        patch(
            "custom_components.pydantic_ai_agent.provider_validation.model_request_stream",
            side_effect=httpx.ReadTimeout("timeout"),
        ),
        pytest.raises(ProviderValidationError) as exc_info,
    ):
        await async_probe_model(
            hass,
            _provider_data(PROVIDER_GOOGLE_GEMINI),
            "gemini-3.1-flash-lite",
        )

    assert exc_info.value.reason == "timeout"
    assert exc_info.value.message == "Request timed out."


@pytest.mark.parametrize(
    ("output_mode", "stream_events", "expected_request_mode"),
    [
        (OUTPUT_MODE_NATIVE, _StructuredTextStream(), "native"),
        (OUTPUT_MODE_PROMPTED, _StructuredTextStream(), "prompted"),
        (OUTPUT_MODE_TOOL, _StructuredToolStream(), "tool"),
    ],
)
async def test_probe_model_can_require_structured_output(
    hass: HomeAssistant,
    output_mode: str,
    stream_events: _StructuredTextStream | _StructuredToolStream,
    expected_request_mode: str,
) -> None:
    """Test provider probing can request each structured-output mode."""

    @asynccontextmanager
    async def stream(
        *_: object, **__: object
    ) -> AsyncGenerator[_StructuredTextStream | _StructuredToolStream]:
        yield stream_events

    with (
        patch(
            "custom_components.pydantic_ai_agent.provider_validation._openai_compatible_model"
        ),
        patch(
            "custom_components.pydantic_ai_agent.provider_validation.model_request_stream",
            side_effect=stream,
        ) as model_request_stream,
    ):
        await async_probe_model(
            hass,
            _provider_data(),
            "gpt-test",
            structured_output_mode=output_mode,
        )

    request_parameters = model_request_stream.call_args.kwargs[
        "model_request_parameters"
    ]
    assert request_parameters.output_mode == expected_request_mode
    if output_mode == OUTPUT_MODE_TOOL:
        assert request_parameters.allow_text_output is False
        assert request_parameters.output_tools[0].kind == "output"
    else:
        assert (
            request_parameters.output_object.name
            == "pydantic_ai_agent_output_probe_response"
        )


async def test_probe_model_rejects_invalid_native_structured_output(
    hass: HomeAssistant,
) -> None:
    """Test native structured output probing rejects non-JSON responses."""

    @asynccontextmanager
    async def stream(*_: object, **__: object) -> AsyncGenerator[_StructuredTextStream]:
        yield _StructuredTextStream("OK")

    with (
        patch(
            "custom_components.pydantic_ai_agent.provider_validation._openai_compatible_model"
        ),
        patch(
            "custom_components.pydantic_ai_agent.provider_validation.model_request_stream",
            side_effect=stream,
        ),
        pytest.raises(ProviderValidationError) as exc_info,
    ):
        await async_probe_model(
            hass,
            _provider_data(),
            "gpt-test",
            structured_output_mode=OUTPUT_MODE_NATIVE,
        )

    assert exc_info.value.reason == "invalid_provider_config"
    assert (
        exc_info.value.message
        == "The provider did not return valid native structured output."
    )


async def test_probe_model_maps_structured_http_400_to_output_mode_error(
    hass: HomeAssistant,
) -> None:
    """Test structured-output HTTP 400 is not reported as an invalid model."""
    with (
        patch(
            "custom_components.pydantic_ai_agent.provider_validation._openai_compatible_model"
        ),
        patch(
            "custom_components.pydantic_ai_agent.provider_validation.model_request_stream",
            return_value=_HTTPErrorStreamContext(status_code=400),
        ),
        pytest.raises(ProviderValidationError) as exc_info,
    ):
        await async_probe_model(
            hass,
            _provider_data(),
            "gpt-test",
            structured_output_mode=OUTPUT_MODE_TOOL,
        )

    assert exc_info.value.reason == "unsupported_output_mode"
    assert exc_info.value.status_code == 400
    assert 'structured output mode "tool"' in exc_info.value.message


async def test_probe_model_merges_configured_model_settings(
    hass: HomeAssistant,
) -> None:
    """Test provider validation preserves configured model settings."""
    stream_events = _SingleEventStream()

    @asynccontextmanager
    async def stream(*_: object, **__: object) -> AsyncGenerator[_SingleEventStream]:
        yield stream_events

    with (
        patch(
            "custom_components.pydantic_ai_agent.provider_validation._openai_compatible_model"
        ),
        patch(
            "custom_components.pydantic_ai_agent.provider_validation.model_request_stream",
            side_effect=stream,
        ) as model_request_stream,
    ):
        await async_probe_model(
            hass,
            _provider_data(),
            "gpt-test",
            {
                "temperature": 0.7,
                "timeout": 30.0,
                CONF_MAX_ITERATIONS: 20,
                CONF_THINKING: "high",
            },
        )

    assert model_request_stream.call_args.kwargs["model_settings"] == {
        "temperature": 0.7,
        "timeout": 30.0,
    }
    assert (
        model_request_stream.call_args.kwargs["model_request_parameters"].thinking
        == "high"
    )


async def test_probe_model_renders_chat_template_kwargs(hass: HomeAssistant) -> None:
    """Test provider validation renders dedicated chat template kwargs."""
    stream_events = _SingleEventStream()

    @asynccontextmanager
    async def stream(*_: object, **__: object) -> AsyncGenerator[_SingleEventStream]:
        yield stream_events

    with (
        patch(
            "custom_components.pydantic_ai_agent.provider_validation._openai_compatible_model"
        ),
        patch(
            "custom_components.pydantic_ai_agent.provider_validation.model_request_stream",
            side_effect=stream,
        ) as model_request_stream,
    ):
        await async_probe_model(
            hass,
            _provider_data() | {CONF_PROVIDER_EXTRA_BODY: {"service_tier": "flex"}},
            "gpt-test",
            {
                CONF_CHAT_TEMPLATE_KWARGS: [
                    {
                        CONF_CHAT_TEMPLATE_KWARG_KEY: "enable_thinking",
                        CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ true }}",
                    }
                ],
            },
        )

    assert model_request_stream.call_args.kwargs["model_settings"] == {
        "extra_body": {
            "service_tier": "flex",
            CONF_CHAT_TEMPLATE_KWARGS: {"enable_thinking": True},
        },
        "timeout": 10.0,
    }


async def test_probe_model_responses_uses_streamed_request(
    hass: HomeAssistant,
) -> None:
    """Test Responses provider validation uses the streaming request path."""
    model = SimpleNamespace()
    stream_events = _SingleEventStream()

    @asynccontextmanager
    async def stream(*_: object, **__: object) -> AsyncGenerator[_SingleEventStream]:
        yield stream_events

    with (
        patch(
            "custom_components.pydantic_ai_agent.provider_validation._openai_compatible_model",
            return_value=model,
        ),
        patch(
            "custom_components.pydantic_ai_agent.provider_validation.model_request_stream",
            side_effect=stream,
        ) as model_request_stream,
    ):
        await async_probe_model(
            hass,
            _provider_data(PROVIDER_OPENAI_COMPATIBLE_RESPONSES),
            "gpt-test",
        )

    model_request_stream.assert_called_once()
    assert model_request_stream.call_args.args[0] is model
    assert stream_events.events_yielded == 1


async def test_probe_model_openai_compatible_uses_normalized_base_url(
    hass: HomeAssistant,
) -> None:
    """Test OpenAI-compatible validation builds a provider with the base URL."""
    data = _provider_data() | {CONF_BASE_URL: "http://localhost:11434/v1/"}
    provider = object()
    model = object()
    stream_events = _SingleEventStream()

    @asynccontextmanager
    async def stream(*_: object, **__: object) -> AsyncGenerator[_SingleEventStream]:
        yield stream_events

    with (
        patch(
            "custom_components.pydantic_ai_agent.provider.OpenAICompatibleProvider",
            return_value=provider,
        ) as compatible_provider,
        patch(
            "custom_components.pydantic_ai_agent.provider.OpenAICompatibleChatModel",
            return_value=model,
        ) as compatible_chat_model,
        patch(
            "custom_components.pydantic_ai_agent.provider_validation.model_request_stream",
            side_effect=stream,
        ) as model_request_stream,
    ):
        await async_probe_model(hass, data, "local-model")

    compatible_provider.assert_called_once()
    assert compatible_provider.call_args.kwargs["api_key"] == "sk-test"
    assert (
        compatible_provider.call_args.kwargs["base_url"] == "http://localhost:11434/v1"
    )
    assert compatible_provider.call_args.kwargs["http_client"] is not None
    compatible_chat_model.assert_called_once_with("local-model", provider=provider)
    assert model_request_stream.call_args.args[0] is model
    assert stream_events.events_yielded == 1
