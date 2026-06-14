"""Test provider model probing for Pydantic AI Agent."""

from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from custom_components.pydantic_ai_agent.const import (
    CONF_BASE_URL,
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
    CONF_MAX_ITERATIONS,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_TEMPLATED_EXTRA_BODY,
    CONF_THINKING,
    OUTPUT_MODE_NATIVE,
    OUTPUT_MODE_PROMPTED,
    OUTPUT_MODE_TOOL,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE_GEMINI,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
)
from custom_components.pydantic_ai_agent.provider_validation import (
    ProviderValidationError,
    async_probe_model,
)
from homeassistant.core import HomeAssistant
from pydantic_ai import ModelResponse, TextPart
from tests.components.pydantic_ai_agent.support.probe_model import (
    FailingStreamContext,
    HTTPErrorStreamContext,
    SingleEventStream,
    StructuredTextStream,
    StructuredToolStream,
    provider_data,
    stream_context,
)


async def test_probe_model_uses_streaming(hass: HomeAssistant) -> None:
    """Test provider validation completes a streaming response."""
    model = object()
    stream_events = SingleEventStream()

    with (
        patch(
            "custom_components.pydantic_ai_agent.provider_validation._openai_compatible_model",
            return_value=model,
        ),
        patch(
            "custom_components.pydantic_ai_agent.provider_validation.model_request_stream",
            side_effect=lambda *_args, **_kwargs: stream_context(stream_events),
        ) as model_request_stream,
    ):
        await async_probe_model(hass, provider_data(), "gpt-test")

    model_request_stream.assert_called_once()
    assert model_request_stream.call_args.kwargs["model_settings"]["timeout"] == 10.0


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
            return_value=FailingStreamContext(),
        ),
        pytest.raises(ProviderValidationError) as exc_info,
    ):
        await async_probe_model(hass, provider_data(), "gpt-test")

    assert exc_info.value.reason == "model_does_not_support_streaming"
    assert exc_info.value.message


async def test_probe_model_uses_non_streaming_request_when_disabled(
    hass: HomeAssistant,
) -> None:
    """Test provider validation can probe via non-streaming requests."""
    model = object()

    with (
        patch(
            "custom_components.pydantic_ai_agent.provider_validation._openai_compatible_model",
            return_value=model,
        ),
        patch(
            "custom_components.pydantic_ai_agent.provider_validation.model_request",
            return_value=ModelResponse(parts=[TextPart(content="OK")]),
        ) as model_request,
        patch(
            "custom_components.pydantic_ai_agent.provider_validation.model_request_stream",
        ) as model_request_stream,
    ):
        await async_probe_model(hass, provider_data(), "gpt-test", stream=False)

    model_request.assert_called_once()
    model_request_stream.assert_not_called()


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
            provider_data(PROVIDER_GOOGLE_GEMINI),
            "gemini-3.1-flash-lite",
        )

    assert exc_info.value.reason == "timeout"
    assert exc_info.value.message


@pytest.mark.parametrize(
    ("output_mode", "stream_events"),
    [
        (OUTPUT_MODE_NATIVE, StructuredTextStream()),
        (OUTPUT_MODE_PROMPTED, StructuredTextStream()),
        (OUTPUT_MODE_TOOL, StructuredToolStream()),
    ],
)
async def test_probe_model_can_require_structured_output(
    hass: HomeAssistant,
    output_mode: str,
    stream_events: StructuredTextStream | StructuredToolStream,
) -> None:
    """Test provider probing can request each structured-output mode."""

    with (
        patch(
            "custom_components.pydantic_ai_agent.provider_validation._openai_compatible_model"
        ),
        patch(
            "custom_components.pydantic_ai_agent.provider_validation.model_request_stream",
            side_effect=lambda *_args, **_kwargs: stream_context(stream_events),
        ),
    ):
        await async_probe_model(
            hass,
            provider_data(),
            "gpt-test",
            structured_output_mode=output_mode,
        )


async def test_probe_model_rejects_invalid_native_structured_output(
    hass: HomeAssistant,
) -> None:
    """Test native structured output probing rejects non-JSON responses."""
    stream_events = StructuredTextStream("OK")

    with (
        patch(
            "custom_components.pydantic_ai_agent.provider_validation._openai_compatible_model"
        ),
        patch(
            "custom_components.pydantic_ai_agent.provider_validation.model_request_stream",
            side_effect=lambda *_args, **_kwargs: stream_context(stream_events),
        ),
        pytest.raises(ProviderValidationError) as exc_info,
    ):
        await async_probe_model(
            hass,
            provider_data(),
            "gpt-test",
            structured_output_mode=OUTPUT_MODE_NATIVE,
        )

    assert exc_info.value.reason == "invalid_provider_config"
    assert exc_info.value.message


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
            return_value=HTTPErrorStreamContext(status_code=400),
        ),
        pytest.raises(ProviderValidationError) as exc_info,
    ):
        await async_probe_model(
            hass,
            provider_data(),
            "gpt-test",
            structured_output_mode=OUTPUT_MODE_TOOL,
        )

    assert exc_info.value.reason == "unsupported_output_mode"
    assert exc_info.value.status_code == 400
    assert "structured output mode" in exc_info.value.message


async def test_probe_model_merges_configured_model_settings(
    hass: HomeAssistant,
) -> None:
    """Test provider validation preserves configured model settings."""
    stream_events = SingleEventStream()

    with (
        patch(
            "custom_components.pydantic_ai_agent.provider_validation._openai_compatible_model"
        ),
        patch(
            "custom_components.pydantic_ai_agent.provider_validation.model_request_stream",
            side_effect=lambda *_args, **_kwargs: stream_context(stream_events),
        ) as model_request_stream,
    ):
        await async_probe_model(
            hass,
            provider_data(),
            "gpt-5",
            {
                "temperature": 0.7,
                "timeout": 30.0,
                CONF_MAX_ITERATIONS: 20,
                CONF_THINKING: "high",
            },
        )

    model_settings = model_request_stream.call_args.kwargs["model_settings"]
    assert model_settings["temperature"] == 0.7
    assert model_settings["timeout"] == 30.0
    assert CONF_MAX_ITERATIONS not in model_settings
    assert (
        model_request_stream.call_args.kwargs["model_request_parameters"].thinking
        == "high"
    )


@pytest.mark.parametrize(
    ("provider_mode", "model_name", "thinking", "expected_thinking"),
    [
        (
            PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            "deepseek-v4-flash",
            "high",
            None,
        ),
        (PROVIDER_GOOGLE_GEMINI, "gemini-2.5-pro", False, None),
        (PROVIDER_ANTHROPIC, "claude-sonnet-4", "high", "high"),
    ],
)
async def test_probe_model_filters_thinking_by_effective_profile_support(
    hass: HomeAssistant,
    provider_mode: str,
    model_name: str,
    thinking: bool | str,
    expected_thinking: bool | str | None,
) -> None:
    """Test probe-time thinking respects effective profile capabilities."""
    stream_events = SingleEventStream()

    with (
        patch(
            "custom_components.pydantic_ai_agent.provider_validation._openai_compatible_model"
        ),
        patch(
            "custom_components.pydantic_ai_agent.provider_validation.model_request_stream",
            side_effect=lambda *_args, **_kwargs: stream_context(stream_events),
        ) as model_request_stream,
    ):
        await async_probe_model(
            hass,
            provider_data(provider_mode),
            model_name,
            {CONF_THINKING: thinking},
        )

    request_parameters = model_request_stream.call_args.kwargs[
        "model_request_parameters"
    ]
    if expected_thinking is None:
        assert request_parameters is None
    else:
        assert request_parameters is not None
        assert request_parameters.thinking == expected_thinking


async def test_probe_model_renders_templated_extra_body(hass: HomeAssistant) -> None:
    """Test provider validation renders templated extra body."""
    stream_events = SingleEventStream()

    with (
        patch(
            "custom_components.pydantic_ai_agent.provider_validation._openai_compatible_model"
        ),
        patch(
            "custom_components.pydantic_ai_agent.provider_validation.model_request_stream",
            side_effect=lambda *_args, **_kwargs: stream_context(stream_events),
        ) as model_request_stream,
    ):
        await async_probe_model(
            hass,
            provider_data()
            | {
                CONF_PROVIDER_EXTRA_BODY: {
                    "service_tier": "flex",
                    "metadata": {"provider": "base"},
                }
            },
            "gpt-test",
            {
                CONF_TEMPLATED_EXTRA_BODY: [
                    {
                        CONF_CHAT_TEMPLATE_KWARG_KEY: (
                            "chat_template_kwargs.enable_thinking"
                        ),
                        CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ true }}",
                    },
                    {
                        CONF_CHAT_TEMPLATE_KWARG_KEY: "metadata.profile",
                        CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: '{{ "rendered" }}',
                    },
                ],
            },
        )

    model_settings = model_request_stream.call_args.kwargs["model_settings"]
    assert model_settings["timeout"] == 10.0
    extra_body = model_settings["extra_body"]
    assert extra_body["service_tier"] == "flex"
    assert extra_body["metadata"]["provider"] == "base"
    assert extra_body["metadata"]["profile"] == "rendered"
    assert extra_body["chat_template_kwargs"]["enable_thinking"] is True


async def test_probe_model_responses_uses_streamed_request(
    hass: HomeAssistant,
) -> None:
    """Test Responses provider validation uses the streaming request path."""
    model = SimpleNamespace()
    stream_events = SingleEventStream()

    with (
        patch(
            "custom_components.pydantic_ai_agent.provider_validation._openai_compatible_model",
            return_value=model,
        ),
        patch(
            "custom_components.pydantic_ai_agent.provider_validation.model_request_stream",
            side_effect=lambda *_args, **_kwargs: stream_context(stream_events),
        ) as model_request_stream,
    ):
        await async_probe_model(
            hass,
            provider_data(PROVIDER_OPENAI_COMPATIBLE_RESPONSES),
            "gpt-test",
        )

    model_request_stream.assert_called_once()


async def test_probe_model_openai_compatible_uses_normalized_base_url(
    hass: HomeAssistant,
) -> None:
    """Test OpenAI-compatible validation builds a provider with the base URL."""
    data = provider_data() | {CONF_BASE_URL: "http://localhost:11434/v1/"}
    provider = object()
    model = object()
    stream_events = SingleEventStream()

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
            side_effect=lambda *_args, **_kwargs: stream_context(stream_events),
        ) as model_request_stream,
    ):
        await async_probe_model(hass, data, "local-model")

    compatible_provider.assert_called_once()
    assert (
        compatible_provider.call_args.kwargs["base_url"] == "http://localhost:11434/v1"
    )
    compatible_chat_model.assert_called_once_with("local-model", provider=provider)
    model_request_stream.assert_called_once()
