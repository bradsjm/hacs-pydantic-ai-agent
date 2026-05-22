"""Test shared entity runtime helper behavior."""

import errno
import socket
import ssl
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from pydantic_ai import (
    AgentRunResultEvent,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.settings import ModelSettings

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from custom_components.pydantic_ai_agent.const import (
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
    CONF_CHAT_TEMPLATE_KWARGS,
    CONF_PROVIDER_EXTRA_BODY,
    PROVIDER_GOOGLE_GEMINI,
)

from custom_components.pydantic_ai_agent.entity import (
    _StreamRunState,
    _agent_events_to_chat_deltas,
    _agent_messages_to_chat_deltas,
    _has_connection_failure,
    _home_assistant_error,
    _model_settings_with_chat_template_kwargs,
    _model_settings_with_provider_extra_body,
    _should_fallback,
)
from custom_components.pydantic_ai_agent.metrics import MetricsStore, record_run_failure
from custom_components.pydantic_ai_agent.mcp import MCPValidationError
from custom_components.pydantic_ai_agent.model_profiles import ModelProfile


async def _collect_deltas(
    messages: list[Any], output_tool_names: set[str]
) -> list[dict[str, Any]]:
    """Collect chat deltas from the async generator."""
    return [
        delta
        async for delta in _agent_messages_to_chat_deltas(messages, output_tool_names)
    ]


async def _collect_event_deltas(events: list[Any]) -> tuple[list[dict[str, Any]], Any]:
    """Collect chat deltas from live Agent events."""

    async def stream() -> AsyncIterator[Any]:
        for event in events:
            yield event

    state = _StreamRunState()
    deltas = [
        delta async for delta in _agent_events_to_chat_deltas(stream(), set(), state)
    ]
    return deltas, state.result


@pytest.mark.parametrize("status_code", [408, 409, 429, 500, 503])
def test_should_fallback_for_retryable_http_errors(status_code: int) -> None:
    """Test transient provider HTTP errors can try fallback profiles."""
    assert _should_fallback(
        ModelHTTPError(status_code=status_code, model_name="gpt-test", body=None)
    )


@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
def test_should_not_fallback_for_non_retryable_http_errors(status_code: int) -> None:
    """Test permanent provider HTTP errors do not try fallback profiles."""
    assert not _should_fallback(
        ModelHTTPError(status_code=status_code, model_name="gpt-test", body=None)
    )


def test_should_fallback_for_timeout_usage_and_transport_api_errors() -> None:
    """Test fallback decisions include runtime and wrapped transport failures."""
    api_error = ModelAPIError("gpt-test", "request failed")
    api_error.__cause__ = httpx.ConnectError("refused")

    assert _should_fallback(TimeoutError())
    assert _should_fallback(httpx.ReadTimeout("timeout"))
    assert _should_fallback(UsageLimitExceeded("too many requests"))
    assert _should_fallback(api_error)
    assert not _should_fallback(ModelAPIError("gpt-test", "bad request"))


@pytest.mark.parametrize(
    "cause",
    [
        httpx.TimeoutException("timeout"),
        httpx.ConnectError("refused"),
        socket.gaierror(),
        ssl.SSLError("tls"),
        OSError(errno.ECONNREFUSED, "refused"),
        OSError(errno.ENETUNREACH, "unreachable"),
        OSError(errno.EHOSTUNREACH, "host unreachable"),
    ],
)
def test_has_connection_failure_detects_transport_cause(cause: BaseException) -> None:
    """Test connection classification walks wrapped exception causes."""
    err = RuntimeError("wrapper")
    err.__cause__ = RuntimeError("middle")
    err.__cause__.__context__ = cause

    assert _has_connection_failure(err)


def test_has_connection_failure_stops_on_cycles() -> None:
    """Test cyclic cause chains do not loop forever."""
    err = RuntimeError("cycle")
    err.__cause__ = err

    assert not _has_connection_failure(err)


@pytest.mark.parametrize(
    ("err", "message"),
    [
        (
            ModelHTTPError(status_code=429, model_name="gpt-test", body=None),
            'The provider returned HTTP 429 for model "gpt-test".',
        ),
        (
            ModelAPIError("gpt-test", "failed"),
            'The provider returned an API error for model "gpt-test".',
        ),
        (UnexpectedModelBehavior("bad"), "Provider returned an unexpected response"),
        (TimeoutError(), "Provider request timed out"),
        (httpx.ReadTimeout("timeout"), "Provider request timed out"),
        (UsageLimitExceeded("too many"), "Model requested too many tool iterations"),
        (
            MCPValidationError("invalid", "MCP failed"),
            "MCP failed",
        ),
        (
            NotImplementedError("missing config"),
            "Invalid provider configuration: missing config",
        ),
        (
            UserError("bad config"),
            "Invalid provider configuration: bad config",
        ),
    ],
)
def test_home_assistant_error_maps_runtime_failures(
    err: Exception, message: str
) -> None:
    """Test provider failures are converted to stable HA-facing errors."""
    assert str(_home_assistant_error(err)) == message


def test_home_assistant_error_preserves_existing_ha_errors() -> None:
    """Test HA errors are not wrapped again."""
    err = HomeAssistantError("already HA")

    assert _home_assistant_error(err) is err


def test_record_run_failure_updates_health_metrics(hass: HomeAssistant) -> None:
    """Test failed runs update native health metric state."""
    store = MetricsStore()

    record_run_failure(
        hass,
        "entry-1",
        store,
        "subentry-1",
        error=TimeoutError(),
    )

    record = store.record_for("subentry-1")
    assert record.last_error_type == "TimeoutError"
    assert record.consecutive_failures == 1
    assert record.provider_healthy is False
    assert record.last_run_succeeded is False


def test_model_settings_with_chat_template_kwargs_renders_without_mutation(
    hass: HomeAssistant,
) -> None:
    """Test dedicated chat template kwargs merge into copied extra_body."""
    profile = ModelProfile(
        ref="provider-1:model-1",
        provider_subentry_id="provider-1",
        profile_id="model-1",
        title="Fast GPT",
        provider_title="Provider",
        provider_mode="openai_compatible_completions",
        model_name="gpt-test",
        model_settings={
            CONF_CHAT_TEMPLATE_KWARGS: [
                {
                    CONF_CHAT_TEMPLATE_KWARG_KEY: "enable_thinking",
                    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ true }}",
                }
            ]
        },
    )
    settings = ModelSettings(extra_body={"service_tier": "flex"})

    result = _model_settings_with_chat_template_kwargs(hass, profile, settings)

    assert result == {
        "extra_body": {
            "service_tier": "flex",
            CONF_CHAT_TEMPLATE_KWARGS: {"enable_thinking": True},
        }
    }
    assert settings == {"extra_body": {"service_tier": "flex"}}


def test_model_settings_with_provider_extra_body_rejects_gemini() -> None:
    """Test unsupported provider body fields fail instead of becoming a no-op."""
    profile = ModelProfile(
        ref="provider-1:model-1",
        provider_subentry_id="provider-1",
        profile_id="model-1",
        title="Gemini",
        provider_title="Provider",
        provider_mode=PROVIDER_GOOGLE_GEMINI,
        model_name="gemini-test",
        model_settings={},
    )
    entry = cast(
        Any,
        SimpleNamespace(
            subentries={
                "provider-1": SimpleNamespace(
                    data={CONF_PROVIDER_EXTRA_BODY: {"service_tier": "flex"}}
                )
            }
        ),
    )

    with pytest.raises(HomeAssistantError, match="OpenAI-compatible and Anthropic"):
        _model_settings_with_provider_extra_body(entry, profile, ModelSettings())


async def test_agent_messages_to_chat_deltas_preserves_assistant_parts() -> None:
    """Test assistant text, thinking, and external tool calls become deltas."""
    deltas = await _collect_deltas(
        [
            ModelResponse(
                parts=[
                    TextPart(content="hello "),
                    ThinkingPart(content="reasoning"),
                    TextPart(content="world"),
                    ToolCallPart(
                        tool_name="HassTurnOn",
                        args={"name": "Kitchen"},
                        tool_call_id="tool-1",
                    ),
                ]
            )
        ],
        output_tool_names=set(),
    )

    assert len(deltas) == 1
    assert deltas[0]["role"] == "assistant"
    assert deltas[0]["content"] == "hello world"
    assert deltas[0]["thinking_content"] == "reasoning"
    tool_call = deltas[0]["tool_calls"][0]
    assert tool_call.tool_name == "HassTurnOn"
    assert tool_call.tool_args == {"name": "Kitchen"}
    assert tool_call.id == "tool-1"


async def test_agent_events_to_chat_deltas_does_not_replay_final_result() -> None:
    """Test live stream deltas are not duplicated by the final run result."""
    result = object()
    deltas, final_result = await _collect_event_deltas(
        [
            PartStartEvent(index=0, part=TextPart(content="hel")),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="lo")),
            AgentRunResultEvent(cast(Any, result)),
        ]
    )

    assert deltas == [{"role": "assistant"}, {"content": "hel"}, {"content": "lo"}]
    assert final_result is result


async def test_agent_messages_to_chat_deltas_converts_output_tool_to_content() -> None:
    """Test structured-output tool calls are rendered as assistant JSON content."""
    deltas = await _collect_deltas(
        [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="generated_data",
                        args={"summary": "ok"},
                        tool_call_id="output-1",
                    )
                ]
            )
        ],
        output_tool_names={"generated_data"},
    )

    assert deltas == [
        {
            "role": "assistant",
            "content": '{"summary": "ok"}',
            "thinking_content": "",
            "tool_calls": [],
        }
    ]


async def test_agent_messages_to_chat_deltas_preserves_tool_returns() -> None:
    """Test tool result request parts become HA tool-result deltas."""
    deltas = await _collect_deltas(
        [
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="HassTurnOn",
                        content={"success": True},
                        tool_call_id="tool-1",
                    )
                ]
            )
        ],
        output_tool_names=set(),
    )

    assert deltas == [
        {
            "role": "tool_result",
            "tool_call_id": "tool-1",
            "tool_name": "HassTurnOn",
            "tool_result": {"success": True},
        }
    ]
