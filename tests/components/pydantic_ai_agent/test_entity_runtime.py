"""Test shared entity runtime helper behavior."""

import errno
import socket
import ssl
from typing import Any

import httpx
import pytest
from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    TextPart,
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

from homeassistant.exceptions import HomeAssistantError

from custom_components.pydantic_ai_agent.entity import (
    _agent_messages_to_chat_deltas,
    _has_connection_failure,
    _home_assistant_error,
    _should_fallback,
)
from custom_components.pydantic_ai_agent.mcp import MCPValidationError


async def _collect_deltas(messages: list[Any], output_tool_names: set[str]) -> list[dict[str, Any]]:
    """Collect chat deltas from the async generator."""
    return [
        delta
        async for delta in _agent_messages_to_chat_deltas(messages, output_tool_names)
    ]


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
