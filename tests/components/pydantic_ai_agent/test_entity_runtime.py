"""Test shared entity runtime helper behavior."""

import errno
import logging
import socket
import ssl
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from pydantic_ai import (
    AgentRunResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
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
from pydantic_ai.usage import UsageLimits

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
    PydanticAIBaseLLMEntity,
    _StreamRunState,
    _agent_events_to_chat_deltas,
    _agent_messages_to_chat_deltas,
    _classify_run_failure,
    _has_connection_failure,
    _home_assistant_error,
    _model_settings_with_chat_template_kwargs,
    _model_settings_with_provider_extra_body,
    _should_fallback,
)
from custom_components.pydantic_ai_agent.metrics import (
    MetricsStore,
    record_run_failure,
    record_run_success,
)
from custom_components.pydantic_ai_agent.mcp import MCPValidationError
from custom_components.pydantic_ai_agent.model_profiles import ModelProfile
from custom_components.pydantic_ai_agent.virtual_workspace.const import (
    TOOL_RETURN_METADATA_SOURCE,
)


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
            "Terminated because the provider quota or rate limit was reached for "
            'model "gpt-test". Check provider quota/rate limits or try again later.',
        ),
        (
            ModelAPIError("gpt-test", "failed"),
            'The provider returned an API error for model "gpt-test".',
        ),
        (
            UnexpectedModelBehavior("bad"),
            "Terminated because the provider returned an unexpected response. Check "
            "model/provider compatibility or try a different model profile.",
        ),
        (
            TimeoutError(),
            "Terminated because the provider request timed out. Check network "
            "connectivity or try again later.",
        ),
        (
            httpx.ReadTimeout("timeout"),
            "Terminated because the provider request timed out. Check network "
            "connectivity or try again later.",
        ),
        (
            UsageLimitExceeded("too many"),
            "Terminated because the model exceeded a configured usage limit. "
            "Increase the relevant model profile limit or reduce the request.",
        ),
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


def test_classify_run_failure_uses_configured_iteration_limit() -> None:
    """Test usage-limit failures include the configured max iterations."""
    failure = _classify_run_failure(
        UsageLimitExceeded("The next request would exceed the request_limit of 24"),
        usage_limits=UsageLimits(request_limit=24),
        partial_response=True,
    )

    assert failure.error_type == "UsageLimitExceeded"
    assert failure.partial_response is True
    assert str(failure.user_message) == (
        "Terminated after a partial response because the model exceeded the "
        "configured maximum of 24 iterations. Increase the model profile max "
        "iterations or fix repeated tool failures."
    )


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


def test_record_run_failure_uses_classified_error_type(hass: HomeAssistant) -> None:
    """Test failed runs can store classified error types for sensors."""
    store = MetricsStore()

    record_run_failure(
        hass,
        "entry-1",
        store,
        "subentry-1",
        error=HomeAssistantError("wrapped"),
        error_type="UsageLimitExceeded",
    )

    record = store.record_for("subentry-1")
    assert record.last_error_type == "UsageLimitExceeded"
    assert record.consecutive_failures == 1


def test_record_agent_run_failure_logs_safe_message(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test terminal failure logs do not include raw provider response bodies."""
    store = MetricsStore()
    entity = SimpleNamespace(
        hass=hass,
        entry=SimpleNamespace(
            entry_id="entry-1",
            runtime_data=SimpleNamespace(metrics=store),
        ),
        subentry=SimpleNamespace(subentry_id="subentry-1"),
        entity_id="conversation.test_agent",
    )
    err = ModelHTTPError(
        status_code=500,
        model_name="gpt-test",
        body="raw provider body with prompt-adjacent content",
    )

    with caplog.at_level(logging.ERROR):
        PydanticAIBaseLLMEntity._record_agent_run_failure(
            cast(Any, entity),
            err,
            model_profile="GPT Test",
        )

    assert "raw provider body" not in caplog.text
    assert "ModelHTTPError" in caplog.text
    assert 'provider service returned HTTP 500 for model "gpt-test"' in caplog.text
    assert store.record_for("subentry-1").last_error_type == "ModelHTTPError"


def test_record_run_success_tracks_priced_costs(hass: HomeAssistant) -> None:
    """Test successful runs compute component and cumulative USD costs."""
    store = MetricsStore()

    record_run_success(
        hass,
        "entry-1",
        store,
        "subentry-1",
        model_profile="GPT Test",
        duration=1.2,
        usage=SimpleNamespace(
            input_tokens=1200,
            output_tokens=300,
            cache_read_tokens=200,
            total_tokens=1500,
            requests=1,
            tool_calls=0,
        ),
        model_pricing={"input": 0.5, "output": 2.0, "cache_read": 0.1},
    )

    record = store.record_for("subentry-1")
    assert record.last_run_input_cost == 1000 * 0.5 / 1_000_000
    assert record.last_run_output_cost == 300 * 2.0 / 1_000_000
    assert record.last_run_cache_read_cost == 200 * 0.1 / 1_000_000
    assert record.last_run_total_cost == pytest.approx(0.00112)
    assert record.cumulative_input_cost == record.last_run_input_cost
    assert record.cumulative_output_cost == record.last_run_output_cost
    assert record.cumulative_cache_read_cost == record.last_run_cache_read_cost
    assert record.cumulative_total_cost == record.last_run_total_cost


def test_record_run_success_leaves_total_cost_unknown_when_pricing_missing(
    hass: HomeAssistant,
) -> None:
    """Test total cost is unknown unless all used token buckets are priced."""
    store = MetricsStore()

    record_run_success(
        hass,
        "entry-1",
        store,
        "subentry-1",
        model_profile="GPT Test",
        duration=1.2,
        usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=300,
            cache_read_tokens=0,
            total_tokens=1300,
        ),
        model_pricing={"input": 0.5},
    )
    record_run_success(
        hass,
        "entry-1",
        store,
        "subentry-1",
        model_profile="GPT Test",
        duration=1.2,
        usage=SimpleNamespace(
            input_tokens=0,
            output_tokens=100,
            cache_read_tokens=0,
            total_tokens=100,
        ),
        model_pricing={"output": 2.0},
    )

    record = store.record_for("subentry-1")
    assert record.last_run_input_cost is None
    assert record.last_run_output_cost == 100 * 2.0 / 1_000_000
    assert record.last_run_total_cost == record.last_run_output_cost
    assert record.cumulative_input_cost == 1000 * 0.5 / 1_000_000
    assert record.cumulative_output_cost == record.last_run_output_cost
    assert record.cumulative_total_cost == record.last_run_output_cost


def test_record_run_success_reads_cached_tokens_from_usage_details(
    hass: HomeAssistant,
) -> None:
    """Test cached-token details are billed as cache reads."""
    store = MetricsStore()

    record_run_success(
        hass,
        "entry-1",
        store,
        "subentry-1",
        model_profile="GPT Test",
        duration=1.2,
        usage=SimpleNamespace(
            input_tokens=1200,
            output_tokens=300,
            total_tokens=1500,
            details={"input_tokens_details.cached_tokens": 200},
        ),
        model_pricing={"input": 0.5, "output": 2.0, "cache_read": 0.1},
    )

    record = store.record_for("subentry-1")
    assert record.last_run_cache_read_tokens == 200
    assert record.last_run_input_cost == 1000 * 0.5 / 1_000_000
    assert record.last_run_cache_read_cost == 200 * 0.1 / 1_000_000
    assert record.last_run_total_cost == pytest.approx(0.00112)


def test_record_run_success_leaves_total_unknown_for_unpriced_token_categories(
    hass: HomeAssistant,
) -> None:
    """Test unsupported token buckets keep total cost unknown."""
    store = MetricsStore()

    record_run_success(
        hass,
        "entry-1",
        store,
        "subentry-1",
        model_profile="GPT Test",
        duration=1.2,
        usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=300,
            cache_read_tokens=0,
            cache_write_tokens=50,
            total_tokens=1300,
        ),
        model_pricing={"input": 0.5, "output": 2.0, "cache_read": 0.1},
    )

    record = store.record_for("subentry-1")
    assert record.last_run_input_cost == 1000 * 0.5 / 1_000_000
    assert record.last_run_output_cost == 300 * 2.0 / 1_000_000
    assert record.last_run_total_cost is None
    assert record.cumulative_total_cost is None


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


async def test_agent_events_to_chat_deltas_streams_tool_call_sequence() -> None:
    """Test live tool call and result events are forwarded in stream order."""
    result = object()
    deltas, final_result = await _collect_event_deltas(
        [
            PartStartEvent(index=0, part=TextPart(content="turning ")),
            FunctionToolCallEvent(
                ToolCallPart(
                    tool_name="HassTurnOn",
                    args={"name": "Kitchen"},
                    tool_call_id="tool-1",
                )
            ),
            FunctionToolResultEvent(
                ToolReturnPart(
                    tool_name="HassTurnOn",
                    content={"success": True},
                    tool_call_id="tool-1",
                )
            ),
            PartStartEvent(index=0, part=TextPart(content="done")),
            AgentRunResultEvent(cast(Any, result)),
        ]
    )

    tool_call = deltas[2]["tool_calls"][0]
    assert deltas[0] == {"role": "assistant"}
    assert deltas[1] == {"content": "turning "}
    assert tool_call.tool_name == "HassTurnOn"
    assert tool_call.tool_args == {"name": "Kitchen"}
    assert tool_call.id == "tool-1"
    assert tool_call.external is True
    assert deltas[3] == {
        "role": "tool_result",
        "tool_call_id": "tool-1",
        "tool_name": "HassTurnOn",
        "tool_result": {"success": True},
    }
    assert deltas[4:] == [{"role": "assistant"}, {"content": "done"}]
    assert final_result is result


async def test_agent_events_to_chat_deltas_logs_tool_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test failed tool result payloads are logged and tracked as context."""

    async def stream() -> AsyncIterator[Any]:
        yield FunctionToolResultEvent(
            ToolReturnPart(
                tool_name="applyPatch",
                content={
                    "success": False,
                    "errors": ["patch must start with *** Begin Patch"],
                },
                tool_call_id="tool-1",
                metadata={"source": TOOL_RETURN_METADATA_SOURCE},
            )
        )

    state = _StreamRunState()

    with caplog.at_level(logging.WARNING):
        deltas = [
            delta
            async for delta in _agent_events_to_chat_deltas(stream(), set(), state)
        ]

    assert deltas == [
        {
            "role": "tool_result",
            "tool_call_id": "tool-1",
            "tool_name": "applyPatch",
            "tool_result": {
                "success": False,
                "errors": ["patch must start with *** Begin Patch"],
            },
        }
    ]
    assert state.latest_tool_problem is not None
    assert state.latest_tool_problem.tool_name == "applyPatch"
    assert state.latest_tool_problem.reason == "patch must start with *** Begin Patch"
    assert "Pydantic AI tool \"applyPatch\" returned failed" in caplog.text


async def test_agent_events_to_chat_deltas_redacts_untrusted_tool_failure_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test arbitrary tool error payloads are not copied into HA logs."""

    async def stream() -> AsyncIterator[Any]:
        yield FunctionToolResultEvent(
            ToolReturnPart(
                tool_name="applyPatch",
                content={
                    "success": False,
                    "error": "raw provider body with private data",
                },
                tool_call_id="tool-1",
            )
        )

    state = _StreamRunState()

    with caplog.at_level(logging.WARNING):
        deltas = [
            delta
            async for delta in _agent_events_to_chat_deltas(stream(), set(), state)
        ]

    assert deltas[0]["tool_name"] == "applyPatch"
    assert state.latest_tool_problem is not None
    assert state.latest_tool_problem.reason is None
    assert "applyPatch" in caplog.text
    assert "no safe detail provided" in caplog.text
    assert "raw provider body" not in caplog.text


async def test_agent_events_to_chat_deltas_tracks_retry_prompts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test retry prompt parts are logged as non-terminal tool problems."""

    async def stream() -> AsyncIterator[Any]:
        yield FunctionToolResultEvent(
            RetryPromptPart(
                tool_name="applyPatch",
                tool_call_id="tool-1",
                content="patch content must follow a file header",
            )
        )

    state = _StreamRunState()

    with caplog.at_level(logging.WARNING):
        deltas = [
            delta
            async for delta in _agent_events_to_chat_deltas(stream(), set(), state)
        ]

    assert deltas[0]["tool_name"] == "applyPatch"
    assert state.latest_tool_problem is not None
    assert state.latest_tool_problem.outcome == "retry"
    assert state.latest_tool_problem.reason is None
    assert "returned retry" in caplog.text
    assert "patch content must follow a file header" not in caplog.text


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
