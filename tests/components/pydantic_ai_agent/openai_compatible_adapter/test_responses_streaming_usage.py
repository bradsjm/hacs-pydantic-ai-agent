"""Tests for Responses streaming lifecycle usage."""

from typing import cast

from custom_components.pydantic_ai_agent.openai_compatible_adapter._responses_streamed_response import (
    OpenAICompatibleResponsesStreamedResponse,
)
from custom_components.pydantic_ai_agent.openai_compatible_client._streaming import (
    ResponseStream,
)
from custom_components.pydantic_ai_agent.openai_compatible_client._types import (
    Response,
    ResponseIncompleteDetails,
    ResponseOutputTokenDetails,
    ResponseStreamEvent,
    ResponseUsage,
)
from pydantic_ai.models import ModelRequestParameters


def _streamed_response() -> OpenAICompatibleResponsesStreamedResponse:
    """Return a response instance suitable for lifecycle event processing."""
    return OpenAICompatibleResponsesStreamedResponse(
        ModelRequestParameters(),
        "initial-model",
        cast(ResponseStream, object()),
        "test-provider",
        "https://example.test",
    )


def _response(*, usage: ResponseUsage | None, status: str | None = None) -> Response:
    """Return one typed Responses lifecycle payload."""
    return Response(
        id="response-1",
        model="final-model",
        status=status,
        usage=usage,
    )


async def test_responses_streaming_uses_latest_cumulative_usage_snapshot() -> None:
    """Later lifecycle snapshots replace rather than inflate prior usage."""
    streamed_response = _streamed_response()
    created_usage = ResponseUsage(input_tokens=1, output_tokens=1, total_tokens=2)
    in_progress_usage = ResponseUsage(input_tokens=4, output_tokens=3, total_tokens=7)
    completed_usage = ResponseUsage(
        input_tokens=10,
        output_tokens=8,
        total_tokens=18,
        output_tokens_details=ResponseOutputTokenDetails(reasoning_tokens=5),
    )

    for event in (
        ResponseStreamEvent(type="response.created", response=_response(usage=created_usage)),
        ResponseStreamEvent(type="response.in_progress", response=_response(usage=in_progress_usage)),
        ResponseStreamEvent(type="response.completed", response=_response(usage=completed_usage)),
    ):
        await streamed_response._process_response_event(event, {}, {})

    assert streamed_response.usage.input_tokens == 10
    assert streamed_response.usage.output_tokens == 8
    assert streamed_response.usage.total_tokens == 18
    assert streamed_response.usage.details == {"output_tokens_details.reasoning_tokens": 5}


async def test_responses_streaming_preserves_usage_when_terminal_event_has_none() -> None:
    """A terminal event without usage retains the most recent lifecycle snapshot."""
    streamed_response = _streamed_response()
    usage = ResponseUsage(input_tokens=10, output_tokens=8, total_tokens=18)

    await streamed_response._process_response_event(
        ResponseStreamEvent(type="response.in_progress", response=_response(usage=usage)),
        {},
        {},
    )
    await streamed_response._process_response_event(
        ResponseStreamEvent(
            type="response.incomplete",
            response=Response(
                id="response-1",
                status="incomplete",
                incomplete_details=ResponseIncompleteDetails(reason="max_output_tokens"),
            ),
        ),
        {},
        {},
    )

    assert streamed_response.usage.total_tokens == 18
    assert streamed_response.finish_reason == "length"
    assert streamed_response.provider_response_id == "response-1"
