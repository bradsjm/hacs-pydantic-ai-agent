"""Pydantic AI streamed response for OpenAI-compatible Chat Completions."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic_ai.messages import (
    FinishReason,
    ModelResponseStreamEvent,
    PartStartEvent,
    ThinkingPart,
)
from pydantic_ai.models import ModelRequestParameters, StreamedResponse

from ..openai_compatible_client._streaming import ChatCompletionStream
from ._usage import map_usage

_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_call",
    "function_call": "tool_call",
    "content_filter": "content_filter",
}


@dataclass
class OpenAICompatibleStreamedResponse(StreamedResponse):
    """Streamed response implementation for Chat Completions chunks."""

    model_request_parameters: ModelRequestParameters
    _model_name: str
    _response: ChatCompletionStream
    _provider_name: str
    _provider_url: str
    _timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    _refusal_text: str = field(default="", init=False)

    async def close_stream(self) -> None:
        """Close the underlying HTTP stream."""
        await self._response.close()

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Yield Pydantic AI stream events for incoming chunks."""
        async for chunk in self._response:
            self._usage += map_usage(chunk)
            if chunk.id:
                self.provider_response_id = chunk.id
            if chunk.model:
                self._model_name = chunk.model
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                self.finish_reason = _FINISH_REASON_MAP.get(choice.finish_reason)
                self.provider_details = {
                    **(self.provider_details or {}),
                    "finish_reason": choice.finish_reason,
                }
            delta = choice.delta
            if delta is None:
                continue
            if delta.refusal:
                self.finish_reason = "content_filter"
                self._refusal_text += delta.refusal
                continue
            if reasoning := _first_text(delta, "reasoning", "reasoning_content"):
                for event in self._parts_manager.handle_thinking_delta(
                    vendor_part_id="reasoning",
                    id="reasoning",
                    content=reasoning,
                    provider_name=self.provider_name,
                ):
                    yield event
            if delta.content:
                for event in self._parts_manager.handle_text_delta(
                    vendor_part_id="content",
                    content=delta.content,
                    thinking_tags=("<think>", "</think>"),
                ):
                    if isinstance(event, PartStartEvent) and isinstance(
                        event.part, ThinkingPart
                    ):
                        event.part.id = "content"
                        event.part.provider_name = self.provider_name
                    yield event
            for tool_call in delta.tool_calls or []:
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=tool_call.index,
                    tool_name=tool_call.function.name if tool_call.function else None,
                    args=tool_call.function.arguments if tool_call.function else None,
                    tool_call_id=tool_call.id,
                )
                if maybe_event is not None:
                    yield maybe_event
        if self._refusal_text:
            self.provider_details = {
                **(self.provider_details or {}),
                "refusal": self._refusal_text,
            }

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return self._provider_name

    @property
    def provider_url(self) -> str:
        """Return the provider URL."""
        return self._provider_url

    @property
    def timestamp(self) -> datetime:
        """Return the local response timestamp."""
        return self._timestamp


def _first_text(data: Any, *names: str) -> str | None:
    """Return the first string attribute found on a chunk delta."""
    for name in names:
        value = getattr(data, name, None)
        if isinstance(value, str) and value:
            return value
    return None
