"""Pydantic AI streamed response for OpenAI-compatible Responses APIs."""

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import FinishReason, ModelResponseStreamEvent
from pydantic_ai.models import ModelRequestParameters, StreamedResponse

from ..openai_compatible_client._streaming import ResponseStream
from ..openai_compatible_client._types import Response, ResponseStreamEvent
from ._usage import map_usage

_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "max_output_tokens": "length",
    "content_filter": "content_filter",
    "completed": "stop",
    "cancelled": "error",
    "failed": "error",
}


@dataclass
class OpenAICompatibleResponsesStreamedResponse(StreamedResponse):
    """Streamed response implementation for Responses API events."""

    model_request_parameters: ModelRequestParameters
    _model_name: str
    _response: ResponseStream
    _provider_name: str
    _provider_url: str
    _timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    _refusal_text: str = field(default="", init=False)
    _tool_arg_delta_item_ids: set[str] = field(default_factory=set, init=False)

    async def close_stream(self) -> None:
        """Close the underlying HTTP stream."""
        await self._response.close()

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Yield Pydantic AI stream events for incoming Responses events."""
        phase_by_item: dict[str, str] = {}
        annotations_by_item: dict[str, list[Any]] = {}
        async for event in self._response:
            event_type = event.type
            if event.response is not None:
                self._store_response_metadata(event.response)

            if event_type in {
                "response.created",
                "response.in_progress",
                "response.completed",
                "response.incomplete",
                "response.failed",
            }:
                if event.response is not None and event_type != "response.created":
                    self._usage += map_usage(event.response)
                continue

            if event_type == "response.output_item.added":
                for stream_event in self._handle_output_item_added(
                    event, phase_by_item
                ):
                    yield stream_event
                continue

            if event_type == "response.output_item.done":
                for stream_event in self._handle_output_item_done(event):
                    yield stream_event
                continue

            if event_type == "response.output_text.delta" and event.delta is not None:
                for stream_event in self._parts_manager.handle_text_delta(
                    vendor_part_id=_event_item_id(event),
                    content=event.delta,
                    id=event.item_id,
                    provider_name=self.provider_name,
                ):
                    yield stream_event
                continue

            if event_type == "response.output_text.done":
                details: dict[str, Any] = {}
                if annotations := annotations_by_item.get(_event_item_id(event)):
                    details["annotations"] = annotations
                if event.logprobs:
                    details["logprobs"] = event.logprobs
                if event.item_id and (phase := phase_by_item.get(event.item_id)):
                    details["phase"] = phase
                if details:
                    for stream_event in self._parts_manager.handle_text_delta(
                        vendor_part_id=_event_item_id(event),
                        content="",
                        provider_name=self.provider_name,
                        provider_details=details,
                    ):
                        yield stream_event
                continue

            if event_type == "response.output_text.annotation.added":
                if event.annotation is not None:
                    annotations_by_item.setdefault(_event_item_id(event), []).append(
                        event.annotation
                    )
                continue

            if event_type == "response.function_call_arguments.delta":
                self._tool_arg_delta_item_ids.add(_event_item_id(event))
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=_event_item_id(event),
                    args=event.delta,
                    provider_name=self.provider_name,
                )
                if maybe_event is not None:
                    yield maybe_event
                continue

            if event_type == "response.function_call_arguments.done":
                continue

            if event_type in {
                "response.reasoning_summary_part.added",
                "response.reasoning_summary_text.delta",
            }:
                text = event.delta or (event.part or {}).get("text")
                if text:
                    for stream_event in self._parts_manager.handle_thinking_delta(
                        vendor_part_id=_reasoning_vendor_id(event),
                        content=text,
                        id=event.item_id,
                        provider_name=self.provider_name,
                    ):
                        yield stream_event
                continue

            if event_type == "response.reasoning_text.delta":
                for stream_event in self._parts_manager.handle_thinking_delta(
                    vendor_part_id=_event_item_id(event),
                    id=event.item_id,
                    provider_name=self.provider_name,
                    provider_details=_raw_content_updater(event.delta or ""),
                ):
                    yield stream_event
                continue

            if event_type in {
                "response.content_part.added",
                "response.content_part.done",
                "response.reasoning_summary_part.done",
                "response.reasoning_summary_text.done",
                "response.reasoning_text.done",
            }:
                continue

            if event_type == "response.refusal.delta":
                self.finish_reason = "content_filter"
                self._refusal_text += event.delta or ""
                continue

            if event_type == "response.refusal.done":
                self.finish_reason = "content_filter"
                self._refusal_text = event.refusal or self._refusal_text
                continue

            raise UnexpectedModelBehavior(
                f"Unsupported Responses stream event: {event_type!r}"
            )

        if self._refusal_text:
            self.provider_details = {
                **(self.provider_details or {}),
                "refusal": self._refusal_text,
            }

    def _handle_output_item_added(
        self, event: ResponseStreamEvent, phase_by_item: dict[str, str]
    ) -> list[ModelResponseStreamEvent]:
        """Handle a Responses output-item-added event."""
        item = event.item or {}
        item_id = _item_id(item, event)
        item_type = item.get("type")
        if item_type == "function_call":
            details = (
                {"namespace": item["namespace"]} if item.get("namespace") else None
            )
            return [
                self._parts_manager.handle_tool_call_part(
                    vendor_part_id=item_id,
                    tool_name=item.get("name") or "",
                    args=item.get("arguments"),
                    tool_call_id=item.get("call_id") or item_id,
                    id=item_id,
                    provider_name=self.provider_name,
                    provider_details=details,
                )
            ]
        if item_type == "message":
            if item_id and item.get("phase") is not None:
                phase_by_item[item_id] = item["phase"]
            return []
        if item_type == "reasoning":
            return []
        raise UnexpectedModelBehavior(
            f"Unsupported Responses output item type: {item_type!r}"
        )

    def _handle_output_item_done(
        self, event: ResponseStreamEvent
    ) -> list[ModelResponseStreamEvent]:
        """Handle a Responses output-item-done event."""
        item = event.item or {}
        item_id = _item_id(item, event)
        item_type = item.get("type")
        if item_type == "function_call":
            args = (
                None
                if item_id in self._tool_arg_delta_item_ids
                else item.get("arguments")
            )
            maybe_event = self._parts_manager.handle_tool_call_delta(
                vendor_part_id=item_id,
                args=args,
                tool_call_id=item.get("call_id") or item_id,
                provider_name=self.provider_name,
            )
            return [maybe_event] if maybe_event is not None else []
        if item_type == "reasoning":
            signature = item.get("encrypted_content")
            if not signature:
                return []
            return list(
                self._parts_manager.handle_thinking_delta(
                    vendor_part_id=item_id,
                    id=item_id,
                    signature=signature,
                    provider_name=self.provider_name,
                )
            )
        if item_type == "message":
            return []
        raise UnexpectedModelBehavior(
            f"Unsupported Responses output item type: {item_type!r}"
        )

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

    def _store_response_metadata(self, response: Response) -> None:
        """Store final response metadata exposed by Responses events."""
        if response.id:
            self.provider_response_id = response.id
        if response.model:
            self._model_name = response.model
        if response.created_at is not None:
            self._timestamp = datetime.fromtimestamp(response.created_at, UTC)
        provider_details = dict(self.provider_details or {})
        raw_finish_reason = (
            response.incomplete_details.reason
            if response.incomplete_details is not None
            and response.incomplete_details.reason is not None
            else response.status
        )
        if raw_finish_reason:
            provider_details["finish_reason"] = raw_finish_reason
            self.finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)
        if response.conversation and response.conversation.id:
            provider_details["conversation_id"] = response.conversation.id
        self.provider_details = provider_details or None


def _item_id(item: dict[str, Any], event: ResponseStreamEvent) -> str:
    """Return the best Responses item identifier for stream part tracking."""
    return str(item.get("id") or event.item_id or event.output_index or "response")


def _event_item_id(event: ResponseStreamEvent) -> str:
    """Return the best event identifier for stream part tracking."""
    return str(event.item_id or event.output_index or event.content_index or "response")


def _reasoning_vendor_id(event: ResponseStreamEvent) -> str:
    """Return the vendor id for reasoning summary chunks."""
    if event.summary_index in (None, 0):
        return _event_item_id(event)
    return f"{_event_item_id(event)}-{event.summary_index}"


def _raw_content_updater(
    delta: str,
) -> Callable[[dict[str, Any] | None], dict[str, Any]]:
    """Return a provider-details updater for raw reasoning deltas."""

    def update(existing: dict[str, Any] | None) -> dict[str, Any]:
        details = dict(existing or {})
        raw_content = list(details.get("raw_content") or [])
        if raw_content:
            raw_content[-1] = f"{raw_content[-1]}{delta}"
        else:
            raw_content.append(delta)
        details["raw_content"] = raw_content
        return details

    return update
