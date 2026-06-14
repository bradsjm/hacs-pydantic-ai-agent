"""Shared probe-model test helpers."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from custom_components.pydantic_ai_agent.const import (
    CONF_PROVIDER_MODE,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
)
from homeassistant.const import CONF_API_KEY, CONF_NAME
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import PartEndEvent, PartStartEvent, TextPart, ToolCallPart


class SingleEventStream:
    """Async stream with one validation event."""

    def __init__(self) -> None:
        """Initialize the stream."""
        self._yielded = False

    def __aiter__(self) -> SingleEventStream:
        """Return the async iterator."""
        return self

    async def __anext__(self) -> object:
        """Return one event, then stop."""
        if self._yielded:
            raise StopAsyncIteration
        self._yielded = True
        return object()


class StructuredTextStream:
    """Async stream with text structured-output events."""

    def __init__(self, content: str = '{"ok":true}') -> None:
        """Initialize the stream."""
        self._events = iter(
            (
                PartStartEvent(index=0, part=TextPart(content=content)),
                PartEndEvent(index=0, part=TextPart(content=content)),
            )
        )

    def __aiter__(self) -> StructuredTextStream:
        """Return the async iterator."""
        return self

    async def __anext__(self) -> object:
        """Return the next stream event."""
        try:
            return next(self._events)
        except StopIteration as err:
            raise StopAsyncIteration from err


class StructuredToolStream:
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

    def __aiter__(self) -> StructuredToolStream:
        """Return the async iterator."""
        return self

    async def __anext__(self) -> object:
        """Return the next stream event."""
        try:
            return next(self._events)
        except StopIteration as err:
            raise StopAsyncIteration from err


class FailingStreamContext:
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


class HTTPErrorStreamContext:
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


def provider_data(
    provider_mode: str = PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
) -> dict[str, object]:
    """Return provider data for model probes."""
    return {
        CONF_NAME: "Hosted OpenAI",
        CONF_PROVIDER_MODE: provider_mode,
        CONF_API_KEY: "sk-test",
    }


@asynccontextmanager
async def stream_context(
    stream_events: SingleEventStream | StructuredTextStream | StructuredToolStream,
) -> AsyncGenerator[SingleEventStream | StructuredTextStream | StructuredToolStream]:
    """Yield a probe stream from an async context manager."""
    yield stream_events
