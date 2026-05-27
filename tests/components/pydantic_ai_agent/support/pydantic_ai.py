"""Reusable Pydantic AI test doubles."""

from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
import json
from typing import Any, cast

from pydantic_ai import (
    AgentRunResultEvent,
    ModelResponse,
    PartStartEvent,
    TextPart,
)


class Usage:
    """Minimal Pydantic AI usage test double."""

    def __init__(
        self,
        *,
        input_tokens: int = 10,
        output_tokens: int = 2,
        total_tokens: int = 12,
        requests: int = 1,
        tool_calls: int = 3,
    ) -> None:
        """Initialize deterministic usage values."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.requests = requests
        self.tool_calls = tool_calls

    def opentelemetry_attributes(self) -> dict[str, int]:
        """Return deterministic token usage attributes."""
        return {
            "gen_ai.usage.input_tokens": self.input_tokens,
            "gen_ai.usage.output_tokens": self.output_tokens,
        }


class TextStream:
    """Async iterator over text chunks or stream events."""

    def __init__(self, *events: object) -> None:
        """Initialize the event stream."""
        self._events = iter(events)

    def __aiter__(self) -> "TextStream":
        """Return the async iterator."""
        return self

    async def __anext__(self) -> object:
        """Return the next stream event."""
        try:
            return next(self._events)
        except StopIteration as err:
            raise StopAsyncIteration from err


class EventStream(TextStream):
    """Async iterator over Pydantic AI stream events."""

    def __init__(self, events: Iterable[object]) -> None:
        """Initialize the event stream."""
        super().__init__(*events)


class StreamResult:
    """Minimal Agent streamed result for tests."""

    def __init__(self, text: str = "runtime response", usage: Usage | None = None) -> None:
        """Initialize the streamed result."""
        self.output = text
        self.usage = Usage() if usage is None else usage

    def stream_text(self, *, delta: bool = False) -> TextStream:
        """Return streamed text chunks."""
        del delta
        return TextStream(self.output)

    def get_output(self) -> str:
        """Return final output."""
        return self.output

    def new_messages(self) -> list[ModelResponse]:
        """Return final Agent messages."""
        return [ModelResponse(parts=[TextPart(content=self.output)])]


class RunResult:
    """Minimal Agent run result for tests."""

    def __init__(
        self,
        output: object,
        messages: list[ModelResponse] | None = None,
        usage: Usage | None = None,
    ) -> None:
        """Initialize the run result."""
        self.output = output
        self._messages = messages
        self.usage = (
            Usage(
                input_tokens=20,
                output_tokens=5,
                total_tokens=25,
                requests=2,
                tool_calls=1,
            )
            if usage is None
            else usage
        )

    def new_messages(self) -> list[ModelResponse]:
        """Return final Agent messages."""
        if self._messages is not None:
            return self._messages
        content = self.output if isinstance(self.output, str) else json.dumps(self.output)
        return [ModelResponse(parts=[TextPart(content=content)])]


class Agent:
    """Minimal async-context Agent test double."""

    def __init__(
        self,
        *,
        stream_text: str = "runtime response",
        output: object = None,
        messages: list[ModelResponse] | None = None,
    ) -> None:
        """Initialize recorded run state."""
        self._stream_text = stream_text
        self._output = stream_text if output is None else output
        self._messages = messages
        self.run_kwargs: dict[str, object] = {}
        self.run_calls = 0
        self.run_stream_events_calls = 0

    async def __aenter__(self) -> "Agent":
        """Enter the agent context."""
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Exit the agent context."""

    @asynccontextmanager
    async def run_stream_events(
        self, *_args: object, **kwargs: object
    ) -> AsyncIterator[EventStream]:
        """Return deterministic streamed Agent events."""
        self.run_stream_events_calls += 1
        self.run_kwargs = kwargs
        result = StreamResult(self._stream_text)
        yield EventStream(
            (
                PartStartEvent(index=0, part=TextPart(content=self._stream_text)),
                AgentRunResultEvent(cast(Any, result)),
            )
        )

    @asynccontextmanager
    async def run_stream(
        self, *_args: object, **_kwargs: object
    ) -> AsyncIterator[StreamResult]:
        """Return a deterministic streamed result."""
        yield StreamResult(self._stream_text)

    async def run(self, *_args: object, **kwargs: object) -> RunResult:
        """Return a deterministic run result."""
        self.run_calls += 1
        self.run_kwargs = kwargs
        return RunResult(self._output, self._messages)


def agent_factory(
    *,
    stream_text: str = "",
    output: object = None,
    messages: list[ModelResponse] | None = None,
):
    """Return an Agent constructor test double."""

    def factory(*_args: object, **_kwargs: object) -> Agent:
        return Agent(stream_text=stream_text, output=output, messages=messages)

    return factory


class ConversationAgent:
    """Minimal conversation Agent test double."""

    def __init__(self, text: str = "runtime response") -> None:
        """Initialize recorded run state."""
        self._text = text
        self.run_kwargs: dict[str, object] = {}
        self.run_calls = 0
        self.run_stream_events_calls = 0

    async def __aenter__(self) -> "ConversationAgent":
        """Enter the agent context."""
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Exit the agent context."""

    @asynccontextmanager
    async def run_stream_events(
        self, *_args: object, **kwargs: object
    ) -> AsyncIterator[EventStream]:
        """Return deterministic streamed Agent events."""
        self.run_stream_events_calls += 1
        self.run_kwargs = kwargs
        result = StreamResult(self._text)
        yield EventStream(
            (
                PartStartEvent(index=0, part=TextPart(content=self._text)),
                AgentRunResultEvent(cast(Any, result)),
            )
        )

    @asynccontextmanager
    async def run_stream(
        self, *_args: object, **_kwargs: object
    ) -> AsyncIterator[StreamResult]:
        """Return a deterministic streamed result."""
        yield StreamResult(self._text)

    async def run(self, *_args: object, **kwargs: object) -> StreamResult:
        """Return a deterministic run result."""
        self.run_calls += 1
        self.run_kwargs = kwargs
        return StreamResult(self._text)
