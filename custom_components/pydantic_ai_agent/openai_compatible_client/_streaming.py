"""SSE streaming support for OpenAI-compatible APIs."""

import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Self, TypeVar, cast

import httpx
from pydantic import BaseModel, ValidationError

from ._exceptions import APIConnectionError, APIStatusError, APITimeoutError
from ._types import ChatCompletionChunk, ResponseStreamEvent


async def response_body(response: httpx.Response) -> Mapping[str, object] | str:
    """Return JSON response body when possible, otherwise text."""
    await response.aread()
    try:
        return response.json()
    except json.JSONDecodeError:
        return response.text


async def raise_for_status(response: httpx.Response) -> None:
    """Raise a lightweight OpenAI-compatible status error for 4xx/5xx."""
    if response.status_code < 400:
        return
    body = await response_body(response)
    message = f"Error code: {response.status_code}"
    if (
        isinstance(body, dict)
        and isinstance(error := body.get("error"), dict)
        and (error_message := error.get("message"))
    ):
        message = str(error_message)
    raise APIStatusError(message=message, response=response, body=body)


_EventT = TypeVar("_EventT", bound=BaseModel)


class _SSEStream[EventT: BaseModel]:
    """Async iterator over typed Server-Sent Events."""

    _event_type: type[EventT]

    def __init__(
        self, response_context: AbstractAsyncContextManager[httpx.Response]
    ) -> None:
        """Initialize the stream."""
        self._response_context = response_context
        self._response: httpx.Response | None = None
        self._lines: AsyncIterator[str] | None = None

    async def __aenter__(self) -> Self:
        """Enter the HTTP stream context."""
        await self._ensure_entered()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        """Close the HTTP stream context."""
        await self.close()
        return False

    def __aiter__(self) -> Self:
        """Return the async stream iterator."""
        return self

    async def __anext__(self) -> EventT:
        """Return the next parsed SSE event."""
        await self._ensure_entered()
        assert self._lines is not None
        try:
            async for line in self._lines:
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    raise StopAsyncIteration
                try:
                    return self._event_type.model_validate_json(data)
                except ValidationError as err:
                    raise APIConnectionError(
                        message=f"Invalid streaming response: {err}",
                    ) from err
        except httpx.TimeoutException as err:
            raise APITimeoutError(message=str(err), request=err.request) from err
        except httpx.RequestError as err:
            raise APIConnectionError(message=str(err), request=err.request) from err
        raise StopAsyncIteration

    async def close(self) -> None:
        """Close the underlying response stream."""
        if self._response is None:
            return
        lines = self._lines
        self._lines = None
        try:
            if lines is not None and (aclose := getattr(lines, "aclose", None)):
                await cast(Callable[[], Awaitable[None]], aclose)()
        finally:
            await self._response_context.__aexit__(None, None, None)
            self._response = None

    async def _ensure_entered(self) -> None:
        """Enter the HTTP stream lazily."""
        if self._response is not None:
            return
        try:
            self._response = await self._response_context.__aenter__()
            try:
                await raise_for_status(self._response)
            except APIStatusError:
                await self.close()
                raise
            response = self._response
            self._lines = response.aiter_lines().__aiter__()
        except httpx.TimeoutException as err:
            raise APITimeoutError(message=str(err), request=err.request) from err
        except httpx.RequestError as err:
            raise APIConnectionError(message=str(err), request=err.request) from err


class ChatCompletionStream(_SSEStream[ChatCompletionChunk]):
    """Async iterator over Chat Completions SSE chunks."""

    _event_type = ChatCompletionChunk


class ResponseStream(_SSEStream[ResponseStreamEvent]):
    """Async iterator over Responses API SSE events."""

    _event_type = ResponseStreamEvent
