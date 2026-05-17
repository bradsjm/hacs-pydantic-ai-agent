"""Chat Completions resource."""

from typing import Any, Literal, overload

import httpx
from pydantic import ValidationError

from ._exceptions import APIConnectionError, APITimeoutError
from ._sentinels import NOT_GIVEN, is_omitted
from ._streaming import ChatCompletionStream, raise_for_status
from ._types import ChatCompletion


def serialize_payload(value: Any) -> Any:
    """Serialize request payloads while preserving None and omitting sentinels."""
    if is_omitted(value):
        return NOT_GIVEN
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            serialized = serialize_payload(item)
            if not is_omitted(serialized):
                result[key] = serialized
        return result
    if isinstance(value, list):
        return [
            item
            for item in (serialize_payload(item) for item in value)
            if not is_omitted(item)
        ]
    return value


class ChatCompletionsResource:
    """OpenAI-compatible ``chat.completions`` resource."""

    def __init__(self, client: Any) -> None:
        """Initialize the resource."""
        self._client = client

    @overload
    async def create(
        self, *, stream: Literal[False] = False, **kwargs: Any
    ) -> ChatCompletion: ...

    @overload
    async def create(
        self, *, stream: Literal[True], **kwargs: Any
    ) -> ChatCompletionStream: ...

    @overload
    async def create(
        self, *, stream: bool, **kwargs: Any
    ) -> ChatCompletion | ChatCompletionStream: ...

    async def create(
        self,
        *,
        stream: bool = False,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None | object = NOT_GIVEN,
        **kwargs: Any,
    ) -> ChatCompletion | ChatCompletionStream:
        """Create a chat completion."""
        body = serialize_payload(kwargs)
        assert isinstance(body, dict)
        body["stream"] = stream
        if extra_body:
            body.update(serialize_payload(extra_body))

        headers = self._client.auth_headers | (extra_headers or {})
        request_timeout = None if is_omitted(timeout) else timeout
        url = self._client.url_for("/chat/completions")
        if stream:
            return ChatCompletionStream(
                self._client.http_client.stream(
                    "POST",
                    url,
                    json=body,
                    headers=headers,
                    timeout=request_timeout,
                )
            )

        try:
            response = await self._client.http_client.post(
                url,
                json=body,
                headers=headers,
                timeout=request_timeout,
            )
            await raise_for_status(response)
            return ChatCompletion.model_validate(response.json())
        except httpx.TimeoutException as err:
            raise APITimeoutError(message=str(err), request=err.request) from err
        except httpx.RequestError as err:
            raise APIConnectionError(message=str(err), request=err.request) from err
        except (ValueError, ValidationError) as err:
            raise APIConnectionError(message=f"Invalid response JSON: {err}") from err


class ChatResource:
    """OpenAI-compatible ``chat`` resource."""

    def __init__(self, client: Any) -> None:
        """Initialize the resource."""
        self.completions = ChatCompletionsResource(client)
