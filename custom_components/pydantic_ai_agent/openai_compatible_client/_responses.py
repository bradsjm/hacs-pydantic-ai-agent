"""Responses resource."""

from typing import TYPE_CHECKING, Any, cast

import httpx
from pydantic import ValidationError

from ._chat import serialize_payload
from ._exceptions import APIConnectionError, APITimeoutError
from ._sentinels import NOT_GIVEN, NotGiven, is_omitted
from ._streaming import ResponseStream, raise_for_status
from ._types import Response

if TYPE_CHECKING:
    from ._client import AsyncOpenAICompatible


class ResponsesResource:
    """OpenAI-compatible ``responses`` resource."""

    def __init__(self, client: AsyncOpenAICompatible) -> None:
        """Initialize the resource."""
        self._client = client

    async def create(
        self,
        *,
        stream: bool = False,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        **kwargs: object,
    ) -> Response | ResponseStream:
        """Create a response."""
        body = serialize_payload(kwargs)
        assert isinstance(body, dict)
        body["stream"] = stream
        if extra_body:
            body.update(cast(dict[str, object], serialize_payload(extra_body)))

        headers = self._client.auth_headers | (extra_headers or {})
        request_timeout = (
            None
            if is_omitted(timeout)
            else cast(float | httpx.Timeout | None, timeout)
        )
        if stream:
            return ResponseStream(
                self._client.http_client.stream(
                    "POST",
                    self._client.url_for("/responses"),
                    json=body,
                    headers=headers,
                    timeout=request_timeout,
                )
            )
        try:
            response = await self._client.http_client.post(
                self._client.url_for("/responses"),
                json=body,
                headers=headers,
                timeout=request_timeout,
            )
            await raise_for_status(response)
            return Response.model_validate(response.json())
        except httpx.TimeoutException as err:
            raise APITimeoutError(message=str(err), request=err.request) from err
        except httpx.RequestError as err:
            raise APIConnectionError(message=str(err), request=err.request) from err
        except (ValueError, ValidationError) as err:
            raise APIConnectionError(message=f"Invalid response JSON: {err}") from err
