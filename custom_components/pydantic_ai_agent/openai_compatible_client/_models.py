"""Models resource."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import httpx

from ._exceptions import APIConnectionError, APITimeoutError
from ._sentinels import NOT_GIVEN, NotGiven, is_omitted
from ._streaming import raise_for_status

if TYPE_CHECKING:
    from ._client import AsyncOpenAICompatible


class ModelsResource:
    """OpenAI-compatible ``models`` resource."""

    def __init__(self, client: AsyncOpenAICompatible) -> None:
        """Initialize the resource."""
        self._client = client

    async def list(
        self,
        *,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> list[str]:
        """Return available model IDs."""
        request_timeout = (
            None if is_omitted(timeout) else cast(float | httpx.Timeout | None, timeout)
        )
        try:
            response = await self._client.http_client.get(
                self._client.url_for("/models"),
                headers=self._client.auth_headers,
                timeout=request_timeout,
            )
            await raise_for_status(response)
            data = response.json()
        except httpx.TimeoutException as err:
            raise APITimeoutError(message=str(err), request=err.request) from err
        except httpx.RequestError as err:
            raise APIConnectionError(message=str(err), request=err.request) from err
        except ValueError as err:
            raise APIConnectionError(message=f"Invalid response JSON: {err}") from err

        if not isinstance(data, Mapping) or not isinstance(
            models := data.get("data"), list
        ):
            raise APIConnectionError(message="Invalid models response JSON")
        model_ids = {
            model["id"]
            for model in models
            if isinstance(model, Mapping) and isinstance(model.get("id"), str)
        }
        return sorted(model_ids)
