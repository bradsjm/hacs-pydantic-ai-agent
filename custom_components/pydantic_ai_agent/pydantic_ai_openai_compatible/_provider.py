"""OpenAI-compatible Pydantic AI provider."""

from typing import overload

import httpx
from pydantic_ai.models import create_async_http_client
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.providers import Provider

from ..openai_compatible_client import AsyncOpenAICompatible

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAICompatibleProvider(Provider[AsyncOpenAICompatible]):
    """Provider for OpenAI-compatible Chat Completions APIs."""

    @property
    def name(self) -> str:
        """Return the provider name."""
        return self._name

    @property
    def base_url(self) -> str:
        """Return the provider base URL."""
        return self.client.base_url

    @property
    def client(self) -> AsyncOpenAICompatible:
        """Return the low-level client."""
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        """Return a Chat Completions compatible model profile."""
        return openai_model_profile(model_name)

    @overload
    def __init__(
        self, *, client: AsyncOpenAICompatible, name: str = "openai-compatible-completions"
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        client: None = None,
        name: str = "openai-compatible-completions",
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        client: AsyncOpenAICompatible | None = None,
        name: str = "openai-compatible-completions",
    ) -> None:
        """Initialize the provider."""
        self._name = name
        if client is not None:
            if (
                api_key is not None
                or base_url is not None
                or headers is not None
                or http_client is not None
            ):
                raise ValueError(
                    "client cannot be combined with api_key, base_url, headers, or http_client"
                )
            self._client = client
            return

        base_url = (base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/")
        if http_client is None:
            http_client = create_async_http_client()
            self._own_http_client = http_client
            self._http_client_factory = create_async_http_client
        self._client = AsyncOpenAICompatible(
            api_key=api_key,
            base_url=base_url,
            headers=headers,
            http_client=http_client,
        )

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        """Update the low-level client's HTTP client when lifecycle recreates it."""
        self._client.http_client = http_client
