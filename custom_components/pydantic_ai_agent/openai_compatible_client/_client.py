"""Async OpenAI-compatible client."""

import httpx

from ._chat import ChatResource


class AsyncOpenAICompatible:
    """Minimal async OpenAI-compatible client."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        http_client: httpx.AsyncClient,
    ) -> None:
        """Initialize the client."""
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.http_client = http_client
        self.chat = ChatResource(self)

    @property
    def auth_headers(self) -> dict[str, str]:
        """Return authentication headers."""
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def url_for(self, path: str) -> str:
        """Return an absolute API URL for a path."""
        return f"{self.base_url}/{path.lstrip('/')}"
