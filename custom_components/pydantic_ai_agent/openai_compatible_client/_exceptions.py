"""OpenAI-compatible client exceptions."""

from typing import Any

import httpx


class OpenAICompatibleError(Exception):
    """Base error for the lightweight OpenAI-compatible client."""


class APIConnectionError(OpenAICompatibleError):
    """Raised when the provider cannot be reached."""

    def __init__(self, *, message: str, request: httpx.Request | None = None) -> None:
        """Initialize the connection error."""
        super().__init__(message)
        self.message = message
        self.request = request


class APITimeoutError(APIConnectionError):
    """Raised when the provider request times out."""


class APIStatusError(OpenAICompatibleError):
    """Raised when the provider returns an HTTP error status."""

    def __init__(
        self,
        *,
        message: str,
        response: httpx.Response,
        body: Any,
    ) -> None:
        """Initialize the status error."""
        super().__init__(message)
        self.message = message
        self.response = response
        self.request = response.request
        self.status_code = response.status_code
        self.body = body
        self.request_id = response.headers.get("x-request-id")
