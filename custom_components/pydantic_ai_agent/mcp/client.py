"""MCP client construction helpers."""

# ruff: noqa: ANN401

from types import TracebackType
from typing import Any

from fastmcp.client import Client as FastMCPClient
from fastmcp.client.transports import StreamableHttpTransport
from homeassistant.helpers.httpx_client import (
    DEFAULT_LIMITS,
    SERVER_SOFTWARE,
    USER_AGENT,
    HassHttpXAsyncClient,
)
from homeassistant.util.ssl import SSL_ALPN_HTTP11, client_context
import httpx

from .models import ValidatedMCPURL
from .validation import _default_port


class _MCPHttpXClient(HassHttpXAsyncClient):
    """HA-configured HTTPX client that FastMCP owns for one session."""

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        """Close the per-session client when FastMCP exits its context."""
        await self.aclose()


def _origin_guard_hook(validated_url: ValidatedMCPURL) -> Any:
    """Return an HTTPX hook that rejects requests outside the MCP origin."""

    async def guard_origin(request: httpx.Request) -> None:
        request_port = _default_port(request.url.scheme, request.url.port)
        if (
            request.url.scheme != validated_url.scheme
            or request.url.host != validated_url.hostname
            or request_port != validated_url.port
        ):
            raise httpx.ConnectError(
                "MCP redirects must stay on the validated origin.",
                request=request,
            )

    return guard_origin


def _mcp_http_client_factory(validated_url: ValidatedMCPURL) -> Any:
    """Return an HTTP client factory for FastMCP Streamable HTTP connections."""

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
        follow_redirects: bool = False,
    ) -> httpx.AsyncClient:
        return _MCPHttpXClient(
            headers={USER_AGENT: SERVER_SOFTWARE, **(headers or {})},
            timeout=timeout,
            auth=auth,
            follow_redirects=follow_redirects,
            trust_env=False,
            limits=DEFAULT_LIMITS,
            verify=client_context(alpn_protocols=SSL_ALPN_HTTP11),
            event_hooks={"request": [_origin_guard_hook(validated_url)]},
        )

    return factory


def _mcp_client(
    validated_url: ValidatedMCPURL,
    headers: dict[str, str],
    timeout: float,
) -> FastMCPClient[Any]:
    """Return a FastMCP client pinned to Streamable HTTP transport."""
    transport = StreamableHttpTransport(
        validated_url.url,
        headers=headers,
        httpx_client_factory=_mcp_http_client_factory(validated_url),
    )
    return FastMCPClient(transport=transport, init_timeout=timeout, timeout=timeout)
