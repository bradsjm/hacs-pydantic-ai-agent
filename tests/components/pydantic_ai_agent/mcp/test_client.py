"""Tests for MCP client helpers."""

import ssl
from unittest.mock import patch

import httpx
import pytest
from custom_components.pydantic_ai_agent.mcp.client import (
    _mcp_http_client_factory,
    _origin_guard_hook,
)
from custom_components.pydantic_ai_agent.mcp.models import ValidatedMCPURL
from homeassistant.util.ssl import SSL_ALPN_HTTP11, client_context


def _validated_url(url: str = "https://mcp.example.com/mcp") -> ValidatedMCPURL:
    """Return a validated MCP URL for tests."""
    return ValidatedMCPURL(url, "https", "mcp.example.com", 443)


async def test_origin_guard_allows_same_origin_requests() -> None:
    hook = _origin_guard_hook(_validated_url())

    await hook(httpx.Request("GET", "https://mcp.example.com/other"))


@pytest.mark.parametrize(
    "url",
    [
        "http://mcp.example.com/mcp",
        "https://other.example.com/mcp",
        "https://mcp.example.com:8443/mcp",
    ],
)
async def test_origin_guard_rejects_cross_origin_requests(url: str) -> None:
    hook = _origin_guard_hook(_validated_url())

    with pytest.raises(httpx.ConnectError):
        await hook(httpx.Request("GET", url))


async def test_mcp_http_client_factory_uses_ha_httpx_helpers() -> None:
    ssl_context = client_context(alpn_protocols=SSL_ALPN_HTTP11)
    with patch(
        "custom_components.pydantic_ai_agent.mcp.client.client_context",
        return_value=ssl_context,
    ) as mock_client_context:
        client = _mcp_http_client_factory(_validated_url())(
            headers={"X-Test": "enabled"}, follow_redirects=True
        )

    try:
        assert client.trust_env is False
        assert client.follow_redirects is True
        assert client.headers["X-Test"] == "enabled"
        assert client.event_hooks["request"]
        mock_client_context.assert_called_once_with(alpn_protocols=SSL_ALPN_HTTP11)
    finally:
        await client.aclose()


async def test_mcp_http_client_factory_does_not_load_ssl_certs_on_loop() -> None:
    client_context(alpn_protocols=SSL_ALPN_HTTP11)

    with patch.object(
        ssl.SSLContext,
        "load_verify_locations",
        side_effect=AssertionError("blocking SSL load"),
    ):
        client = _mcp_http_client_factory(_validated_url())()

    await client.aclose()
