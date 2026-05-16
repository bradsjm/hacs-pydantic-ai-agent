"""Test MCP helpers for Pydantic AI Agent."""

import ssl
from unittest.mock import patch

import httpx
import pytest

from homeassistant.util.ssl import SSL_ALPN_HTTP11, client_context

from custom_components.pydantic_ai_agent.mcp import (
    MCPValidationError,
    ValidatedMCPURL,
    _mcp_http_client_factory,
    _origin_guard_hook,
    normalise_mcp_url,
    redact_for_log,
)


def _validated_url(url: str = "https://mcp.example.com/mcp") -> ValidatedMCPURL:
    """Return a validated MCP URL for tests."""
    return ValidatedMCPURL(url, "https", "mcp.example.com", 443)


async def test_origin_guard_allows_same_origin_requests() -> None:
    """Test MCP origin guard allows same-origin requests."""
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
    """Test MCP origin guard rejects scheme, host, and port changes."""
    hook = _origin_guard_hook(_validated_url())

    with pytest.raises(httpx.ConnectError):
        await hook(httpx.Request("GET", url))


async def test_origin_guard_rejects_cross_origin_redirects() -> None:
    """Test HTTPX request hooks block redirected cross-origin requests."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "mcp.example.com":
            return httpx.Response(
                302,
                headers={"location": "https://other.example.com/mcp"},
                request=request,
            )
        return httpx.Response(200, request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
        event_hooks={"request": [_origin_guard_hook(_validated_url())]},
    ) as client:
        with pytest.raises(httpx.ConnectError):
            await client.get("https://mcp.example.com/mcp")


async def test_mcp_http_client_factory_uses_ha_httpx_helpers() -> None:
    """Test MCP HTTP clients use Home Assistant's pre-warmed SSL context."""
    ssl_context = client_context(alpn_protocols=SSL_ALPN_HTTP11)
    with patch(
        "custom_components.pydantic_ai_agent.mcp.client_context",
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
    """Test MCP client creation does not synchronously load CA certs."""
    client_context(alpn_protocols=SSL_ALPN_HTTP11)

    with patch.object(
        ssl.SSLContext,
        "load_verify_locations",
        side_effect=AssertionError("blocking SSL load"),
    ):
        client = _mcp_http_client_factory(_validated_url())()

    await client.aclose()


async def test_mcp_http_client_factory_closes_fastmcp_owned_clients() -> None:
    """Test FastMCP context-manager cleanup closes per-session clients."""
    client = _mcp_http_client_factory(_validated_url())()

    async with client:
        assert not client.is_closed

    assert client.is_closed


def test_mcp_log_redaction_uses_shared_sensitive_key_handling() -> None:
    """Test MCP log redaction handles nested sensitive keys."""
    redacted = redact_for_log(
        {
            "mcp_url": "https://mcp.example.com/mcp?token=visible",
            "headers": {"Authorization": "Bearer secret"},
            "result": {"session_token": "secret", "value": "safe"},
        }
    )

    assert redacted["mcp_url"] == "**REDACTED**"
    assert redacted["headers"] == "**REDACTED**"
    assert redacted["result"]["session_token"] == "**REDACTED**"
    assert redacted["result"]["value"] == "safe"


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("not a url", "invalid_mcp_url"),
        ("ftp://mcp.example.com/mcp", "invalid_mcp_url"),
        ("https://mcp.example.com/mcp#fragment", "invalid_mcp_url"),
        ("http://user:pass@mcp.example.com/mcp", "invalid_mcp_url"),
        ("https://user:pass@mcp.example.com/mcp", "invalid_mcp_url"),
    ],
)
def test_normalise_mcp_url_uses_ha_url_validation_for_baseline(
    url: str, reason: str
) -> None:
    """Test MCP URL validation combines HA baseline validation with MCP policy."""
    with pytest.raises(MCPValidationError) as err:
        normalise_mcp_url(url)

    assert getattr(err.value, "reason") == reason
