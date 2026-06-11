"""Test MCP helpers for Pydantic AI Agent."""

import ssl
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import httpx
import pytest
import voluptuous as vol
from custom_components.pydantic_ai_agent._redaction import redact_data
from custom_components.pydantic_ai_agent.const import (
    CONF_MCP_ALLOWED_TOOLS,
    CONF_MCP_DEFERRED_LOADING,
    CONF_MCP_HEADERS,
    CONF_MCP_INCLUDE_RETURN_SCHEMA,
    CONF_MCP_URL,
    DOMAIN,
    SUBENTRY_TYPE_MCP_SERVER,
)
from custom_components.pydantic_ai_agent.mcp import (
    MCPValidationError,
    ValidatedMCPURL,
    _cache_key,
    _mcp_http_client_factory,
    _origin_guard_hook,
    async_discover_mcp_tools_from_config,
    async_runtime_mcp_toolsets,
    cached_mcp_tools,
    mcp_config_from_subentry,
    normalise_mcp_url,
    parse_allowed_tools,
    parse_mcp_headers,
    schema_hash,
)
from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.util.ssl import SSL_ALPN_HTTP11, client_context
from pytest_homeassistant_custom_component.common import MockConfigEntry


def _validated_url(url: str = "https://mcp.example.com/mcp") -> ValidatedMCPURL:
    """Return a validated MCP URL for tests."""
    return ValidatedMCPURL(url, "https", "mcp.example.com", 443)


def _mcp_entry(
    *,
    subentry_id: str = "mcp_server_1",
    allowed_tools: list[str] | None = None,
    include_return_schema: bool | None = None,
    deferred_loading: bool | None = None,
) -> MockConfigEntry:
    """Return a config entry with one MCP server subentry."""
    data: dict[str, object] = {
        CONF_NAME: "Echo MCP",
        CONF_MCP_URL: "https://mcp.example.com/mcp",
        CONF_MCP_HEADERS: {"Authorization": "Bearer secret"},
        CONF_MCP_ALLOWED_TOOLS: allowed_tools or [],
    }
    if include_return_schema is not None:
        data[CONF_MCP_INCLUDE_RETURN_SCHEMA] = include_return_schema
    if deferred_loading is not None:
        data[CONF_MCP_DEFERRED_LOADING] = deferred_loading
    return MockConfigEntry(
        domain=DOMAIN,
        title="Workspace",
        data={CONF_NAME: "Workspace"},
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "subentry_id": subentry_id,
                "data": data,
                "subentry_type": SUBENTRY_TYPE_MCP_SERVER,
                "title": "Echo MCP",
                "unique_id": None,
            },
        ),
        options={},
        unique_id=None,
    )


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
    client_context(alpn_protocols=SSL_ALPN_HTTP11)

    with patch.object(
        ssl.SSLContext,
        "load_verify_locations",
        side_effect=AssertionError("blocking SSL load"),
    ):
        client = _mcp_http_client_factory(_validated_url())()

    await client.aclose()


def test_mcp_log_redaction_uses_shared_sensitive_key_handling() -> None:
    redacted = redact_data(
        {
            "mcp_url": "https://mcp.example.com/mcp?token=visible",
            "headers": {"Authorization": "Bearer secret"},
            "result": {"token": "secret", "session_token": "visible", "value": "safe"},
        }
    )

    assert redacted["mcp_url"] == "**REDACTED**"
    assert redacted["headers"] == "**REDACTED**"
    assert redacted["result"]["token"] == "**REDACTED**"
    assert redacted["result"]["session_token"] == "visible"


def test_parse_mcp_headers_accepts_multiline_header_lines() -> None:
    assert parse_mcp_headers(
        "Authorization: Bearer secret\n\nX-Trace: value:with:colons"
    ) == {
        "Authorization": "Bearer secret",
        "X-Trace": "value:with:colons",
    }


@pytest.mark.parametrize(
    "headers",
    [
        '{"X-Test": "enabled"}',
        "Missing separator",
        "Bad Header: value",
        {"Bad Header": "value"},
        {"X-Test": 1},
    ],
)
def test_parse_mcp_headers_reject_invalid_values(headers: object) -> None:
    with pytest.raises(vol.Invalid):
        parse_mcp_headers(headers)


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("not a url", "invalid_mcp_url"),
        ("ftp://mcp.example.com/mcp", "invalid_mcp_url"),
        ("https://mcp.example.com/mcp#fragment", "invalid_mcp_url"),
        ("https://user:pass@mcp.example.com/mcp", "invalid_mcp_url"),
    ],
)
def test_normalise_mcp_url_rejects_invalid_inputs(url: str, reason: str) -> None:
    with pytest.raises(MCPValidationError) as err:
        normalise_mcp_url(url)

    assert err.value.reason == reason


def test_parse_allowed_tools_normalizes_strings_and_sequences() -> None:
    assert parse_allowed_tools(" read_file, list_files\nread_file ") == [
        "list_files",
        "read_file",
    ]
    assert parse_allowed_tools(["echo", "  list_files ", "echo", ""]) == [
        "echo",
        "list_files",
    ]
    assert parse_allowed_tools(None) == []

    with pytest.raises(vol.Invalid):
        parse_allowed_tools(123)


@pytest.mark.parametrize(
    ("stored_value", "expected_value"),
    [(None, True), (True, True), (False, False)],
)
def test_mcp_config_from_subentry_defaults_return_schema_preference(
    stored_value: bool | None, expected_value: bool
) -> None:
    entry = _mcp_entry(include_return_schema=stored_value)
    subentry = next(iter(entry.subentries.values()))

    assert (
        mcp_config_from_subentry(subentry)[CONF_MCP_INCLUDE_RETURN_SCHEMA]
        is expected_value
    )


@pytest.mark.parametrize(
    ("stored_value", "expected_value"),
    [(None, False), (True, True), (False, False)],
)
def test_mcp_config_from_subentry_defaults_deferred_loading_preference(
    stored_value: bool | None, expected_value: bool
) -> None:
    entry = _mcp_entry(deferred_loading=stored_value)
    subentry = next(iter(entry.subentries.values()))

    assert (
        mcp_config_from_subentry(subentry)[CONF_MCP_DEFERRED_LOADING] is expected_value
    )


def test_schema_hash_is_stable_for_json_equivalent_schemas() -> None:
    schema_a = {"type": "object", "properties": {"name": {"type": "string"}}}
    schema_b = {"properties": {"name": {"type": "string"}}, "type": "object"}
    schema_c = {"type": "object", "properties": {"id": {"type": "integer"}}}

    assert schema_hash(schema_a) == schema_hash(schema_b)
    assert schema_hash(schema_a) != schema_hash(schema_c)


def test_cached_mcp_tools_returns_copy_and_validates_entry_state() -> None:
    entry = _mcp_entry()

    with pytest.raises(MCPValidationError, match="config entry is not loaded"):
        cached_mcp_tools(entry, "mcp_server_1")

    entry.runtime_data = SimpleNamespace(mcp_tool_cache={})
    assert cached_mcp_tools(entry, "mcp_server_1") is None

    cache_key = _cache_key(entry, "mcp_server_1")
    entry.runtime_data.mcp_tool_cache[cache_key] = [{"name": "echo"}]
    cached = cached_mcp_tools(entry, "mcp_server_1")
    assert cached == [{"name": "echo"}]
    assert cached is not entry.runtime_data.mcp_tool_cache[cache_key]


async def test_discover_mcp_tools_from_config_shapes_and_filters_tools(
    hass: HomeAssistant,
) -> None:
    class FakeMCPToolset:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakeMCPToolset:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def list_tools(self) -> list[dict[str, object]]:
            return [
                {
                    "name": "echo",
                    "description": "Echo text",
                    "inputSchema": {"type": "object"},
                },
                {"name": "ignored", "inputSchema": {"type": "object"}},
                {"description": "missing name"},
            ]

    with (
        patch("custom_components.pydantic_ai_agent.mcp.MCPToolset", FakeMCPToolset),
        patch(
            "custom_components.pydantic_ai_agent.mcp._mcp_client",
            return_value=object(),
        ),
    ):
        tools = await async_discover_mcp_tools_from_config(
            hass,
            {
                CONF_NAME: "Echo MCP",
                CONF_MCP_URL: "https://mcp.example.com/mcp",
                CONF_MCP_ALLOWED_TOOLS: ["echo"],
            },
            server_id="server-1",
        )

    assert tools == [
        {
            "server_id": "server-1",
            "server_name": "Echo MCP",
            "name": "echo",
            "description": "Echo text",
            "input_schema": {"type": "object"},
            "schema_hash": schema_hash({"type": "object"}),
        }
    ]


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (TimeoutError(), "timeout"),
        (
            httpx.HTTPStatusError(
                "Unauthorized",
                request=httpx.Request("GET", "https://mcp.example.com/mcp"),
                response=httpx.Response(401),
            ),
            "invalid_auth",
        ),
        (RuntimeError("down"), "cannot_connect"),
    ],
)
async def test_discover_mcp_tools_from_config_maps_connection_errors(
    hass: HomeAssistant, error: BaseException, reason: str
) -> None:
    class FakeMCPToolset:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakeMCPToolset:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def list_tools(self) -> list[dict[str, object]]:
            raise error

    with (
        patch("custom_components.pydantic_ai_agent.mcp.MCPToolset", FakeMCPToolset),
        patch(
            "custom_components.pydantic_ai_agent.mcp._mcp_client",
            return_value=object(),
        ),
        pytest.raises(MCPValidationError) as err,
    ):
        await async_discover_mcp_tools_from_config(
            hass,
            {CONF_NAME: "Echo MCP", CONF_MCP_URL: "https://mcp.example.com/mcp"},
        )

    assert err.value.reason == reason


async def test_runtime_mcp_toolsets_require_selected_allowlisted_servers(
    hass: HomeAssistant,
) -> None:
    assert await async_runtime_mcp_toolsets(hass, _mcp_entry(), []) == []

    with pytest.raises(MCPValidationError) as err:
        await async_runtime_mcp_toolsets(hass, _mcp_entry(), ["missing"])
    assert err.value.reason == "mcp_server_not_found"

    with pytest.raises(MCPValidationError) as err:
        await async_runtime_mcp_toolsets(hass, _mcp_entry(), ["mcp_server_1"])
    assert err.value.reason == "mcp_tools_not_allowlisted"


async def test_runtime_mcp_toolsets_enforce_allowlist_and_deferred_loading(
    hass: HomeAssistant,
) -> None:
    class FakePrefixedToolset:
        def __init__(self, toolset: object, prefix: str) -> None:
            self.toolset = toolset
            self.prefix = prefix
            self.deferred = False

        def defer_loading(self) -> FakePrefixedToolset:
            self.deferred = True
            return self

    class FakeMCPToolset:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            self.process_tool_call = kwargs["process_tool_call"]
            self.include_return_schema = kwargs["include_return_schema"]
            self.filter_func = None

        def filtered(self, filter_func: object) -> FakeMCPToolset:
            self.filter_func = filter_func
            return self

        def prefixed(self, prefix: str) -> FakePrefixedToolset:
            return FakePrefixedToolset(self, prefix)

    entry = _mcp_entry(
        allowed_tools=["echo"],
        include_return_schema=False,
        deferred_loading=True,
    )
    with (
        patch("custom_components.pydantic_ai_agent.mcp.MCPToolset", FakeMCPToolset),
        patch(
            "custom_components.pydantic_ai_agent.mcp._mcp_client",
            return_value=object(),
        ),
    ):
        toolsets = await async_runtime_mcp_toolsets(hass, entry, ["mcp_server_1"])

    toolset = cast(Any, toolsets[0])
    assert len(toolsets) == 1
    assert toolset.prefix == "mcp_mcp_server_1"
    assert toolset.deferred is True
    assert toolset.toolset.include_return_schema is False
    assert toolset.toolset.filter_func is not None
    assert toolset.toolset.filter_func(None, SimpleNamespace(name="echo")) is True
    assert toolset.toolset.filter_func(None, SimpleNamespace(name="hidden")) is False

    async def call_tool(
        tool_name: str, tool_args: dict[str, object]
    ) -> dict[str, object]:
        return {"tool": tool_name, "args": tool_args}

    assert await toolset.toolset.process_tool_call(
        None, call_tool, "echo", {"message": "hi"}
    ) == {"tool": "echo", "args": {"message": "hi"}}

    with pytest.raises(MCPValidationError) as err:
        await toolset.toolset.process_tool_call(
            None, call_tool, "read_file", {"path": "/tmp/x"}
        )
    assert err.value.reason == "mcp_tool_not_allowed"
    assert err.value.tool_name == "read_file"
