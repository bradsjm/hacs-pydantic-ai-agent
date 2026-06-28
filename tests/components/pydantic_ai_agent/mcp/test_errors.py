"""Tests for MCP error classification helpers."""

from custom_components.pydantic_ai_agent.mcp.errors import is_mcp_timeout_error
import httpx
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData
import pytest


@pytest.mark.parametrize(
    ("err", "expected"),
    [
        (
            McpError(
                ErrorData(
                    code=httpx.codes.REQUEST_TIMEOUT,
                    message="Request timed out. Waited 10 seconds.",
                )
            ),
            True,
        ),
        (McpError(ErrorData(code=500, message="Server error")), False),
        (McpError(ErrorData(code=-32603, message="Internal error")), False),
        (ValueError("not an MCP error"), False),
    ],
)
def test_is_mcp_timeout_error_classifies_timeout_codes(
    err: BaseException, expected: bool
) -> None:
    """Only MCP request-timeout errors are classified as retryable timeouts."""
    assert is_mcp_timeout_error(err) is expected
