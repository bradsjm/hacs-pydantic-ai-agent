"""MCP exception types."""

from dataclasses import dataclass

import httpx


@dataclass(slots=True)
class MCPValidationError(Exception):
    """MCP validation or discovery failed with a stable reason."""

    reason: str
    message: str
    status_code: int | None = None
    server_id: str | None = None
    tool_name: str | None = None


def is_mcp_timeout_error(err: BaseException) -> bool:
    """Return whether an MCP SDK error represents a request read timeout."""
    error = getattr(err, "error", None)
    code = getattr(error, "code", None)
    return code == httpx.codes.REQUEST_TIMEOUT
