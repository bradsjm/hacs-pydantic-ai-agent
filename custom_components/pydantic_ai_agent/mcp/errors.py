"""MCP exception types."""

from dataclasses import dataclass


@dataclass(slots=True)
class MCPValidationError(Exception):
    """MCP validation or discovery failed with a stable reason."""

    reason: str
    message: str
    status_code: int | None = None
    server_id: str | None = None
    tool_name: str | None = None
