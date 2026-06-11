"""MCP models."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValidatedMCPURL:
    """MCP URL plus its exact origin."""

    url: str
    scheme: str
    hostname: str
    port: int
