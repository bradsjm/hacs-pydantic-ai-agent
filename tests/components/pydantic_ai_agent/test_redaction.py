"""Test shared redaction helpers for Pydantic AI Agent."""

from homeassistant.components.diagnostics import REDACTED

from custom_components.pydantic_ai_agent._redaction import redact_data


def test_redact_data_uses_explicit_shared_sensitive_keys() -> None:
    """Test shared redaction is key-based and redacts MCP URLs."""
    redacted = redact_data(
        {
            "api_key": "secret",
            "mcp_url": "https://mcp.example.com/mcp?token=visible",
            "nested": {
                "Authorization": "Bearer secret",
                "session_token": "visible",
            },
        }
    )

    assert redacted["api_key"] == REDACTED
    assert redacted["mcp_url"] == REDACTED
    assert redacted["nested"]["Authorization"] == REDACTED
    assert redacted["nested"]["session_token"] == "visible"
