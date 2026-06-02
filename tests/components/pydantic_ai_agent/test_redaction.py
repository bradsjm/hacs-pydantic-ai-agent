"""Test shared redaction helpers for Pydantic AI Agent."""

from homeassistant.components.diagnostics import REDACTED

from custom_components.pydantic_ai_agent._redaction import redact_data


def test_redact_data_uses_explicit_shared_sensitive_keys() -> None:
    """Test shared redaction is key-based and redacts sensitive headers."""
    redacted = redact_data(
        {
            "api_key": "secret",
            "provider_headers": {"Authorization": "Bearer secret"},
            "nested": {
                "Authorization": "Bearer secret",
                "session_token": "visible",
            },
        }
    )

    assert redacted["api_key"] == REDACTED
    assert redacted["provider_headers"] == REDACTED
    assert redacted["nested"]["Authorization"] == REDACTED
    assert redacted["nested"]["session_token"] == "visible"
