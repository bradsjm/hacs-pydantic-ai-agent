"""Tests for provider validation error mapping."""

from custom_components.pydantic_ai_agent.models._provider_validation_errors import (
    format_api_error,
    map_http_error,
)
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
import pytest


def test_format_api_error_maps_generic_api_error() -> None:
    """Non-HTTP provider API failures use the provider_error reason."""
    error = format_api_error(ModelAPIError(model_name="model-1", message="boom"))

    assert error.reason == "provider_error"
    assert error.status_code is None
    assert "model-1" in error.message


@pytest.mark.parametrize(
    ("status_code", "reason"),
    [
        (400, "invalid_model"),
        (401, "invalid_auth"),
        (403, "permission_denied"),
        (404, "invalid_model"),
        (408, "timeout"),
        (429, "rate_limited"),
        (504, "timeout"),
        (500, "provider_error"),
    ],
)
def test_map_http_error_classifies_status_codes(status_code: int, reason: str) -> None:
    """HTTP status codes map to stable config-flow reason keys."""
    error = map_http_error(
        ModelHTTPError(status_code=status_code, model_name="model-1", body=None)
    )

    assert error.reason == reason
    assert error.status_code == status_code
    assert str(status_code) in error.message


def test_map_http_error_redacts_sensitive_metadata() -> None:
    """Metadata shown in validation messages omits secret-bearing keys."""
    error = map_http_error(
        ModelHTTPError(
            status_code=422,
            model_name="model-1",
            body={
                "metadata": {
                    "request_id": "req-123",
                    "api_key": "secret-token",
                    "nested": {"authorization": "Bearer secret"},
                }
            },
        )
    )

    assert error.reason == "provider_error"
    assert error.status_code == 422
    assert "req-123" in error.message
    assert "secret-token" not in error.message
    assert "Bearer secret" not in error.message
