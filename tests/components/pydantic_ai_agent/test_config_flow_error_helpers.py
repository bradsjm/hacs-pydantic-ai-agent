"""Test config-flow provider error helper behavior."""

import errno
import socket
import ssl

import httpx
import pytest
from custom_components.pydantic_ai_agent.provider_validation import (
    _format_api_error,
    _map_http_error,
)
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError


def test_http_error_formats_redacted_compact_metadata() -> None:
    err = ModelHTTPError(
        status_code=402,
        model_name="deepseek/deepseek-v4-flash:free",
        body={
            "message": "Provider returned error",
            "metadata": {
                "provider_name": "Crucible",
                "access_token": "secret-token",
                "request_headers": {"Authorization": "Bearer nested-secret"},
            },
        },
    )
    result = _map_http_error(err)
    assert result.reason == "provider_error"
    assert "access_token" not in result.message
    assert "nested-secret" not in result.message


@pytest.mark.parametrize(
    ("status_code", "expected_reason", "expected_label"),
    [
        (400, "invalid_model", "invalid request"),
        (401, "invalid_auth", "authentication issue"),
        (403, "permission_denied", "permission issue"),
        (404, "invalid_model", "model not found"),
        (408, "timeout", "timeout"),
        (429, "rate_limited", "rate limit"),
        (500, "provider_error", "provider server issue"),
    ],
)
def test_http_error_status_categories(
    status_code: int, expected_reason: str, expected_label: str
) -> None:
    err = ModelHTTPError(status_code=status_code, model_name="gpt-test", body=None)
    result = _map_http_error(err)
    assert result.reason == expected_reason
    assert expected_label in result.message


@pytest.mark.parametrize(
    ("cause", "expected_reason", "expected_message"),
    [
        (socket.gaierror(), "cannot_connect", "Host not found."),
        (
            OSError(errno.ECONNREFUSED, "refused"),
            "cannot_connect",
            "Connection refused.",
        ),
        (
            OSError(errno.ENETUNREACH, "unreachable"),
            "cannot_connect",
            "Network unreachable.",
        ),
        (ssl.SSLError("certificate verify failed"), "cannot_connect", "TLS error."),
        (TimeoutError(), "timeout", "Request timed out."),
        (httpx.ReadTimeout("timeout"), "timeout", "Request timed out."),
    ],
)
def test_api_error_connection_categories(
    cause: BaseException, expected_reason: str, expected_message: str
) -> None:
    err = ModelAPIError("gpt-test", "probe failed")
    err.__cause__ = cause
    result = _format_api_error(err)
    assert result.reason == expected_reason
    assert expected_message in result.message


def test_api_error_fallback_is_concise() -> None:
    err = ModelAPIError("gpt-test", "status_code: 500, body: {'huge': 'payload'}")
    result = _format_api_error(err)
    assert result.reason == "provider_error"
