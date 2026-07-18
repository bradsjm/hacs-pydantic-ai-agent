from custom_components.pydantic_ai_agent.observability.run_failures import (
    _AgentRunFailed,
    _AgentRunFailure,
    _classify_run_failure,
    _home_assistant_error,
    _should_fallback,
)
from homeassistant.exceptions import HomeAssistantError
import httpx
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UsageLimitExceeded
from pydantic_ai.usage import UsageLimits
import pytest


@pytest.mark.parametrize(
    ("status", "fragment"),
    [
        (401, "credentials or permissions"),
        (403, "credentials or permissions"),
        (429, "quota or rate limit"),
        (500, "provider service returned HTTP 500"),
        (418, "HTTP 418"),
    ],
)
def test_classify_http_errors_by_status(status: int, fragment: str) -> None:
    failure = _classify_run_failure(ModelHTTPError(status, "test-model"))

    assert failure.error_type == "ModelHTTPError"
    assert fragment in failure.user_message
    assert "test-model" in failure.user_message


def test_classify_api_connection_failure_is_actionable() -> None:
    err = ModelAPIError("test-model", "wrapped")
    err.__cause__ = httpx.ConnectError("connect")

    failure = _classify_run_failure(err)

    assert failure.error_type == "ModelAPIError"
    assert "provider connection failed" in failure.user_message
    assert "test-model" in failure.user_message


def test_classify_generic_api_error_keeps_provider_api_category() -> None:
    failure = _classify_run_failure(ModelAPIError("test-model", "bad request"))

    assert failure.error_type == "ModelAPIError"
    assert "API error" in failure.user_message
    assert "test-model" in failure.user_message


@pytest.mark.parametrize("err", [TimeoutError(), httpx.ReadTimeout("timeout")])
def test_classify_timeouts(err: Exception) -> None:
    failure = _classify_run_failure(err)

    assert failure.error_type == type(err).__name__
    assert "timed out" in failure.user_message


def test_classify_usage_limit_with_request_limit() -> None:
    failure = _classify_run_failure(
        UsageLimitExceeded("too many"), usage_limits=UsageLimits(request_limit=3)
    )

    assert failure.error_type == "UsageLimitExceeded"
    assert "maximum of 3 iterations" in failure.user_message


def test_home_assistant_error_passthrough() -> None:
    err = HomeAssistantError("already safe")

    assert _home_assistant_error(err) is err


@pytest.mark.parametrize(
    "err",
    [
        TimeoutError(),
        httpx.ReadTimeout("timeout"),
        UsageLimitExceeded("too many"),
    ],
)
def test_should_fallback_for_retryable_non_http_failures(err: Exception) -> None:
    assert _should_fallback(err) is True


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
def test_should_fallback_for_retryable_http_statuses(status: int) -> None:
    assert _should_fallback(ModelHTTPError(status, "test-model")) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_should_not_fallback_for_non_retryable_http_statuses(status: int) -> None:
    assert _should_fallback(ModelHTTPError(status, "test-model")) is False


def test_should_fallback_for_api_connection_failure_only() -> None:
    connection = ModelAPIError("test-model", "wrapped")
    connection.__cause__ = httpx.ConnectError("connect")

    assert _should_fallback(connection) is True
    assert _should_fallback(ModelAPIError("test-model", "api")) is False


def test_partial_response_suppresses_fallback() -> None:
    err = _AgentRunFailed(
        _AgentRunFailure(
            error_type="ModelHTTPError",
            user_message="partial",
            log_message="partial",
            partial_response=True,
        )
    )
    err.__cause__ = ModelHTTPError(500, "test-model")

    assert _should_fallback(err) is False


def test_non_partial_classified_failure_can_fallback_from_cause() -> None:
    err = _AgentRunFailed(
        _AgentRunFailure(
            error_type="ModelHTTPError",
            user_message="retry",
            log_message="retry",
        )
    )
    err.__cause__ = ModelHTTPError(500, "test-model")

    assert _should_fallback(err) is True
