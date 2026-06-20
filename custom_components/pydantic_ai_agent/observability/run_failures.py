"""Pydantic AI run failure classification helpers."""

from dataclasses import dataclass
from typing import cast

import httpx
from homeassistant.exceptions import HomeAssistantError
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.usage import UsageLimits

from ..agent.tool_errors import HAToolRetryExhausted
from ..runtime.error_classification import has_connection_failure


@dataclass(frozen=True, kw_only=True)
class _ToolProblem:
    """Safe summary of a tool result problem."""

    tool_name: str | None
    tool_call_id: str | None
    outcome: str
    reason: str | None = None


@dataclass(frozen=True, kw_only=True)
class _AgentRunFailure:
    """Single classified source for terminal agent run failures."""

    error_type: str
    user_message: str
    log_message: str
    partial_response: bool = False
    tool_problem: _ToolProblem | None = None


class _AgentRunFailed(HomeAssistantError):
    """Home Assistant error carrying classified run-failure details."""

    def __init__(self, failure: _AgentRunFailure) -> None:
        """Initialize the error with the safe user-facing message."""
        super().__init__(failure.user_message)
        self.failure = failure


def _format_http_error(err: ModelHTTPError) -> str:
    """Return a user-facing provider HTTP error message."""
    return f'The provider returned HTTP {err.status_code} for model "{err.model_name}".'


def _format_api_error(err: ModelAPIError) -> str:
    """Return a user-facing provider API error message."""
    return f'The provider returned an API error for model "{err.model_name}".'


def _run_failure_cause(err: BaseException) -> BaseException:
    """Return the underlying failure cause for classified run errors."""
    if isinstance(err, _AgentRunFailed) and err.__cause__ is not None:
        return err.__cause__
    return err


def _classify_run_failure(
    err: BaseException,
    *,
    usage_limits: UsageLimits | None = None,
    partial_response: bool = False,
    tool_problem: _ToolProblem | None = None,
) -> _AgentRunFailure:
    """Classify a run failure into safe user, log, event, and metric details."""
    if isinstance(err, _AgentRunFailed):
        return err.failure

    cause = _run_failure_cause(err)
    error_type = type(cause).__name__
    context = _tool_problem_context(tool_problem)
    prefix = (
        "Terminated after a partial response because "
        if partial_response
        else "Terminated because "
    )

    if isinstance(cause, UsageLimitExceeded):
        return _build_usage_limit_failure(
            cause,
            error_type,
            prefix,
            context,
            usage_limits,
            partial_response,
            tool_problem,
        )

    message = _build_failure_message(cause, prefix, error_type)
    return _AgentRunFailure(
        error_type=error_type,
        user_message=message + context,
        log_message=message + context,
        partial_response=partial_response,
        tool_problem=tool_problem,
    )


def _build_usage_limit_failure(
    cause: UsageLimitExceeded,
    error_type: str,
    prefix: str,
    context: str,
    usage_limits: UsageLimits | None,
    partial_response: bool,
    tool_problem: _ToolProblem | None,
) -> _AgentRunFailure:
    """Build a failure result for usage limit exceeded errors."""
    request_limit = usage_limits.request_limit if usage_limits is not None else None
    if request_limit is not None:
        message = (
            f"{prefix}the model exceeded the configured maximum of "
            f"{request_limit} iterations. Increase the run max "
            "iterations or fix repeated tool failures."
        )
    else:
        message = (
            f"{prefix}the model exceeded a configured usage limit. "
            "Increase the relevant run limit or reduce the request."
        )
    return _AgentRunFailure(
        error_type=error_type,
        user_message=message + context,
        log_message=message + context,
        partial_response=partial_response,
        tool_problem=tool_problem,
    )


def _build_failure_message(
    cause: BaseException,
    prefix: str,
    error_type: str,
) -> str:
    """Build a user-facing failure message from the root exception cause."""
    if isinstance(cause, ModelHTTPError):
        return _http_failure_message(cause, prefix)
    if isinstance(cause, ModelAPIError):
        api_error = cast(ModelAPIError, cause)
        if _has_connection_failure(cause):
            return (
                f"{prefix}the provider connection failed for model "
                f'"{api_error.model_name}". Check network connectivity and provider '
                "availability."
            )
        return _format_api_error(api_error)
    if isinstance(cause, HAToolRetryExhausted):
        return str(cause)
    if isinstance(cause, UnexpectedModelBehavior):
        return (
            f"{prefix}the provider returned an unexpected response. Check "
            "model/provider compatibility or try a different model profile."
        )
    if isinstance(cause, TimeoutError | httpx.TimeoutException):
        return (
            f"{prefix}the provider request timed out. Check network "
            "connectivity or try again later."
        )
    if isinstance(cause, NotImplementedError | UserError):
        return f"Invalid provider configuration: {cause}"
    if isinstance(cause, HomeAssistantError):
        return str(cause)
    return str(cause) or error_type


def _http_failure_message(err: ModelHTTPError, prefix: str) -> str:
    """Return an actionable HTTP provider failure message."""
    if err.status_code == 429:
        return (
            f"{prefix}the provider quota or rate limit was reached for model "
            f'"{err.model_name}". Check provider quota/rate limits or try '
            "again later."
        )
    if err.status_code in {401, 403}:
        return (
            f"{prefix}the provider rejected credentials or permissions for "
            f'model "{err.model_name}". Check the provider API key and '
            "account access."
        )
    if 500 <= err.status_code <= 599:
        return (
            f"{prefix}the provider service returned HTTP {err.status_code} "
            f'for model "{err.model_name}". Try again later or use a fallback '
            "model profile."
        )
    return _format_http_error(err)


def _tool_problem_context(tool_problem: _ToolProblem | None) -> str:
    """Return safe user-facing context for the latest tool problem."""
    if tool_problem is None:
        return ""
    name = tool_problem.tool_name or "unknown tool"
    if tool_problem.reason:
        return f" Last tool failure: {name} reported {tool_problem.reason}."
    return f" Last tool failure: {name} returned {tool_problem.outcome}."


def _home_assistant_error(err: Exception) -> HomeAssistantError:
    """Convert provider/runtime failures into HA-facing errors."""
    if isinstance(err, _AgentRunFailed):
        return err
    if isinstance(err, HomeAssistantError):
        return err
    return HomeAssistantError(_classify_run_failure(err).user_message)


def _should_fallback(err: Exception) -> bool:
    """Return if a failed model attempt should try the next profile."""
    if isinstance(err, _AgentRunFailed):
        return not err.failure.partial_response and _should_fallback(
            cast(Exception, _run_failure_cause(err))
        )
    if isinstance(err, ModelHTTPError):
        http_error = cast(ModelHTTPError, err)
        return (
            http_error.status_code
            in {
                408,
                409,
                429,
            }
            or 500 <= http_error.status_code <= 599
        )
    if isinstance(err, TimeoutError | httpx.TimeoutException | UsageLimitExceeded):
        return True
    if isinstance(err, ModelAPIError):
        return _has_connection_failure(err)
    return False


def _has_connection_failure(err: BaseException) -> bool:
    """Return if an exception cause chain indicates transport failure."""
    return has_connection_failure(err)
