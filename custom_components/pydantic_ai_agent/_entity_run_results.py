"""Result and failure helpers for entity-backed agent runs."""

import logging
from collections.abc import Mapping
from typing import Any

from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from opentelemetry.trace import Span
from pydantic_ai import AgentRunResult
from pydantic_ai.usage import UsageLimits

from ._entity_auth import (
    _async_create_runtime_auth_issue,
    _auth_status_code,
    _clear_runtime_auth_failure,
)
from ._types import PydanticAIAgentConfigEntry
from .metrics import (
    EVENT_AGENT_RUN_COMPLETED,
    EVENT_AGENT_RUN_FAILED,
    fire_integration_event,
    record_run_failure,
    record_run_success,
)
from .run_diagnostics import RunDiagnosticsRecorder
from .run_failures import (
    _AgentRunFailed,
    _AgentRunFailure,
    _classify_run_failure,
    _should_fallback,
)
from .run_state import AgentRunOutcome

_LOGGER = logging.getLogger(__name__)


def handle_profile_error(
    hass: HomeAssistant,
    entry: PydanticAIAgentConfigEntry,
    subentry: ConfigSubentry,
    *,
    err: Exception,
    index: int,
    is_last_attempt: bool,
    profile: Any,  # noqa: ANN401
    usage_limits: UsageLimits,
    agent_id: str,
    run_recorder: RunDiagnosticsRecorder,
    errors: list[BaseException],
) -> None:
    """Handle a model profile attempt failure with fallback logic."""
    if _auth_status_code(err) is None:
        _clear_runtime_auth_failure(hass, entry, profile)
    if is_last_attempt or not _should_fallback(err):
        failure = _classify_run_failure(err, usage_limits=usage_limits)
        if not _async_create_runtime_auth_issue(
            hass, entry, profile, err, failure.user_message
        ):
            _clear_runtime_auth_failure(hass, entry, profile)
        record_agent_run_failure(
            hass,
            entry,
            subentry,
            err,
            agent_id,
            model_profile=profile.title,
            failure=failure,
        )
        run_recorder.record(
            phase="failure",
            event="run_failed",
            data={
                "error": err,
                "failure": failure,
                "model_profile": profile.title,
            },
        )
        store_run_diagnostics(
            entry,
            subentry,
            run_recorder,
            status="failed",
            summary={
                "error": err,
                "failure": failure,
                "model_profile": profile.title,
            },
        )
        raise _AgentRunFailed(failure) from err
    errors.append(err)
    failure = _classify_run_failure(err, usage_limits=usage_limits)
    run_recorder.record(
        phase="attempt",
        event="model_profile_attempt_failed_retrying",
        data={
            "error": err,
            "failure": failure,
            "model_profile": profile.title,
        },
    )
    _LOGGER.warning(
        'Model profile "%s" failed with retryable %s; trying fallback: %s',
        profile.title,
        failure.error_type,
        failure.log_message,
    )


def record_agent_run_success(
    hass: HomeAssistant,
    entry: PydanticAIAgentConfigEntry,
    subentry: ConfigSubentry,
    outcome: AgentRunOutcome,
    agent_id: str | None = None,
) -> None:
    """Record successful run metrics and fire the completion event."""
    entity_id = agent_id or subentry.subentry_id
    record_run_success(
        hass,
        entry.entry_id,
        entry.runtime_data.metrics,
        subentry.subentry_id,
        model_profile=outcome.model_profile,
        duration=outcome.duration,
        usage=outcome.usage,
        model_pricing=outcome.model_pricing,
    )
    fire_integration_event(
        hass,
        EVENT_AGENT_RUN_COMPLETED,
        {
            "config_entry_id": entry.entry_id,
            "subentry_id": subentry.subentry_id,
            "entity_id": entity_id,
            "model_profile": outcome.model_profile,
        },
    )


def record_agent_run_failure(
    hass: HomeAssistant,
    entry: PydanticAIAgentConfigEntry,
    subentry: ConfigSubentry,
    err: BaseException,
    agent_id: str | None = None,
    *,
    model_profile: str | None = None,
    failure: _AgentRunFailure | None = None,
) -> None:
    """Record failed run metrics and fire the failure event."""
    failure = failure or _classify_run_failure(err)
    entity_id = agent_id or subentry.subentry_id
    _LOGGER.error(
        'Pydantic AI agent run failed for model profile "%s" (%s): %s',
        model_profile or "unknown",
        failure.error_type,
        failure.log_message,
    )
    record_run_failure(
        hass,
        entry.entry_id,
        entry.runtime_data.metrics,
        subentry.subentry_id,
        error=err,
        error_type=failure.error_type,
    )
    event_data: dict[str, object] = {
        "config_entry_id": entry.entry_id,
        "subentry_id": subentry.subentry_id,
        "entity_id": entity_id,
        "error_type": failure.error_type,
        "error_message": failure.user_message,
        "partial_response": failure.partial_response,
    }
    if failure.tool_problem is not None:
        event_data["tool_name"] = failure.tool_problem.tool_name
        event_data["tool_call_id"] = failure.tool_problem.tool_call_id
    if model_profile is not None:
        event_data["model_profile"] = model_profile
    fire_integration_event(hass, EVENT_AGENT_RUN_FAILED, event_data)


def store_run_diagnostics(
    entry: PydanticAIAgentConfigEntry,
    subentry: ConfigSubentry,
    recorder: RunDiagnosticsRecorder,
    *,
    status: str,
    summary: Mapping[str, Any],
) -> None:
    """Store latest bounded last-run diagnostics for this subentry."""
    entry.runtime_data.latest_run_diagnostics[subentry.subentry_id] = recorder.payload(
        status=status, summary=summary
    )


def set_span_usage_attributes(span: Span, result: AgentRunResult[Any]) -> None:
    """Copy aggregate Pydantic AI usage to the wrapper span without blocking runs."""
    try:
        usage_attributes = result.usage.opentelemetry_attributes()
        if usage_attributes:
            span.set_attributes(usage_attributes)
    except Exception:
        _LOGGER.exception("Failed to add usage attributes to Logfire span")
