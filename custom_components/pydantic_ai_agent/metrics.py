"""Runtime metrics for Pydantic AI Agent entities."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .const import DOMAIN


EVENT_AGENT_RUN_COMPLETED = "agent_run_completed"
EVENT_AGENT_RUN_FAILED = "agent_run_failed"
EVENT_MCP_TOOL_REFRESH_COMPLETED = "mcp_tool_refresh_completed"
EVENT_MCP_TOOL_REFRESH_FAILED = "mcp_tool_refresh_failed"
EVENT_STRUCTURED_AI_TASK_OUTPUT_GENERATED = "structured_ai_task_output_generated"


def metrics_signal(entry_id: str, subentry_id: str) -> str:
    """Return the dispatcher signal for one metrics record."""
    return f"{DOMAIN}_metrics_{entry_id}_{subentry_id}"


@dataclass(kw_only=True)
class AgentRunMetrics:
    """Metrics captured for one conversation or AI task subentry."""

    last_run_model_profile: str | None = None
    last_run_input_tokens: int | None = None
    last_run_output_tokens: int | None = None
    last_run_total_tokens: int | None = None
    last_run_model_request_count: int | None = None
    last_run_tool_use_count: int | None = None
    cumulative_input_tokens: int = 0
    cumulative_output_tokens: int = 0
    cumulative_total_tokens: int = 0
    last_run_duration: float | None = None
    last_error_type: str | None = None
    consecutive_failures: int = 0
    provider_healthy: bool | None = None
    last_run_succeeded: bool | None = None


@dataclass(kw_only=True)
class MetricsStore:
    """Mutable per-entry metrics store."""

    _records: dict[str, AgentRunMetrics] = field(default_factory=dict)

    def record_for(self, subentry_id: str) -> AgentRunMetrics:
        """Return the metrics record for a subentry."""
        return self._records.setdefault(subentry_id, AgentRunMetrics())


def record_run_success(
    hass: HomeAssistant,
    entry_id: str,
    store: MetricsStore,
    subentry_id: str,
    *,
    model_profile: str,
    duration: float,
    usage: Any,
) -> None:
    """Record a successful agent run."""
    record = store.record_for(subentry_id)
    input_tokens = _int_usage_value(usage, "input_tokens")
    output_tokens = _int_usage_value(usage, "output_tokens")
    total_tokens = _int_usage_value(usage, "total_tokens")
    model_requests = _int_usage_value(usage, "requests")
    tool_uses = _int_usage_value(usage, "tool_calls")

    record.last_run_model_profile = model_profile
    record.last_run_input_tokens = input_tokens
    record.last_run_output_tokens = output_tokens
    record.last_run_total_tokens = total_tokens
    record.last_run_model_request_count = model_requests
    record.last_run_tool_use_count = tool_uses
    record.cumulative_input_tokens += input_tokens
    record.cumulative_output_tokens += output_tokens
    record.cumulative_total_tokens += total_tokens
    record.last_run_duration = duration
    record.last_error_type = None
    record.consecutive_failures = 0
    record.provider_healthy = True
    record.last_run_succeeded = True
    async_dispatcher_send(hass, metrics_signal(entry_id, subentry_id))


def record_run_failure(
    hass: HomeAssistant,
    entry_id: str,
    store: MetricsStore,
    subentry_id: str,
    *,
    error: BaseException,
) -> None:
    """Record a failed agent run."""
    record = store.record_for(subentry_id)
    record.last_error_type = type(error).__name__
    record.consecutive_failures += 1
    record.provider_healthy = False
    record.last_run_succeeded = False
    async_dispatcher_send(hass, metrics_signal(entry_id, subentry_id))


def fire_integration_event(
    hass: HomeAssistant,
    event_type: str,
    event_data: dict[str, Any],
) -> None:
    """Fire a pydantic_ai_agent integration event."""
    hass.bus.async_fire(f"{DOMAIN}_{event_type}", event_data)


def metric_value(record: AgentRunMetrics, key: str) -> int | float | str | None:
    """Return a metrics value by key."""
    return getattr(record, key)


def metric_bool(record: AgentRunMetrics, key: str) -> bool | None:
    """Return a binary metrics value by key."""
    value = getattr(record, key)
    if value is None:
        return None
    return bool(value)


def _int_usage_value(usage: Any, attr: str) -> int:
    """Return an integer Pydantic AI usage value from a result test double safely."""
    value = getattr(usage, attr, 0)
    if callable(value):
        value = value()
    if isinstance(value, int | float):
        return int(value)
    return 0


type MetricValueFn = Callable[[AgentRunMetrics], int | float | str | None]
type MetricBoolFn = Callable[[AgentRunMetrics], bool | None]
