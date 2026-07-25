"""Runtime metrics for Pydantic AI Agent entities."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send

from ..const import DOMAIN

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
    last_run_cache_read_tokens: int | None = None
    last_run_total_tokens: int | None = None
    last_run_input_cost: float | None = None
    last_run_output_cost: float | None = None
    last_run_cache_read_cost: float | None = None
    last_run_total_cost: float | None = None
    last_run_model_request_count: int | None = None
    last_run_tool_use_count: int | None = None
    last_mcp_tool_call: str | None = None
    cumulative_input_tokens: int = 0
    cumulative_output_tokens: int = 0
    cumulative_cache_read_tokens: int = 0
    cumulative_total_tokens: int = 0
    cumulative_input_cost: float | None = None
    cumulative_output_cost: float | None = None
    cumulative_cache_read_cost: float | None = None
    cumulative_total_cost: float | None = None
    last_run_duration: float | None = None
    last_error_type: str | None = None
    consecutive_failures: int = 0
    provider_healthy: bool | None = None
    last_run_succeeded: bool | None = None


@dataclass(kw_only=True)
class MetricsStore:
    """Mutable per-entry metrics store."""

    _records: dict[str, AgentRunMetrics] = field(default_factory=dict)

    @property
    def records(self) -> Mapping[str, AgentRunMetrics]:
        """Return a read-only view of the metrics records."""
        return MappingProxyType(self._records)

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
    usage: object,
    model_pricing: Mapping[str, float] | None = None,
) -> None:
    """Record a successful agent run."""
    record = store.record_for(subentry_id)
    input_tokens = _int_usage_value(usage, "input_tokens")
    output_tokens = _int_usage_value(usage, "output_tokens")
    cache_read_tokens = _cache_read_token_usage(usage)
    total_tokens = _int_usage_value(usage, "total_tokens")
    model_requests = _int_usage_value(usage, "requests")
    tool_uses = _int_usage_value(usage, "tool_calls")
    cost = usage_costs(usage, model_pricing)

    record.last_run_model_profile = model_profile
    record.last_run_input_tokens = input_tokens
    record.last_run_output_tokens = output_tokens
    record.last_run_cache_read_tokens = cache_read_tokens
    record.last_run_total_tokens = total_tokens
    record.last_run_input_cost = cost.input
    record.last_run_output_cost = cost.output
    record.last_run_cache_read_cost = cost.cache_read
    record.last_run_total_cost = cost.total
    record.last_run_model_request_count = model_requests
    record.last_run_tool_use_count = tool_uses
    record.cumulative_input_tokens += input_tokens
    record.cumulative_output_tokens += output_tokens
    record.cumulative_cache_read_tokens += cache_read_tokens
    record.cumulative_total_tokens += total_tokens
    record.cumulative_input_cost = _add_optional_cost(record.cumulative_input_cost, cost.input)
    record.cumulative_output_cost = _add_optional_cost(record.cumulative_output_cost, cost.output)
    record.cumulative_cache_read_cost = _add_optional_cost(record.cumulative_cache_read_cost, cost.cache_read)
    record.cumulative_total_cost = _add_optional_cost(record.cumulative_total_cost, cost.total)
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
    error_type: str | None = None,
) -> None:
    """Record a failed agent run."""
    record = store.record_for(subentry_id)
    record.last_error_type = error_type or type(error).__name__
    record.consecutive_failures += 1
    record.provider_healthy = False
    record.last_run_succeeded = False
    async_dispatcher_send(hass, metrics_signal(entry_id, subentry_id))


def record_mcp_tool_call(
    hass: HomeAssistant,
    entry_id: str,
    store: MetricsStore,
    subentry_id: str,
    *,
    tool_name: str,
) -> None:
    """Record the last MCP tool call for one agent subentry."""
    record = store.record_for(subentry_id)
    record.last_mcp_tool_call = tool_name
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
    return cast(int | float | str | None, getattr(record, key))


def metric_bool(record: AgentRunMetrics, key: str) -> bool | None:
    """Return a binary metrics value by key."""
    value = getattr(record, key)
    if value is None:
        return None
    return bool(value)


def _int_usage_value(usage: object, attr: str) -> int:
    """Return an integer Pydantic AI usage value from a result test double safely."""
    value = getattr(usage, attr, 0)
    if callable(value):
        value = value()
    if isinstance(value, int | float):
        return int(value)
    return 0


def _cache_read_token_usage(usage: object) -> int:
    """Return cached input tokens from Pydantic AI usage or provider details."""
    return max(
        _int_usage_value(usage, "cache_read_tokens"),
        _int_usage_detail(usage, "cache_read_tokens"),
        _int_usage_detail(usage, "prompt_tokens_details.cached_tokens"),
        _int_usage_detail(usage, "input_tokens_details.cached_tokens"),
        _int_usage_detail(usage, "cached_tokens"),
    )


def _unsupported_cost_token_usage(usage: object) -> int:
    """Return known token buckets this integration cannot price separately."""
    return sum(
        (
            max(
                _int_usage_value(usage, "cache_write_tokens"),
                _int_usage_detail(usage, "cache_write_tokens"),
            ),
            max(
                _int_usage_value(usage, "input_audio_tokens"),
                _int_usage_detail(usage, "input_audio_tokens"),
            ),
            max(
                _int_usage_value(usage, "cache_audio_read_tokens"),
                _int_usage_detail(usage, "cache_audio_read_tokens"),
            ),
            max(
                _int_usage_value(usage, "output_audio_tokens"),
                _int_usage_detail(usage, "output_audio_tokens"),
            ),
        )
    )


def _int_usage_detail(usage: object, key: str) -> int:
    """Return an integer Pydantic AI usage detail value safely."""
    details = getattr(usage, "details", {})
    if callable(details):
        details = details()
    if not isinstance(details, Mapping):
        return 0
    value = details.get(key, 0)
    if isinstance(value, int | float):
        return int(value)
    return 0


@dataclass(frozen=True, kw_only=True)
class _RunCosts:
    """Cost components for one completed model run."""

    input: float | None
    output: float | None
    cache_read: float | None
    total: float | None


def usage_costs(usage: object, model_pricing: Mapping[str, float] | None = None) -> _RunCosts:
    """Return USD costs for one usage object from configured pricing."""
    input_tokens = _int_usage_value(usage, "input_tokens")
    output_tokens = _int_usage_value(usage, "output_tokens")
    cache_read_tokens = _cache_read_token_usage(usage)
    unsupported_tokens = _unsupported_cost_token_usage(usage)
    return _run_costs(
        input_tokens,
        output_tokens,
        cache_read_tokens,
        unsupported_tokens,
        model_pricing,
    )


def _run_costs(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    unsupported_tokens: int,
    model_pricing: Mapping[str, float] | None,
) -> _RunCosts:
    """Return USD costs for one run when pricing covers each used token bucket."""
    pricing = model_pricing or {}
    non_cached_input_tokens = max(input_tokens - cache_read_tokens, 0)
    input_cost = _token_cost(non_cached_input_tokens, pricing.get("input"))
    output_cost = _token_cost(output_tokens, pricing.get("output"))
    cache_read_cost = _token_cost(cache_read_tokens, pricing.get("cache_read"))
    total = _total_cost(
        (
            (non_cached_input_tokens, input_cost),
            (output_tokens, output_cost),
            (cache_read_tokens, cache_read_cost),
            (unsupported_tokens, None),
        )
    )
    return _RunCosts(
        input=input_cost,
        output=output_cost,
        cache_read=cache_read_cost,
        total=total,
    )


def _token_cost(tokens: int, price_per_million: float | None) -> float | None:
    """Return the cost for a priced non-zero token bucket."""
    if tokens <= 0 or price_per_million is None:
        return None
    return tokens * price_per_million / 1_000_000


def _total_cost(components: tuple[tuple[int, float | None], ...]) -> float | None:
    """Return total cost only when all non-zero token buckets are priced."""
    total = 0.0
    saw_costed_component = False
    for tokens, cost in components:
        if tokens <= 0:
            continue
        if cost is None:
            return None
        total += cost
        saw_costed_component = True
    if not saw_costed_component:
        return None
    return total


def _add_optional_cost(current: float | None, value: float | None) -> float | None:
    """Increment a cumulative cost only when this run produced that cost."""
    if value is None:
        return current
    if current is None:
        return value
    return current + value


type MetricValueFn = Callable[[AgentRunMetrics], int | float | str | None]
type MetricBoolFn = Callable[[AgentRunMetrics], bool | None]
