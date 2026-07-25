from types import SimpleNamespace

from custom_components.pydantic_ai_agent.observability.metrics import (
    AgentRunMetrics,
    MetricsStore,
    metric_bool,
    metric_value,
    metrics_signal,
    usage_costs,
)
import pytest


def test_records_returns_read_only_live_view() -> None:
    store = MetricsStore()
    records = store.records

    with pytest.raises(TypeError):
        records["new"] = AgentRunMetrics()

    record = store.record_for("subentry")
    assert records["subentry"] is record


def test_record_for_lazily_creates_and_reuses_record() -> None:
    store = MetricsStore()

    first = store.record_for("agent")
    second = store.record_for("agent")

    assert first is second


def test_usage_costs_prices_supported_token_buckets() -> None:
    usage = SimpleNamespace(
        input_tokens=1_000,
        output_tokens=2_000,
        details={"cache_read_tokens": 250},
    )

    costs = usage_costs(usage, {"input": 1.0, "output": 2.0, "cache_read": 0.5})

    assert costs.input == 0.00075
    assert costs.output == 0.004
    assert costs.cache_read == 0.000125
    assert costs.total == 0.004875


def test_usage_costs_total_is_unknown_when_used_bucket_has_no_price() -> None:
    usage = SimpleNamespace(input_tokens=100, output_tokens=50, details={})

    costs = usage_costs(usage, {"input": 1.0})

    assert costs.input == 0.0001
    assert costs.output is None
    assert costs.total is None


def test_usage_costs_total_is_unknown_for_unsupported_token_buckets() -> None:
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=0,
        details={"cache_write_tokens": 10},
    )

    costs = usage_costs(usage, {"input": 1.0})

    assert costs.input == 0.0001
    assert costs.total is None


def test_metric_value_and_bool_read_record_attributes() -> None:
    record = AgentRunMetrics(
        last_run_model_profile="provider:model",
        provider_healthy=False,
        last_run_succeeded=None,
    )

    assert metric_value(record, "last_run_model_profile") == "provider:model"
    assert metric_bool(record, "provider_healthy") is False
    assert metric_bool(record, "last_run_succeeded") is None


def test_metrics_signal_includes_domain_entry_and_subentry() -> None:
    assert metrics_signal("entry-1", "subentry-2") == ("pydantic_ai_agent_metrics_entry-1_subentry-2")
