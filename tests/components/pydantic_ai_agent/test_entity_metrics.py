"""Test entity metrics and pricing behavior."""

from types import SimpleNamespace

import pytest
from custom_components.pydantic_ai_agent.metrics import (
    MetricsStore,
    record_mcp_tool_call,
    record_run_failure,
    record_run_success,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError


def test_record_run_failure_updates_health_metrics(hass: HomeAssistant) -> None:
    """Test failed runs update native health metric state."""
    store = MetricsStore()

    record_run_failure(
        hass,
        "entry-1",
        store,
        "subentry-1",
        error=TimeoutError(),
    )

    record = store.record_for("subentry-1")
    assert record.last_error_type == "TimeoutError"
    assert record.consecutive_failures == 1
    assert record.provider_healthy is False
    assert record.last_run_succeeded is False


def test_metrics_store_records_is_read_only_view() -> None:
    """Test metrics store exposes existing records without mutable access."""
    store = MetricsStore()

    assert store.records == {}
    record = store.record_for("subentry-1")

    assert store.records["subentry-1"] is record
    with pytest.raises(TypeError):
        store.records["subentry-2"] = record  # type: ignore[index]
    assert list(store.records) == ["subentry-1"]


def test_record_run_failure_uses_classified_error_type(hass: HomeAssistant) -> None:
    """Test failed runs can store classified error types for sensors."""
    store = MetricsStore()

    record_run_failure(
        hass,
        "entry-1",
        store,
        "subentry-1",
        error=HomeAssistantError("wrapped"),
        error_type="UsageLimitExceeded",
    )

    record = store.record_for("subentry-1")
    assert record.last_error_type == "UsageLimitExceeded"
    assert record.consecutive_failures == 1


def test_record_mcp_tool_call_tracks_last_tool_name(hass: HomeAssistant) -> None:
    """Test MCP tool calls update the last called tool metric immediately."""
    store = MetricsStore()

    record_mcp_tool_call(
        hass,
        "entry-1",
        store,
        "subentry-1",
        tool_name="echo",
    )
    record_mcp_tool_call(
        hass,
        "entry-1",
        store,
        "subentry-1",
        tool_name="list_files",
    )

    record = store.record_for("subentry-1")
    assert record.last_mcp_tool_call == "list_files"


def test_record_run_success_tracks_priced_costs(hass: HomeAssistant) -> None:
    """Test successful runs compute component and cumulative USD costs."""
    store = MetricsStore()

    record_run_success(
        hass,
        "entry-1",
        store,
        "subentry-1",
        model_profile="GPT Test",
        duration=1.2,
        usage=SimpleNamespace(
            input_tokens=1200,
            output_tokens=300,
            cache_read_tokens=200,
            total_tokens=1500,
            requests=1,
            tool_calls=0,
        ),
        model_pricing={"input": 0.5, "output": 2.0, "cache_read": 0.1},
    )

    record = store.record_for("subentry-1")
    assert record.last_run_input_cost == 1000 * 0.5 / 1_000_000
    assert record.last_run_output_cost == 300 * 2.0 / 1_000_000
    assert record.last_run_cache_read_cost == 200 * 0.1 / 1_000_000
    assert record.last_run_total_cost == pytest.approx(0.00112)
    assert record.cumulative_input_cost == record.last_run_input_cost
    assert record.cumulative_output_cost == record.last_run_output_cost
    assert record.cumulative_cache_read_cost == record.last_run_cache_read_cost
    assert record.cumulative_total_cost == record.last_run_total_cost


def test_record_run_success_leaves_total_cost_unknown_when_pricing_missing(
    hass: HomeAssistant,
) -> None:
    """Test total cost is unknown unless all used token buckets are priced."""
    store = MetricsStore()

    record_run_success(
        hass,
        "entry-1",
        store,
        "subentry-1",
        model_profile="GPT Test",
        duration=1.2,
        usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=300,
            cache_read_tokens=0,
            total_tokens=1300,
        ),
        model_pricing={"input": 0.5},
    )
    record_run_success(
        hass,
        "entry-1",
        store,
        "subentry-1",
        model_profile="GPT Test",
        duration=1.2,
        usage=SimpleNamespace(
            input_tokens=0,
            output_tokens=100,
            cache_read_tokens=0,
            total_tokens=100,
        ),
        model_pricing={"output": 2.0},
    )

    record = store.record_for("subentry-1")
    assert record.last_run_input_cost is None
    assert record.last_run_output_cost == 100 * 2.0 / 1_000_000
    assert record.last_run_total_cost == record.last_run_output_cost
    assert record.cumulative_input_cost == 1000 * 0.5 / 1_000_000
    assert record.cumulative_output_cost == record.last_run_output_cost
    assert record.cumulative_total_cost == record.last_run_output_cost


def test_record_run_success_reads_cached_tokens_from_usage_details(
    hass: HomeAssistant,
) -> None:
    """Test cached-token details are billed as cache reads."""
    store = MetricsStore()

    record_run_success(
        hass,
        "entry-1",
        store,
        "subentry-1",
        model_profile="GPT Test",
        duration=1.2,
        usage=SimpleNamespace(
            input_tokens=1200,
            output_tokens=300,
            total_tokens=1500,
            details={"input_tokens_details.cached_tokens": 200},
        ),
        model_pricing={"input": 0.5, "output": 2.0, "cache_read": 0.1},
    )

    record = store.record_for("subentry-1")
    assert record.last_run_cache_read_tokens == 200
    assert record.last_run_input_cost == 1000 * 0.5 / 1_000_000
    assert record.last_run_cache_read_cost == 200 * 0.1 / 1_000_000
    assert record.last_run_total_cost == pytest.approx(0.00112)


def test_record_run_success_leaves_total_unknown_for_unpriced_token_categories(
    hass: HomeAssistant,
) -> None:
    """Test unsupported token buckets keep total cost unknown."""
    store = MetricsStore()

    record_run_success(
        hass,
        "entry-1",
        store,
        "subentry-1",
        model_profile="GPT Test",
        duration=1.2,
        usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=300,
            cache_read_tokens=0,
            cache_write_tokens=50,
            total_tokens=1300,
        ),
        model_pricing={"input": 0.5, "output": 2.0, "cache_read": 0.1},
    )

    record = store.record_for("subentry-1")
    assert record.last_run_input_cost == 1000 * 0.5 / 1_000_000
    assert record.last_run_output_cost == 300 * 2.0 / 1_000_000
    assert record.last_run_total_cost is None
    assert record.cumulative_total_cost is None
