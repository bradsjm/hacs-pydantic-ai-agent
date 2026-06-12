"""Focused tests for Logfire support state management."""

import sys
import warnings
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
from custom_components.pydantic_ai_agent._entity_run_results import (
    set_span_usage_attributes,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_LLM_HASS_API,
    CONF_LOGFIRE_INCLUDE_CONTENT,
    CONF_LOGFIRE_TOKEN,
    DOMAIN,
    PROVIDER_GOOGLE_GEMINI,
)
from custom_components.pydantic_ai_agent.logfire_support import (
    _configure_logfire_sync,
    _entry_logfire_token,
    _logfire_state,
    agent_run_span,
    async_configure_logfire,
    async_release_logfire,
    instrument_agent,
    logfire_active_for_entry,
    logfire_include_content,
    logfire_token_conflict,
)
from custom_components.pydantic_ai_agent.model_profiles import primary_model_profile
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir
from tests.components.pydantic_ai_agent.support.builders import (
    ai_task_subentry_data,
    conversation_subentry_data,
    provider_subentry_data,
    workspace_entry,
)
from tests.components.pydantic_ai_agent.support.pydantic_ai import RunResult, Usage


class _Span:
    """Simple synchronous context manager for Logfire span tests."""

    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}

    def __enter__(self) -> _Span:
        """Enter the span context."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Exit the span context."""

    def set_attributes(self, attributes: dict[str, object]) -> None:
        """Record attributes set after span creation."""
        self.attributes.update(attributes)


async def test_async_configure_logfire_sets_first_owner(
    hass: HomeAssistant,
) -> None:
    """Test the first Logfire owner configures process-global state."""
    entry = workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-a"})
    entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent.logfire_support._configure_logfire_sync"
    ) as configure_logfire:
        assert await async_configure_logfire(hass, entry) is True

    state = _logfire_state(hass)
    configure_logfire.assert_called_once_with("token-a")
    assert state.configured_token == "token-a"
    assert state.owner_include_content_by_entry_id == {entry.entry_id: False}
    assert logfire_active_for_entry(hass, entry) is True


async def test_async_configure_logfire_shares_same_token(
    hass: HomeAssistant,
) -> None:
    """Test same-token entries share the active Logfire configuration."""
    first_entry = workspace_entry(
        data={CONF_LOGFIRE_TOKEN: "token-a", CONF_LOGFIRE_INCLUDE_CONTENT: True}
    )
    second_entry = workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-a"})
    first_entry.add_to_hass(hass)
    second_entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent.logfire_support._configure_logfire_sync"
    ) as configure_logfire:
        assert await async_configure_logfire(hass, first_entry) is True
        assert await async_configure_logfire(hass, second_entry) is True

    state = _logfire_state(hass)
    assert configure_logfire.call_count == 1
    assert logfire_active_for_entry(hass, first_entry) is True
    assert logfire_active_for_entry(hass, second_entry) is True
    assert logfire_include_content(hass, first_entry) is True
    assert logfire_include_content(hass, second_entry) is False
    assert state.configured_include_content is True


async def test_async_configure_logfire_creates_conflict_issue(
    hass: HomeAssistant,
) -> None:
    """Test conflicting Logfire tokens create a repair issue."""
    first_entry = workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-a"})
    second_entry = workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-b"})
    first_entry.add_to_hass(hass)
    second_entry.add_to_hass(hass)
    issue_id = f"logfire_token_conflict_{second_entry.entry_id}"

    with patch(
        "custom_components.pydantic_ai_agent.logfire_support._configure_logfire_sync"
    ):
        assert await async_configure_logfire(hass, first_entry) is True
        assert await async_configure_logfire(hass, second_entry) is False

    assert logfire_token_conflict(hass, second_entry) is True
    assert logfire_active_for_entry(hass, second_entry) is False
    assert ir.async_get(hass).async_get_issue(DOMAIN, issue_id) is not None


async def test_async_release_logfire_keeps_same_token_owner_active(
    hass: HomeAssistant,
) -> None:
    """Test releasing one shared owner leaves the other active."""
    first_entry = workspace_entry(
        data={CONF_LOGFIRE_TOKEN: "token-a", CONF_LOGFIRE_INCLUDE_CONTENT: True}
    )
    second_entry = workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-a"})
    first_entry.add_to_hass(hass)
    second_entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent.logfire_support._configure_logfire_sync"
    ):
        await async_configure_logfire(hass, first_entry)
        await async_configure_logfire(hass, second_entry)

    await async_release_logfire(hass, first_entry)

    state = _logfire_state(hass)
    assert logfire_active_for_entry(hass, first_entry) is False
    assert logfire_active_for_entry(hass, second_entry) is True
    assert state.configured_token == "token-a"
    assert state.configured_include_content is False


async def test_async_release_logfire_promotes_loaded_conflicting_entries(
    hass: HomeAssistant,
) -> None:
    """Test releasing the last owner promotes loaded conflicting entries."""
    first_entry = workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-a"})
    second_entry = workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-b"})
    third_entry = workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-b"})
    first_entry.add_to_hass(hass)
    second_entry.add_to_hass(hass)
    third_entry.add_to_hass(hass)
    second_entry.mock_state(hass, config_entries.ConfigEntryState.LOADED)
    third_entry.mock_state(hass, config_entries.ConfigEntryState.LOADED)

    with patch(
        "custom_components.pydantic_ai_agent.logfire_support._configure_logfire_sync"
    ) as configure_logfire:
        assert await async_configure_logfire(hass, first_entry) is True
        assert await async_configure_logfire(hass, second_entry) is False
        assert await async_configure_logfire(hass, third_entry) is False

        await async_release_logfire(hass, first_entry)

    assert configure_logfire.call_count == 2
    assert logfire_active_for_entry(hass, second_entry) is True
    assert logfire_active_for_entry(hass, third_entry) is True


def test_entry_logfire_token_normalizes_strings() -> None:
    """Test Logfire token parsing trims whitespace and rejects empty values."""
    assert _entry_logfire_token(
        workspace_entry(data={CONF_LOGFIRE_TOKEN: " token "})
    ) == ("token")
    assert (
        _entry_logfire_token(workspace_entry(data={CONF_LOGFIRE_TOKEN: "   "})) is None
    )
    assert _entry_logfire_token(workspace_entry(data={CONF_LOGFIRE_TOKEN: 123})) is None


def test_configure_logfire_sync_suppresses_known_passlib_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Logfire setup suppresses only the known passlib warning."""

    def configure(**kwargs: object) -> None:
        del kwargs
        warnings.warn_explicit(
            (
                "handler names should be lower-case, and use underscores instead "
                "of hyphens: 'LambdaRuntimeClient' => 'lambdaruntimeclient'"
            ),
            category=Warning,
            filename="/tmp/passlib/registry.py",
            lineno=43,
            module="passlib.registry",
        )

    monkeypatch.setitem(
        sys.modules,
        "logfire",
        SimpleNamespace(configure=Mock(side_effect=configure)),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _configure_logfire_sync("token-a")

    assert not caught
    logfire = cast(Any, sys.modules["logfire"])
    logfire.configure.assert_called_once_with(
        send_to_logfire=True,
        token="token-a",
        service_name=DOMAIN,
        console=False,
        inspect_arguments=False,
    )


def test_configure_logfire_sync_does_not_suppress_unrelated_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Logfire setup still surfaces unrelated warnings."""

    def configure(**kwargs: object) -> None:
        del kwargs
        warnings.warn("another warning", UserWarning, stacklevel=2)

    monkeypatch.setitem(
        sys.modules,
        "logfire",
        SimpleNamespace(configure=Mock(side_effect=configure)),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _configure_logfire_sync("token-a")

    assert len(caught) == 1
    assert str(caught[0].message) == "another warning"


async def test_instrument_agent_only_runs_for_active_owner(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test agent instrumentation is skipped unless the entry owns Logfire."""
    instrument = Mock()
    monkeypatch.setitem(
        sys.modules,
        "logfire",
        SimpleNamespace(instrument_pydantic_ai=instrument),
    )
    active_entry = workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-a"})
    inactive_entry = workspace_entry(data={CONF_LOGFIRE_TOKEN: "token-b"})
    active_entry.add_to_hass(hass)
    inactive_entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent.logfire_support._configure_logfire_sync"
    ):
        await async_configure_logfire(hass, active_entry)

    agent = cast(Any, object())
    instrument_agent(hass, inactive_entry, agent)
    instrument_agent(hass, active_entry, agent)

    instrument.assert_called_once()
    assert instrument.call_args.kwargs["include_content"] is False


async def test_agent_run_span_is_safe_without_active_logfire(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test agent_run_span is a no-op when the entry is not active."""
    span = Mock(return_value=_Span())
    monkeypatch.setitem(sys.modules, "logfire", SimpleNamespace(span=span))
    entry = workspace_entry(
        (
            conversation_subentry_data(
                "provider-1:profile-1",
                subentry_id="conversation-1",
                llm_hass_api=[CONF_LLM_HASS_API],
            ),
            provider_subentry_data(subentry_id="provider-1"),
        ),
        data={CONF_LOGFIRE_TOKEN: "token-a"},
    )
    subentry = entry.subentries["conversation-1"]

    with agent_run_span(
        hass,
        entry,
        subentry,
        profile=primary_model_profile(entry, subentry),
        attempt_index=0,
        attempt_count=1,
        entity_id="conversation.test",
        conversation_id="conversation-id",
    ) as run_span:
        assert run_span is None

    span.assert_not_called()


async def test_agent_run_span_adds_ai_task_specific_metadata(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test active spans include AI task context and a dynamic title."""
    span = Mock(return_value=_Span())
    monkeypatch.setitem(sys.modules, "logfire", SimpleNamespace(span=span))
    entry = workspace_entry(
        (
            ai_task_subentry_data(
                "provider-1:profile-1",
                subentry_id="ai-task-1",
                title="Report {task}",
                task_name="Morning Report",
                web_fetch_enabled=True,
                todo_workspace_entity_id="todo.demo",
            ),
            provider_subentry_data(subentry_id="provider-1", title="Hosted OpenAI"),
        ),
        data={CONF_LOGFIRE_TOKEN: "token-a", CONF_LOGFIRE_INCLUDE_CONTENT: True},
    )
    entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent.logfire_support._configure_logfire_sync"
    ):
        assert await async_configure_logfire(hass, entry) is True

    subentry = entry.subentries["ai-task-1"]
    profile = primary_model_profile(entry, subentry)

    with agent_run_span(
        hass,
        entry,
        subentry,
        profile=profile,
        attempt_index=1,
        attempt_count=2,
        entity_id="ai_task.report_task",
        conversation_id=None,
    ) as run_span:
        assert run_span is span.return_value

    assert span.call_args.args == ("Report {{task}} → Fast GPT (fallback 2/2)",)
    assert span.call_args.kwargs["ha.ai_task_name"] == "Morning Report"
    assert span.call_args.kwargs["ha.todo_workspace_enabled"] is True
    assert span.call_args.kwargs["ha.web_fetch_enabled"] is True
    assert span.call_args.kwargs["ha.is_fallback_attempt"] is True
    assert span.call_args.kwargs["ha.attempt_index"] == 1
    assert span.call_args.kwargs["ha.attempt_count"] == 2
    assert span.call_args.kwargs["ha.conversation_id"] is None
    assert span.call_args.kwargs["gen_ai.operation.name"] == "chat"
    assert span.call_args.kwargs["gen_ai.system"] == "openai"
    assert span.call_args.kwargs["gen_ai.provider.name"] == "openai"


async def test_agent_run_span_maps_gemini_operation_name(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Gemini runs use Generate Content tracing metadata."""
    span = Mock(return_value=_Span())
    monkeypatch.setitem(sys.modules, "logfire", SimpleNamespace(span=span))
    entry = workspace_entry(
        (
            conversation_subentry_data(
                "provider-1:profile-1",
                subentry_id="conversation-1",
            ),
            provider_subentry_data(
                subentry_id="provider-1",
                provider_mode=PROVIDER_GOOGLE_GEMINI,
            ),
        ),
        data={CONF_LOGFIRE_TOKEN: "token-a"},
    )
    entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent.logfire_support._configure_logfire_sync"
    ):
        assert await async_configure_logfire(hass, entry) is True

    subentry = entry.subentries["conversation-1"]
    profile = primary_model_profile(entry, subentry)

    with agent_run_span(
        hass,
        entry,
        subentry,
        profile=profile,
        attempt_index=0,
        attempt_count=1,
        entity_id="conversation.test",
        conversation_id="conversation-id",
    ):
        pass

    assert span.call_args.kwargs["gen_ai.operation.name"] == "generate_content"
    assert span.call_args.kwargs["gen_ai.system"] == "gcp.gemini"
    assert span.call_args.kwargs["gen_ai.provider.name"] == "gcp.gemini"


def test_set_span_usage_attributes_adds_cost_and_response_model() -> None:
    """Test post-run span enrichment includes pricing and response metadata."""
    span = _Span()
    result = RunResult(
        "ok",
        usage=Usage(input_tokens=20, output_tokens=5, total_tokens=25),
    )

    set_span_usage_attributes(
        cast(Any, span),
        cast(Any, result),
        model_name="gpt-test",
        model_pricing={"input": 1.0, "output": 2.0},
    )

    assert span.attributes["gen_ai.response.model"] == "gpt-test"
    assert span.attributes["gen_ai.usage.total_tokens"] == 25
    assert span.attributes["ha.cost_currency"] == "USD"
    assert span.attributes["ha.input_cost"] == pytest.approx(0.00002)
    assert span.attributes["ha.output_cost"] == pytest.approx(0.00001)
    assert span.attributes["ha.total_cost"] == pytest.approx(0.00003)
    assert span.attributes["gen_ai.usage.cost"] == pytest.approx(0.00003)


def test_set_span_usage_attributes_omits_total_cost_when_pricing_incomplete() -> None:
    """Test cached token usage without cache pricing omits total cost."""
    span = _Span()
    result = RunResult(
        "ok",
        usage=Usage(
            input_tokens=20,
            output_tokens=5,
            total_tokens=25,
            cache_read_tokens=4,
            details={"cache_read_tokens": 4},
        ),
    )

    set_span_usage_attributes(
        cast(Any, span),
        cast(Any, result),
        model_name="gpt-test",
        model_pricing={"input": 1.0, "output": 2.0},
    )

    assert span.attributes["ha.input_cost"] == pytest.approx(0.000016)
    assert span.attributes["ha.output_cost"] == pytest.approx(0.00001)
    assert "ha.cache_read_cost" not in span.attributes
    assert "ha.total_cost" not in span.attributes
    assert "gen_ai.usage.cost" not in span.attributes
