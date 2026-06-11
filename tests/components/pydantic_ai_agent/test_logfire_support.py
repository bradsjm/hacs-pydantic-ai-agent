"""Focused tests for Logfire support state management."""

import sys
import warnings
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
from custom_components.pydantic_ai_agent.const import (
    CONF_LLM_HASS_API,
    CONF_LOGFIRE_INCLUDE_CONTENT,
    CONF_LOGFIRE_TOKEN,
    DOMAIN,
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
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir
from tests.components.pydantic_ai_agent.support.builders import (
    conversation_subentry_data,
    workspace_entry,
)


class _Span:
    """Simple synchronous context manager for Logfire span tests."""

    def __enter__(self) -> _Span:
        """Enter the span context."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Exit the span context."""


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
        ),
        data={CONF_LOGFIRE_TOKEN: "token-a"},
    )
    subentry = entry.subentries["conversation-1"]

    with agent_run_span(
        hass,
        entry,
        subentry,
        entity_id="conversation.test",
        conversation_id="conversation-id",
        model_name="gpt-test",
    ) as run_span:
        assert run_span is None

    span.assert_not_called()
