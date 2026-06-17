"""Test Pydantic AI Agent conversation trace recording and diagnostics."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast
from unittest.mock import patch

from custom_components.pydantic_ai_agent.agent.chat_deltas import _StreamTraceRecorder
from custom_components.pydantic_ai_agent.diagnostics import (
    async_get_config_entry_diagnostics,
)
from homeassistant.components import conversation
from homeassistant.components.conversation.trace import async_get_traces
from homeassistant.core import Context, HomeAssistant
from pydantic_ai import (
    AgentRunResultEvent,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    ThinkingPart,
    ThinkingPartDelta,
)
from pytest_homeassistant_custom_component.common import MockConfigEntry
from tests.components.pydantic_ai_agent.support.builders import (
    conversation_subentry_data,
    provider_runtime_data,
    provider_subentry_data,
    workspace_entry,
    workspace_runtime_data,
)
from tests.components.pydantic_ai_agent.support.pydantic_ai import (
    Agent as _Agent,
)
from tests.components.pydantic_ai_agent.support.pydantic_ai import (
    EventStream,
    Usage,
)

_PROVIDER_SUBENTRY_ID = "provider-1"
_MODEL_PROFILE_ID = "model-profile-1"
_MODEL_PROFILE_REF = f"{_PROVIDER_SUBENTRY_ID}:{_MODEL_PROFILE_ID}"


class _ResultWithMessages:
    """Minimal Agent result with explicit final messages."""

    def __init__(self, output: str, messages: list[ModelResponse]) -> None:
        self.output = output
        self.usage = Usage()
        self._messages = messages

    def new_messages(self) -> list[ModelResponse]:
        return self._messages


def _entry() -> MockConfigEntry:
    """Return a config entry with one conversation subentry."""
    entry = workspace_entry(
        (
            conversation_subentry_data(_MODEL_PROFILE_REF),
            provider_subentry_data(
                subentry_id=_PROVIDER_SUBENTRY_ID,
                title="Hosted OpenAI",
                profile_id=_MODEL_PROFILE_ID,
            ),
        )
    )
    entry.runtime_data = workspace_runtime_data(
        providers={
            _PROVIDER_SUBENTRY_ID: provider_runtime_data(
                subentry_id=_PROVIDER_SUBENTRY_ID, name="Hosted OpenAI"
            )
        },
    )
    return entry


class TracedAgent(_Agent):
    """Agent override that streams thinking without final TextPart events."""

    @asynccontextmanager
    async def run_stream_events(
        self, *_args: object, **kwargs: object
    ) -> AsyncIterator[EventStream]:
        self.run_stream_events_calls += 1
        self.run_kwargs = kwargs
        result = _ResultWithMessages(
            "Hello. How can I help you?",
            [ModelResponse(parts=[TextPart(content="Hello. How can I help you?")])],
        )

        async def stream() -> AsyncIterator[object]:
            yield PartStartEvent(index=0, part=ThinkingPart(content=""))
            yield PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta="thinking about greeting"),
            )
            yield AgentRunResultEvent(cast(Any, result))

        yield cast(EventStream, stream())


async def test_streaming_records_safe_trace_payload(
    hass: HomeAssistant,
) -> None:
    """Test streaming records bounded HA trace and diagnostic details."""
    entry = _entry()
    entry.add_to_hass(hass)
    agent = TracedAgent()

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    entity_id = next(
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id != "conversation.home_assistant"
    )
    with (
        patch(
            "custom_components.pydantic_ai_agent.entity.chat_model_for_profile",
            return_value=object(),
        ),
        patch(
            "custom_components.pydantic_ai_agent.entity.Agent",
            return_value=agent,
        ),
    ):
        result = await conversation.async_converse(
            hass,
            "hello",
            None,
            Context(),
            agent_id=entity_id,
        )

    assert result.response.speech["plain"]["speech"] == "Hello. How can I help you?"
    trace_events = async_get_traces()[-1].as_dict()["events"]
    payload = next(
        event["data"]["pydantic_ai_stream"]
        for event in trace_events
        if event["data"] and "pydantic_ai_stream" in event["data"]
    )
    json.dumps(payload)
    assert payload["events"]
    assert all("event_type" in event for event in payload["events"])
    assert payload["chat_deltas"]
    assert payload["final_new_messages"]
    assert payload["backfill"]["changed"] is True
    assert (
        "Hello. How can I help you?" in payload["final_chat_content"]["content_preview"]
    )

    subentry_id = next(iter(entry.subentries))
    diagnostics = await async_get_config_entry_diagnostics(hass, entry)
    diagnostic_trace = diagnostics["runtime"]["latest_stream_traces"][subentry_id]
    assert diagnostic_trace["events_total"] >= len(diagnostic_trace["events"])
    run_diagnostics = diagnostics["runtime"]["latest_run_diagnostics"][subentry_id]
    assert run_diagnostics["status"] == "success"
    assert run_diagnostics["subentry_id"] == subentry_id
    assert run_diagnostics["timeline_event_count"] >= 4
    timeline = run_diagnostics["timeline"]
    assert all("timestamp" in event for event in timeline)
    assert all("elapsed_ms" in event for event in timeline)
    assert all("delta_ms" in event for event in timeline)
    assert all("phase" in event for event in timeline)


def test_stream_trace_recorder_keeps_tail_for_long_streams() -> None:
    """Test long traces retain the beginning and end of the stream."""
    recorder = _StreamTraceRecorder()
    for index in range(250):
        recorder.record_event(
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=str(index)),
            )
        )
    recorder.record_event(PartStartEvent(index=1, part=TextPart(content="done")))

    payload = recorder.payload(
        final_messages=[ModelResponse(parts=[TextPart(content="done")])],
        backfill={"changed": False},
        final_chat_content=None,
    )

    assert payload["events_truncated"] is True
    assert payload["events_omitted_middle_count"] > 0
    assert payload["events"]
    assert payload["events_tail"]
    assert payload["events"][0]["order"] == 1
    assert payload["events_tail"][-1]["order"] == payload["events_total"]
    assert len(payload["events"]) + len(payload["events_tail"]) <= 200
