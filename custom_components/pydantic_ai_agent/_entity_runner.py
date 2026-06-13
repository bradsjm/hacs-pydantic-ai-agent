"""Agent runner functions extracted from PydanticAIBaseLLMEntity methods."""

from collections.abc import AsyncIterable, Callable, Sequence
from typing import Any, cast

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.capabilities import AbstractCapability, ToolSearch
from pydantic_ai.messages import ModelMessage
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset, DeferredLoadingToolset
from pydantic_ai.usage import UsageLimits

from ._entity_auth import (
    _join_instructions,
)
from ._entity_run_results import set_span_usage_attributes
from ._types import WorkspaceRuntimeData
from .chat_deltas import (
    _agent_events_to_chat_deltas,
    _append_agent_messages,
    _append_missing_final_text,
    _append_text,
    _json_output,
)
from .const import CONF_MCP_SERVER_IDS
from .ha_toolset import tools_from_llm_api_with_diagnostics
from .logfire_support import agent_run_span, instrument_agent
from .mcp import async_runtime_mcp_toolsets
from .model_profiles import (
    ModelProfile,
    chat_model_for_profile,
    model_settings,
    thinking_capability,
)
from .model_request_settings import (
    _model_settings_with_provider_extra_body,
    _model_settings_with_templated_extra_body,
)
from .run_diagnostics import RunDiagnosticsRecorder
from .run_failures import (
    _AgentRunFailed,
    _classify_run_failure,
)
from .run_state import AgentRunOutcome, _StreamRunState
from .virtual_workspace import virtual_workspace_enabled, virtual_workspace_parts

type PydanticAIAgentConfigEntry = ConfigEntry[WorkspaceRuntimeData]


async def run_model_profile(
    hass: HomeAssistant,
    entry: PydanticAIAgentConfigEntry,
    subentry: ConfigSubentry,
    *,
    index: int,
    attempt_count: int = 1,
    profile: ModelProfile,
    usage_limits: UsageLimits,
    chat_log: conversation.ChatLog,
    agent_id: str,
    user_prompt: str | Sequence[Any] | None,
    message_history: list[ModelMessage],
    structured_output_tool_names: set[str],
    has_structure: bool,
    stream: bool,
    run_recorder: RunDiagnosticsRecorder,
    agent_output_type: object,
    capabilities: list[AbstractCapability],
    extra_toolsets: Sequence[AbstractToolset[Any]],
    extra_instructions: str | None,
    tool_retries: int,
    agent_factory: type[Agent[Any, Any]] = Agent,
    model_factory: Callable[..., object] = chat_model_for_profile,
    virtual_workspace_parts_factory: Callable[..., Any] = virtual_workspace_parts,
) -> AgentRunOutcome:
    """Run one model profile attempt and return the outcome."""
    virtual_toolsets: Sequence[AbstractToolset[Any]] = ()
    virtual_instructions: str | None = None
    if virtual_workspace_enabled(subentry.data):
        parts = virtual_workspace_parts_factory()
        virtual_toolsets = parts.toolsets
        virtual_instructions = parts.instructions
    instructions = _join_instructions(virtual_instructions, extra_instructions)
    settings = model_settings(profile, subentry.data)
    settings = _model_settings_with_provider_extra_body(entry, profile, settings)
    settings = _model_settings_with_templated_extra_body(hass, profile, settings)
    mcp_toolsets = await async_runtime_mcp_toolsets(
        hass,
        entry,
        subentry.subentry_id,
        subentry.data.get(CONF_MCP_SERVER_IDS),
    )
    toolsets = [*mcp_toolsets, *virtual_toolsets, *extra_toolsets]
    run_capabilities = list(capabilities)
    if any(isinstance(toolset, DeferredLoadingToolset) for toolset in toolsets):
        run_capabilities = [
            capability
            for capability in run_capabilities
            if not isinstance(capability, ToolSearch)
        ]
        run_capabilities.append(ToolSearch(strategy="keywords"))
    if thinking := thinking_capability(subentry.data, profile):
        run_capabilities.append(thinking)
    run_recorder.record(
        phase="attempt",
        event="model_profile_attempt_started",
        data={
            "attempt_index": index,
            "model_profile": profile,
            "model_settings": settings,
            "usage_limits": usage_limits,
            "mcp_toolset_count": len(mcp_toolsets),
            "extra_toolset_count": len(extra_toolsets),
            "virtual_workspace_enabled": virtual_workspace_enabled(subentry.data),
            "capability_types": [
                type(capability).__name__ for capability in run_capabilities
            ],
        },
    )
    agent = agent_factory(
        cast(Any, model_factory(hass, entry, profile)),
        output_type=cast(Any, agent_output_type),
        instructions=instructions,
        model_settings=settings,
        tool_retries=tool_retries,
        output_retries=2,
        tools=tools_from_llm_api_with_diagnostics(chat_log.llm_api, run_recorder),
        toolsets=toolsets,
        max_concurrency=1,
        capabilities=run_capabilities,
    )
    instrument_agent(hass, entry, agent)
    return await run_agent_try(
        hass,
        entry,
        subentry,
        agent,
        profile,
        settings,
        chat_log,
        agent_id,
        user_prompt,
        message_history,
        usage_limits,
        structured_output_tool_names,
        has_structure,
        stream,
        run_recorder,
        attempt_index=index,
        attempt_count=attempt_count,
    )


async def run_agent_try(
    hass: HomeAssistant,
    entry: PydanticAIAgentConfigEntry,
    subentry: ConfigSubentry,
    agent: Agent[Any, Any],
    profile: ModelProfile,
    settings: ModelSettings,
    chat_log: conversation.ChatLog,
    agent_id: str,
    user_prompt: str | Sequence[Any] | None,
    message_history: list[ModelMessage],
    usage_limits: UsageLimits,
    structured_output_tool_names: set[str],
    has_structure: bool,
    stream: bool,
    run_recorder: RunDiagnosticsRecorder,
    *,
    attempt_index: int = 0,
    attempt_count: int = 1,
) -> AgentRunOutcome:
    """Run one model profile attempt and append only successful messages."""
    try:
        async with agent:
            with agent_run_span(
                hass,
                entry,
                subentry,
                profile=profile,
                attempt_index=attempt_index,
                attempt_count=attempt_count,
                entity_id=agent_id,
                conversation_id=chat_log.conversation_id,
            ) as span:
                start = hass.loop.time()
                run_recorder.record(
                    phase="llm_request",
                    source="provider",
                    event="request_started",
                    data={
                        "model_profile": profile,
                        "model_settings": settings,
                        "user_prompt": user_prompt,
                        "message_history": message_history,
                        "stream": stream and not has_structure,
                        "has_structure": has_structure,
                    },
                )
                if stream and not has_structure:
                    result = await run_agent_stream(
                        agent,
                        user_prompt,
                        message_history,
                        settings,
                        usage_limits,
                        chat_log,
                        agent_id,
                        structured_output_tool_names,
                        run_recorder,
                        entry,
                        subentry,
                    )
                    if span is not None:
                        set_span_usage_attributes(
                            span,
                            result,
                            model_name=profile.model_name,
                            model_pricing=profile.model_pricing,
                        )
                    duration = hass.loop.time() - start
                    run_recorder.record(
                        phase="llm_response",
                        source="provider",
                        event="stream_finished",
                        data={
                            "new_messages": result.new_messages(),
                            "output": result.output,
                            "usage": result.usage,
                            "duration": duration,
                        },
                    )
                    return AgentRunOutcome(
                        output=result.output,
                        usage=result.usage,
                        duration=duration,
                        model_profile=profile.title,
                        model_profile_ref=profile.ref,
                        provider_subentry_id=profile.provider_subentry_id,
                        model_pricing=profile.model_pricing,
                        run_recorder=run_recorder,
                    )

                if not has_structure:
                    result = await agent.run(
                        user_prompt,
                        message_history=message_history,
                        model_settings=settings,
                        usage_limits=usage_limits,
                    )
                    if span is not None:
                        set_span_usage_attributes(
                            span,
                            result,
                            model_name=profile.model_name,
                            model_pricing=profile.model_pricing,
                        )
                    duration = hass.loop.time() - start
                    await _append_agent_messages(
                        chat_log, agent_id, result.new_messages()
                    )
                    run_recorder.record(
                        phase="llm_response",
                        source="provider",
                        event="run_finished",
                        data={
                            "new_messages": result.new_messages(),
                            "output": result.output,
                            "usage": result.usage,
                            "duration": duration,
                        },
                    )
                    return AgentRunOutcome(
                        output=result.output,
                        usage=result.usage,
                        duration=duration,
                        model_profile=profile.title,
                        model_profile_ref=profile.ref,
                        provider_subentry_id=profile.provider_subentry_id,
                        model_pricing=profile.model_pricing,
                        run_recorder=run_recorder,
                    )

                result = await agent.run(
                    user_prompt,
                    message_history=message_history,
                    model_settings=settings,
                    usage_limits=usage_limits,
                )
                if span is not None:
                    set_span_usage_attributes(
                        span,
                        result,
                        model_name=profile.model_name,
                        model_pricing=profile.model_pricing,
                    )
                duration = hass.loop.time() - start
                output = result.output
                await _append_agent_messages(
                    chat_log,
                    agent_id,
                    result.new_messages(),
                    output_tool_names=structured_output_tool_names,
                )
                if not isinstance(chat_log.content[-1], conversation.AssistantContent):
                    await _append_text(chat_log, agent_id, _json_output(output))
                run_recorder.record(
                    phase="llm_response",
                    source="provider",
                    event="structured_run_finished",
                    data={
                        "new_messages": result.new_messages(),
                        "output": output,
                        "usage": result.usage,
                        "duration": duration,
                        "final_chat_content": chat_log.content[-1]
                        if chat_log.content
                        else None,
                    },
                )
                return AgentRunOutcome(
                    output=output,
                    usage=result.usage,
                    duration=duration,
                    model_profile=profile.title,
                    model_profile_ref=profile.ref,
                    provider_subentry_id=profile.provider_subentry_id,
                    model_pricing=profile.model_pricing,
                    run_recorder=run_recorder,
                )
    except Exception:
        raise


async def run_agent_stream(
    agent: Agent[Any, Any],
    user_prompt: str | Sequence[Any] | None,
    message_history: list[ModelMessage],
    settings: ModelSettings,
    usage_limits: UsageLimits,
    chat_log: conversation.ChatLog,
    agent_id: str,
    structured_output_tool_names: set[str],
    run_recorder: RunDiagnosticsRecorder,
    entry: PydanticAIAgentConfigEntry,
    subentry: ConfigSubentry,
) -> AgentRunResult[Any]:
    """Run one Agent attempt and stream live deltas into the HA ChatLog."""
    from ._stream_trace import _StreamTraceRecorder

    state = _StreamRunState()
    trace_recorder = _StreamTraceRecorder(run_recorder=run_recorder)
    try:
        async with agent.run_stream_events(
            user_prompt,
            message_history=message_history,
            model_settings=settings,
            usage_limits=usage_limits,
        ) as events:
            async for _content in chat_log.async_add_delta_content_stream(
                agent_id,
                cast(
                    AsyncIterable[Any],
                    _agent_events_to_chat_deltas(
                        events,
                        structured_output_tool_names,
                        state,
                        trace_recorder,
                    ),
                ),
            ):
                pass
    except Exception as err:
        if state.emitted_deltas:
            failure = _classify_run_failure(
                err,
                usage_limits=usage_limits,
                partial_response=True,
                tool_problem=state.latest_tool_problem,
            )
            raise _AgentRunFailed(failure) from err
        raise
    if state.result is None:
        raise HomeAssistantError("Agent stream did not produce a final result")
    final_messages = state.result.new_messages()
    backfill = await _append_missing_final_text(
        chat_log,
        agent_id,
        final_messages,
    )
    run_recorder.record(
        phase="output",
        event="final_text_reconciled",
        data={
            "final_messages": final_messages,
            "backfill": backfill,
            "final_chat_content": chat_log.content[-1] if chat_log.content else None,
        },
    )
    trace_payload = trace_recorder.payload(
        final_messages=final_messages,
        backfill=backfill,
        final_chat_content=chat_log.content[-1] if chat_log.content else None,
    )
    chat_log.async_trace({"pydantic_ai_stream": trace_payload})
    entry.runtime_data.latest_stream_traces[subentry.subentry_id] = trace_payload
    return state.result
