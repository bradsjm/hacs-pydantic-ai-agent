"""Shared Pydantic AI entity runtime."""

import logging
from collections.abc import AsyncIterable, Mapping, Sequence
from functools import cached_property
from typing import Any, cast

import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import llm
from opentelemetry.trace import Span
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.capabilities import AbstractCapability, ToolSearch, WebFetch
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset, DeferredLoadingToolset
from pydantic_ai.usage import UsageLimits

from . import PydanticAIAgentConfigEntry
from .chat_deltas import (
    _agent_events_to_chat_deltas,
    _append_agent_messages,
    _append_missing_final_text,
    _append_text,
    _json_output,
    _StreamTraceRecorder,
)
from .chat_deltas import (
    _agent_messages_to_chat_deltas as _agent_messages_to_chat_deltas,
)
from .const import (
    CONF_MCP_SERVER_IDS,
    CONF_OUTPUT_MODE,
    CONF_SKILLS,
    CONF_WEB_FETCH_ENABLED,
    DOMAIN,
)
from .context_management import SlidingWindowContextCapability
from .ha_toolset import (
    tool_definitions_from_llm_api,
    tools_from_llm_api_with_diagnostics,
)
from .history import chat_log_content_to_model_messages, split_last_user_prompt
from .logfire_support import agent_run_span, instrument_agent
from .mcp import async_runtime_mcp_toolsets
from .metrics import (
    EVENT_AGENT_RUN_COMPLETED,
    EVENT_AGENT_RUN_FAILED,
    fire_integration_event,
    record_run_failure,
    record_run_success,
)
from .model_profiles import (
    ModelProfile,
    chat_model_for_profile,
    model_display_names,
    model_profile_chain,
    model_settings,
    primary_model_profile,
    thinking_capability,
)
from .model_profiles import (
    max_iterations as run_max_iterations,
)
from .model_request_settings import (
    _model_settings_with_chat_template_kwargs,
    _model_settings_with_provider_extra_body,
)
from .provider_validation import ProviderValidationError
from .repair_issues import (
    async_create_provider_auth_issue,
    async_delete_provider_auth_issue,
)
from .run_diagnostics import RunDiagnosticsRecorder
from .run_failures import (
    _AgentRunFailed,
    _AgentRunFailure,
    _classify_run_failure,
    _should_fallback,
)
from .run_failures import (
    _has_connection_failure as _has_connection_failure,
)
from .run_failures import (
    _home_assistant_error as _home_assistant_error,
)
from .run_state import AgentRunOutcome, _StreamRunState
from .skills import async_skills_capabilities
from .structured_output import (
    default_structure_serializer,
    output_tool_names,
    structured_agent_output_type,
    structured_output_json_schema,
    structured_output_mode,
    structured_output_name,
)
from .virtual_workspace import virtual_workspace_enabled, virtual_workspace_parts

_LOGGER = logging.getLogger(__name__)
_AUTH_VALIDATION_FAILURE_REASONS = {"invalid_auth", "permission_denied"}


def _join_instructions(*parts: str | None) -> str | None:
    """Join optional instruction blocks for one agent run."""
    instructions = [part.strip() for part in parts if part and part.strip()]
    return "\n\n".join(instructions) if instructions else None


def _has_provider_auth_validation_failure(
    failures: Mapping[str, str], provider_subentry_id: str
) -> bool:
    """Return if setup validation still has an auth issue for a provider."""
    return any(
        reason in _AUTH_VALIDATION_FAILURE_REASONS
        and len(parts := failure_key.split(":")) >= 3
        and parts[1] == provider_subentry_id
        for failure_key, reason in failures.items()
    )


def _has_provider_auth_failure(
    entry: PydanticAIAgentConfigEntry, provider_subentry_id: str
) -> bool:
    """Return if setup or runtime has a current auth issue for a provider."""
    return _has_provider_auth_validation_failure(
        entry.runtime_data.model_validation_failures,
        provider_subentry_id,
    ) or bool(
        entry.runtime_data.runtime_provider_auth_failures.get(provider_subentry_id)
    )


def _record_runtime_auth_failure(
    entry: PydanticAIAgentConfigEntry, profile: ModelProfile
) -> None:
    """Record a runtime auth issue for one provider/profile pair."""
    failures = entry.runtime_data.runtime_provider_auth_failures.setdefault(
        profile.provider_subentry_id, []
    )
    if profile.ref not in failures:
        failures.append(profile.ref)


def _clear_runtime_auth_failure(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry, profile: ModelProfile
) -> None:
    """Clear a runtime auth issue for one provider/profile pair when safe."""
    _clear_runtime_auth_failure_for_ref(
        hass, entry, profile.provider_subentry_id, profile.ref
    )


def _clear_runtime_auth_failure_for_ref(
    hass: HomeAssistant,
    entry: PydanticAIAgentConfigEntry,
    provider_subentry_id: str,
    profile_ref: str,
) -> None:
    """Clear a runtime auth issue by provider/profile reference when safe."""
    failures = entry.runtime_data.runtime_provider_auth_failures.get(
        provider_subentry_id
    )
    if failures is not None and profile_ref in failures:
        failures.remove(profile_ref)
        if not failures:
            entry.runtime_data.runtime_provider_auth_failures.pop(
                provider_subentry_id, None
            )
    if not _has_provider_auth_failure(entry, provider_subentry_id):
        async_delete_provider_auth_issue(hass, entry, provider_subentry_id)


class PydanticAIBaseLLMEntity:
    """Shared Pydantic AI streaming runtime for subentry-backed entities."""

    entry: PydanticAIAgentConfigEntry
    hass: HomeAssistant
    subentry: ConfigSubentry

    def __init__(
        self,
        entry: PydanticAIAgentConfigEntry,
        subentry: ConfigSubentry,
        *,
        name: str,
        device_name: str | None = None,
    ) -> None:
        """Initialize shared entity metadata."""
        self.entry = entry
        self.subentry = subentry
        profile = primary_model_profile(entry, subentry)
        self._attr_unique_id = unique_id_for_subentry(entry, subentry)
        self._attr_device_info = dr.DeviceInfo(
            identifiers={device_identifier_for_subentry(entry, subentry)},
            name=device_name or name,
            manufacturer="Pydantic AI",
            model=profile.model_name,
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @cached_property
    def available(self) -> bool:
        """Return if the primary model profile validated at setup."""
        failures = self.entry.runtime_data.model_validation_failures
        try:
            primary_profile = model_profile_chain(self.entry, self.subentry)[0]
        except HomeAssistantError, IndexError:
            return False
        return f"{self.subentry.subentry_id}:{primary_profile.ref}" not in failures

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure_name: str | None = None,
        structure: vol.Schema | None = None,
        max_iterations: int = 10,
        record_success: bool = True,
        stream: bool = False,
        extra_toolsets: Sequence[AbstractToolset[Any]] = (),
        extra_instructions: str | None = None,
    ) -> object | None:
        """Run a Pydantic AI Agent and stream its response into ChatLog."""
        profiles = model_profile_chain(self.entry, self.subentry)
        # ChatLog needs a stable agent id for deltas, but entity_id can be absent
        # before Home Assistant has fully registered the entity.
        agent_id = getattr(self, "entity_id", None) or getattr(self, "unique_id", None)
        if agent_id is None:
            raise HomeAssistantError("Entity is not ready")
        run_recorder = RunDiagnosticsRecorder(
            subentry_id=self.subentry.subentry_id,
            subentry_type=self.subentry.subentry_type,
            conversation_id=chat_log.conversation_id,
        )
        run_recorder.record(
            phase="run",
            event="run_started",
            data={
                "agent_id": agent_id,
                "subentry_title": self.subentry.title,
                "stream": stream,
                "record_success": record_success,
            },
        )

        output_mode = structured_output_mode(self.subentry.data.get(CONF_OUTPUT_MODE))
        output_name = self._structured_output_name(chat_log.llm_api, structure_name)
        agent_output_type: object = str
        structured_output_tool_names: set[str] = set()
        if structure is not None:
            structured_output_tool_names = output_tool_names(output_mode, output_name)
            agent_output_type = structured_agent_output_type(
                output_mode=output_mode,
                output_name=output_name,
                json_schema=structured_output_json_schema(
                    structure,
                    custom_serializer=default_structure_serializer(chat_log.llm_api),
                ),
            )

        messages = await chat_log_content_to_model_messages(self.hass, chat_log.content)
        user_prompt, message_history = split_last_user_prompt(messages)
        run_recorder.record(
            phase="input",
            event="messages_prepared",
            data={
                "user_prompt": user_prompt,
                "message_history": message_history,
                "chat_log_content": chat_log.content,
                "llm_tool_definitions": tool_definitions_from_llm_api(chat_log.llm_api),
            },
        )
        capabilities: list[AbstractCapability] = [
            *await async_skills_capabilities(
                self.hass,
                self.entry,
                self.subentry.data.get(CONF_SKILLS),
            )
        ]
        if self.subentry.data.get(CONF_WEB_FETCH_ENABLED):
            capabilities.append(WebFetch(local=True))
        capabilities.append(SlidingWindowContextCapability())
        errors: list[BaseException] = []
        for index, profile in enumerate(profiles):
            usage_limits = UsageLimits(
                request_limit=run_max_iterations(self.subentry.data, max_iterations),
            )
            try:
                outcome = await self._async_try_model_profile(
                    index=index,
                    profile=profile,
                    usage_limits=usage_limits,
                    chat_log=chat_log,
                    agent_id=agent_id,
                    user_prompt=user_prompt,
                    message_history=message_history,
                    structured_output_tool_names=structured_output_tool_names,
                    has_structure=structure is not None,
                    stream=stream,
                    run_recorder=run_recorder,
                    agent_output_type=agent_output_type,
                    capabilities=capabilities,
                    extra_toolsets=extra_toolsets,
                    extra_instructions=extra_instructions,
                )
                if record_success:
                    _clear_runtime_auth_failure(self.hass, self.entry, profile)
                    self._record_agent_run_success(outcome, agent_id)
                    self._store_run_diagnostics(
                        run_recorder,
                        status="success",
                        summary={
                            "output": outcome.output,
                            "usage": outcome.usage,
                            "model_profile": outcome.model_profile,
                            "duration": outcome.duration,
                        },
                    )
                    return outcome.output
                return outcome
            except Exception as err:
                self._async_handle_profile_error(
                    err=err,
                    index=index,
                    is_last_attempt=index == len(profiles) - 1,
                    profile=profile,
                    usage_limits=usage_limits,
                    agent_id=agent_id,
                    run_recorder=run_recorder,
                    errors=errors,
                )
        raise HomeAssistantError(
            "All configured model profiles failed: "
            + ", ".join(model_display_names(profiles))
        ) from (errors[-1] if errors else None)

    async def _async_try_model_profile(
        self,
        *,
        index: int,
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
    ) -> AgentRunOutcome:
        """Run one model profile attempt and return the outcome."""
        virtual_toolsets: Sequence[AbstractToolset[Any]] = ()
        virtual_instructions: str | None = None
        if virtual_workspace_enabled(self.subentry.data):
            parts = virtual_workspace_parts()
            virtual_toolsets = parts.toolsets
            virtual_instructions = parts.instructions
        instructions = _join_instructions(virtual_instructions, extra_instructions)
        settings = model_settings(profile, self.subentry.data)
        settings = _model_settings_with_provider_extra_body(
            self.entry, profile, settings
        )
        settings = _model_settings_with_chat_template_kwargs(
            self.hass, profile, settings
        )
        mcp_toolsets = await async_runtime_mcp_toolsets(
            self.hass,
            self.entry,
            self.subentry.data.get(CONF_MCP_SERVER_IDS),
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
        if thinking := thinking_capability(self.subentry.data):
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
                "virtual_workspace_enabled": virtual_workspace_enabled(
                    self.subentry.data
                ),
                "capability_types": [
                    type(capability).__name__ for capability in run_capabilities
                ],
            },
        )
        agent = Agent(
            chat_model_for_profile(self.hass, self.entry, profile),
            output_type=cast(Any, agent_output_type),
            instructions=instructions,
            model_settings=settings,
            tool_retries=0,
            output_retries=2,
            tools=tools_from_llm_api_with_diagnostics(chat_log.llm_api, run_recorder),
            toolsets=toolsets,
            max_concurrency=1,
            capabilities=run_capabilities,
        )
        instrument_agent(self.hass, self.entry, agent)
        return await self._async_run_agent(
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
        )

    def _async_handle_profile_error(
        self,
        *,
        err: Exception,
        index: int,
        is_last_attempt: bool,
        profile: ModelProfile,
        usage_limits: UsageLimits,
        agent_id: str,
        run_recorder: RunDiagnosticsRecorder,
        errors: list[BaseException],
    ) -> None:
        """Handle a model profile attempt failure with fallback logic."""
        if _auth_status_code(err) is None:
            _clear_runtime_auth_failure(self.hass, self.entry, profile)
        if is_last_attempt or not _should_fallback(err):
            failure = _classify_run_failure(err, usage_limits=usage_limits)
            if not _async_create_runtime_auth_issue(
                self.hass, self.entry, profile, err, failure.user_message
            ):
                _clear_runtime_auth_failure(self.hass, self.entry, profile)
            self._record_agent_run_failure(
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
            self._store_run_diagnostics(
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

    async def _async_run_agent(
        self,
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
    ) -> AgentRunOutcome:
        """Run one model profile attempt and append only successful messages."""
        try:
            async with agent:
                with agent_run_span(
                    self.hass,
                    self.entry,
                    self.subentry,
                    entity_id=agent_id,
                    conversation_id=chat_log.conversation_id,
                    model_name=profile.model_name,
                ) as span:
                    start = self.hass.loop.time()
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
                        result = await self._async_run_agent_streaming(
                            agent,
                            user_prompt,
                            message_history,
                            settings,
                            usage_limits,
                            chat_log,
                            agent_id,
                            structured_output_tool_names,
                            run_recorder,
                        )
                        if span is not None:
                            _set_span_usage_attributes(span, result)
                        duration = self.hass.loop.time() - start
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
                            _set_span_usage_attributes(span, result)
                        duration = self.hass.loop.time() - start
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
                        _set_span_usage_attributes(span, result)
                    duration = self.hass.loop.time() - start
                    output = result.output
                    await _append_agent_messages(
                        chat_log,
                        agent_id,
                        result.new_messages(),
                        output_tool_names=structured_output_tool_names,
                    )
                    if not isinstance(
                        chat_log.content[-1], conversation.AssistantContent
                    ):
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
            _LOGGER.debug("Model profile attempt failed: %s", profile.title)
            raise

    async def _async_run_agent_streaming(
        self,
        agent: Agent[Any, Any],
        user_prompt: str | Sequence[Any] | None,
        message_history: list[ModelMessage],
        settings: ModelSettings,
        usage_limits: UsageLimits,
        chat_log: conversation.ChatLog,
        agent_id: str,
        structured_output_tool_names: set[str],
        run_recorder: RunDiagnosticsRecorder,
    ) -> AgentRunResult[Any]:
        """Run one Agent attempt and stream live deltas into the HA ChatLog."""
        state = _StreamRunState()
        trace_recorder = _StreamTraceRecorder(
            run_recorder=run_recorder,
        )
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
                "final_chat_content": chat_log.content[-1]
                if chat_log.content
                else None,
            },
        )
        trace_payload = trace_recorder.payload(
            final_messages=final_messages,
            backfill=backfill,
            final_chat_content=chat_log.content[-1] if chat_log.content else None,
        )
        chat_log.async_trace({"pydantic_ai_stream": trace_payload})
        self.entry.runtime_data.latest_stream_traces[self.subentry.subentry_id] = (
            trace_payload
        )
        return state.result

    def _record_agent_run_success(
        self, outcome: AgentRunOutcome, agent_id: str | None = None
    ) -> None:
        """Record successful run metrics and fire the completion event."""
        entity_id = (
            agent_id or getattr(self, "entity_id", None) or self.subentry.subentry_id
        )
        record_run_success(
            self.hass,
            self.entry.entry_id,
            self.entry.runtime_data.metrics,
            self.subentry.subentry_id,
            model_profile=outcome.model_profile,
            duration=outcome.duration,
            usage=outcome.usage,
            model_pricing=outcome.model_pricing,
        )
        fire_integration_event(
            self.hass,
            EVENT_AGENT_RUN_COMPLETED,
            {
                "config_entry_id": self.entry.entry_id,
                "subentry_id": self.subentry.subentry_id,
                "entity_id": entity_id,
                "model_profile": outcome.model_profile,
            },
        )

    def _store_run_diagnostics(
        self,
        recorder: RunDiagnosticsRecorder,
        *,
        status: str,
        summary: Mapping[str, Any],
    ) -> None:
        """Store latest bounded last-run diagnostics for this subentry."""
        self.entry.runtime_data.latest_run_diagnostics[self.subentry.subentry_id] = (
            recorder.payload(status=status, summary=summary)
        )

    def _record_agent_run_failure(
        self,
        err: BaseException,
        agent_id: str | None = None,
        *,
        model_profile: str | None = None,
        failure: _AgentRunFailure | None = None,
    ) -> None:
        """Record failed run metrics and fire the failure event."""
        failure = failure or _classify_run_failure(err)
        entity_id = (
            agent_id or getattr(self, "entity_id", None) or self.subentry.subentry_id
        )
        _LOGGER.error(
            'Pydantic AI agent run failed for model profile "%s" (%s): %s',
            model_profile or "unknown",
            failure.error_type,
            failure.log_message,
        )
        record_run_failure(
            self.hass,
            self.entry.entry_id,
            self.entry.runtime_data.metrics,
            self.subentry.subentry_id,
            error=err,
            error_type=failure.error_type,
        )
        event_data: dict[str, object] = {
            "config_entry_id": self.entry.entry_id,
            "subentry_id": self.subentry.subentry_id,
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
        fire_integration_event(self.hass, EVENT_AGENT_RUN_FAILED, event_data)

    def _structured_output_name(
        self, api_instance: llm.APIInstance | None, structure_name: str | None
    ) -> str:
        """Return an output name that cannot shadow configured HA tools."""
        return structured_output_name(
            structure_name,
            "generated_data",
            reserved_names=(
                tool.name for tool in tool_definitions_from_llm_api(api_instance)
            ),
        )


def unique_id_for_subentry(
    entry: PydanticAIAgentConfigEntry, subentry: ConfigSubentry
) -> str:
    """Return the unique ID for one subentry-backed entity."""
    return f"{DOMAIN}_{entry.entry_id}_{subentry.subentry_type}_{subentry.subentry_id}"


def unique_id_for_subentry_entity(
    entry: PydanticAIAgentConfigEntry, subentry: ConfigSubentry, key: str
) -> str:
    """Return the unique ID for one subentry-backed diagnostic entity."""
    return f"{unique_id_for_subentry(entry, subentry)}_{key}"


def _async_create_runtime_auth_issue(
    hass: HomeAssistant,
    entry: PydanticAIAgentConfigEntry,
    profile: ModelProfile,
    err: BaseException,
    message: str,
) -> bool:
    """Create a provider auth repair issue for runtime credential failures."""
    status_code = _auth_status_code(err)
    if status_code is None:
        return False
    reason = "invalid_auth" if status_code == 401 else "permission_denied"
    _record_runtime_auth_failure(entry, profile)
    async_create_provider_auth_issue(
        hass,
        entry,
        profile.provider_subentry_id,
        profile.provider_title,
        ProviderValidationError(reason, message, status_code),
    )
    return True


def _auth_status_code(err: BaseException) -> int | None:
    """Return 401/403 from a runtime error cause chain when present."""
    seen: set[int] = set()
    current: BaseException | None = err
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ModelHTTPError) and current.status_code in {401, 403}:
            return current.status_code
        current = current.__cause__ or current.__context__
    return None


def device_identifier_for_subentry(
    entry: PydanticAIAgentConfigEntry, subentry: ConfigSubentry
) -> tuple[str, str]:
    """Return the device identifier for one subentry-backed entity."""
    return (
        DOMAIN,
        f"{entry.entry_id}:{subentry.subentry_type}:{subentry.subentry_id}",
    )


def _set_span_usage_attributes(span: Span, result: AgentRunResult[Any]) -> None:
    """Copy aggregate Pydantic AI usage to the wrapper span without blocking runs."""
    try:
        usage_attributes = result.usage.opentelemetry_attributes()
        if usage_attributes:
            span.set_attributes(usage_attributes)
    except Exception:
        _LOGGER.exception("Failed to add usage attributes to Logfire span")
