"""Shared Pydantic AI entity runtime."""

from collections.abc import AsyncIterable, AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
import json
import logging
from typing import Any, cast

import httpx
from pydantic_ai import Agent, AgentRunResultEvent
from pydantic_ai.capabilities import AbstractCapability, ToolSearch, WebFetch
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UserError,
    UsageLimitExceeded,
)
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    OutputToolCallEvent,
    OutputToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset, DeferredLoadingToolset
from pydantic_ai.usage import UsageLimits
import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm

from . import PydanticAIAgentConfigEntry
from .const import (
    CONF_CHAT_TEMPLATE_KWARGS,
    CONF_MCP_SERVER_IDS,
    CONF_OUTPUT_MODE,
    CONF_SKILLS,
    CONF_WEB_FETCH_ENABLED,
    DOMAIN,
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
)
from .chat_template_kwargs import (
    reject_chat_template_kwargs_in_extra_body,
    render_chat_template_kwargs,
)
from .context_management import SlidingWindowContextCapability
from .error_classification import has_connection_failure
from .ha_toolset import tool_definitions_from_llm_api, tools_from_llm_api
from .history import chat_log_content_to_model_messages, split_last_user_prompt
from .logfire_support import agent_run_span, instrument_agent
from .metrics import (
    EVENT_AGENT_RUN_COMPLETED,
    EVENT_AGENT_RUN_FAILED,
    fire_integration_event,
    record_run_failure,
    record_run_success,
)
from .mcp import MCPValidationError, async_runtime_mcp_toolsets
from .model_profiles import (
    ModelProfile,
    chat_model_for_profile,
    max_iterations as profile_max_iterations,
    model_display_names,
    model_profile_chain,
    model_settings,
    primary_model_profile,
    provider_extra_body,
    thinking_capability,
)
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
from .virtual_workspace.const import TOOL_RETURN_METADATA_SOURCE

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class AgentRunOutcome:
    """Successful agent run data needed for metrics after validation."""

    output: object | None
    usage: Any
    duration: float
    model_profile: str
    model_pricing: dict[str, float]


@dataclass
class _StreamRunState:
    """Mutable state shared with the ChatLog streaming delta generator."""

    result: Any | None = None
    emitted_deltas: bool = False
    latest_tool_problem: "_ToolProblem | None" = None


@dataclass(frozen=True, kw_only=True)
class _ToolProblem:
    """Safe summary of a tool result problem."""

    tool_name: str | None
    tool_call_id: str | None
    outcome: str
    reason: str | None = None


@dataclass(frozen=True, kw_only=True)
class _AgentRunFailure:
    """Single classified source for terminal agent run failures."""

    error_type: str
    user_message: str
    log_message: str
    partial_response: bool = False
    tool_problem: _ToolProblem | None = None


class _AgentRunFailed(HomeAssistantError):
    """Home Assistant error carrying classified run-failure details."""

    def __init__(self, failure: _AgentRunFailure) -> None:
        """Initialize the error with the safe user-facing message."""
        super().__init__(failure.user_message)
        self.failure = failure


def _join_instructions(*parts: str | None) -> str | None:
    """Join optional instruction blocks for one agent run."""
    instructions = [part.strip() for part in parts if part and part.strip()]
    return "\n\n".join(instructions) if instructions else None


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
        if device_name is not None and device_name != name:
            self._attr_has_entity_name = False
            self._attr_name = name
        profile = primary_model_profile(entry, subentry)
        self._attr_unique_id = unique_id_for_subentry(entry, subentry)
        self._attr_device_info = dr.DeviceInfo(
            identifiers={device_identifier_for_subentry(entry, subentry)},
            name=device_name or name,
            manufacturer="Pydantic AI",
            model=profile.model_name,
            entry_type=dr.DeviceEntryType.SERVICE,
        )

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
        use_virtual_workspace = virtual_workspace_enabled(self.subentry.data)
        errors: list[BaseException] = []
        for index, profile in enumerate(profiles):
            usage_limits = UsageLimits(
                request_limit=profile_max_iterations(profile, max_iterations),
            )
            try:
                virtual_toolsets: Sequence[AbstractToolset[Any]] = ()
                virtual_instructions: str | None = None
                if use_virtual_workspace:
                    parts = virtual_workspace_parts()
                    virtual_toolsets = parts.toolsets
                    virtual_instructions = parts.instructions
                instructions = _join_instructions(virtual_instructions, extra_instructions)
                settings = model_settings(profile)
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
                if any(
                    isinstance(toolset, DeferredLoadingToolset) for toolset in toolsets
                ):
                    run_capabilities = [
                        capability
                        for capability in run_capabilities
                        if not isinstance(capability, ToolSearch)
                    ]
                    run_capabilities.append(ToolSearch(strategy="keywords"))
                if thinking := thinking_capability(profile):
                    run_capabilities.append(thinking)
                agent = Agent(
                    chat_model_for_profile(self.hass, self.entry, profile),
                    output_type=cast(Any, agent_output_type),
                    instructions=instructions,
                    model_settings=settings,
                    tool_retries=0,
                    output_retries=2,
                    tools=tools_from_llm_api(chat_log.llm_api),
                    toolsets=toolsets,
                    max_concurrency=1,
                    capabilities=run_capabilities,
                )
                instrument_agent(self.hass, self.entry, agent)
                outcome = await self._async_run_agent(
                    agent,
                    profile,
                    settings,
                    chat_log,
                    agent_id,
                    user_prompt,
                    message_history,
                    usage_limits,
                    structured_output_tool_names,
                    structure is not None,
                    stream,
                )
                if record_success:
                    self._record_agent_run_success(outcome, agent_id)
                    return outcome.output
                return outcome
            except Exception as err:
                if index == len(profiles) - 1 or not _should_fallback(err):
                    failure = _classify_run_failure(
                        err,
                        usage_limits=usage_limits,
                    )
                    self._record_agent_run_failure(
                        err,
                        agent_id,
                        model_profile=profile.title,
                        failure=failure,
                    )
                    raise _AgentRunFailed(failure) from err
                errors.append(err)
                failure = _classify_run_failure(err, usage_limits=usage_limits)
                _LOGGER.warning(
                    'Model profile "%s" failed with retryable %s; trying fallback: %s',
                    profile.title,
                    failure.error_type,
                    failure.log_message,
                )
        raise HomeAssistantError(
            "All configured model profiles failed: "
            + ", ".join(model_display_names(profiles))
        ) from (errors[-1] if errors else None)

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
                        )
                        if span is not None:
                            _set_span_usage_attributes(span, result)
                        duration = self.hass.loop.time() - start
                        return AgentRunOutcome(
                            output=result.output,
                            usage=result.usage,
                            duration=duration,
                            model_profile=profile.title,
                            model_pricing=profile.model_pricing,
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
                        return AgentRunOutcome(
                            output=result.output,
                            usage=result.usage,
                            duration=duration,
                            model_profile=profile.title,
                            model_pricing=profile.model_pricing,
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
                    return AgentRunOutcome(
                        output=output,
                        usage=result.usage,
                        duration=duration,
                        model_profile=profile.title,
                        model_pricing=profile.model_pricing,
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
    ) -> Any:
        """Run one Agent attempt and stream live deltas into the HA ChatLog."""
        state = _StreamRunState()
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


def device_identifier_for_subentry(
    entry: PydanticAIAgentConfigEntry, subentry: ConfigSubentry
) -> tuple[str, str]:
    """Return the device identifier for one subentry-backed entity."""
    return (
        DOMAIN,
        f"{entry.entry_id}:{subentry.subentry_type}:{subentry.subentry_id}",
    )


def _model_settings_with_chat_template_kwargs(
    hass: HomeAssistant, profile: ModelProfile, settings: ModelSettings
) -> ModelSettings:
    """Return request settings with rendered chat-template kwargs injected."""
    rendered_kwargs = render_chat_template_kwargs(
        hass, profile.model_settings.get(CONF_CHAT_TEMPLATE_KWARGS)
    )
    if not rendered_kwargs:
        reject_chat_template_kwargs_in_extra_body(settings.get("extra_body"))
        return settings

    request_settings = dict(settings)
    extra_body = request_settings.get("extra_body")
    reject_chat_template_kwargs_in_extra_body(extra_body)
    request_extra_body = dict(extra_body) if isinstance(extra_body, Mapping) else {}
    request_extra_body[CONF_CHAT_TEMPLATE_KWARGS] = rendered_kwargs
    request_settings["extra_body"] = request_extra_body
    return ModelSettings(**cast(Any, request_settings))


def _model_settings_with_provider_extra_body(
    entry: PydanticAIAgentConfigEntry, profile: ModelProfile, settings: ModelSettings
) -> ModelSettings:
    """Return request settings with provider-level extra body merged."""
    extra_body = provider_extra_body(entry, profile)
    if not extra_body:
        return settings
    if profile.provider_mode not in {
        PROVIDER_ANTHROPIC,
        PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
    }:
        raise HomeAssistantError(
            "Provider extra body is only supported by OpenAI-compatible and Anthropic provider modes"
        )
    reject_chat_template_kwargs_in_extra_body(extra_body)
    request_settings = dict(settings)
    request_settings["extra_body"] = extra_body
    return ModelSettings(**cast(Any, request_settings))


def _format_http_error(err: ModelHTTPError) -> str:
    """Return a user-facing provider HTTP error message."""
    return f'The provider returned HTTP {err.status_code} for model "{err.model_name}".'


def _format_api_error(err: ModelAPIError) -> str:
    """Return a user-facing provider API error message."""
    return f'The provider returned an API error for model "{err.model_name}".'


def _run_failure_cause(err: BaseException) -> BaseException:
    """Return the underlying failure cause for classified run errors."""
    if isinstance(err, _AgentRunFailed) and err.__cause__ is not None:
        return err.__cause__
    return err


def _classify_run_failure(
    err: BaseException,
    *,
    usage_limits: UsageLimits | None = None,
    partial_response: bool = False,
    tool_problem: _ToolProblem | None = None,
) -> _AgentRunFailure:
    """Classify a run failure into safe user, log, event, and metric details."""
    if isinstance(err, _AgentRunFailed):
        return err.failure

    cause = _run_failure_cause(err)
    error_type = type(cause).__name__
    context = _tool_problem_context(tool_problem)
    prefix = (
        "Terminated after a partial response because "
        if partial_response
        else "Terminated because "
    )

    if isinstance(cause, UsageLimitExceeded):
        request_limit = usage_limits.request_limit if usage_limits is not None else None
        if request_limit is not None:
            message = (
                f"{prefix}the model exceeded the configured maximum of "
                f"{request_limit} iterations. Increase the model profile max "
                "iterations or fix repeated tool failures."
            )
        else:
            message = (
                f"{prefix}the model exceeded a configured usage limit. "
                "Increase the relevant model profile limit or reduce the request."
            )
        return _AgentRunFailure(
            error_type=error_type,
            user_message=message + context,
            log_message=message + context,
            partial_response=partial_response,
            tool_problem=tool_problem,
        )

    if isinstance(cause, ModelHTTPError):
        message = _http_failure_message(cause, prefix)
    elif isinstance(cause, ModelAPIError):
        if _has_connection_failure(cause):
            message = (
                f'{prefix}the provider connection failed for model '
                f'"{cause.model_name}". Check network connectivity and provider '
                "availability."
            )
        else:
            message = _format_api_error(cause)
    elif isinstance(cause, UnexpectedModelBehavior):
        message = (
            f"{prefix}the provider returned an unexpected response. Check "
            "model/provider compatibility or try a different model profile."
        )
    elif isinstance(cause, TimeoutError | httpx.TimeoutException):
        message = (
            f"{prefix}the provider request timed out. Check network "
            "connectivity or try again later."
        )
    elif isinstance(cause, MCPValidationError):
        message = cause.message
    elif isinstance(cause, NotImplementedError | UserError):
        message = f"Invalid provider configuration: {cause}"
    elif isinstance(cause, HomeAssistantError):
        message = str(cause)
    else:
        message = str(cause) or error_type

    return _AgentRunFailure(
        error_type=error_type,
        user_message=message + context,
        log_message=message + context,
        partial_response=partial_response,
        tool_problem=tool_problem,
    )


def _http_failure_message(err: ModelHTTPError, prefix: str) -> str:
    """Return an actionable HTTP provider failure message."""
    if err.status_code == 429:
        return (
            f'{prefix}the provider quota or rate limit was reached for model '
            f'"{err.model_name}". Check provider quota/rate limits or try '
            "again later."
        )
    if err.status_code in {401, 403}:
        return (
            f'{prefix}the provider rejected credentials or permissions for '
            f'model "{err.model_name}". Check the provider API key and '
            "account access."
        )
    if 500 <= err.status_code <= 599:
        return (
            f'{prefix}the provider service returned HTTP {err.status_code} '
            f'for model "{err.model_name}". Try again later or use a fallback '
            "model profile."
        )
    return _format_http_error(err)


def _tool_problem_context(tool_problem: _ToolProblem | None) -> str:
    """Return safe user-facing context for the latest tool problem."""
    if tool_problem is None:
        return ""
    name = tool_problem.tool_name or "unknown tool"
    if tool_problem.reason:
        return f" Last tool failure: {name} reported {tool_problem.reason}."
    return f" Last tool failure: {name} returned {tool_problem.outcome}."


def _home_assistant_error(err: Exception) -> HomeAssistantError:
    """Convert provider/runtime failures into HA-facing errors."""
    if isinstance(err, _AgentRunFailed):
        return err
    if isinstance(err, HomeAssistantError):
        return err
    return HomeAssistantError(_classify_run_failure(err).user_message)


def _should_fallback(err: Exception) -> bool:
    """Return if a failed model attempt should try the next profile."""
    if isinstance(err, _AgentRunFailed):
        return not err.failure.partial_response and _should_fallback(
            cast(Exception, _run_failure_cause(err))
        )
    if isinstance(err, ModelHTTPError):
        return err.status_code in {408, 409, 429} or 500 <= err.status_code <= 599
    if isinstance(err, TimeoutError | httpx.TimeoutException | UsageLimitExceeded):
        return True
    if isinstance(err, ModelAPIError):
        return _has_connection_failure(err)
    return False


def _has_connection_failure(err: BaseException) -> bool:
    """Return if an exception cause chain indicates transport failure."""
    return has_connection_failure(err)


def _set_span_usage_attributes(span: Any, result: Any) -> None:
    """Copy aggregate Pydantic AI usage to the wrapper span without blocking runs."""
    try:
        usage_attributes = result.usage.opentelemetry_attributes()
        if usage_attributes:
            span.set_attributes(usage_attributes)
    except Exception:
        _LOGGER.exception("Failed to add usage attributes to Logfire span")


async def _append_agent_messages(
    chat_log: conversation.ChatLog,
    agent_id: str,
    messages: list[ModelMessage],
    output_tool_names: set[str] | None = None,
) -> None:
    """Append Agent-produced assistant/tool messages to the Home Assistant log."""
    async for _content in chat_log.async_add_delta_content_stream(
        agent_id,
        cast(
            AsyncIterable[Any],
            _agent_messages_to_chat_deltas(messages, output_tool_names or set()),
        ),
    ):
        pass


async def _append_text(
    chat_log: conversation.ChatLog,
    agent_id: str,
    text: str,
) -> None:
    """Append one assistant text response to the Home Assistant log."""
    async for _content in chat_log.async_add_delta_content_stream(
        agent_id,
        cast(AsyncIterable[Any], _text_stream_to_chat_deltas(_single_text(text))),
    ):
        pass


async def _text_stream_to_chat_deltas(
    text_stream: AsyncIterable[str],
) -> AsyncIterator[dict[str, str]]:
    """Yield ChatLog text deltas from a Pydantic AI text stream."""
    async for chunk in text_stream:
        if chunk:
            yield {"content": chunk}


async def _single_text(text: str) -> AsyncIterator[str]:
    """Yield one text chunk as an async iterator."""
    yield text


async def _agent_events_to_chat_deltas(
    events: AsyncIterable[Any],
    output_tool_names: set[str],
    state: _StreamRunState,
) -> AsyncIterator[dict[str, Any]]:
    """Yield HA ChatLog deltas from live Pydantic AI Agent events."""
    assistant_open = False
    emitted_tool_call_ids: set[str] = set()
    async for event in events:
        if isinstance(event, AgentRunResultEvent):
            state.result = event.result
            continue
        if isinstance(event, PartStartEvent):
            if event.index == 0:
                state.emitted_deltas = True
                yield {"role": "assistant"}
                assistant_open = True
            async for delta in _part_start_to_chat_deltas(event, output_tool_names):
                state.emitted_deltas = True
                yield delta
            continue
        if isinstance(event, PartDeltaEvent):
            if not assistant_open:
                state.emitted_deltas = True
                yield {"role": "assistant"}
                assistant_open = True
            async for delta in _part_delta_to_chat_deltas(event):
                state.emitted_deltas = True
                yield delta
            continue
        if isinstance(event, FunctionToolCallEvent | OutputToolCallEvent):
            if not assistant_open:
                state.emitted_deltas = True
                yield {"role": "assistant"}
                assistant_open = True
            async for delta in _tool_call_event_to_chat_deltas(
                event.part,
                output_tool_names,
                emitted_tool_call_ids,
            ):
                state.emitted_deltas = True
                yield delta
            continue
        if isinstance(event, FunctionToolResultEvent | OutputToolResultEvent):
            if event.part is None:
                continue
            tool_problem = _tool_problem_from_part(event.part)
            if tool_problem is not None:
                state.latest_tool_problem = tool_problem
                _log_tool_problem(tool_problem)
            state.emitted_deltas = True
            yield {
                "role": "tool_result",
                "tool_call_id": event.part.tool_call_id,
                "tool_name": event.part.tool_name,
                "tool_result": event.part.content,
            }
            assistant_open = False


def _tool_problem_from_part(
    part: ToolReturnPart | RetryPromptPart,
) -> _ToolProblem | None:
    """Return a safe tool problem summary from a Pydantic AI tool result part."""
    if isinstance(part, RetryPromptPart):
        return _ToolProblem(
            tool_name=part.tool_name,
            tool_call_id=part.tool_call_id,
            outcome="retry",
            reason=_safe_tool_result_reason(part.content, getattr(part, "metadata", None)),
        )
    outcome = getattr(part, "outcome", "success")
    reason = _safe_tool_result_reason(part.content, part.metadata)
    if outcome != "success":
        return _ToolProblem(
            tool_name=part.tool_name,
            tool_call_id=part.tool_call_id,
            outcome=outcome,
            reason=reason,
        )
    if isinstance(part.content, Mapping) and part.content.get("success") is False:
        return _ToolProblem(
            tool_name=part.tool_name,
            tool_call_id=part.tool_call_id,
            outcome="failed",
            reason=reason,
        )
    return None


def _safe_tool_result_reason(content: object, metadata: object) -> str | None:
    """Extract a short safe reason from structured tool failure content."""
    if not (
        isinstance(metadata, Mapping)
        and metadata.get("source") == TOOL_RETURN_METADATA_SOURCE
    ):
        return None
    reason: object | None = None
    if isinstance(content, Mapping):
        errors = content.get("errors")
        if isinstance(errors, Sequence) and not isinstance(errors, str | bytes):
            reason = next((item for item in errors if isinstance(item, str)), None)
        if reason is None:
            for key in ("error", "message"):
                value = content.get(key)
                if isinstance(value, str):
                    reason = value
                    break
    elif isinstance(content, str):
        reason = content
    if not isinstance(reason, str) or not reason:
        return None
    return reason[:200]


def _log_tool_problem(problem: _ToolProblem) -> None:
    """Log a non-terminal tool problem without exposing tool arguments."""
    _LOGGER.warning(
        'Pydantic AI tool "%s" returned %s for call "%s": %s',
        problem.tool_name or "unknown",
        problem.outcome,
        problem.tool_call_id or "unknown",
        problem.reason or "no safe detail provided",
    )


async def _part_start_to_chat_deltas(
    event: PartStartEvent,
    output_tool_names: set[str],
) -> AsyncIterator[dict[str, Any]]:
    """Yield initial HA deltas for a Pydantic AI part-start event."""
    part = event.part
    if isinstance(part, TextPart) and part.content:
        yield {"content": part.content}
    elif isinstance(part, ThinkingPart) and part.content:
        yield {"thinking_content": part.content}
    elif isinstance(part, ToolCallPart) and part.tool_name in output_tool_names:
        yield {"content": json.dumps(part.args_as_dict())}


async def _part_delta_to_chat_deltas(
    event: PartDeltaEvent,
) -> AsyncIterator[dict[str, Any]]:
    """Yield incremental HA deltas for a Pydantic AI part-delta event."""
    delta = event.delta
    if isinstance(delta, TextPartDelta) and delta.content_delta:
        yield {"content": delta.content_delta}
    elif isinstance(delta, ThinkingPartDelta) and delta.content_delta:
        yield {"thinking_content": delta.content_delta}


async def _tool_call_event_to_chat_deltas(
    part: ToolCallPart,
    output_tool_names: set[str],
    emitted_tool_call_ids: set[str],
) -> AsyncIterator[dict[str, Any]]:
    """Yield a HA tool-call delta when Pydantic AI starts executing a tool."""
    if part.tool_name in output_tool_names:
        yield {"content": json.dumps(part.args_as_dict())}
        return
    if part.tool_call_id in emitted_tool_call_ids:
        return
    emitted_tool_call_ids.add(part.tool_call_id)
    yield {
        "tool_calls": [
            llm.ToolInput(
                id=part.tool_call_id,
                tool_name=part.tool_name,
                tool_args=part.args_as_dict(),
                external=True,
            )
        ]
    }


def _json_output(output: object) -> str:
    """Return a JSON string for structured Agent output."""
    if isinstance(output, str):
        return output
    return json.dumps(output)


async def _agent_messages_to_chat_deltas(
    messages: list[ModelMessage],
    output_tool_names: set[str],
) -> AsyncIterator[dict[str, Any]]:
    """Yield ChatLog deltas from Agent messages without re-executing tools."""
    for message in messages:
        if isinstance(message, ModelResponse):
            content = ""
            thinking_content = ""
            tool_calls: list[llm.ToolInput] = []
            for part in message.parts:
                if isinstance(part, TextPart):
                    content += part.content
                elif isinstance(part, ThinkingPart):
                    thinking_content += part.content
                elif isinstance(part, ToolCallPart):
                    if part.tool_name in output_tool_names:
                        content += json.dumps(part.args_as_dict())
                        continue
                    tool_calls.append(
                        llm.ToolInput(
                            id=part.tool_call_id,
                            tool_name=part.tool_name,
                            tool_args=part.args_as_dict(),
                            external=True,
                        )
                    )
            if content or thinking_content or tool_calls:
                yield {
                    "role": "assistant",
                    "content": content,
                    "thinking_content": thinking_content,
                    "tool_calls": tool_calls,
                }
        elif isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, ToolReturnPart | RetryPromptPart):
                    tool_problem = _tool_problem_from_part(part)
                    if tool_problem is not None:
                        _log_tool_problem(tool_problem)
                    yield {
                        "role": "tool_result",
                        "tool_call_id": part.tool_call_id,
                        "tool_name": part.tool_name,
                        "tool_result": part.content,
                    }
