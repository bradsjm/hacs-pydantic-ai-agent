"""Shared Pydantic AI entity runtime."""

from collections.abc import Sequence
from functools import cached_property
from typing import Any

import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import llm
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, WebFetch
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import UsageLimits

from ._entity_auth import _clear_runtime_auth_failure
from ._entity_run_results import (
    handle_profile_error,
    record_agent_run_failure,
    record_agent_run_success,
    store_run_diagnostics,
)
from ._entity_runner import run_model_profile
from ._types import WorkspaceRuntimeData
from .chat_deltas import _agent_events_to_chat_deltas as _agent_events_to_chat_deltas
from .const import (
    CONF_SKILLS,
    CONF_TOOL_RETRIES,
    CONF_WEB_FETCH_ENABLED,
    DEFAULT_TOOL_RETRIES,
    DOMAIN,
)
from .context_management import SlidingWindowContextCapability
from .ha_toolset import tool_definitions_from_llm_api
from .history import chat_log_content_to_model_messages, split_last_user_prompt
from .model_profiles import (
    chat_model_for_profile,
    model_display_names,
    model_profile_chain,
    primary_model_profile,
)
from .model_profiles import max_iterations as run_max_iterations
from .run_diagnostics import RunDiagnosticsRecorder
from .run_failures import _AgentRunFailure
from .run_state import AgentRunOutcome
from .skills import async_skills_capabilities
from .structured_output import (
    default_structure_serializer,
    output_tool_names,
    resolved_structured_output_mode,
    structured_agent_output_type,
    structured_output_json_schema,
    structured_output_name,
)
from .virtual_workspace import virtual_workspace_parts

type PydanticAIAgentConfigEntry = ConfigEntry[WorkspaceRuntimeData]


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
        """Return if the primary model profile resolves at runtime."""
        try:
            primary_model_profile(self.entry, self.subentry)
        except HomeAssistantError:
            return False
        return True

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

        output_name: str | None = None
        output_json_schema: dict[str, Any] | None = None
        if structure is not None:
            output_name = self._structured_output_name(chat_log.llm_api, structure_name)
            output_json_schema = structured_output_json_schema(
                structure,
                custom_serializer=default_structure_serializer(chat_log.llm_api),
            )
        agent_output_type: object = str

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
            structured_output_tool_names: set[str] = set()
            if output_name is not None and output_json_schema is not None:
                output_mode = resolved_structured_output_mode(profile)
                structured_output_tool_names = output_tool_names(
                    output_mode, output_name
                )
                agent_output_type = structured_agent_output_type(
                    output_mode=output_mode,
                    output_name=output_name,
                    json_schema=output_json_schema,
                    output_tool_retries=self._tool_retries(),
                )
            else:
                agent_output_type = str
            usage_limits = UsageLimits(
                request_limit=run_max_iterations(self.subentry.data, max_iterations),
            )
            try:
                outcome = await run_model_profile(
                    self.hass,
                    self.entry,
                    self.subentry,
                    index=index,
                    attempt_count=len(profiles),
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
                    tool_retries=self._tool_retries(),
                    agent_factory=Agent,
                    model_factory=chat_model_for_profile,
                    virtual_workspace_parts_factory=virtual_workspace_parts,
                )
                if record_success:
                    _clear_runtime_auth_failure(self.hass, self.entry, profile)
                    record_agent_run_success(
                        self.hass, self.entry, self.subentry, outcome, agent_id
                    )
                    store_run_diagnostics(
                        self.entry,
                        self.subentry,
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
                handle_profile_error(
                    self.hass,
                    self.entry,
                    self.subentry,
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

    def _tool_retries(self) -> int:
        """Return the configured tool retry budget for one run."""
        value = self.subentry.data.get(CONF_TOOL_RETRIES)
        if type(value) is int and value >= 0:
            return value
        return DEFAULT_TOOL_RETRIES

    def _store_run_diagnostics(
        self,
        recorder: RunDiagnosticsRecorder,
        *,
        status: str,
        summary: dict[str, Any],
    ) -> None:
        """Store latest bounded last-run diagnostics for this subentry."""
        store_run_diagnostics(
            self.entry,
            self.subentry,
            recorder,
            status=status,
            summary=summary,
        )

    def _record_agent_run_success(
        self, outcome: AgentRunOutcome, agent_id: str | None = None
    ) -> None:
        """Record successful run metrics and fire the completion event."""
        record_agent_run_success(
            self.hass,
            self.entry,
            self.subentry,
            outcome,
            agent_id,
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
        record_agent_run_failure(
            self.hass,
            self.entry,
            self.subentry,
            err,
            agent_id,
            model_profile=model_profile,
            failure=failure,
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
