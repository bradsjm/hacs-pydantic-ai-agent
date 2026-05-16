"""Shared Pydantic AI entity runtime."""

from collections.abc import AsyncIterable, AsyncIterator, Mapping
import json
import logging
from typing import Any, cast

from pydantic_ai import Agent
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UserError,
    UsageLimitExceeded,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits
import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm

from . import PydanticAIAgentConfigEntry
from .const import (
    CONF_MODEL,
    CONF_MODEL_SETTINGS,
    CONF_MCP_SERVER_IDS,
    CONF_OUTPUT_MODE,
    DEFAULT_TIMEOUT,
    DOMAIN,
    SUBENTRY_TYPE_CONVERSATION,
)
from .ha_toolset import tool_definitions_from_llm_api, tools_from_llm_api
from .history import chat_log_content_to_model_messages, split_last_user_prompt
from .mcp import MCPValidationError, async_runtime_mcp_toolsets
from .provider import openai_chat_model
from .structured_output import (
    default_structure_serializer,
    output_tool_names,
    structured_agent_output_type,
    structured_output_json_schema,
    structured_output_mode,
    structured_output_name,
)

_LOGGER = logging.getLogger(__name__)


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
    ) -> None:
        """Initialize shared entity metadata."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=name,
            manufacturer="Pydantic AI",
            model=subentry.data[CONF_MODEL],
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure_name: str | None = None,
        structure: vol.Schema | None = None,
        max_iterations: int = 10,
    ) -> object | None:
        """Run a Pydantic AI Agent and stream its response into ChatLog."""
        runtime_data = self.entry.runtime_data
        model = openai_chat_model(
            self.hass,
            api_key=runtime_data.api_key,
            base_url=runtime_data.base_url,
            model_name=self.subentry.data[CONF_MODEL],
        )
        model_settings = ModelSettings(**self._model_settings())
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
        mcp_toolsets, mcp_http_clients = await async_runtime_mcp_toolsets(
            self.hass,
            self.entry,
            self.subentry.data.get(CONF_MCP_SERVER_IDS),
        )
        agent = Agent(
            model,
            output_type=cast(Any, agent_output_type),
            model_settings=model_settings,
            tool_retries=0,
            output_retries=2,
            tools=tools_from_llm_api(chat_log.llm_api),
            toolsets=mcp_toolsets,
            max_concurrency=1,
        )
        usage_limits = UsageLimits(
            request_limit=max_iterations,
        )
        try:
            async with agent:
                if structure is None:
                    result = await agent.run(
                        user_prompt,
                        message_history=message_history,
                        model_settings=model_settings,
                        usage_limits=usage_limits,
                    )
                    await _append_agent_messages(
                        chat_log, agent_id, result.new_messages()
                    )
                    return result.output

                result = await agent.run(
                    user_prompt,
                    message_history=message_history,
                    model_settings=model_settings,
                    usage_limits=usage_limits,
                )
                output = result.output
                await _append_agent_messages(
                    chat_log,
                    agent_id,
                    result.new_messages(),
                    output_tool_names=structured_output_tool_names,
                )
                if not isinstance(chat_log.content[-1], conversation.AssistantContent):
                    await _append_text(chat_log, agent_id, _json_output(output))
                return output
        except ModelHTTPError as err:
            # Convert provider/runtime failures into HA-facing errors so
            # conversation and AI task platforms report consistent failures.
            raise HomeAssistantError(_format_http_error(err)) from err
        except ModelAPIError as err:
            raise HomeAssistantError(_format_api_error(err)) from err
        except UnexpectedModelBehavior as err:
            raise HomeAssistantError(
                "Provider returned an unexpected response"
            ) from err
        except TimeoutError as err:
            raise HomeAssistantError("Provider request timed out") from err
        except UsageLimitExceeded as err:
            raise HomeAssistantError(
                "Model requested too many tool iterations"
            ) from err
        except MCPValidationError as err:
            raise HomeAssistantError(err.message) from err
        except (NotImplementedError, UserError) as err:
            raise HomeAssistantError(f"Invalid provider configuration: {err}") from err
        finally:
            for http_client in mcp_http_clients:
                await http_client.aclose()

    def _model_settings(self) -> dict[str, Any]:
        """Return subentry model settings with the integration timeout default."""
        settings: Mapping[str, Any] | None = None
        if self.subentry.subentry_type == SUBENTRY_TYPE_CONVERSATION:
            raw_settings = self.subentry.data.get(CONF_MODEL_SETTINGS)
            if isinstance(raw_settings, Mapping):
                settings = raw_settings

        model_settings = dict(settings or {})
        model_settings.setdefault("timeout", DEFAULT_TIMEOUT)
        return model_settings

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


def _format_http_error(err: ModelHTTPError) -> str:
    """Return a user-facing provider HTTP error message."""
    return f'The provider returned HTTP {err.status_code} for model "{err.model_name}".'


def _format_api_error(err: ModelAPIError) -> str:
    """Return a user-facing provider API error message."""
    return f'The provider returned an API error for model "{err.model_name}".'


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
                if isinstance(part, ToolReturnPart):
                    yield {
                        "role": "tool_result",
                        "tool_call_id": part.tool_call_id,
                        "tool_name": part.tool_name,
                        "tool_result": part.content,
                    }
