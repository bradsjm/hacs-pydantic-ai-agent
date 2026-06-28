"""Pydantic AI Responses model without the OpenAI SDK."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    FinishReason,
    ModelMessage,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)
from pydantic_ai.models import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
    check_allow_model_requests,
    get_user_agent,
)
from pydantic_ai.profiles import ModelProfile, ModelProfileSpec
from pydantic_ai.providers import Provider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT

from ..openai_compatible_client import NOT_GIVEN, AsyncOpenAICompatible, Response, omit
from ..openai_compatible_client._streaming import ResponseStream
from ._chat_model import _map_api_errors
from ._provider import OpenAICompatibleProvider
from ._responses_message_mapping import (
    map_json_schema,
    map_messages,
    map_tool_definition,
)
from ._responses_streamed_response import OpenAICompatibleResponsesStreamedResponse
from ._usage import map_usage

_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "max_output_tokens": "length",
    "content_filter": "content_filter",
    "completed": "stop",
    "cancelled": "error",
    "failed": "error",
}


@dataclass(init=False)
class OpenAICompatibleResponsesModel(Model[AsyncOpenAICompatible]):
    """Pydantic AI model for OpenAI-compatible Responses APIs."""

    _model_name: str = field(repr=False)
    _provider: Provider[AsyncOpenAICompatible] = field(repr=False)

    def __init__(
        self,
        model_name: str,
        *,
        provider: OpenAICompatibleProvider,
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ) -> None:
        """Initialize the model."""
        self._model_name = model_name
        self._provider = provider
        super().__init__(settings=settings, profile=cast(ModelProfile | None, profile))

    @property
    def client(self) -> AsyncOpenAICompatible:
        """Return the low-level client."""
        return self._provider.client

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """Return the provider system name."""
        return self._provider.name

    @property
    def base_url(self) -> str:
        """Return the provider base URL."""
        return self._provider.base_url

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make a non-streamed model request."""
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        settings = model_settings or {}
        with _map_api_errors(self.model_name):
            response = await self._responses_create(
                messages, settings, model_request_parameters
            )
        assert isinstance(response, Response)
        return self._process_response(response)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: AgentDepsT | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a streamed model request."""
        _ = run_context
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        settings = model_settings or {}
        with _map_api_errors(self.model_name):
            response = await self._responses_create(
                messages, settings, model_request_parameters, stream=True
            )
            assert isinstance(response, ResponseStream)
            async with response:
                yield OpenAICompatibleResponsesStreamedResponse(
                    model_request_parameters=model_request_parameters,
                    _model_name=self.model_name,
                    _response=response,
                    _provider_name=self._provider.name,
                    _provider_url=self._provider.base_url,
                )

    async def _responses_create(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings,
        model_request_parameters: ModelRequestParameters,
        *,
        stream: bool = False,
    ) -> Response | ResponseStream:
        """Create a Responses request using the low-level client."""
        tools, tool_choice = self._get_tools_and_tool_choice(
            model_settings, model_request_parameters
        )
        strict_supported = bool(
            self.profile.get("openai_supports_strict_tool_definition")
        )
        text: dict[str, Any] | None = None
        if model_request_parameters.output_mode == "native":
            output_object = model_request_parameters.output_object
            assert output_object is not None
            text = {
                "format": map_json_schema(
                    output_object,
                    strict_supported=strict_supported,
                )
            }
        elif model_request_parameters.output_mode == "prompted" and self.profile.get(
            "supports_json_object_output"
        ):
            text = {"format": {"type": "json_object"}}

        instructions, input_items = await map_messages(
            self, messages, model_request_parameters
        )
        if text and text.get("format", {}).get("type") == "json_object":
            json_instruction = (
                instructions if isinstance(instructions, str) else "Return JSON."
            )
            input_items.insert(
                0,
                {
                    "role": _system_role(self),
                    "content": json_instruction,
                },
            )
            instructions = omit
        if not input_items:
            input_items.append({"role": "user", "content": ""})

        extra_headers = {}
        if not any(key.lower() == "user-agent" for key in self.client.auth_headers):
            extra_headers["User-Agent"] = get_user_agent()
        return await self.client.responses.create(
            model=self.model_name,
            input=input_items,
            instructions=instructions,
            tools=tools or omit,
            tool_choice=tool_choice or omit,
            parallel_tool_calls=model_settings.get("parallel_tool_calls", omit)
            if tools
            else omit,
            max_output_tokens=model_settings.get("max_tokens", omit),
            stream=stream,
            timeout=model_settings.get("timeout", NOT_GIVEN),
            reasoning=_reasoning(model_settings, model_request_parameters),
            user=model_settings.get("user", model_settings.get("openai_user", omit)),
            temperature=model_settings.get("temperature", omit),
            top_p=model_settings.get("top_p", omit),
            text=text or omit,
            extra_headers=extra_headers,
            extra_body=cast(dict[str, Any] | None, model_settings.get("extra_body")),
        )

    def _get_tools_and_tool_choice(
        self,
        model_settings: ModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[list[dict[str, Any]], str | dict[str, Any] | None]:
        """Return request tools and Responses tool_choice."""
        if model_request_parameters.native_tools:
            raise UnexpectedModelBehavior(
                "Native tools are not supported by the Responses adapter"
            )
        strict_supported = bool(
            self.profile.get("openai_supports_strict_tool_definition")
        )
        tools = [
            map_tool_definition(tool, strict_supported=strict_supported)
            for tool in model_request_parameters.tool_defs.values()
        ]
        if not tools:
            return tools, None
        choice = model_settings.get("tool_choice")
        if isinstance(choice, str) and choice in ("none", "auto", "required"):
            return tools, choice
        if isinstance(choice, list) and len(choice) == 1:
            return tools, {"type": "function", "name": choice[0]}
        return tools, "auto"

    def _process_response(self, response: Response) -> ModelResponse:
        """Process a non-streamed Responses API response."""
        timestamp = _timestamp(response.created_at)
        parts: list[Any] = []
        refusal: str | None = None
        for item in response.output:
            item_type = item.get("type")
            if item_type == "reasoning":
                parts.extend(_reasoning_parts(item, self.system))
            elif item_type == "message":
                message_parts, message_refusal = _message_parts(item, self.system)
                parts.extend(message_parts)
                refusal = refusal or message_refusal
            elif item_type == "function_call":
                parts.append(_tool_call_part(item, self.system))
            else:
                raise UnexpectedModelBehavior(
                    f"Unsupported Responses output item type: {item_type!r}"
                )

        provider_details: dict[str, Any] = {}
        raw_finish_reason = (
            response.incomplete_details.reason
            if response.incomplete_details is not None
            and response.incomplete_details.reason is not None
            else response.status
        )
        finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason or "")
        if raw_finish_reason:
            provider_details["finish_reason"] = raw_finish_reason
        if response.conversation and response.conversation.id:
            provider_details["conversation_id"] = response.conversation.id
        if refusal is not None:
            parts = []
            provider_details.pop("finish_reason", None)
            provider_details["refusal"] = refusal
            finish_reason = "content_filter"
        return ModelResponse(
            parts=parts,
            usage=map_usage(response),
            model_name=response.model or self.model_name,
            timestamp=timestamp,
            provider_details=provider_details or None,
            provider_response_id=response.id,
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            finish_reason=finish_reason,
        )


def _system_role(model: Model[Any]) -> str:
    role = model.profile.get("openai_system_prompt_role")
    return role if isinstance(role, str) and role else "system"


def _reasoning(
    model_settings: ModelSettings,
    model_request_parameters: ModelRequestParameters,
) -> dict[str, str] | object:
    effort = model_settings.get("openai_reasoning_effort")
    if model_request_parameters.thinking is not None:
        thinking = model_request_parameters.thinking
        effort = (
            None
            if thinking in (False, "none")
            else "medium"
            if thinking is True
            else thinking
        )
    summary = model_settings.get("openai_reasoning_summary")
    reasoning: dict[str, str] = {}
    if isinstance(effort, str) and effort and effort != "none":
        reasoning["effort"] = effort
    if isinstance(summary, str) and summary:
        reasoning["summary"] = summary
    return reasoning or omit


def _timestamp(value: float | int | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    return datetime.fromtimestamp(value, UTC)


def _reasoning_parts(item: dict[str, Any], provider_name: str) -> list[ThinkingPart]:
    signature = item.get("encrypted_content")
    raw_content = [
        content.get("text", "")
        for content in item.get("content") or []
        if isinstance(content, dict) and content.get("text") is not None
    ]
    provider_details = {"raw_content": raw_content} if raw_content else None
    summaries = item.get("summary") or []
    if summaries:
        parts: list[ThinkingPart] = []
        for summary in summaries:
            if isinstance(summary, dict):
                parts.append(
                    ThinkingPart(
                        content=summary.get("text") or "",
                        id=item.get("id"),
                        signature=signature,
                        provider_name=provider_name,
                        provider_details=provider_details,
                    )
                )
                signature = None
                provider_details = None
        return parts
    if signature or provider_details:
        return [
            ThinkingPart(
                content="",
                id=item.get("id"),
                signature=signature,
                provider_name=provider_name,
                provider_details=provider_details,
            )
        ]
    return []


def _message_parts(
    item: dict[str, Any], provider_name: str
) -> tuple[list[TextPart], str | None]:
    parts: list[TextPart] = []
    refusal: str | None = None
    for content in item.get("content") or []:
        if not isinstance(content, dict):
            continue
        content_type = content.get("type")
        if content_type == "refusal":
            refusal = content.get("refusal") or ""
        elif content_type == "output_text":
            details: dict[str, Any] = {}
            if item.get("phase") is not None:
                details["phase"] = item["phase"]
            if content.get("annotations"):
                details["annotations"] = content["annotations"]
            if content.get("logprobs"):
                details["logprobs"] = content["logprobs"]
            parts.append(
                TextPart(
                    content.get("text") or "",
                    id=item.get("id"),
                    provider_name=provider_name,
                    provider_details=details or None,
                )
            )
        else:
            raise UnexpectedModelBehavior(
                f"Unsupported Responses message content type: {content_type!r}"
            )
    return parts, refusal


def _tool_call_part(item: dict[str, Any], provider_name: str) -> ToolCallPart:
    details = {"namespace": item["namespace"]} if item.get("namespace") else None
    return ToolCallPart(
        item.get("name") or "",
        item.get("arguments") or "{}",
        tool_call_id=item.get("call_id") or item.get("id") or "",
        id=item.get("id"),
        provider_name=provider_name,
        provider_details=details,
    )
