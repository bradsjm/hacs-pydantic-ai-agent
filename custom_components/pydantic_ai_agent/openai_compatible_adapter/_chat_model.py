"""Pydantic AI Chat Completions model without the OpenAI SDK."""

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, assert_never, cast

from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
)
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
from pydantic_ai.profiles import ModelProfileSpec
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT

from ..openai_compatible_client import (
    NOT_GIVEN,
    APIConnectionError,
    APIStatusError,
    AsyncOpenAICompatible,
    ChatCompletion,
    ChatCompletionStream,
    omit,
)
from ._message_mapping import map_json_schema, map_messages, map_tool_definition
from ._provider import OpenAICompatibleProvider
from ._streamed_response import OpenAICompatibleStreamedResponse
from ._usage import map_usage

_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_call",
    "function_call": "tool_call",
    "content_filter": "content_filter",
}


@contextmanager
def _map_api_errors(model_name: str) -> Iterator[None]:
    """Map low-level client errors to Pydantic AI errors."""
    try:
        yield
    except APIStatusError as err:
        if err.status_code >= 400:
            raise ModelHTTPError(
                status_code=err.status_code,
                model_name=model_name,
                body=err.body,
            ) from err
        raise ModelAPIError(model_name=model_name, message=err.message) from err
    except APIConnectionError as err:
        raise ModelAPIError(model_name=model_name, message=err.message) from err


@dataclass(init=False)
class OpenAICompatibleChatModel(Model[AsyncOpenAICompatible]):
    """Pydantic AI model for OpenAI-compatible Chat Completions APIs."""

    _model_name: str = field(repr=False)
    _provider: OpenAICompatibleProvider = field(repr=False)

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
        super().__init__(settings=settings, profile=profile or provider.model_profile)

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
            response = await self._completions_create(
                messages, False, settings, model_request_parameters
            )
        return self._process_response(cast(ChatCompletion, response))

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: AgentDepsT | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a streamed model request."""
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        settings = model_settings or {}
        with _map_api_errors(self.model_name):
            response = await self._completions_create(
                messages, True, settings, model_request_parameters
            )
            streamed_response = cast(ChatCompletionStream, response)
            async with streamed_response:
                yield OpenAICompatibleStreamedResponse(
                    model_request_parameters=model_request_parameters,
                    _model_name=self.model_name,
                    _response=streamed_response,
                    _provider_name=self._provider.name,
                    _provider_url=self._provider.base_url,
                )

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: ModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> ChatCompletion | ChatCompletionStream:
        """Create a Chat Completions request using the low-level client."""
        tools, tool_choice = self._get_tools_and_tool_choice(
            model_settings, model_request_parameters
        )
        response_format = None
        if model_request_parameters.output_mode == "native":
            output_object = model_request_parameters.output_object
            assert output_object is not None
            response_format = map_json_schema(output_object)
        elif (
            model_request_parameters.output_mode == "prompted"
            and self.profile.supports_json_object_output
        ):
            response_format = {"type": "json_object"}

        extra_headers = {}
        if not any(key.lower() == "user-agent" for key in self.client.auth_headers):
            extra_headers["User-Agent"] = get_user_agent()
        return await self.client.chat.completions.create(
            model=self.model_name,
            messages=await map_messages(self, messages, model_request_parameters),
            tools=tools or omit,
            tool_choice=tool_choice or omit,
            stream=stream,
            stream_options={"include_usage": True} if stream else omit,
            stop=model_settings.get("stop_sequences", omit),
            max_completion_tokens=model_settings.get("max_tokens", omit),
            timeout=model_settings.get("timeout", NOT_GIVEN),
            response_format=response_format or omit,
            seed=model_settings.get("seed", omit),
            reasoning_effort=_reasoning_effort(model_request_parameters),
            user=model_settings.get("user", model_settings.get("openai_user", omit)),
            temperature=model_settings.get("temperature", omit),
            top_p=model_settings.get("top_p", omit),
            presence_penalty=model_settings.get("presence_penalty", omit),
            frequency_penalty=model_settings.get("frequency_penalty", omit),
            parallel_tool_calls=model_settings.get("parallel_tool_calls", omit)
            if tools
            else omit,
            extra_headers=extra_headers,
            extra_body=cast(dict[str, Any] | None, model_settings.get("extra_body")),
        )

    def _get_tools_and_tool_choice(
        self,
        model_settings: ModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[list[dict[str, Any]], str | dict[str, Any] | None]:
        """Return request tools and OpenAI-compatible tool_choice."""
        if model_request_parameters.native_tools:
            raise UnexpectedModelBehavior(
                "Native tools are not supported by the Chat Completions adapter"
            )
        tool_defs = model_request_parameters.tool_defs
        tools = [
            map_tool_definition(
                tool,
                strict_supported=OpenAIModelProfile.from_profile(
                    self.profile
                ).openai_supports_strict_tool_definition,
            )
            for tool in tool_defs.values()
        ]
        if not tools:
            return tools, None
        choice = model_settings.get("tool_choice")
        if isinstance(choice, str) and choice in ("none", "auto", "required"):
            return tools, choice
        if isinstance(choice, list) and len(choice) == 1:
            return tools, {"type": "function", "function": {"name": choice[0]}}
        return tools, "auto"

    def _process_response(self, response: ChatCompletion) -> ModelResponse:
        """Process a non-streamed Chat Completions response."""
        if not response.choices:
            raise UnexpectedModelBehavior(
                "Chat Completions response did not include choices"
            )
        timestamp = datetime.now(UTC)
        choice = response.choices[0]
        if choice.message.refusal:
            return ModelResponse(
                parts=[],
                usage=map_usage(response),
                model_name=response.model or self.model_name,
                timestamp=timestamp,
                provider_details={"refusal": choice.message.refusal},
                provider_response_id=response.id,
                provider_name=self._provider.name,
                provider_url=self._provider.base_url,
                finish_reason="content_filter",
            )
        parts: list[Any] = []
        for field_name in ("reasoning", "reasoning_content"):
            reasoning = getattr(choice.message, field_name, None)
            if isinstance(reasoning, str) and reasoning:
                parts.append(
                    ThinkingPart(
                        id=field_name, content=reasoning, provider_name=self.system
                    )
                )
                break
        if choice.message.content:
            parts.append(TextPart(choice.message.content))
        for tool_call in choice.message.tool_calls or []:
            if tool_call.type == "function":
                parts.append(
                    ToolCallPart(
                        tool_call.function.name or "",
                        tool_call.function.arguments or "{}",
                        tool_call_id=tool_call.id,
                    )
                )
            else:
                assert_never(tool_call.type)
        provider_details = (
            {"finish_reason": choice.finish_reason} if choice.finish_reason else None
        )
        return ModelResponse(
            parts=parts,
            usage=map_usage(response),
            model_name=response.model or self.model_name,
            timestamp=timestamp,
            provider_details=provider_details,
            provider_response_id=response.id,
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            finish_reason=_FINISH_REASON_MAP.get(choice.finish_reason or ""),
        )


def _reasoning_effort(model_request_parameters: ModelRequestParameters) -> str | object:
    """Map Pydantic AI thinking configuration to reasoning_effort."""
    thinking = model_request_parameters.thinking
    if thinking is None:
        return omit
    if thinking is True:
        return "medium"
    if thinking is False:
        return "none"
    return thinking
