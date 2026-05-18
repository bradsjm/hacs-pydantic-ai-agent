"""Pydantic models for OpenAI-compatible API responses."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class OpenAICompatibleModel(BaseModel):
    """Base model that preserves provider-specific extension fields."""

    model_config = ConfigDict(extra="allow")


class CompletionTokenDetails(OpenAICompatibleModel):
    """Token detail counts returned by some providers."""

    reasoning_tokens: int | None = None
    audio_tokens: int | None = None
    accepted_prediction_tokens: int | None = None
    rejected_prediction_tokens: int | None = None


class PromptTokenDetails(OpenAICompatibleModel):
    """Prompt token detail counts returned by some providers."""

    audio_tokens: int | None = None
    cached_tokens: int | None = None


class CompletionUsage(OpenAICompatibleModel):
    """Token usage for a Chat Completions response."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    completion_tokens_details: CompletionTokenDetails | None = None
    prompt_tokens_details: PromptTokenDetails | None = None


class ResponseOutputTokenDetails(OpenAICompatibleModel):
    """Token detail counts returned by Responses APIs."""

    reasoning_tokens: int | None = None


class ResponseUsage(OpenAICompatibleModel):
    """Token usage for a Responses API response."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    output_tokens_details: ResponseOutputTokenDetails | None = None


class ResponseIncompleteDetails(OpenAICompatibleModel):
    """Incomplete response details."""

    reason: str | None = None


class ResponseConversation(OpenAICompatibleModel):
    """Responses API conversation metadata."""

    id: str | None = None


class Response(OpenAICompatibleModel):
    """Non-streamed Responses API response."""

    id: str | None = None
    object: str | None = None
    created_at: float | int | None = None
    model: str | None = None
    status: str | None = None
    output: list[dict[str, Any]] = []
    usage: ResponseUsage | None = None
    incomplete_details: ResponseIncompleteDetails | None = None
    conversation: ResponseConversation | None = None


class ResponseStreamEvent(OpenAICompatibleModel):
    """One Responses API streaming event."""

    type: str
    response: Response | None = None
    item: dict[str, Any] | None = None
    item_id: str | None = None
    output_index: int | None = None
    content_index: int | None = None
    summary_index: int | None = None
    delta: str | None = None
    text: str | None = None
    refusal: str | None = None
    part: dict[str, Any] | None = None
    annotation: Any = None
    logprobs: Any = None


class FunctionCall(OpenAICompatibleModel):
    """Function call name and JSON argument payload."""

    name: str | None = None
    arguments: str | None = None


class ChatCompletionMessageToolCall(OpenAICompatibleModel):
    """Tool call returned in a non-streamed assistant message."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatCompletionMessage(OpenAICompatibleModel):
    """Assistant message returned by Chat Completions."""

    role: str | None = None
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[ChatCompletionMessageToolCall] | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None


class ChatCompletionChoice(OpenAICompatibleModel):
    """One Chat Completions choice."""

    index: int
    message: ChatCompletionMessage
    finish_reason: str | None = None
    logprobs: Any = None


class ChatCompletion(OpenAICompatibleModel):
    """Non-streamed Chat Completions response."""

    id: str | None = None
    object: str | None = None
    created: int | None = None
    model: str | None = None
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage | None = None


class ChatCompletionChunkToolCall(OpenAICompatibleModel):
    """Tool-call delta returned by streaming Chat Completions."""

    index: int
    id: str | None = None
    type: Literal["function"] | None = None
    function: FunctionCall | None = None


class ChatCompletionChunkDelta(OpenAICompatibleModel):
    """Streaming assistant delta."""

    role: str | None = None
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[ChatCompletionChunkToolCall] | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None


class ChatCompletionChunkChoice(OpenAICompatibleModel):
    """One streaming Chat Completions choice."""

    index: int
    delta: ChatCompletionChunkDelta | None = None
    finish_reason: str | None = None
    logprobs: Any = None


class ChatCompletionChunk(OpenAICompatibleModel):
    """Streaming Chat Completions response chunk."""

    id: str | None = None
    object: str | None = None
    created: int | None = None
    model: str | None = None
    choices: list[ChatCompletionChunkChoice] = []
    usage: CompletionUsage | None = None
