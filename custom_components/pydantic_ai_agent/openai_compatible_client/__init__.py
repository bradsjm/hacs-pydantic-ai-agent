"""Lightweight OpenAI-compatible async client.

This package intentionally mirrors the small subset of the OpenAI Python SDK
shape needed by the integration without importing or depending on ``openai``.
Some sentinel, exception, response-model, and SSE semantics are adapted from
the OpenAI Python SDK, which is Apache-2.0 licensed.
"""

from ._client import AsyncOpenAICompatible
from ._exceptions import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAICompatibleError,
)
from ._sentinels import NOT_GIVEN, NotGiven, Omit, omit
from ._types import ChatCompletion, ChatCompletionChunk, Response, ResponseStreamEvent

__all__ = [
    "APIConnectionError",
    "APIStatusError",
    "APITimeoutError",
    "AsyncOpenAICompatible",
    "ChatCompletion",
    "ChatCompletionChunk",
    "NOT_GIVEN",
    "NotGiven",
    "Omit",
    "OpenAICompatibleError",
    "Response",
    "ResponseStreamEvent",
    "omit",
]
