"""Pydantic AI adapter for OpenAI-compatible Chat Completions APIs."""

from ._chat_model import OpenAICompatibleChatModel
from ._provider import OpenAICompatibleProvider
from ._responses_model import OpenAICompatibleResponsesModel

__all__ = [
    "OpenAICompatibleChatModel",
    "OpenAICompatibleProvider",
    "OpenAICompatibleResponsesModel",
]
