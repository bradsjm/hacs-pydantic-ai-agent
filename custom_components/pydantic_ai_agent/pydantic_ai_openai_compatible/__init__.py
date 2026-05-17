"""Pydantic AI adapter for OpenAI-compatible Chat Completions APIs."""

from ._chat_model import OpenAICompatibleChatModel
from ._provider import OpenAICompatibleProvider

__all__ = ["OpenAICompatibleChatModel", "OpenAICompatibleProvider"]
