"""Pydantic AI context management helpers."""

from typing import Final

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestContext

DEFAULT_CONTEXT_TRIGGER_MESSAGE_COUNT: Final = 100
DEFAULT_CONTEXT_KEEP_MESSAGE_COUNT: Final = 50
DEFAULT_CONTEXT_KEEP_HEAD_MESSAGE_COUNT: Final = 1


class SlidingWindowContextCapability(AbstractCapability[object]):
    """Trim model-request history with a zero-cost sliding message window."""

    def __init__(
        self,
        *,
        trigger_message_count: int = DEFAULT_CONTEXT_TRIGGER_MESSAGE_COUNT,
        keep_message_count: int = DEFAULT_CONTEXT_KEEP_MESSAGE_COUNT,
        keep_head_message_count: int = DEFAULT_CONTEXT_KEEP_HEAD_MESSAGE_COUNT,
    ) -> None:
        """Initialize the fixed-size sliding window."""
        self._trigger_message_count = trigger_message_count
        self._keep_message_count = keep_message_count
        self._keep_head_message_count = keep_head_message_count

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Keep this integration-internal capability out of agent specs."""
        return None

    async def before_model_request(
        self,
        ctx: RunContext[object],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Trim messages immediately before the provider request."""
        request_context.messages = await self.process_messages(
            request_context.messages, preserve_run_id=ctx.run_id
        )
        return request_context

    async def process_messages(
        self, messages: list[ModelMessage], *, preserve_run_id: str | None = None
    ) -> list[ModelMessage]:
        """Return a safely trimmed copy of messages when the trigger is exceeded."""
        history, current_run_messages = _split_current_run_messages(
            messages, preserve_run_id
        )
        if len(history) <= self._trigger_message_count:
            return messages

        head_count = min(self._keep_head_message_count, len(history))
        cutoff_index = len(history) - self._keep_message_count
        if cutoff_index <= head_count:
            return messages

        while cutoff_index > head_count:
            kept_messages = history[:head_count] + history[cutoff_index:]
            dropped_messages = history[head_count:cutoff_index]
            if not _splits_tool_call_return_pair(kept_messages, dropped_messages):
                return kept_messages + current_run_messages
            cutoff_index -= 1

        return messages


def _split_current_run_messages(
    messages: list[ModelMessage], run_id: str | None
) -> tuple[list[ModelMessage], list[ModelMessage]]:
    """Split prior history from messages created during the active Agent run."""
    if run_id is None:
        return messages, []

    for index, message in enumerate(messages):
        if message.run_id == run_id:
            return messages[:index], messages[index:]
    return messages, []


def _splits_tool_call_return_pair(
    kept_messages: list[ModelMessage], dropped_messages: list[ModelMessage]
) -> bool:
    """Return whether dropping messages would separate a tool call and result."""
    kept_call_ids = _tool_call_ids(kept_messages)
    kept_return_ids = _tool_return_ids(kept_messages)
    dropped_call_ids = _tool_call_ids(dropped_messages)
    dropped_return_ids = _tool_return_ids(dropped_messages)
    return bool(
        kept_call_ids & dropped_return_ids or dropped_call_ids & kept_return_ids
    )


def _tool_call_ids(messages: list[ModelMessage]) -> set[str]:
    """Return tool call ids contained in assistant response history."""
    return {
        part.tool_call_id
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    }


def _tool_return_ids(messages: list[ModelMessage]) -> set[str]:
    """Return tool call ids referenced by tool return history."""
    return {
        part.tool_call_id
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }
