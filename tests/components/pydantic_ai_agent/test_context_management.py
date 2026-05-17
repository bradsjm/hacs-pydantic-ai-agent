"""Test Pydantic AI context management helpers."""

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from custom_components.pydantic_ai_agent.context_management import (
    SlidingWindowContextCapability,
)


def _system_message(content: str) -> ModelRequest:
    """Return a system prompt message."""
    return ModelRequest(parts=[SystemPromptPart(content=content)])


def _user_message(content: str) -> ModelRequest:
    """Return a user prompt message."""
    return ModelRequest(parts=[UserPromptPart(content=content)])


def _assistant_message(content: str) -> ModelResponse:
    """Return an assistant text message."""
    return ModelResponse(parts=[TextPart(content=content)])


async def test_sliding_window_leaves_messages_under_trigger_unchanged() -> None:
    """Test context management is a no-op until the trigger is exceeded."""
    messages: list[ModelMessage] = [_user_message("hello")]
    capability = SlidingWindowContextCapability(
        trigger_message_count=2,
        keep_message_count=1,
        keep_head_message_count=0,
    )

    trimmed = await capability.process_messages(messages)

    assert trimmed is messages


async def test_sliding_window_trims_old_messages_and_preserves_head() -> None:
    """Test old middle history is removed while the head and tail remain."""
    messages: list[ModelMessage] = [
        _system_message("system"),
        _user_message("old user"),
        _assistant_message("old assistant"),
        _user_message("recent user"),
        _assistant_message("recent assistant"),
        _user_message("current user"),
    ]
    capability = SlidingWindowContextCapability(
        trigger_message_count=4,
        keep_message_count=2,
        keep_head_message_count=1,
    )

    trimmed = await capability.process_messages(messages)

    assert trimmed == [messages[0], messages[4], messages[5]]


async def test_sliding_window_does_not_split_tool_call_result_pairs() -> None:
    """Test trimming expands the tail to keep tool calls with their returns."""
    tool_call = ModelResponse(
        parts=[
            ToolCallPart(
                tool_name="HassTurnOn",
                args={"name": "Kitchen"},
                tool_call_id="tool-1",
            )
        ]
    )
    tool_return = ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name="HassTurnOn",
                content={"success": True},
                tool_call_id="tool-1",
            )
        ]
    )
    messages: list[ModelMessage] = [
        _system_message("system"),
        _user_message("old user"),
        tool_call,
        _assistant_message("filler"),
        tool_return,
        _user_message("current user"),
    ]
    capability = SlidingWindowContextCapability(
        trigger_message_count=4,
        keep_message_count=2,
        keep_head_message_count=1,
    )

    trimmed = await capability.process_messages(messages)

    assert messages[1] not in trimmed
    assert tool_call in trimmed
    assert tool_return in trimmed


async def test_sliding_window_preserves_active_run_messages() -> None:
    """Test only prior history is trimmed during an active Agent run."""
    current_user = ModelRequest(
        parts=[UserPromptPart(content="current user")], run_id="run-1"
    )
    current_response = ModelResponse(
        parts=[TextPart(content="current response")], run_id="run-1"
    )
    messages: list[ModelMessage] = [
        _user_message("old one"),
        _assistant_message("old two"),
        _user_message("old three"),
        current_user,
        current_response,
    ]
    capability = SlidingWindowContextCapability(
        trigger_message_count=2,
        keep_message_count=1,
        keep_head_message_count=0,
    )

    trimmed = await capability.process_messages(messages, preserve_run_id="run-1")

    assert trimmed == [messages[2], current_user, current_response]


async def test_sliding_window_returns_original_when_head_and_tail_overlap() -> None:
    """Test trimming is skipped when preserving the head would overlap the tail."""
    messages: list[ModelMessage] = [
        _system_message("system"),
        _user_message("one"),
        _assistant_message("two"),
        _user_message("three"),
    ]
    capability = SlidingWindowContextCapability(
        trigger_message_count=3,
        keep_message_count=3,
        keep_head_message_count=1,
    )

    trimmed = await capability.process_messages(messages)

    assert trimmed is messages
