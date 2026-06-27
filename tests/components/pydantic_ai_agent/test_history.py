"""Test Home Assistant ChatLog history conversion."""

import base64
from pathlib import Path

from custom_components.pydantic_ai_agent.agent.history import (
    chat_log_content_to_model_messages,
    split_last_user_prompt,
)
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from pydantic_ai import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
import pytest


async def test_chat_log_content_converts_to_model_messages(
    hass: HomeAssistant,
) -> None:
    """Test ChatLog content converts to Pydantic AI message history."""
    messages = await chat_log_content_to_model_messages(
        hass,
        [
            conversation.SystemContent("system prompt"),
            conversation.UserContent("hello"),
            conversation.AssistantContent(
                agent_id="conversation.test",
                content="hi",
                thinking_content="thinking",
                tool_calls=[
                    llm.ToolInput(
                        tool_name="HassTurnOn",
                        tool_args={"name": "Kitchen"},
                        id="tool-1",
                    )
                ],
            ),
            conversation.ToolResultContent(
                agent_id="conversation.test",
                tool_call_id="tool-1",
                tool_name="HassTurnOn",
                tool_result={"success": True},
            ),
        ],
    )

    system_message = messages[0]
    assert isinstance(system_message, ModelRequest)
    system_part = system_message.parts[0]
    assert isinstance(system_part, SystemPromptPart)
    assert system_part.content == "system prompt"

    user_message = messages[1]
    assert isinstance(user_message, ModelRequest)
    user_part = user_message.parts[0]
    assert isinstance(user_part, UserPromptPart)
    assert user_part.content == "hello"

    assistant_message = messages[2]
    assert isinstance(assistant_message, ModelResponse)
    assert assistant_message.parts == [
        TextPart(content="hi"),
        ThinkingPart(content="thinking"),
        ToolCallPart(
            tool_name="HassTurnOn",
            args={"name": "Kitchen"},
            tool_call_id="tool-1",
        ),
    ]

    tool_message = messages[3]
    assert isinstance(tool_message, ModelRequest)
    tool_part = tool_message.parts[0]
    assert isinstance(tool_part, ToolReturnPart)
    assert tool_part.tool_name == "HassTurnOn"
    assert tool_part.content == {"success": True}
    assert tool_part.tool_call_id == "tool-1"


async def test_user_attachments_convert_to_binary_content(
    hass: HomeAssistant, tmp_path: Path
) -> None:
    """Test supported attachments become Pydantic AI binary content."""
    attachment_path = tmp_path / "image.png"
    attachment_path.write_bytes(b"image-bytes")

    messages = await chat_log_content_to_model_messages(
        hass,
        [
            conversation.UserContent(
                "describe this",
                attachments=[
                    conversation.Attachment(
                        media_content_id="media-source://camera/test",
                        mime_type="image/png",
                        path=attachment_path,
                    )
                ],
            )
        ],
    )

    message = messages[0]
    assert isinstance(message, ModelRequest)
    user_part = message.parts[0]
    assert isinstance(user_part, UserPromptPart)
    assert isinstance(user_part.content, list)
    assert user_part.content[0] == "describe this"
    assert user_part.content[1] == BinaryContent(
        data=b"image-bytes", media_type="image/png"
    )


async def test_unsupported_attachment_type_raises(
    hass: HomeAssistant, tmp_path: Path
) -> None:
    """Test unsupported attachments raise a Home Assistant error."""
    attachment_path = tmp_path / "notes.txt"
    attachment_path.write_text("notes")

    with pytest.raises(HomeAssistantError):
        await chat_log_content_to_model_messages(
            hass,
            [
                conversation.UserContent(
                    "read this",
                    attachments=[
                        conversation.Attachment(
                            media_content_id="media-source://local/notes.txt",
                            mime_type="text/plain",
                            path=attachment_path,
                        )
                    ],
                )
            ],
        )


async def test_tool_result_multimodal_sentinel_rehydrates_to_binary_content(
    hass: HomeAssistant,
) -> None:
    """Test stored multimodal tool results replay as text plus binary content."""
    messages = await chat_log_content_to_model_messages(
        hass,
        [
            conversation.ToolResultContent(
                agent_id="conversation.test",
                tool_call_id="tool-1",
                tool_name="camera_snapshot",
                tool_result={
                    "_type": "ha_multimodal_tool_result",
                    "text": "Snapshot",
                    "attachments": [
                        {
                            "kind": "inline_image",
                            "mime_type": "image/jpeg",
                            "base64": base64.b64encode(b"jpeg-bytes").decode(),
                        }
                    ],
                },
            )
        ],
    )

    message = messages[0]
    assert isinstance(message, ModelRequest)
    tool_part = message.parts[0]
    assert isinstance(tool_part, ToolReturnPart)
    assert tool_part.content == [
        "Snapshot",
        BinaryContent(data=b"jpeg-bytes", media_type="image/jpeg"),
    ]


def test_split_last_user_prompt_returns_none_without_user_prompt() -> None:
    """Test history is unchanged when no user prompt exists."""
    messages: list[ModelMessage] = [
        ModelResponse(parts=[TextPart(content="assistant")])
    ]

    prompt, history = split_last_user_prompt(messages)

    assert prompt is None
    assert history is messages


def test_split_last_user_prompt_removes_latest_user_part_only() -> None:
    """Test latest user prompt is split while surrounding request parts remain."""
    first_user = ModelRequest(parts=[UserPromptPart(content="old user")])
    assistant = ModelResponse(parts=[TextPart(content="assistant")])
    latest_request = ModelRequest(
        parts=[
            SystemPromptPart(content="system tail"),
            UserPromptPart(content="current user"),
            ToolReturnPart(
                tool_name="HassTurnOn",
                content={"success": True},
                tool_call_id="tool-1",
            ),
        ]
    )
    trailing_assistant = ModelResponse(parts=[TextPart(content="after")])
    messages = [first_user, assistant, latest_request, trailing_assistant]

    prompt, history = split_last_user_prompt(messages)

    assert prompt == "current user"
    assert history[0] is first_user
    assert history[1] is assistant
    assert history[3] is trailing_assistant
    preserved_request = history[2]
    assert isinstance(preserved_request, ModelRequest)
    system_part = preserved_request.parts[0]
    assert isinstance(system_part, SystemPromptPart)
    assert system_part.content == "system tail"
    tool_part = preserved_request.parts[1]
    assert isinstance(tool_part, ToolReturnPart)
    assert tool_part.tool_name == "HassTurnOn"
    assert tool_part.content == {"success": True}
    assert tool_part.tool_call_id == "tool-1"
