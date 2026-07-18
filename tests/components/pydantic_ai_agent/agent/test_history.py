"""Tests for Home Assistant chat-log history conversion."""

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


async def test_chat_log_content_to_model_messages_converts_core_content(
    hass: HomeAssistant,
) -> None:
    """System, user, assistant, and tool-result content convert to model messages."""
    messages = await chat_log_content_to_model_messages(
        hass,
        [
            conversation.SystemContent("system prompt"),
            conversation.UserContent("turn on light"),
            conversation.AssistantContent(
                agent_id="agent-1",
                content="done",
                thinking_content="reasoning",
                tool_calls=[
                    llm.ToolInput(
                        tool_name="light.turn_on",
                        tool_args={"entity_id": "light.kitchen"},
                        id="call-1",
                    )
                ],
            ),
            conversation.ToolResultContent(
                agent_id="agent-1",
                tool_call_id="call-1",
                tool_name="light.turn_on",
                tool_result={"ok": True},
            ),
        ],
    )

    assert len(messages) == 4
    assert isinstance(messages[0], ModelRequest)
    assert isinstance(messages[0].parts[0], SystemPromptPart)
    assert messages[0].parts[0].content == "system prompt"
    assert isinstance(messages[1], ModelRequest)
    assert isinstance(messages[1].parts[0], UserPromptPart)
    assert messages[1].parts[0].content == "turn on light"
    assert isinstance(messages[2], ModelResponse)
    assert [type(part) for part in messages[2].parts] == [
        TextPart,
        ThinkingPart,
        ToolCallPart,
    ]
    text_part, thinking_part, tool_call_part = messages[2].parts
    assert text_part.content == "done"
    assert thinking_part.content == "reasoning"
    assert tool_call_part.tool_name == "light.turn_on"
    assert tool_call_part.args == {"entity_id": "light.kitchen"}
    assert tool_call_part.tool_call_id == "call-1"
    assert isinstance(messages[3], ModelRequest)
    assert isinstance(messages[3].parts[0], ToolReturnPart)
    tool_return_part = messages[3].parts[0]
    assert tool_return_part.tool_name == "light.turn_on"
    assert tool_return_part.content == {"ok": True}
    assert tool_return_part.tool_call_id == "call-1"



@pytest.mark.parametrize(
    ("supports_images", "expected"),
    [
        (
            None,
            [
                "describe this",
                BinaryContent(data=b"image-bytes", media_type="image/png"),
            ],
        ),
        (
            True,
            [
                "describe this",
                BinaryContent(data=b"image-bytes", media_type="image/png"),
            ],
        ),
        (False, ["describe this"]),
    ],
)
async def test_chat_log_content_to_model_messages_gates_image_attachments(
    hass: HomeAssistant,
    tmp_path: Path,
    supports_images: bool | None,
    expected: str | list[object],
) -> None:
    """Image attachments are dropped for image-unsupported models."""
    image_path = tmp_path / "snapshot.png"
    image_path.write_bytes(b"image-bytes")

    messages = await chat_log_content_to_model_messages(
        hass,
        [
            conversation.UserContent(
                "describe this",
                attachments=[
                    conversation.Attachment(
                        media_content_id="media-source://snapshot",
                        mime_type="image/png",
                        path=image_path,
                    )
                ],
            )
        ],
        supports_images=supports_images,
    )

    assert len(messages) == 1
    assert isinstance(messages[0], ModelRequest)
    assert isinstance(messages[0].parts[0], UserPromptPart)
    assert messages[0].parts[0].content == expected


async def test_chat_log_content_to_model_messages_rejects_unsupported_attachment(
    hass: HomeAssistant,
    tmp_path: Path,
) -> None:
    """Unsupported attachment MIME types raise HomeAssistantError."""
    attachment_path = tmp_path / "data.txt"
    attachment_path.write_text("plain text")

    with pytest.raises(HomeAssistantError):
        await chat_log_content_to_model_messages(
            hass,
            [
                conversation.UserContent(
                    "read this",
                    attachments=[
                        conversation.Attachment(
                            media_content_id="media-source://data",
                            mime_type="text/plain",
                            path=attachment_path,
                        )
                    ],
                )
            ],
        )


@pytest.mark.parametrize(
    ("supports_images", "expected_content"),
    [
        (
            None,
            [
                "camera snapshot",
                BinaryContent(data=b"png-data", media_type="image/png"),
            ],
        ),
        (
            True,
            [
                "camera snapshot",
                BinaryContent(data=b"png-data", media_type="image/png"),
            ],
        ),
        (False, "camera snapshot"),
    ],
)
async def test_chat_log_content_to_model_messages_gates_tool_result_images(
    hass: HomeAssistant,
    supports_images: bool | None,
    expected_content: str | list[object],
) -> None:
    """Multimodal tool-result images drop for image-unsupported models."""
    messages = await chat_log_content_to_model_messages(
        hass,
        [
            conversation.ToolResultContent(
                agent_id="agent-1",
                tool_call_id="call-1",
                tool_name="camera",
                tool_result={
                    "_type": "ha_multimodal_tool_result",
                    "text": "camera snapshot",
                    "attachments": [
                        {
                            "kind": "inline_image",
                            "mime_type": "image/png",
                            "base64": base64.b64encode(b"png-data").decode(),
                        }
                    ],
                },
            )
        ],
        supports_images=supports_images,
    )

    assert isinstance(messages[0], ModelRequest)
    assert isinstance(messages[0].parts[0], ToolReturnPart)
    assert messages[0].parts[0].content == expected_content


def test_split_last_user_prompt_removes_only_latest_prompt() -> None:
    """Only the latest user prompt is split away from preserved history."""
    first_user = ModelRequest(parts=[UserPromptPart(content="first prompt")])
    assistant = ModelResponse(parts=[TextPart(content="first response")])
    mixed_request = ModelRequest(
        parts=[
            SystemPromptPart(content="keep system"),
            UserPromptPart(content="latest prompt"),
            ToolReturnPart(
                tool_name="tool",
                content="tool result",
                tool_call_id="call-1",
            ),
        ]
    )

    prompt, history = split_last_user_prompt([first_user, assistant, mixed_request])

    assert prompt == "latest prompt"
    assert history[:2] == [first_user, assistant]
    assert len(history) == 3
    assert isinstance(history[2], ModelRequest)
    assert [type(part) for part in history[2].parts] == [
        SystemPromptPart,
        ToolReturnPart,
    ]
    system_part, tool_return_part = history[2].parts
    assert isinstance(system_part, SystemPromptPart)
    assert system_part.content == "keep system"
    assert isinstance(tool_return_part, ToolReturnPart)
    assert tool_return_part.tool_name == "tool"
    assert tool_return_part.content == "tool result"
    assert tool_return_part.tool_call_id == "call-1"


def test_split_last_user_prompt_without_user_prompt_returns_original() -> None:
    """History without a user prompt is returned unchanged."""
    messages = [ModelResponse(parts=[TextPart(content="assistant only")])]

    prompt, history = split_last_user_prompt(messages)

    assert prompt is None
    assert history is messages
