"""Home Assistant ChatLog to Pydantic AI message conversion."""

from collections.abc import Iterable, Sequence
from typing import Any

from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
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

_SUPPORTED_ATTACHMENT_MIME_TYPES = {"application/pdf"}
_SUPPORTED_ATTACHMENT_MIME_PREFIXES = ("image/",)


async def chat_log_content_to_model_messages(
    hass: HomeAssistant, content: Iterable[conversation.Content]
) -> list[ModelMessage]:
    """Convert Home Assistant chat log content into Pydantic AI messages."""
    messages: list[ModelMessage] = []
    for item in content:
        if isinstance(item, conversation.SystemContent):
            messages.append(
                ModelRequest(parts=[SystemPromptPart(content=item.content)])
            )
        elif isinstance(item, conversation.UserContent):
            messages.append(
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content=await _user_prompt_content_from_ha_content(
                                hass, item
                            )
                        )
                    ]
                )
            )
        elif isinstance(item, conversation.AssistantContent):
            messages.append(ModelResponse(parts=_assistant_parts_from_ha_content(item)))
        elif isinstance(item, conversation.ToolResultContent):
            # Tool results are model input in Pydantic AI, so they are replayed as
            # a request part rather than assistant response history.
            messages.append(
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name=item.tool_name,
                            content=item.tool_result,
                            tool_call_id=item.tool_call_id,
                        )
                    ]
                )
            )
    return messages


def split_last_user_prompt(
    messages: list[ModelMessage],
) -> tuple[str | Sequence[Any] | None, list[ModelMessage]]:
    """Split the latest user prompt from message history for Agent runs."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, ModelRequest):
            continue
        for part_index in range(len(message.parts) - 1, -1, -1):
            part = message.parts[part_index]
            if not isinstance(part, UserPromptPart):
                continue
            before_parts = list(message.parts[:part_index])
            after_parts = list(message.parts[part_index + 1 :])
            history = list(messages[:index])
            if before_parts or after_parts:
                history.append(ModelRequest(parts=before_parts + after_parts))
            history.extend(messages[index + 1 :])
            return part.content, history
    return None, messages


async def _user_prompt_content_from_ha_content(
    hass: HomeAssistant, content: conversation.UserContent
) -> str | Sequence[Any]:
    """Return text-only or mixed text/binary content for a user message."""
    if not content.attachments:
        return content.content

    # Pydantic AI represents multimodal user prompts as text mixed with binary
    # content parts; Home Assistant stores attachments separately on the message.
    parts: list[Any] = [content.content]
    for attachment in content.attachments:
        parts.append(await _binary_content_from_attachment(hass, attachment))
    return parts


async def _binary_content_from_attachment(
    hass: HomeAssistant, attachment: conversation.Attachment
) -> BinaryContent:
    """Read supported Home Assistant attachments off the event loop."""
    mime_type = attachment.mime_type
    if not _is_supported_attachment_mime_type(mime_type):
        raise HomeAssistantError(f"Unsupported attachment type: {mime_type}")

    try:
        # Attachment payloads may be backed by files, so keep reads off the event
        # loop before handing bytes to Pydantic AI.
        data = await hass.async_add_executor_job(attachment.path.read_bytes)
    except OSError as err:
        raise HomeAssistantError("Unable to read attachment") from err
    return BinaryContent(data=data, media_type=mime_type)


def _is_supported_attachment_mime_type(mime_type: str) -> bool:
    """Return whether the attachment MIME type is supported."""
    return mime_type in _SUPPORTED_ATTACHMENT_MIME_TYPES or mime_type.startswith(
        _SUPPORTED_ATTACHMENT_MIME_PREFIXES
    )


def _assistant_parts_from_ha_content(
    content: conversation.AssistantContent,
) -> list[TextPart | ThinkingPart | ToolCallPart]:
    """Preserve assistant text, thinking, and tool calls for model history."""
    parts: list[TextPart | ThinkingPart | ToolCallPart] = []
    # Preserve HA's assistant-content order when reconstructing Pydantic AI
    # response parts for the next streamed request.
    if content.content:
        parts.append(TextPart(content=content.content))
    if content.thinking_content:
        parts.append(ThinkingPart(content=content.thinking_content))
    if content.tool_calls:
        parts.extend(
            ToolCallPart(
                tool_name=tool_call.tool_name,
                args=tool_call.tool_args,
                tool_call_id=tool_call.id,
            )
            for tool_call in content.tool_calls
        )
    return parts
