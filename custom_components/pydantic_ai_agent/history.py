"""Home Assistant ChatLog to Pydantic AI message conversion."""

from collections.abc import Iterable, Sequence
from typing import Any

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

from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

_SUPPORTED_ATTACHMENT_MIME_TYPES = {"application/pdf"}
_SUPPORTED_ATTACHMENT_MIME_PREFIXES = ("image/",)


async def chat_log_content_to_model_messages(
    hass: HomeAssistant, content: Iterable[conversation.Content]
) -> list[ModelMessage]:
    """Convert Home Assistant chat log content into Pydantic AI messages."""
    messages: list[ModelMessage] = []
    for item in content:
        if isinstance(item, conversation.SystemContent):
            messages.append(ModelRequest(parts=[SystemPromptPart(content=item.content)]))
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


async def _user_prompt_content_from_ha_content(
    hass: HomeAssistant, content: conversation.UserContent
) -> str | Sequence[Any]:
    """Return a Pydantic AI user prompt content payload."""
    if not content.attachments:
        return content.content

    parts: list[Any] = [content.content]
    for attachment in content.attachments:
        parts.append(await _binary_content_from_attachment(hass, attachment))
    return parts


async def _binary_content_from_attachment(
    hass: HomeAssistant, attachment: conversation.Attachment
) -> BinaryContent:
    """Read a Home Assistant attachment as Pydantic AI binary content."""
    mime_type = attachment.mime_type
    if not _is_supported_attachment_mime_type(mime_type):
        raise HomeAssistantError(f"Unsupported attachment type: {mime_type}")

    try:
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
    """Convert Home Assistant assistant content into Pydantic AI response parts."""
    parts: list[TextPart | ThinkingPart | ToolCallPart] = []
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
