"""Map Pydantic AI messages to OpenAI-compatible Chat Completions payloads."""

from collections.abc import AsyncIterable, Sequence
import base64
from typing import Any

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    CachePoint,
    CompactionPart,
    DocumentUrl,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryPromptPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.tools import ToolDefinition
from typing_extensions import assert_never


def map_tool_definition(
    tool: ToolDefinition, *, strict_supported: bool
) -> dict[str, Any]:
    """Map a Pydantic AI tool definition to an OpenAI-compatible function tool."""
    function: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description or "",
        "parameters": tool.parameters_json_schema,
    }
    if tool.strict and strict_supported:
        function["strict"] = tool.strict
    return {"type": "function", "function": function}


def map_json_schema(output_object: Any) -> dict[str, Any]:
    """Map a native structured-output schema to response_format."""
    json_schema: dict[str, Any] = {
        "name": output_object.name or "final_result",
        "schema": output_object.json_schema,
    }
    if output_object.description:
        json_schema["description"] = output_object.description
    if output_object.strict is not None:
        json_schema["strict"] = output_object.strict
    return {"type": "json_schema", "json_schema": json_schema}


async def map_messages(
    model: Model[Any],
    messages: Sequence[ModelMessage],
    model_request_parameters: ModelRequestParameters,
) -> list[dict[str, Any]]:
    """Map Pydantic AI messages to Chat Completions messages."""
    mapped_messages: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            async for item in _map_model_request(model, message):
                mapped_messages.append(item)
        elif isinstance(message, ModelResponse):
            if (mapped := _map_model_response(message)) is not None:
                mapped_messages.append(mapped)
        else:
            assert_never(message)

    if instruction_parts := model._get_instruction_parts(
        messages, model_request_parameters
    ):
        system_role = _system_role(model)
        first_non_system = next(
            (
                i
                for i, item in enumerate(mapped_messages)
                if item.get("role") != system_role
            ),
            len(mapped_messages),
        )
        mapped_messages[first_non_system:first_non_system] = [
            {"role": system_role, "content": part.content} for part in instruction_parts
        ]
    return mapped_messages


def _system_role(model: Model[Any]) -> str:
    """Return the OpenAI-compatible system prompt role."""
    return (
        OpenAIModelProfile.from_profile(model.profile).openai_system_prompt_role
        or "system"
    )


async def _map_model_request(
    model: Model[Any], message: ModelRequest
) -> AsyncIterable[dict[str, Any]]:
    file_content: list[Any] = []
    for part in message.parts:
        if isinstance(part, SystemPromptPart):
            yield {"role": _system_role(model), "content": part.content}
        elif isinstance(part, UserPromptPart):
            yield _map_user_prompt(part)
        elif isinstance(part, ToolReturnPart):
            tool_text, tool_file_content = part.model_response_str_and_user_content()
            file_content.extend(tool_file_content)
            yield {
                "role": "tool",
                "tool_call_id": part.tool_call_id,
                "content": tool_text,
            }
        elif isinstance(part, RetryPromptPart):
            if part.tool_name is None:
                yield {"role": "user", "content": part.model_response()}
            else:
                yield {
                    "role": "tool",
                    "tool_call_id": part.tool_call_id,
                    "content": part.model_response(),
                }
        else:
            assert_never(part)
    if file_content:
        yield _map_user_prompt(UserPromptPart(content=file_content))


def _map_model_response(message: ModelResponse) -> dict[str, Any] | None:
    """Map a prior model response to an assistant message."""
    texts: list[str] = []
    thinking_fields: dict[str, list[str]] = {}
    tool_calls: list[dict[str, Any]] = []
    for item in message.parts:
        if isinstance(item, TextPart):
            texts.append(item.content)
        elif isinstance(item, ThinkingPart):
            if item.id in {"reasoning", "reasoning_content"}:
                thinking_fields.setdefault(item.id, []).append(item.content)
            else:
                start_tag, end_tag = ("<think>", "</think>")
                texts.append("\n".join([start_tag, item.content, end_tag]))
        elif isinstance(item, ToolCallPart):
            tool_calls.append(_map_tool_call(item))
        elif isinstance(
            item, NativeToolCallPart | NativeToolReturnPart | FilePart | CompactionPart
        ):
            pass
        else:
            assert_never(item)
    if not texts and not tool_calls:
        return None
    mapped: dict[str, Any] = {
        "role": "assistant",
        "content": "\n\n".join(texts) if texts else "",
    }
    for field_name, contents in thinking_fields.items():
        mapped[field_name] = "\n\n".join(contents)
    if tool_calls:
        mapped["tool_calls"] = tool_calls
    return mapped


def _map_tool_call(part: ToolCallPart) -> dict[str, Any]:
    """Map a prior tool call to OpenAI-compatible assistant history."""
    return {
        "id": part.tool_call_id,
        "type": "function",
        "function": {"name": part.tool_name, "arguments": part.args_as_json_str()},
    }


def _map_user_prompt(part: UserPromptPart) -> dict[str, Any]:
    """Map a user prompt part."""
    if isinstance(part.content, str):
        content: str | list[dict[str, Any]] = part.content
    else:
        content = [
            mapped
            for item in part.content
            if (mapped := _map_content_item(item)) is not None
        ]
    return {"role": "user", "content": content}


def _map_content_item(item: Any) -> dict[str, Any] | None:
    """Map one multimodal user content item."""
    if isinstance(item, str | TextContent):
        text = item if isinstance(item, str) else item.content
        return {"type": "text", "text": text}
    if isinstance(item, ImageUrl):
        image_url: dict[str, Any] = {"url": item.url}
        if item.vendor_metadata:
            image_url["detail"] = item.vendor_metadata.get("detail", "auto")
        return {"type": "image_url", "image_url": image_url}
    if isinstance(item, BinaryContent):
        return _map_binary_content(item)
    if isinstance(item, DocumentUrl):
        return {
            "type": "file",
            "file": {"file_data": item.url, "filename": f"filename.{item.format}"},
        }
    if isinstance(item, UploadedFile):
        return {"type": "file", "file": {"file_id": item.file_id}}
    if isinstance(item, CachePoint):
        return None
    if isinstance(item, AudioUrl | VideoUrl):
        raise UserError(
            "Audio and video URL inputs are not supported by this Chat Completions adapter"
        )
    assert_never(item)


def _map_binary_content(item: BinaryContent) -> dict[str, Any]:
    """Map binary content as image, file, or inline text."""
    if item.media_type.startswith("text/") or item.media_type in {
        "application/json",
        "application/xml",
        "application/yaml",
        "application/toml",
    }:
        return {"type": "text", "text": item.data.decode("utf-8")}
    if item.is_image:
        image_url: dict[str, Any] = {"url": item.data_uri}
        if item.vendor_metadata:
            image_url["detail"] = item.vendor_metadata.get("detail", "auto")
        return {"type": "image_url", "image_url": image_url}
    if item.is_document:
        data = f"data:{item.media_type};base64,{base64.b64encode(item.data).decode()}"
        return {
            "type": "file",
            "file": {"file_data": data, "filename": f"filename.{item.format}"},
        }
    raise UserError(f"Unsupported binary content type: {item.media_type}")
