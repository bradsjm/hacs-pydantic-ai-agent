"""Map Pydantic AI messages to OpenAI-compatible Responses payloads."""

import base64
from collections.abc import AsyncIterable, Sequence
from typing import Any, assert_never

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
from pydantic_ai.tools import ToolDefinition

from ..openai_compatible_client import omit


def map_tool_definition(
    tool: ToolDefinition, *, strict_supported: bool
) -> dict[str, Any]:
    """Map a Pydantic AI tool definition to a Responses function tool."""
    mapped: dict[str, Any] = {
        "type": "function",
        "name": tool.name,
        "description": tool.description or "",
        "parameters": tool.parameters_json_schema,
    }
    if tool.strict and strict_supported:
        mapped["strict"] = tool.strict
    return mapped


def map_json_schema(output_object: Any, *, strict_supported: bool) -> dict[str, Any]:  # noqa: ANN401
    """Map a native structured-output schema to Responses text.format."""
    mapped: dict[str, Any] = {
        "type": "json_schema",
        "name": output_object.name or "final_result",
        "schema": output_object.json_schema,
    }
    if output_object.description:
        mapped["description"] = output_object.description
    if strict_supported and output_object.strict is not None:
        mapped["strict"] = output_object.strict
    return mapped


async def map_messages(
    model: Model[Any],
    messages: Sequence[ModelMessage],
    model_request_parameters: ModelRequestParameters,
) -> tuple[str | object, list[dict[str, Any]]]:
    """Map Pydantic AI messages to Responses input items."""
    mapped_messages: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            async for item in _map_model_request(model, message):
                mapped_messages.append(item)
        elif isinstance(message, ModelResponse):
            mapped_messages.extend(_map_model_response(model, message))
        else:
            assert_never(message)

    instruction_parts = (
        model._get_instruction_parts(messages, model_request_parameters) or []
    )
    instructions = "\n\n".join(part.content for part in instruction_parts)
    return instructions or omit, mapped_messages


def _system_role(model: Model[Any]) -> str:
    """Return the OpenAI-compatible system prompt role."""
    role = model.profile.get("openai_system_prompt_role")
    return role if isinstance(role, str) and role else "system"


async def _map_model_request(
    model: Model[Any], message: ModelRequest
) -> AsyncIterable[dict[str, Any]]:
    for part in message.parts:
        if isinstance(part, SystemPromptPart):
            yield {"role": _system_role(model), "content": part.content}
        elif isinstance(part, UserPromptPart):
            yield _map_user_prompt(model, part)
        elif isinstance(part, ToolReturnPart):
            call_id = _split_combined_tool_call_id(part.tool_call_id)[0]
            tool_text, tool_file_content = part.model_response_str_and_user_content()
            yield {
                "type": "function_call_output",
                "call_id": call_id,
                "output": tool_text,
            }
            if tool_file_content:
                yield _map_user_prompt(model, UserPromptPart(content=tool_file_content))
        elif isinstance(part, RetryPromptPart):
            if part.tool_name is None:
                yield {
                    "role": "user",
                    "content": [{"type": "input_text", "text": part.model_response()}],
                }
            else:
                call_id = _split_combined_tool_call_id(part.tool_call_id)[0]
                yield {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": part.model_response(),
                }
        else:
            assert_never(part)


def _map_model_response(
    model: Model[Any], message: ModelResponse
) -> list[dict[str, Any]]:
    """Map a prior model response to Responses input items."""
    mapped: list[dict[str, Any]] = []
    for item in message.parts:
        should_send_item_id = item.provider_name == model.system or (
            item.provider_name is None and message.provider_name == model.system
        )
        if isinstance(item, TextPart):
            if item.id and should_send_item_id:
                content: dict[str, Any] = {
                    "type": "output_text",
                    "text": item.content,
                    "annotations": [],
                }
                mapped_item: dict[str, Any] = {
                    "type": "message",
                    "role": "assistant",
                    "id": item.id,
                    "status": "completed",
                    "content": [content],
                }
                if item.provider_details and (
                    phase := item.provider_details.get("phase")
                ):
                    mapped_item["phase"] = phase
                mapped.append(mapped_item)
            else:
                mapped.append({"role": "assistant", "content": item.content})
        elif isinstance(item, ThinkingPart):
            mapped.extend(_map_thinking_part(model, item, should_send_item_id))
        elif isinstance(item, ToolCallPart):
            mapped.append(_map_tool_call(model, item, should_send_item_id))
        elif isinstance(
            item, NativeToolCallPart | NativeToolReturnPart | FilePart | CompactionPart
        ):
            pass
        else:
            assert_never(item)
    return mapped


def _map_thinking_part(
    model: Model[Any], item: ThinkingPart, should_send_item_id: bool
) -> list[dict[str, Any]]:
    raw_content = None
    if item.provider_name == model.system:
        raw_content = (item.provider_details or {}).get("raw_content")
    if item.id and (should_send_item_id or raw_content):
        mapped: dict[str, Any] = {
            "type": "reasoning",
            "id": item.id,
            "summary": [],
        }
        if item.signature and item.provider_name == model.system:
            mapped["encrypted_content"] = item.signature
        if item.content:
            mapped["summary"] = [{"type": "summary_text", "text": item.content}]
        if raw_content:
            mapped["content"] = [
                {"type": "reasoning_text", "text": text} for text in raw_content
            ]
        return [mapped]
    return [{"role": "assistant", "content": f"<think>\n{item.content}\n</think>"}]


def _map_tool_call(
    model: Model[Any], part: ToolCallPart, should_send_item_id: bool
) -> dict[str, Any]:
    """Map a prior tool call to Responses function-call history."""
    call_id, item_id = _split_combined_tool_call_id(part.tool_call_id)
    mapped: dict[str, Any] = {
        "type": "function_call",
        "name": part.tool_name,
        "arguments": part.args_as_json_str(),
        "call_id": call_id,
    }
    if should_send_item_id and (item_id or part.id):
        mapped["id"] = item_id or part.id
    if (
        part.provider_name == model.system
        and part.provider_details
        and (namespace := part.provider_details.get("namespace"))
    ):
        mapped["namespace"] = namespace
    return mapped


def _map_user_prompt(model: Model[Any], part: UserPromptPart) -> dict[str, Any]:
    """Map a user prompt part."""
    if isinstance(part.content, str):
        content: str | list[dict[str, Any]] = part.content
    else:
        mapped_content = [_map_content_item(model, item) for item in part.content]
        content = [item for item in mapped_content if item is not None]
    return {"role": "user", "content": content}


def _map_content_item(model: Model[Any], item: Any) -> dict[str, Any] | None:  # noqa: ANN401
    """Map one multimodal user content item."""
    if isinstance(item, str | TextContent):
        text = item if isinstance(item, str) else item.content
        return {"type": "input_text", "text": text}
    if isinstance(item, ImageUrl):
        mapped: dict[str, Any] = {
            "type": "input_image",
            "image_url": item.url,
            "detail": (item.vendor_metadata or {}).get("detail", "auto"),
        }
        return mapped
    if isinstance(item, BinaryContent):
        return _map_binary_content(item)
    if isinstance(item, DocumentUrl | AudioUrl):
        return {"type": "input_file", "file_url": item.url}
    if isinstance(item, UploadedFile):
        if item.provider_name != model.system:
            raise UserError(
                f"UploadedFile with provider_name={item.provider_name!r} cannot be "
                f"used with {model.system!r}."
            )
        return {"type": "input_file", "file_id": item.file_id}
    if isinstance(item, CachePoint):
        return None
    if isinstance(item, VideoUrl):
        raise UserError("Video URL inputs are not supported by this Responses adapter")
    assert_never(item)


def _map_binary_content(item: BinaryContent) -> dict[str, Any]:
    """Map binary content as image, file, or inline text."""
    if item.media_type.startswith("text/") or item.media_type in {
        "application/json",
        "application/xml",
        "application/yaml",
        "application/toml",
    }:
        return {
            "type": "input_text",
            "text": item.data.decode("utf-8", errors="replace"),
        }
    if item.is_image:
        return {
            "type": "input_image",
            "image_url": item.data_uri,
            "detail": (item.vendor_metadata or {}).get("detail", "auto"),
        }
    if item.is_document:
        data = f"data:{item.media_type};base64,{base64.b64encode(item.data).decode()}"
        return {
            "type": "input_file",
            "file_data": data,
            "filename": f"filename.{item.format}",
        }
    raise UserError(f"Unsupported binary content type: {item.media_type}")


def _split_combined_tool_call_id(combined_id: str) -> tuple[str, str | None]:
    if "|" in combined_id:
        call_id, item_id = combined_id.split("|", 1)
        return call_id, item_id
    return combined_id, None
