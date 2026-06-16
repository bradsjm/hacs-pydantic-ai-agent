"""Helpers for Home Assistant multimodal tool-result sentinels."""

import base64
import binascii
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic_ai import BinaryContent

_MULTIMODAL_TOOL_RESULT_TYPE = "ha_multimodal_tool_result"
_INLINE_IMAGE_KIND = "inline_image"
_ALLOWED_INLINE_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}
_MAX_ATTACHMENTS = 4
_MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024
_MAX_TOTAL_ATTACHMENT_BYTES = 10 * 1024 * 1024
_DEFAULT_TEXT = "Tool returned image attachments."


def normalize_multimodal_tool_result(value: object) -> object:
    """Convert a recognized HA multimodal sentinel into model-facing content."""
    if (sentinel := _as_multimodal_tool_result(value)) is None:
        return value

    text = _sentinel_text(sentinel)
    attachments = sentinel.get("attachments")
    if not isinstance(attachments, Sequence) or isinstance(attachments, str | bytes):
        return text or _DEFAULT_TEXT
    if not attachments or len(attachments) > _MAX_ATTACHMENTS:
        return text or _DEFAULT_TEXT

    total_bytes = 0
    content: list[object] = [text or _DEFAULT_TEXT]
    for attachment in attachments:
        binary_content = _binary_content_from_attachment(attachment, total_bytes)
        if binary_content is None:
            return text or _DEFAULT_TEXT
        total_bytes += len(binary_content.data)
        content.append(binary_content)
    return content


def serialize_multimodal_tool_result(value: object) -> object:
    """Serialize model-facing multimodal content back to the HA sentinel."""
    if not isinstance(value, list):
        return value

    text_parts: list[str] = []
    attachments: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, str):
            if item:
                text_parts.append(item)
            continue
        if not isinstance(item, BinaryContent) or (
            item.media_type not in _ALLOWED_INLINE_IMAGE_MIME_TYPES
        ):
            return value
        attachment: dict[str, str] = {
            "kind": _INLINE_IMAGE_KIND,
            "mime_type": item.media_type,
            "base64": base64.b64encode(item.data).decode(),
        }
        if isinstance(item.vendor_metadata, Mapping) and isinstance(
            detail := item.vendor_metadata.get("detail"), str
        ):
            attachment["detail"] = detail
        attachments.append(attachment)

    if not attachments:
        return value
    return {
        "_type": _MULTIMODAL_TOOL_RESULT_TYPE,
        "text": "\n\n".join(text_parts),
        "attachments": attachments,
    }


def _as_multimodal_tool_result(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    if value.get("_type") != _MULTIMODAL_TOOL_RESULT_TYPE:
        return None
    return value


def _sentinel_text(value: Mapping[str, object]) -> str:
    text = value.get("text")
    return text if isinstance(text, str) else ""


def _binary_content_from_attachment(
    value: object, total_bytes: int
) -> BinaryContent | None:
    if not isinstance(value, Mapping):
        return None
    if value.get("kind") != _INLINE_IMAGE_KIND:
        return None
    mime_type = value.get("mime_type")
    encoded = value.get("base64")
    if not isinstance(mime_type, str) or (
        mime_type not in _ALLOWED_INLINE_IMAGE_MIME_TYPES
    ):
        return None
    if not isinstance(encoded, str):
        return None
    try:
        data = base64.b64decode(encoded, validate=True)
    except binascii.Error:
        return None
    if not data or len(data) > _MAX_ATTACHMENT_BYTES:
        return None
    if total_bytes + len(data) > _MAX_TOTAL_ATTACHMENT_BYTES:
        return None

    vendor_metadata: dict[str, Any] | None = None
    if isinstance(detail := value.get("detail"), str):
        vendor_metadata = {"detail": detail}
    return BinaryContent(
        data=data,
        media_type=mime_type,
        vendor_metadata=vendor_metadata,
    )
