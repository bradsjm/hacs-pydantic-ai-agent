"""Tests for multimodal tool-result sentinel conversion."""

import base64

from custom_components.pydantic_ai_agent.agent.multimodal_tool_result import (
    normalize_multimodal_tool_result,
    serialize_multimodal_tool_result,
)
from pydantic_ai import BinaryContent


def _encoded(data: bytes) -> str:
    """Return base64-encoded test data."""
    return base64.b64encode(data).decode()


def test_normalize_non_sentinel_unchanged() -> None:
    """Unknown values are returned unchanged."""
    value = {"_type": "ordinary_result", "text": "hello"}

    assert normalize_multimodal_tool_result(value) is value


def test_normalize_valid_sentinel_returns_text_and_binary_content() -> None:
    """Valid inline image attachments become BinaryContent parts."""
    result = normalize_multimodal_tool_result(
        {
            "_type": "ha_multimodal_tool_result",
            "text": "camera snapshot",
            "attachments": [
                {
                    "kind": "inline_image",
                    "mime_type": "image/png",
                    "base64": _encoded(b"png-data"),
                    "detail": "high",
                },
                {
                    "kind": "inline_image",
                    "mime_type": "image/webp",
                    "base64": _encoded(b"webp-data"),
                },
            ],
        }
    )

    assert isinstance(result, list)
    assert result[0] == "camera snapshot"
    assert result[1] == BinaryContent(
        data=b"png-data",
        media_type="image/png",
        vendor_metadata={"detail": "high"},
    )
    assert result[2] == BinaryContent(data=b"webp-data", media_type="image/webp")


def test_normalize_invalid_attachment_falls_back_to_text() -> None:
    """Malformed or unsupported attachments degrade to text-only results."""
    for attachment in (
        {
            "kind": "inline_image",
            "mime_type": "image/gif",
            "base64": _encoded(b"gif"),
        },
        {"kind": "inline_image", "mime_type": "image/png", "base64": "not-base64"},
    ):
        assert (
            normalize_multimodal_tool_result(
                {
                    "_type": "ha_multimodal_tool_result",
                    "text": "fallback text",
                    "attachments": [attachment],
                }
            )
            == "fallback text"
        )


def test_normalize_invalid_without_text_uses_default_text() -> None:
    """Invalid sentinels with no text still produce a stable text fallback."""
    assert (
        normalize_multimodal_tool_result(
            {"_type": "ha_multimodal_tool_result", "attachments": "bad"}
        )
        == "Tool returned image attachments."
    )


def test_serialize_binary_content_to_sentinel() -> None:
    """Allowed image BinaryContent values serialize back to HA's sentinel."""
    result = serialize_multimodal_tool_result(
        [
            "first text",
            "second text",
            BinaryContent(
                data=b"jpeg-data",
                media_type="image/jpeg",
                vendor_metadata={"detail": "low"},
            ),
            BinaryContent(data=b"webp-data", media_type="image/webp"),
        ]
    )

    assert result == {
        "_type": "ha_multimodal_tool_result",
        "text": "first text\n\nsecond text",
        "attachments": [
            {
                "kind": "inline_image",
                "mime_type": "image/jpeg",
                "base64": _encoded(b"jpeg-data"),
                "detail": "low",
            },
            {
                "kind": "inline_image",
                "mime_type": "image/webp",
                "base64": _encoded(b"webp-data"),
            },
        ],
    }


def test_serialize_unsupported_list_unchanged() -> None:
    """Lists containing unsupported parts are not serialized."""
    value = ["text", BinaryContent(data=b"gif", media_type="image/gif")]

    assert serialize_multimodal_tool_result(value) is value
