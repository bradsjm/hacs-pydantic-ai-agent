"""Usage mapping helpers."""

from typing import Any

from pydantic_ai.usage import RequestUsage

from ..openai_compatible_client import ChatCompletion, ChatCompletionChunk, Response
from ..openai_compatible_client._types import CompletionUsage, ResponseUsage


def map_usage(response: ChatCompletion | ChatCompletionChunk | Response) -> RequestUsage:
    """Map OpenAI-compatible usage to Pydantic AI request usage."""
    raw_usage = response.usage
    if raw_usage is None:
        return RequestUsage()
    if isinstance(raw_usage, ResponseUsage):
        return RequestUsage(
            input_tokens=raw_usage.input_tokens or 0,
            output_tokens=raw_usage.output_tokens or 0,
            details=_usage_details(raw_usage),
        )
    details = _usage_details(raw_usage)
    return RequestUsage(
        input_tokens=raw_usage.prompt_tokens or 0,
        output_tokens=raw_usage.completion_tokens or 0,
        cache_read_tokens=(
            raw_usage.prompt_tokens_details.cached_tokens
            if raw_usage.prompt_tokens_details is not None
            and raw_usage.prompt_tokens_details.cached_tokens is not None
            else 0
        ),
        input_audio_tokens=(
            raw_usage.prompt_tokens_details.audio_tokens
            if raw_usage.prompt_tokens_details is not None
            and raw_usage.prompt_tokens_details.audio_tokens is not None
            else 0
        ),
        output_audio_tokens=(
            raw_usage.completion_tokens_details.audio_tokens
            if raw_usage.completion_tokens_details is not None
            and raw_usage.completion_tokens_details.audio_tokens is not None
            else 0
        ),
        details=details,
    )


def _usage_details(raw_usage: CompletionUsage | ResponseUsage) -> dict[str, int]:
    """Return extra integer usage details."""
    usage_data = raw_usage.model_dump(exclude_none=True)
    details: dict[str, int] = {}
    for key, value in usage_data.items():
        if key in {
            "prompt_tokens",
            "completion_tokens",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        }:
            continue
        if isinstance(value, int):
            details[key] = value
        elif isinstance(value, dict):
            _flatten_details(details, key, value)
    return details


def _flatten_details(
    details: dict[str, int], prefix: str, data: dict[str, Any]
) -> None:
    """Flatten nested integer detail values."""
    for key, value in data.items():
        if isinstance(value, int):
            details[f"{prefix}.{key}"] = value
