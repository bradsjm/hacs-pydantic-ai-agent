"""Low-level streaming provider integration tests."""

import pytest
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_stream
from pydantic_ai.settings import ModelSettings

from homeassistant.core import HomeAssistant

from custom_components.pydantic_ai_agent.provider import (
    openai_compatible_completions_model_from_config,
)

from .capture import append_text_event
from .config import (
    PROVIDER_INTEGRATION_TIMEOUT,
    STREAM_SENTINEL,
    ProviderIntegrationConfig,
)
from .entries import drain_stream_cleanup

pytestmark = [
    pytest.mark.provider_integration,
    pytest.mark.usefixtures("socket_enabled"),
]


async def test_provider_stream_events_include_text(
    hass: HomeAssistant, provider_config: ProviderIntegrationConfig
) -> None:
    """Test the provider emits usable Pydantic AI text stream events."""
    model = openai_compatible_completions_model_from_config(
        hass, provider_config.provider_data, provider_config.model
    )
    text_parts: list[str] = []
    event_count = 0

    async with model_request_stream(
        model,
        [
            ModelRequest.user_text_prompt(
                f"Reply with exactly {STREAM_SENTINEL}. No punctuation."
            )
        ],
        model_settings=ModelSettings(timeout=PROVIDER_INTEGRATION_TIMEOUT),
    ) as stream:
        async for event in stream:
            event_count += 1
            append_text_event(text_parts, event)

    await drain_stream_cleanup(hass)
    assert event_count > 0
    assert STREAM_SENTINEL in "".join(text_parts)
