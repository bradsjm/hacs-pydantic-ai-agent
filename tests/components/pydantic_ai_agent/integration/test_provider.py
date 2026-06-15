"""Provider probe integration tests."""

import pytest
from custom_components.pydantic_ai_agent.provider_validation import async_probe_model
from homeassistant.core import HomeAssistant

from .config import (
    MODEL_PROFILE_ID,
    PROVIDER_INTEGRATION_TIMEOUT,
    ProviderIntegrationConfig,
)
from .entries import drain_stream_cleanup

pytestmark = [
    pytest.mark.provider_integration,
    pytest.mark.usefixtures("socket_enabled"),
]


async def test_provider_probe_succeeds(
    hass: HomeAssistant, provider_config: ProviderIntegrationConfig
) -> None:
    """Test configured credentials and model pass provider probing."""
    await async_probe_model(
        hass,
        provider_config.provider_data,
        provider_config.model,
        {"timeout": PROVIDER_INTEGRATION_TIMEOUT},
        profile_id=MODEL_PROFILE_ID,
    )
    await drain_stream_cleanup(hass)
