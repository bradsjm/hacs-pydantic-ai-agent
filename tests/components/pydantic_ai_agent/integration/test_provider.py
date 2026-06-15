"""Provider integration tests for live discovery APIs."""

import pytest
from custom_components.pydantic_ai_agent.provider_validation import (
    async_list_provider_model_names,
)
from homeassistant.core import HomeAssistant

from .config import (
    ProviderIntegrationConfig,
)

pytestmark = [
    pytest.mark.provider_integration,
    pytest.mark.usefixtures("socket_enabled"),
]


async def test_provider_model_listing_succeeds(
    hass: HomeAssistant, provider_config: ProviderIntegrationConfig
) -> None:
    """Test configured credentials can list available provider models."""
    model_names = await async_list_provider_model_names(
        hass, provider_config.provider_data
    )

    assert model_names
    assert provider_config.model in model_names
