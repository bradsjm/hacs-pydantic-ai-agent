"""AI task provider integration tests."""

import pytest
import voluptuous as vol
from homeassistant.components import ai_task
from homeassistant.core import HomeAssistant

from .config import (
    AI_TASK_SENTINEL,
    AI_TASK_STRUCTURED_SENTINEL,
    ProviderIntegrationConfig,
)
from .entries import ai_task_entity_id, drain_stream_cleanup

pytestmark = [
    pytest.mark.provider_integration,
    pytest.mark.usefixtures("socket_enabled"),
]


async def test_ai_task_plain_generation(
    hass: HomeAssistant, provider_config: ProviderIntegrationConfig
) -> None:
    """Test a live provider can generate plain AI task data."""
    entity_id = await ai_task_entity_id(hass, provider_config)

    result = await ai_task.async_generate_data(
        hass,
        task_name="Integration plain task",
        entity_id=entity_id,
        instructions=f"Return exactly {AI_TASK_SENTINEL}. No punctuation.",
    )

    await drain_stream_cleanup(hass)
    assert AI_TASK_SENTINEL in str(result.data)


async def test_ai_task_structured_generation(
    hass: HomeAssistant,
    provider_config: ProviderIntegrationConfig,
) -> None:
    """Test a live provider can generate schema-validated AI task data."""
    entity_id = await ai_task_entity_id(hass, provider_config)

    result = await ai_task.async_generate_data(
        hass,
        task_name="Integration structured task",
        entity_id=entity_id,
        instructions=(
            f"Generate data where result is exactly {AI_TASK_STRUCTURED_SENTINEL}."
        ),
        structure=vol.Schema({vol.Required("result"): str}),
    )

    await drain_stream_cleanup(hass)
    assert isinstance(result.data, dict)
    assert result.data["result"] == AI_TASK_STRUCTURED_SENTINEL
