"""Test workspace-first diagnostics."""

from types import SimpleNamespace
from typing import cast

from custom_components.pydantic_ai_agent.const import (
    DOMAIN,
    SUBENTRY_TYPE_CONVERSATION,
)
from custom_components.pydantic_ai_agent.diagnostics import async_get_device_diagnostics
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from tests.components.pydantic_ai_agent.support.builders import (
    conversation_subentry_data,
    workspace_entry,
)


async def test_device_diagnostics_resolves_subentry_from_device_identifier(
    hass: HomeAssistant,
) -> None:
    """Test device diagnostics map compound device identifiers to subentries."""
    subentry_id = "conversation-1"
    entry = workspace_entry(
        (
            conversation_subentry_data(
                "provider-1:profile-1",
                subentry_id=subentry_id,
            ),
        )
    )
    entry.add_to_hass(hass)
    device = cast(
        dr.DeviceEntry,
        SimpleNamespace(
            identifiers={
                (DOMAIN, f"{entry.entry_id}:{SUBENTRY_TYPE_CONVERSATION}:{subentry_id}")
            }
        ),
    )

    diagnostics = await async_get_device_diagnostics(hass, entry, device)

    assert diagnostics["device"] == {"subentry_id": subentry_id}
    assert [item["subentry_id"] for item in diagnostics["subentries"]] == [subentry_id]
