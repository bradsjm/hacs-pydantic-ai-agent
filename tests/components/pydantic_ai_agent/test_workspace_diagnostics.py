"""Test workspace-first diagnostics."""

from types import SimpleNamespace
from typing import cast

from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_PRIMARY_MODEL_REF,
    DOMAIN,
    SUBENTRY_TYPE_CONVERSATION,
)
from custom_components.pydantic_ai_agent.diagnostics import async_get_device_diagnostics


async def test_device_diagnostics_resolves_subentry_from_device_identifier(
    hass: HomeAssistant,
) -> None:
    """Test device diagnostics map compound device identifiers to subentries."""
    subentry_id = "conversation-1"
    entry = MockConfigEntry(
        version=2,
        minor_version=0,
        domain=DOMAIN,
        title="Workspace",
        data={CONF_NAME: "Workspace"},
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "subentry_id": subentry_id,
                "data": {
                    CONF_AGENT_NAME: "Kitchen Agent",
                    CONF_PRIMARY_MODEL_REF: "provider-1:profile-1",
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
            },
        ),
        unique_id=None,
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
    assert [item["subentry_id"] for item in diagnostics["subentries"]] == [
        subentry_id
    ]
