"""Diagnostic binary sensors for Pydantic AI Agent metrics."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from homeassistant.components.binary_sensor import (
    BinarySensorEntity,
    BinarySensorEntityDescription,
)
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import PydanticAIAgentConfigEntry
from .const import (
    CONF_AGENT_NAME,
    CONF_AI_TASK_NAME,
    DOMAIN,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
)
from .metrics import AgentRunMetrics, metric_bool, metrics_signal
from .model_profiles import model_profile_chain


@dataclass(frozen=True, kw_only=True)
class PydanticAIMetricBinarySensorDescription(BinarySensorEntityDescription):
    """Description for one Pydantic AI metric binary sensor."""

    value_fn: Callable[[AgentRunMetrics], bool | None]


BINARY_SENSOR_DESCRIPTIONS: tuple[PydanticAIMetricBinarySensorDescription, ...] = (
    PydanticAIMetricBinarySensorDescription(
        key="provider_healthy",
        name="Provider healthy",
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_bool(record, "provider_healthy"),
    ),
    PydanticAIMetricBinarySensorDescription(
        key="last_run_succeeded",
        name="Last run succeeded",
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_bool(record, "last_run_succeeded"),
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: PydanticAIAgentConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Pydantic AI Agent metric binary sensors."""
    for subentry in _agent_subentries(config_entry):
        async_add_entities(
            [
                PydanticAIMetricBinarySensor(config_entry, subentry, description)
                for description in BINARY_SENSOR_DESCRIPTIONS
            ],
            config_subentry_id=subentry.subentry_id,
        )


class PydanticAIMetricBinarySensor(BinarySensorEntity):
    """Binary sensor that exposes one Pydantic AI health metric."""

    _attr_has_entity_name = True

    entity_description: PydanticAIMetricBinarySensorDescription

    def __init__(
        self,
        entry: PydanticAIAgentConfigEntry,
        subentry: ConfigSubentry,
        description: PydanticAIMetricBinarySensorDescription,
    ) -> None:
        """Initialize the metric binary sensor."""
        self.entry = entry
        self.subentry = subentry
        self.entity_description = description
        profiles = model_profile_chain(entry, subentry)
        self._attr_unique_id = f"{subentry.subentry_id}_{description.key}"
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=_subentry_name(subentry),
            manufacturer="Pydantic AI",
            model=profiles[0].model_name,
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    async def async_added_to_hass(self) -> None:
        """Subscribe to metric updates."""
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                metrics_signal(self.entry.entry_id, self.subentry.subentry_id),
                self._handle_metrics_update,
            )
        )

    @property
    def is_on(self) -> bool | None:
        """Return the current binary metric value."""
        record = self.entry.runtime_data.metrics.record_for(self.subentry.subentry_id)
        return self.entity_description.value_fn(record)

    @callback
    def _handle_metrics_update(self) -> None:
        """Write the updated metric state."""
        self.async_write_ha_state()


def _agent_subentries(entry: PydanticAIAgentConfigEntry) -> Iterable[ConfigSubentry]:
    """Yield configured conversation and AI task subentries with valid models."""
    for subentry in entry.subentries.values():
        if subentry.subentry_type not in (
            SUBENTRY_TYPE_CONVERSATION,
            SUBENTRY_TYPE_AI_TASK,
        ):
            continue
        try:
            model_profile_chain(entry, subentry)
        except Exception:
            continue
        yield subentry


def _subentry_name(subentry: ConfigSubentry) -> str:
    """Return the service device name for a subentry."""
    if subentry.subentry_type == SUBENTRY_TYPE_CONVERSATION:
        return str(subentry.data[CONF_AGENT_NAME])
    return str(subentry.data.get(CONF_AI_TASK_NAME, subentry.title))
