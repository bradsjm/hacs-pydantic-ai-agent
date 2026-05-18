"""Diagnostic sensors for Pydantic AI Agent metrics."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import UnitOfTime
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import PydanticAIAgentConfigEntry
from .const import (
    CONF_AGENT_NAME,
    CONF_AI_TASK_NAME,
    CONF_MCP_SERVER_IDS,
    CONF_OUTPUT_MODE,
    DOMAIN,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
)
from .metrics import AgentRunMetrics, metric_value, metrics_signal
from .entity import unique_id_for_subentry_entity
from .model_profiles import model_profile_chain
from .structured_output import structured_output_mode


@dataclass(frozen=True, kw_only=True)
class PydanticAIMetricSensorDescription(SensorEntityDescription):
    """Description for one Pydantic AI metric sensor."""

    value_fn: Callable[[AgentRunMetrics], int | float | str | None]


@dataclass(frozen=True, kw_only=True)
class PydanticAIConfigSensorDescription(SensorEntityDescription):
    """Description for one Pydantic AI configuration sensor."""

    value_fn: Callable[[PydanticAIAgentConfigEntry, ConfigSubentry], int | str | None]
    subentry_types: tuple[str, ...] = (
        SUBENTRY_TYPE_CONVERSATION,
        SUBENTRY_TYPE_AI_TASK,
    )


SENSOR_DESCRIPTIONS: tuple[PydanticAIMetricSensorDescription, ...] = (
    PydanticAIMetricSensorDescription(
        key="last_run_model_profile",
        name="Last run model profile",
        icon="mdi:brain",
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_model_profile"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_run_input_tokens",
        name="Last run input tokens",
        icon="mdi:calculator",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_input_tokens"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_run_output_tokens",
        name="Last run output tokens",
        icon="mdi:calculator",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_output_tokens"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_run_total_tokens",
        name="Last run total tokens",
        icon="mdi:counter",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_total_tokens"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_run_model_request_count",
        name="Last run model request count",
        icon="mdi:api",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_model_request_count"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_run_tool_use_count",
        name="Last run tool use count",
        icon="mdi:toolbox-outline",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_tool_use_count"),
    ),
    PydanticAIMetricSensorDescription(
        key="cumulative_input_tokens",
        name="Cumulative input tokens",
        icon="mdi:calculator",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "cumulative_input_tokens"),
    ),
    PydanticAIMetricSensorDescription(
        key="cumulative_output_tokens",
        name="Cumulative output tokens",
        icon="mdi:calculator",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "cumulative_output_tokens"),
    ),
    PydanticAIMetricSensorDescription(
        key="cumulative_total_tokens",
        name="Cumulative total tokens",
        icon="mdi:counter",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "cumulative_total_tokens"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_run_duration",
        name="Last run duration",
        native_unit_of_measurement=UnitOfTime.SECONDS,
        device_class=SensorDeviceClass.DURATION,
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=3,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_duration"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_error_type",
        name="Last error type",
        icon="mdi:alert-circle-outline",
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_error_type"),
    ),
    PydanticAIMetricSensorDescription(
        key="consecutive_failures",
        name="Consecutive failures",
        icon="mdi:alert-circle-outline",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "consecutive_failures"),
    ),
)

CONFIG_SENSOR_DESCRIPTIONS: tuple[PydanticAIConfigSensorDescription, ...] = (
    PydanticAIConfigSensorDescription(
        key="primary_language_model",
        name="Primary language model",
        icon="mdi:brain",
        value_fn=lambda entry, subentry: (
            model_profile_chain(entry, subentry)[0].model_name
        ),
    ),
    PydanticAIConfigSensorDescription(
        key="mcp_servers_enabled",
        name="MCP servers enabled",
        icon="mdi:server-network-outline",
        native_unit_of_measurement="servers",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda _entry, subentry: len(
            subentry.data.get(CONF_MCP_SERVER_IDS, [])
        ),
    ),
    PydanticAIConfigSensorDescription(
        key="structured_output_mode",
        name="Structured output mode",
        icon="mdi:code-json",
        subentry_types=(SUBENTRY_TYPE_AI_TASK,),
        value_fn=lambda _entry, subentry: structured_output_mode(
            subentry.data.get(CONF_OUTPUT_MODE)
        ),
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: PydanticAIAgentConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Pydantic AI Agent metric sensors."""
    for subentry in _agent_subentries(config_entry):
        async_add_entities(
            [
                PydanticAIMetricSensor(config_entry, subentry, description)
                for description in SENSOR_DESCRIPTIONS
            ],
            config_subentry_id=subentry.subentry_id,
        )
        async_add_entities(
            [
                PydanticAIConfigSensor(config_entry, subentry, description)
                for description in CONFIG_SENSOR_DESCRIPTIONS
                if subentry.subentry_type in description.subentry_types
            ],
            config_subentry_id=subentry.subentry_id,
        )


class PydanticAIMetricSensor(SensorEntity):
    """Sensor that exposes one native Pydantic AI runtime metric."""

    _attr_has_entity_name = True

    entity_description: PydanticAIMetricSensorDescription

    def __init__(
        self,
        entry: PydanticAIAgentConfigEntry,
        subentry: ConfigSubentry,
        description: PydanticAIMetricSensorDescription,
    ) -> None:
        """Initialize the metric sensor."""
        self.entry = entry
        self.subentry = subentry
        self.entity_description = description
        profiles = model_profile_chain(entry, subentry)
        self._attr_unique_id = unique_id_for_subentry_entity(
            subentry, description.key
        )
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
    def native_value(self) -> int | float | str | None:
        """Return the current metric value."""
        record = self.entry.runtime_data.metrics.record_for(self.subentry.subentry_id)
        return self.entity_description.value_fn(record)

    @callback
    def _handle_metrics_update(self) -> None:
        """Write the updated metric state."""
        self.async_write_ha_state()


class PydanticAIConfigSensor(SensorEntity):
    """Sensor that exposes one static Pydantic AI configuration value."""

    _attr_has_entity_name = True

    entity_description: PydanticAIConfigSensorDescription

    def __init__(
        self,
        entry: PydanticAIAgentConfigEntry,
        subentry: ConfigSubentry,
        description: PydanticAIConfigSensorDescription,
    ) -> None:
        """Initialize the configuration sensor."""
        self.entry = entry
        self.subentry = subentry
        self.entity_description = description
        profiles = model_profile_chain(entry, subentry)
        self._attr_unique_id = unique_id_for_subentry_entity(
            subentry, description.key
        )
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=_subentry_name(subentry),
            manufacturer="Pydantic AI",
            model=profiles[0].model_name,
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def native_value(self) -> int | str | None:
        """Return the configured value."""
        return self.entity_description.value_fn(self.entry, self.subentry)


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
