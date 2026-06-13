"""Diagnostic sensors for Pydantic AI Agent metrics."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import EntityCategory, UnitOfTime
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import PydanticAIAgentConfigEntry
from .agent_subentries import ValidAgentSubentry, iter_valid_agent_subentries
from .const import (
    CONF_AGENT_NAME,
    CONF_AI_TASK_NAME,
    CONF_MCP_SERVER_IDS,
    CONF_OUTPUT_MODE,
    CONF_SKILLS,
    DOMAIN,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
)
from .entity import device_identifier_for_subentry, unique_id_for_subentry_entity
from .metrics import AgentRunMetrics, metric_value, metrics_signal
from .model_profiles import ModelProfile, primary_model_profile
from .structured_output import structured_output_mode


@dataclass(frozen=True, kw_only=True)
class PydanticAIMetricSensorDescription(SensorEntityDescription):
    """Description for one Pydantic AI metric sensor."""

    value_fn: Callable[[AgentRunMetrics], int | float | str | None]
    entity_registry_enabled_default: bool = False


@dataclass(frozen=True, kw_only=True)
class PydanticAIConfigSensorDescription(SensorEntityDescription):
    """Description for one Pydantic AI configuration sensor."""

    value_fn: Callable[[PydanticAIAgentConfigEntry, ConfigSubentry], int | str | None]
    entity_registry_enabled_default: bool = False
    subentry_types: tuple[str, ...] = (
        SUBENTRY_TYPE_CONVERSATION,
        SUBENTRY_TYPE_AI_TASK,
    )


SENSOR_DESCRIPTIONS: tuple[PydanticAIMetricSensorDescription, ...] = (
    PydanticAIMetricSensorDescription(
        key="last_run_model_profile",
        translation_key="last_run_model_profile",
        icon="mdi:brain",
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_model_profile"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_run_input_tokens",
        translation_key="last_run_input_tokens",
        icon="mdi:calculator",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_input_tokens"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_run_output_tokens",
        translation_key="last_run_output_tokens",
        icon="mdi:calculator",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_output_tokens"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_run_cache_read_tokens",
        translation_key="last_run_cache_read_tokens",
        icon="mdi:cached",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_cache_read_tokens"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_run_total_tokens",
        translation_key="last_run_total_tokens",
        icon="mdi:counter",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_total_tokens"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_run_input_cost",
        translation_key="last_run_input_cost",
        icon="mdi:cash",
        native_unit_of_measurement="USD",
        device_class=SensorDeviceClass.MONETARY,
        suggested_display_precision=3,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_input_cost"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_run_output_cost",
        translation_key="last_run_output_cost",
        icon="mdi:cash",
        native_unit_of_measurement="USD",
        device_class=SensorDeviceClass.MONETARY,
        suggested_display_precision=3,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_output_cost"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_run_cache_read_cost",
        translation_key="last_run_cache_read_cost",
        icon="mdi:cash-sync",
        native_unit_of_measurement="USD",
        device_class=SensorDeviceClass.MONETARY,
        suggested_display_precision=3,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_cache_read_cost"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_run_total_cost",
        translation_key="last_run_total_cost",
        icon="mdi:cash-multiple",
        native_unit_of_measurement="USD",
        device_class=SensorDeviceClass.MONETARY,
        suggested_display_precision=3,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_total_cost"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_run_model_request_count",
        translation_key="last_run_model_request_count",
        icon="mdi:api",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_model_request_count"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_run_tool_use_count",
        translation_key="last_run_tool_use_count",
        icon="mdi:toolbox-outline",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_tool_use_count"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_mcp_tool_call",
        translation_key="last_mcp_tool_call",
        icon="mdi:server-network",
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=True,
        value_fn=lambda record: metric_value(record, "last_mcp_tool_call"),
    ),
    PydanticAIMetricSensorDescription(
        key="cumulative_input_tokens",
        translation_key="cumulative_input_tokens",
        icon="mdi:calculator",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "cumulative_input_tokens"),
    ),
    PydanticAIMetricSensorDescription(
        key="cumulative_output_tokens",
        translation_key="cumulative_output_tokens",
        icon="mdi:calculator",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "cumulative_output_tokens"),
    ),
    PydanticAIMetricSensorDescription(
        key="cumulative_cache_read_tokens",
        translation_key="cumulative_cache_read_tokens",
        icon="mdi:cached",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "cumulative_cache_read_tokens"),
    ),
    PydanticAIMetricSensorDescription(
        key="cumulative_total_tokens",
        translation_key="cumulative_total_tokens",
        icon="mdi:counter",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=True,
        value_fn=lambda record: metric_value(record, "cumulative_total_tokens"),
    ),
    PydanticAIMetricSensorDescription(
        key="cumulative_input_cost",
        translation_key="cumulative_input_cost",
        icon="mdi:cash",
        native_unit_of_measurement="USD",
        device_class=SensorDeviceClass.MONETARY,
        state_class=SensorStateClass.TOTAL,
        suggested_display_precision=3,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "cumulative_input_cost"),
    ),
    PydanticAIMetricSensorDescription(
        key="cumulative_output_cost",
        translation_key="cumulative_output_cost",
        icon="mdi:cash",
        native_unit_of_measurement="USD",
        device_class=SensorDeviceClass.MONETARY,
        state_class=SensorStateClass.TOTAL,
        suggested_display_precision=3,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "cumulative_output_cost"),
    ),
    PydanticAIMetricSensorDescription(
        key="cumulative_cache_read_cost",
        translation_key="cumulative_cache_read_cost",
        icon="mdi:cash-sync",
        native_unit_of_measurement="USD",
        device_class=SensorDeviceClass.MONETARY,
        state_class=SensorStateClass.TOTAL,
        suggested_display_precision=3,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "cumulative_cache_read_cost"),
    ),
    PydanticAIMetricSensorDescription(
        key="cumulative_total_cost",
        translation_key="cumulative_total_cost",
        icon="mdi:cash-multiple",
        native_unit_of_measurement="USD",
        device_class=SensorDeviceClass.MONETARY,
        state_class=SensorStateClass.TOTAL,
        suggested_display_precision=3,
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=True,
        value_fn=lambda record: metric_value(record, "cumulative_total_cost"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_run_duration",
        translation_key="last_run_duration",
        native_unit_of_measurement=UnitOfTime.SECONDS,
        device_class=SensorDeviceClass.DURATION,
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=3,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_duration"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_error_type",
        translation_key="last_error_type",
        icon="mdi:alert-circle-outline",
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=True,
        value_fn=lambda record: metric_value(record, "last_error_type"),
    ),
    PydanticAIMetricSensorDescription(
        key="consecutive_failures",
        translation_key="consecutive_failures",
        icon="mdi:alert-circle-outline",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=True,
        value_fn=lambda record: metric_value(record, "consecutive_failures"),
    ),
)

CONFIG_SENSOR_DESCRIPTIONS: tuple[PydanticAIConfigSensorDescription, ...] = (
    PydanticAIConfigSensorDescription(
        key="primary_language_model",
        translation_key="primary_language_model",
        icon="mdi:brain",
        value_fn=lambda entry, subentry: (
            primary_model_profile(entry, subentry).model_name
        ),
    ),
    PydanticAIConfigSensorDescription(
        key="skills_enabled",
        translation_key="skills_enabled",
        icon="mdi:school-outline",
        native_unit_of_measurement="skills",
        state_class=SensorStateClass.MEASUREMENT,
        entity_registry_enabled_default=True,
        value_fn=lambda _entry, subentry: len(subentry.data.get(CONF_SKILLS, [])),
    ),
    PydanticAIConfigSensorDescription(
        key="mcp_servers_enabled",
        translation_key="mcp_servers_enabled",
        icon="mdi:server-network-outline",
        native_unit_of_measurement="servers",
        state_class=SensorStateClass.MEASUREMENT,
        entity_registry_enabled_default=True,
        value_fn=lambda _entry, subentry: len(
            subentry.data.get(CONF_MCP_SERVER_IDS, [])
        ),
    ),
    PydanticAIConfigSensorDescription(
        key="structured_output_mode",
        translation_key="structured_output_mode",
        icon="mdi:code-json",
        subentry_types=(SUBENTRY_TYPE_AI_TASK,),
        entity_registry_enabled_default=True,
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
    for valid in _agent_subentries(config_entry):
        subentry = valid.subentry
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

    entity_description: SensorEntityDescription

    def __init__(
        self,
        entry: PydanticAIAgentConfigEntry,
        subentry: ConfigSubentry,
        description: PydanticAIMetricSensorDescription,
    ) -> None:
        """Initialize the metric sensor."""
        self.entry = entry
        self.subentry = subentry
        self._description = description
        self.entity_description = description
        self._attr_entity_registry_enabled_default = (
            description.entity_registry_enabled_default
        )
        profile = primary_model_profile(entry, subentry)
        self._attr_unique_id = unique_id_for_subentry_entity(
            entry, subentry, description.key
        )
        self._attr_device_info = dr.DeviceInfo(
            identifiers={device_identifier_for_subentry(entry, subentry)},
            name=_subentry_name(subentry),
            manufacturer="Pydantic AI",
            model=profile.model_name,
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        self._attr_native_value = self._description.value_fn(
            self.entry.runtime_data.metrics.record_for(self.subentry.subentry_id)
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

    @callback
    def _handle_metrics_update(self) -> None:
        """Write the updated metric state."""
        self._attr_native_value = self._description.value_fn(
            self.entry.runtime_data.metrics.record_for(self.subentry.subentry_id)
        )
        self.async_write_ha_state()


class PydanticAIConfigSensor(SensorEntity):
    """Sensor that exposes one static Pydantic AI configuration value."""

    _attr_has_entity_name = True

    entity_description: SensorEntityDescription

    def __init__(
        self,
        entry: PydanticAIAgentConfigEntry,
        subentry: ConfigSubentry,
        description: PydanticAIConfigSensorDescription,
    ) -> None:
        """Initialize the configuration sensor."""
        self.entry = entry
        self.subentry = subentry
        self._description = description
        self.entity_description = description
        self._attr_entity_registry_enabled_default = (
            description.entity_registry_enabled_default
        )
        profile = primary_model_profile(entry, subentry)
        self._attr_unique_id = unique_id_for_subentry_entity(
            entry, subentry, description.key
        )
        self._attr_device_info = dr.DeviceInfo(
            identifiers={device_identifier_for_subentry(entry, subentry)},
            name=_subentry_name(subentry),
            manufacturer="Pydantic AI",
            model=profile.model_name,
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        self._attr_native_value = self._description.value_fn(self.entry, self.subentry)


def _agent_subentries(
    entry: PydanticAIAgentConfigEntry,
) -> Iterator[ValidAgentSubentry[ModelProfile]]:
    """Yield configured conversation and AI task subentries with valid models."""
    yield from iter_valid_agent_subentries(
        entry,
        subentry_type=SUBENTRY_TYPE_CONVERSATION,
        platform=DOMAIN,
        resolver=primary_model_profile,
    )
    yield from iter_valid_agent_subentries(
        entry,
        subentry_type=SUBENTRY_TYPE_AI_TASK,
        platform=DOMAIN,
        resolver=primary_model_profile,
    )


def _subentry_name(subentry: ConfigSubentry) -> str:
    """Return the service device name for a subentry."""
    if subentry.subentry_type == SUBENTRY_TYPE_CONVERSATION:
        return str(subentry.data[CONF_AGENT_NAME])
    return str(subentry.data.get(CONF_AI_TASK_NAME, subentry.title))
