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
from homeassistant.const import UnitOfTime
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import EntityCategory
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
from .home_semantic.manager import HomeSemanticIndexManager, semantic_index_signal
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


@dataclass(frozen=True, kw_only=True)
class PydanticAISemanticSensorDescription(SensorEntityDescription):
    """Description for one workspace semantic index sensor."""

    value_fn: Callable[[HomeSemanticIndexManager | None], int | float | None]


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
        key="last_run_cache_read_tokens",
        name="Last run cache read tokens",
        icon="mdi:cached",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_cache_read_tokens"),
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
        key="last_run_input_cost",
        name="Last run input cost",
        icon="mdi:cash",
        native_unit_of_measurement="USD",
        device_class=SensorDeviceClass.MONETARY,
        suggested_display_precision=3,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_input_cost"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_run_output_cost",
        name="Last run output cost",
        icon="mdi:cash",
        native_unit_of_measurement="USD",
        device_class=SensorDeviceClass.MONETARY,
        suggested_display_precision=3,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_output_cost"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_run_cache_read_cost",
        name="Last run cache read cost",
        icon="mdi:cash-sync",
        native_unit_of_measurement="USD",
        device_class=SensorDeviceClass.MONETARY,
        suggested_display_precision=3,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_cache_read_cost"),
    ),
    PydanticAIMetricSensorDescription(
        key="last_run_total_cost",
        name="Last run total cost",
        icon="mdi:cash-multiple",
        native_unit_of_measurement="USD",
        device_class=SensorDeviceClass.MONETARY,
        suggested_display_precision=3,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "last_run_total_cost"),
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
        key="cumulative_cache_read_tokens",
        name="Cumulative cache read tokens",
        icon="mdi:cached",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda record: metric_value(record, "cumulative_cache_read_tokens"),
    ),
    PydanticAIMetricSensorDescription(
        key="cumulative_total_tokens",
        name="Cumulative total tokens",
        icon="mdi:counter",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=True,
        value_fn=lambda record: metric_value(record, "cumulative_total_tokens"),
    ),
    PydanticAIMetricSensorDescription(
        key="cumulative_input_cost",
        name="Cumulative input cost",
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
        name="Cumulative output cost",
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
        name="Cumulative cache read cost",
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
        name="Cumulative total cost",
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
        entity_registry_enabled_default=True,
        value_fn=lambda record: metric_value(record, "last_error_type"),
    ),
    PydanticAIMetricSensorDescription(
        key="consecutive_failures",
        name="Consecutive failures",
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
        name="Primary language model",
        icon="mdi:brain",
        value_fn=lambda entry, subentry: (
            primary_model_profile(entry, subentry).model_name
        ),
    ),
    PydanticAIConfigSensorDescription(
        key="mcp_servers_enabled",
        name="MCP servers enabled",
        icon="mdi:server-network-outline",
        native_unit_of_measurement="servers",
        state_class=SensorStateClass.MEASUREMENT,
        entity_registry_enabled_default=True,
        value_fn=lambda _entry, subentry: len(
            subentry.data.get(CONF_MCP_SERVER_IDS, [])
        ),
    ),
    PydanticAIConfigSensorDescription(
        key="skills_enabled",
        name="Skills enabled",
        icon="mdi:school-outline",
        native_unit_of_measurement="skills",
        state_class=SensorStateClass.MEASUREMENT,
        entity_registry_enabled_default=True,
        value_fn=lambda _entry, subentry: len(subentry.data.get(CONF_SKILLS, [])),
    ),
    PydanticAIConfigSensorDescription(
        key="structured_output_mode",
        name="Structured output mode",
        icon="mdi:code-json",
        subentry_types=(SUBENTRY_TYPE_AI_TASK,),
        entity_registry_enabled_default=True,
        value_fn=lambda _entry, subentry: structured_output_mode(
            subentry.data.get(CONF_OUTPUT_MODE)
        ),
    ),
)

SEMANTIC_SENSOR_DESCRIPTIONS: tuple[PydanticAISemanticSensorDescription, ...] = (
    PydanticAISemanticSensorDescription(
        key="semantic_index_generation",
        name="Semantic index generation",
        icon="mdi:counter",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda manager: None if manager is None else manager.generation,
    ),
    PydanticAISemanticSensorDescription(
        key="semantic_document_count",
        name="Semantic document count",
        icon="mdi:file-tree-outline",
        native_unit_of_measurement="documents",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda manager: None
        if manager is None or manager.index is None
        else len(manager.index.documents),
    ),
    PydanticAISemanticSensorDescription(
        key="semantic_last_refresh_duration",
        name="Semantic last refresh duration",
        icon="mdi:timer-outline",
        native_unit_of_measurement=UnitOfTime.SECONDS,
        device_class=SensorDeviceClass.DURATION,
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=3,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda manager: None
        if manager is None or manager.last_duration_ms is None
        else manager.last_duration_ms / 1000,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: PydanticAIAgentConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Pydantic AI Agent metric sensors."""
    async_add_entities(
        [
            PydanticAISemanticSensor(config_entry, description)
            for description in SEMANTIC_SENSOR_DESCRIPTIONS
        ]
    )
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

    @property
    def native_value(self) -> int | str | None:
        """Return the configured value."""
        return self.entity_description.value_fn(self.entry, self.subentry)


class PydanticAISemanticSensor(SensorEntity):
    """Workspace-level sensor for the Home Semantic Index."""

    _attr_has_entity_name = True

    entity_description: PydanticAISemanticSensorDescription

    def __init__(
        self,
        entry: PydanticAIAgentConfigEntry,
        description: PydanticAISemanticSensorDescription,
    ) -> None:
        """Initialize the semantic index sensor."""
        self.entry = entry
        self.entity_description = description
        self._attr_unique_id = f"{DOMAIN}_{entry.entry_id}_{description.key}"
        self._attr_device_info = _workspace_device_info(entry)

    async def async_added_to_hass(self) -> None:
        """Subscribe to semantic index updates."""
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                semantic_index_signal(self.entry.entry_id),
                self._handle_semantic_update,
            )
        )

    @property
    def native_value(self) -> int | float | None:
        """Return the current semantic index diagnostic value."""
        return self.entity_description.value_fn(self.entry.runtime_data.home_semantic)

    @property
    def extra_state_attributes(self) -> dict[str, object]:
        """Return compact semantic index status attributes."""
        manager = self.entry.runtime_data.home_semantic
        if manager is None:
            return {"ready": False, "status": "not_loaded"}
        return {
            "ready": manager.index is not None,
            "status": manager.status,
            "generation": manager.generation,
            "last_error_type": manager.last_error_type,
        }

    @callback
    def _handle_semantic_update(self) -> None:
        """Write updated semantic index sensor state."""
        self.async_write_ha_state()


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


def _workspace_device_info(entry: PydanticAIAgentConfigEntry) -> dr.DeviceInfo:
    """Return the workspace-level diagnostic device info."""
    return dr.DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name=entry.title,
        manufacturer="Pydantic AI",
        entry_type=dr.DeviceEntryType.SERVICE,
    )
