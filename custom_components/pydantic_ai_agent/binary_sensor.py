"""Diagnostic binary sensors for Pydantic AI Agent metrics."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from homeassistant.components.binary_sensor import (
    BinarySensorEntity,
    BinarySensorEntityDescription,
)
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import PydanticAIAgentConfigEntry
from .agent_subentries import ValidAgentSubentry, iter_valid_agent_subentries
from .const import (
    CONF_AGENT_NAME,
    CONF_AI_TASK_NAME,
    CONF_TODO_LIST_ENTITY_ID,
    CONF_VIRTUAL_WORKSPACE_ENABLED,
    CONF_WEB_FETCH_ENABLED,
    DOMAIN,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
)
from .entity import device_identifier_for_subentry, unique_id_for_subentry_entity
from .metrics import AgentRunMetrics, metric_bool, metrics_signal
from .model_profiles import ModelProfile, primary_model_profile


@dataclass(frozen=True, kw_only=True)
class PydanticAIMetricBinarySensorDescription(BinarySensorEntityDescription):
    """Description for one Pydantic AI metric binary sensor."""

    value_fn: Callable[[AgentRunMetrics], bool | None]
    entity_registry_enabled_default: bool = False


@dataclass(frozen=True, kw_only=True)
class PydanticAIConfigBinarySensorDescription(BinarySensorEntityDescription):
    """Description for one Pydantic AI configuration binary sensor."""

    value_fn: Callable[[ConfigSubentry], bool]
    entity_registry_enabled_default: bool = False
    subentry_types: tuple[str, ...] = (
        SUBENTRY_TYPE_CONVERSATION,
        SUBENTRY_TYPE_AI_TASK,
    )


BINARY_SENSOR_DESCRIPTIONS: tuple[PydanticAIMetricBinarySensorDescription, ...] = (
    PydanticAIMetricBinarySensorDescription(
        key="provider_healthy",
        translation_key="provider_healthy",
        icon="mdi:heart-pulse",
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=True,
        value_fn=lambda record: metric_bool(record, "provider_healthy"),
    ),
    PydanticAIMetricBinarySensorDescription(
        key="last_run_succeeded",
        translation_key="last_run_succeeded",
        icon="mdi:check-circle-outline",
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=True,
        value_fn=lambda record: metric_bool(record, "last_run_succeeded"),
    ),
)

CONFIG_BINARY_SENSOR_DESCRIPTIONS: tuple[
    PydanticAIConfigBinarySensorDescription, ...
] = (
    PydanticAIConfigBinarySensorDescription(
        key="assist_enabled",
        translation_key="assist_enabled",
        icon="mdi:assistant",
        subentry_types=(SUBENTRY_TYPE_CONVERSATION, SUBENTRY_TYPE_AI_TASK),
        entity_registry_enabled_default=True,
        value_fn=lambda subentry: bool(subentry.data.get(CONF_LLM_HASS_API)),
    ),
    PydanticAIConfigBinarySensorDescription(
        key="web_fetch_enabled",
        translation_key="web_fetch_enabled",
        icon="mdi:web",
        entity_registry_enabled_default=True,
        value_fn=lambda subentry: bool(
            subentry.data.get(CONF_WEB_FETCH_ENABLED, False)
        ),
    ),
    PydanticAIConfigBinarySensorDescription(
        key="virtual_workspace_enabled",
        translation_key="virtual_workspace_enabled",
        icon="mdi:folder-cog-outline",
        entity_registry_enabled_default=True,
        value_fn=lambda subentry: (
            subentry.data.get(CONF_VIRTUAL_WORKSPACE_ENABLED) is True
        ),
    ),
    PydanticAIConfigBinarySensorDescription(
        key="todo_workspace_enabled",
        translation_key="todo_workspace_enabled",
        icon="mdi:format-list-checks",
        subentry_types=(SUBENTRY_TYPE_AI_TASK,),
        entity_registry_enabled_default=True,
        value_fn=lambda subentry: bool(subentry.data.get(CONF_TODO_LIST_ENTITY_ID)),
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: PydanticAIAgentConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Pydantic AI Agent metric binary sensors."""
    for valid in _agent_subentries(config_entry):
        subentry = valid.subentry
        async_add_entities(
            [
                PydanticAIMetricBinarySensor(config_entry, subentry, description)
                for description in BINARY_SENSOR_DESCRIPTIONS
            ],
            config_subentry_id=subentry.subentry_id,
        )
        async_add_entities(
            [
                PydanticAIConfigBinarySensor(config_entry, subentry, description)
                for description in CONFIG_BINARY_SENSOR_DESCRIPTIONS
                if subentry.subentry_type in description.subentry_types
            ],
            config_subentry_id=subentry.subentry_id,
        )


class PydanticAIMetricBinarySensor(BinarySensorEntity):
    """Binary sensor that exposes one Pydantic AI health metric."""

    _attr_has_entity_name = True

    entity_description: BinarySensorEntityDescription

    def __init__(
        self,
        entry: PydanticAIAgentConfigEntry,
        subentry: ConfigSubentry,
        description: PydanticAIMetricBinarySensorDescription,
    ) -> None:
        """Initialize the metric binary sensor."""
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
        self._attr_is_on = self._description.value_fn(
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
        self._attr_is_on = self._description.value_fn(
            self.entry.runtime_data.metrics.record_for(self.subentry.subentry_id)
        )
        self.async_write_ha_state()


class PydanticAIConfigBinarySensor(BinarySensorEntity):
    """Binary sensor that exposes one static Pydantic AI configuration value."""

    _attr_has_entity_name = True

    entity_description: BinarySensorEntityDescription

    def __init__(
        self,
        entry: PydanticAIAgentConfigEntry,
        subentry: ConfigSubentry,
        description: PydanticAIConfigBinarySensorDescription,
    ) -> None:
        """Initialize the configuration binary sensor."""
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
        self._attr_is_on = self._description.value_fn(self.subentry)


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
