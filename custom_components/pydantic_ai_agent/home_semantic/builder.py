"""Build the local Home Semantic Index from Home Assistant state."""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, TypedDict

from homeassistant.const import (
    ATTR_DEVICE_CLASS,
    ATTR_SUPPORTED_FEATURES,
    ATTR_UNIT_OF_MEASUREMENT,
)
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    floor_registry as fr,
    label_registry as lr,
)

from .index import HomeSemanticIndex, normalize_tokens
from .models import (
    AreaSource,
    CapabilitySummary,
    DeviceSource,
    EntitySource,
    FloorSource,
    GraphEdge,
    HomeSemanticDocument,
    HomeSemanticSource,
    SemanticRankFeatures,
)

_CAPABILITY_BY_DOMAIN = {
    "alarm_control_panel": "alarms",
    "automation": "automations",
    "button": "buttons",
    "climate": "climate",
    "cover": "covers",
    "fan": "fans",
    "humidifier": "humidifiers",
    "input_boolean": "helpers",
    "light": "lights",
    "lock": "locks",
    "media_player": "media players",
    "number": "numbers",
    "scene": "scenes",
    "script": "scripts",
    "select": "selects",
    "switch": "switches",
    "vacuum": "vacuums",
}
_HIGH_CHURN_DOMAINS = {"binary_sensor", "sensor"}
_PHYSICAL_CONTROL_DOMAINS = {
    "alarm_control_panel",
    "button",
    "climate",
    "cover",
    "fan",
    "humidifier",
    "light",
    "lock",
    "media_player",
    "number",
    "select",
    "switch",
    "vacuum",
}


class _StateSnapshot(TypedDict):
    """Primitive state data copied from HA before executor work."""

    entity_id: str
    domain: str
    state: str
    attributes: dict[str, Any]


class _EntityRegistrySnapshot(TypedDict):
    """Primitive entity registry data copied from HA before executor work."""

    entity_id: str
    name: str | None
    original_name: str | None
    aliases: tuple[str, ...]
    labels: tuple[str, ...]
    area_id: str | None
    device_id: str | None
    domain: str
    platform: str
    device_class: str | None
    original_device_class: str | None
    entity_category: str | None
    unit_of_measurement: str | None
    supported_features: int
    disabled: bool
    hidden: bool


class _HomeSemanticSnapshot(TypedDict):
    """Primitive HA snapshot safe for executor-side source construction."""

    floors: tuple[FloorSource, ...]
    areas: tuple[AreaSource, ...]
    devices: tuple[DeviceSource, ...]
    entities: tuple[_EntityRegistrySnapshot, ...]
    states: dict[str, _StateSnapshot]


async def async_build_home_semantic_index(hass: HomeAssistant) -> HomeSemanticIndex:
    """Build an entry-scoped semantic index from HA registries and states."""
    snapshot = _home_semantic_snapshot(hass)
    return await hass.async_add_executor_job(
        _build_home_semantic_index_from_snapshot,
        snapshot,
    )


def _home_semantic_snapshot(hass: HomeAssistant) -> _HomeSemanticSnapshot:
    """Copy HA-owned registry and state fields needed by the executor builder."""
    entity_registry = er.async_get(hass)
    device_registry = dr.async_get(hass)
    area_registry = ar.async_get(hass)
    floor_registry = fr.async_get(hass)
    label_registry = lr.async_get(hass)
    labels_by_id = {
        label.label_id: label.name for label in label_registry.labels.values()
    }
    states = {
        snapshot["entity_id"]: snapshot
        for snapshot in (_state_snapshot(state) for state in hass.states.async_all())
    }
    return {
        "floors": tuple(
            _floor_source(floor) for floor in floor_registry.floors.values()
        ),
        "areas": tuple(
            _area_source(area, labels_by_id) for area in area_registry.areas.values()
        ),
        "devices": tuple(
            _device_source(device, labels_by_id)
            for device in device_registry.devices.values()
        ),
        "entities": tuple(
            _entity_registry_snapshot(hass, entry, labels_by_id)
            for entry in entity_registry.entities.values()
        ),
        "states": states,
    }


def _build_home_semantic_index_from_snapshot(
    snapshot: _HomeSemanticSnapshot,
) -> HomeSemanticIndex:
    """Build the index from primitive HA snapshot data off the event loop."""
    registry_entity_ids = {entry["entity_id"] for entry in snapshot["entities"]}
    source = HomeSemanticSource(
        floors=snapshot["floors"],
        areas=snapshot["areas"],
        devices=snapshot["devices"],
        entities=tuple(
            [
                _entity_source(entry, snapshot["states"].get(entry["entity_id"]))
                for entry in snapshot["entities"]
            ]
            + [
                _state_entity_source(state)
                for entity_id, state in snapshot["states"].items()
                if entity_id not in registry_entity_ids
            ]
        ),
    )
    return build_home_semantic_index(source)


def _state_snapshot(state: State) -> _StateSnapshot:
    """Return primitive state fields needed by semantic source construction."""
    return {
        "entity_id": state.entity_id,
        "domain": state.domain,
        "state": state.state,
        "attributes": {
            key: state.attributes[key]
            for key in (
                "friendly_name",
                ATTR_DEVICE_CLASS,
                ATTR_SUPPORTED_FEATURES,
                ATTR_UNIT_OF_MEASUREMENT,
            )
            if key in state.attributes
        },
    }


def _entity_registry_snapshot(
    hass: HomeAssistant,
    entry: er.RegistryEntry,
    labels_by_id: Mapping[str, str],
) -> _EntityRegistrySnapshot:
    """Return primitive registry fields needed by semantic source construction."""
    return {
        "entity_id": entry.entity_id,
        "name": entry.name,
        "original_name": entry.original_name,
        "aliases": _sorted_tuple(er.async_get_entity_aliases(hass, entry)),
        "labels": _label_names(entry.labels, labels_by_id),
        "area_id": entry.area_id,
        "device_id": entry.device_id,
        "domain": entry.domain,
        "platform": entry.platform,
        "device_class": entry.device_class,
        "original_device_class": entry.original_device_class,
        "entity_category": str(entry.entity_category)
        if entry.entity_category is not None
        else None,
        "unit_of_measurement": entry.unit_of_measurement,
        "supported_features": entry.supported_features,
        "disabled": entry.disabled_by is not None,
        "hidden": entry.hidden_by is not None,
    }


def _state_entity_source(state: _StateSnapshot) -> EntitySource:
    """Return normalized entity source data for a state-only entity."""
    return EntitySource(
        entity_id=state["entity_id"],
        name=state_name(state) or state["entity_id"],
        domain=state["domain"],
        device_class=_state_attribute(state, ATTR_DEVICE_CLASS),
        unit_of_measurement=_state_attribute(state, ATTR_UNIT_OF_MEASUREMENT),
        supported_features=_state_supported_features(state),
        current_state=state["state"],
    )


def _state_supported_features(state: _StateSnapshot) -> int:
    """Return a state's supported features value."""
    value = state["attributes"].get(ATTR_SUPPORTED_FEATURES)
    return value if isinstance(value, int) else 0


def build_home_semantic_index(source: HomeSemanticSource) -> HomeSemanticIndex:
    """Build a symbolic semantic index from normalized source data."""
    documents: list[HomeSemanticDocument] = []
    edges: list[GraphEdge] = []
    areas = {area.area_id: area for area in source.areas}
    devices = {device.device_id: device for device in source.devices}
    areas_by_device = {
        device_id: device.area_id
        for device_id, device in devices.items()
        if device.area_id is not None
    }
    entities_by_area_capability: dict[tuple[str, str], list[EntitySource]] = (
        defaultdict(list)
    )

    for floor in source.floors:
        documents.append(
            HomeSemanticDocument(
                document_id=f"floor:{floor.floor_id}",
                document_type="floor",
                name=floor.name,
                aliases=_sorted_tuple(floor.aliases),
                text=" ".join((floor.name, "floor", *(floor.aliases))),
                floor_id=floor.floor_id,
            )
        )
    for area in source.areas:
        if area.floor_id is not None:
            edges.append(
                GraphEdge(
                    source_id=f"area:{area.area_id}",
                    relation="on_floor",
                    target_id=f"floor:{area.floor_id}",
                )
            )
    for device in source.devices:
        area_id = device.area_id
        text_parts = [device.name, "device", *(device.aliases), *(device.labels)]
        if device.manufacturer is not None:
            text_parts.append(device.manufacturer)
        if device.model is not None:
            text_parts.append(device.model)
        documents.append(
            HomeSemanticDocument(
                document_id=f"device:{device.device_id}",
                document_type="device",
                name=device.name,
                aliases=_sorted_tuple(device.aliases),
                labels=_sorted_tuple(device.labels),
                text=" ".join(text_parts),
                area_id=area_id,
                floor_id=_floor_id_for_area(areas, area_id),
                device_id=device.device_id,
                rank=SemanticRankFeatures(disabled=device.disabled),
            )
        )
        if area_id is not None:
            edges.append(
                GraphEdge(
                    source_id=f"device:{device.device_id}",
                    relation="in_area",
                    target_id=f"area:{area_id}",
                )
            )

    for entity in source.entities:
        entity_area_id = entity.area_id or (
            areas_by_device.get(entity.device_id)
            if entity.device_id is not None
            else None
        )
        capability = _capability_for_entity(entity)
        if entity_area_id is not None and capability is not None:
            entities_by_area_capability[(entity_area_id, capability)].append(entity)

    preferred_targets = _preferred_targets(entities_by_area_capability)
    for area in source.areas:
        capabilities = tuple(
            CapabilitySummary(
                capability=capability,
                entity_count=len(entities),
                preferred_target=preferred_targets.get((area.area_id, capability)),
            )
            for (area_id, capability), entities in sorted(
                entities_by_area_capability.items()
            )
            if area_id == area.area_id
        )
        capability_text = " ".join(summary.capability for summary in capabilities)
        documents.append(
            HomeSemanticDocument(
                document_id=f"area:{area.area_id}",
                document_type="area",
                name=area.name,
                aliases=_sorted_tuple(area.aliases),
                labels=_sorted_tuple(area.labels),
                text=" ".join(
                    (
                        area.name,
                        "area",
                        *(area.aliases),
                        *(area.labels),
                        capability_text,
                    )
                ),
                floor_id=area.floor_id,
                area_id=area.area_id,
                capabilities=capabilities,
            )
        )
        for summary in capabilities:
            documents.append(
                HomeSemanticDocument(
                    document_id=f"capability:{area.area_id}:{summary.capability}",
                    document_type="capability",
                    name=f"{area.name} {summary.capability}",
                    aliases=_sorted_tuple(area.aliases),
                    labels=_sorted_tuple(area.labels),
                    text=f"{area.name} {summary.capability} area capability",
                    floor_id=area.floor_id,
                    area_id=area.area_id,
                    capability=summary.capability,
                    target_entity_id=summary.preferred_target,
                    entity_count=summary.entity_count,
                    rank=SemanticRankFeatures(
                        preferred_target=summary.preferred_target is not None
                    ),
                )
            )
            edges.append(
                GraphEdge(
                    source_id=f"capability:{area.area_id}:{summary.capability}",
                    relation="in_area",
                    target_id=f"area:{area.area_id}",
                )
            )

    for entity in source.entities:
        entity_area_id = entity.area_id or (
            areas_by_device.get(entity.device_id)
            if entity.device_id is not None
            else None
        )
        floor_id = _floor_id_for_area(areas, entity_area_id)
        capability = _capability_for_entity(entity)
        preferred = (
            entity_area_id is not None
            and capability is not None
            and preferred_targets.get((entity_area_id, capability)) == entity.entity_id
        )
        group = _is_group_entity(entity)
        document_type = "group" if group else "entity"
        documents.append(
            HomeSemanticDocument(
                document_id=f"{document_type}:{entity.entity_id}",
                document_type=document_type,
                name=entity.name,
                aliases=_sorted_tuple(entity.aliases),
                labels=_sorted_tuple(entity.labels),
                text=_entity_text(
                    entity,
                    capability,
                    areas.get(entity_area_id) if entity_area_id is not None else None,
                    devices.get(entity.device_id)
                    if entity.device_id is not None
                    else None,
                ),
                floor_id=floor_id,
                area_id=entity_area_id,
                device_id=entity.device_id,
                entity_id=entity.entity_id,
                domain=entity.domain,
                capability=capability,
                current_state=entity.current_state,
                unit_of_measurement=entity.unit_of_measurement,
                device_class=entity.device_class,
                supported_features=entity.supported_features,
                target_entity_id=entity.entity_id if preferred else None,
                rank=SemanticRankFeatures(
                    disabled=entity.disabled,
                    hidden=entity.hidden,
                    diagnostic=entity.entity_category == "diagnostic",
                    group=group,
                    high_churn=entity.domain in _HIGH_CHURN_DOMAINS,
                    preferred_target=preferred,
                    physical_control=entity.domain in _PHYSICAL_CONTROL_DOMAINS,
                ),
            )
        )
        if entity_area_id is not None:
            edges.append(
                GraphEdge(
                    source_id=f"{document_type}:{entity.entity_id}",
                    relation="in_area",
                    target_id=f"area:{entity_area_id}",
                )
            )
        if entity.device_id is not None:
            edges.append(
                GraphEdge(
                    source_id=f"{document_type}:{entity.entity_id}",
                    relation="belongs_to_device",
                    target_id=f"device:{entity.device_id}",
                )
            )
    return HomeSemanticIndex(documents, edges)


def _floor_source(entry: fr.FloorEntry) -> FloorSource:
    """Return normalized floor source data."""
    return FloorSource(
        floor_id=entry.floor_id,
        name=entry.name,
        aliases=_sorted_tuple(entry.aliases),
        level=entry.level,
    )


def _area_source(entry: ar.AreaEntry, labels_by_id: Mapping[str, str]) -> AreaSource:
    """Return normalized area source data."""
    return AreaSource(
        area_id=entry.id,
        name=entry.name,
        aliases=_sorted_tuple(entry.aliases),
        labels=_label_names(entry.labels, labels_by_id),
        floor_id=entry.floor_id,
    )


def _device_source(
    entry: dr.DeviceEntry, labels_by_id: Mapping[str, str]
) -> DeviceSource:
    """Return normalized device source data."""
    name = entry.name_by_user or entry.name or entry.id
    aliases: tuple[str, ...] = ()
    if entry.name_by_user is not None and entry.name is not None:
        aliases = (entry.name,)
    return DeviceSource(
        device_id=entry.id,
        name=name,
        aliases=aliases,
        labels=_label_names(entry.labels, labels_by_id),
        area_id=entry.area_id,
        manufacturer=entry.manufacturer,
        model=entry.model,
        disabled=entry.disabled_by is not None,
    )


def _entity_source(
    entry: _EntityRegistrySnapshot, state: _StateSnapshot | None
) -> EntitySource:
    """Return normalized entity source data."""
    name = (
        entry["name"]
        or entry["original_name"]
        or state_name(state)
        or entry["entity_id"]
    )
    return EntitySource(
        entity_id=entry["entity_id"],
        name=name,
        aliases=entry["aliases"],
        labels=entry["labels"],
        area_id=entry["area_id"],
        device_id=entry["device_id"],
        domain=entry["domain"],
        platform=entry["platform"],
        device_class=entry["device_class"]
        or entry["original_device_class"]
        or _state_attribute(state, ATTR_DEVICE_CLASS),
        entity_category=entry["entity_category"],
        unit_of_measurement=entry["unit_of_measurement"]
        or _state_attribute(state, ATTR_UNIT_OF_MEASUREMENT),
        supported_features=entry["supported_features"],
        current_state=state["state"] if state is not None else None,
        disabled=entry["disabled"],
        hidden=entry["hidden"],
    )


def state_name(state: _StateSnapshot | None) -> str | None:
    """Return a state's friendly name if it has one."""
    if state is None:
        return None
    value = state["attributes"].get("friendly_name")
    return value if isinstance(value, str) else None


def _state_attribute(state: _StateSnapshot | None, key: str) -> str | None:
    """Return a string state attribute value."""
    if state is None:
        return None
    value = state["attributes"].get(key)
    return value if isinstance(value, str) else None


def _capability_for_entity(entity: EntitySource) -> str | None:
    """Return the high-level capability represented by an entity."""
    if entity.domain is None:
        return None
    return _CAPABILITY_BY_DOMAIN.get(entity.domain)


def _floor_id_for_area(areas: dict[str, AreaSource], area_id: str | None) -> str | None:
    """Return the floor id for an area id, if known."""
    if area_id is None:
        return None
    area = areas.get(area_id)
    return area.floor_id if area is not None else None


def _preferred_targets(
    entities_by_area_capability: dict[tuple[str, str], list[EntitySource]],
) -> dict[tuple[str, str], str]:
    """Choose a deterministic preferred target per area capability."""
    preferred: dict[tuple[str, str], str] = {}
    for key, entities in entities_by_area_capability.items():
        candidates = [
            entity for entity in entities if not entity.disabled and not entity.hidden
        ]
        if not candidates:
            continue
        group_candidates = [entity for entity in candidates if _is_group_entity(entity)]
        if group_candidates:
            group_candidates.sort(key=_preferred_entity_sort_key)
            preferred[key] = group_candidates[0].entity_id
        elif len(candidates) == 1:
            preferred[key] = candidates[0].entity_id
    return preferred


def _preferred_entity_sort_key(entity: EntitySource) -> tuple[int, int, int, str]:
    """Sort preferred target candidates by safety and usefulness."""
    return (
        0 if _is_group_entity(entity) else 1,
        1 if entity.domain in _HIGH_CHURN_DOMAINS else 0,
        1 if entity.entity_category == "diagnostic" else 0,
        entity.entity_id,
    )


def _is_group_entity(entity: EntitySource) -> bool:
    """Return whether an entity should be treated as a grouped target."""
    if entity.domain == "group" or entity.platform == "group":
        return True
    tokens = set(normalize_tokens(entity.entity_id)) | set(
        normalize_tokens(entity.name)
    )
    return bool(tokens & {"group", "groups"})


def _entity_text(
    entity: EntitySource,
    capability: str | None,
    area: AreaSource | None,
    device: DeviceSource | None,
) -> str:
    """Return compact text used for symbolic search."""
    parts = [
        entity.name,
        entity.entity_id,
        entity.domain,
        entity.platform,
        entity.device_class,
        entity.entity_category,
        capability,
        *(entity.aliases),
        *(entity.labels),
    ]
    if area is not None:
        parts.extend((area.name, *area.aliases, *area.labels))
    if device is not None:
        parts.extend((device.name, *device.aliases, *device.labels))
    return " ".join(part for part in parts if part)


def _sorted_tuple(values: Iterable[str]) -> tuple[str, ...]:
    """Return stable sorted string values."""
    return tuple(sorted(value for value in values if value))


def _label_names(
    label_ids: Iterable[str], labels_by_id: Mapping[str, str]
) -> tuple[str, ...]:
    """Return current label names for registry label IDs."""
    return _sorted_tuple(labels_by_id.get(label_id, label_id) for label_id in label_ids)
