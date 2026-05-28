"""Typed models for the local Home Semantic Index."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

DocumentType = Literal["floor", "area", "device", "group", "entity", "capability"]


@dataclass(frozen=True, kw_only=True)
class SemanticRankFeatures:
    """Ranking signals that should remain local and deterministic."""

    disabled: bool = False
    hidden: bool = False
    diagnostic: bool = False
    group: bool = False
    high_churn: bool = False
    preferred_target: bool = False
    physical_control: bool = False


@dataclass(frozen=True, kw_only=True)
class CapabilitySummary:
    """Compact summary of a capability available in one scope."""

    capability: str
    entity_count: int
    preferred_target: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable diagnostics data."""
        data: dict[str, Any] = {
            "capability": self.capability,
            "entity_count": self.entity_count,
        }
        if self.preferred_target is not None:
            data["preferred_target"] = self.preferred_target
        return data


@dataclass(frozen=True, kw_only=True)
class HomeSemanticDocument:
    """One searchable symbolic document for home retrieval."""

    document_id: str
    document_type: DocumentType
    name: str
    text: str
    aliases: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    floor_id: str | None = None
    area_id: str | None = None
    device_id: str | None = None
    entity_id: str | None = None
    domain: str | None = None
    capability: str | None = None
    current_state: str | None = None
    unit_of_measurement: str | None = None
    device_class: str | None = None
    supported_features: int = 0
    target_entity_id: str | None = None
    entity_count: int | None = None
    capabilities: tuple[CapabilitySummary, ...] = ()
    rank: SemanticRankFeatures = field(default_factory=SemanticRankFeatures)

    def searchable_parts(self) -> tuple[str, ...]:
        """Return all local text used by the symbolic index."""
        parts = [
            self.name,
            self.text,
            self.document_type,
            self.domain,
            self.capability,
            self.device_class,
            self.unit_of_measurement,
            self.target_entity_id,
        ]
        parts.extend(self.aliases)
        parts.extend(self.labels)
        parts.extend(summary.capability for summary in self.capabilities)
        return tuple(part for part in parts if part)

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable diagnostics data."""
        return {
            "document_id": self.document_id,
            "document_type": self.document_type,
            "name": self.name,
            "aliases": list(self.aliases),
            "labels": list(self.labels),
            "floor_id": self.floor_id,
            "area_id": self.area_id,
            "device_id": self.device_id,
            "entity_id": self.entity_id,
            "domain": self.domain,
            "capability": self.capability,
            "current_state": self.current_state,
            "unit_of_measurement": self.unit_of_measurement,
            "device_class": self.device_class,
            "supported_features": self.supported_features,
            "target_entity_id": self.target_entity_id,
            "entity_count": self.entity_count,
            "capabilities": [summary.as_dict() for summary in self.capabilities],
            "rank": {
                "disabled": self.rank.disabled,
                "hidden": self.rank.hidden,
                "diagnostic": self.rank.diagnostic,
                "group": self.rank.group,
                "high_churn": self.rank.high_churn,
                "preferred_target": self.rank.preferred_target,
                "physical_control": self.rank.physical_control,
            },
        }


@dataclass(frozen=True, kw_only=True)
class GraphEdge:
    """A typed relationship between semantic documents."""

    source_id: str
    relation: str
    target_id: str

    def as_dict(self) -> dict[str, str]:
        """Return JSON-serializable diagnostics data."""
        return {
            "source_id": self.source_id,
            "relation": self.relation,
            "target_id": self.target_id,
        }


@dataclass(frozen=True, kw_only=True)
class FloorSource:
    """Registry data needed to build one floor document."""

    floor_id: str
    name: str
    aliases: tuple[str, ...] = ()
    level: int | None = None


@dataclass(frozen=True, kw_only=True)
class AreaSource:
    """Registry data needed to build one area document."""

    area_id: str
    name: str
    aliases: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    floor_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class DeviceSource:
    """Registry data needed to build one device document."""

    device_id: str
    name: str
    aliases: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    area_id: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    disabled: bool = False


@dataclass(frozen=True, kw_only=True)
class EntitySource:
    """Registry and state data needed to build one entity document."""

    entity_id: str
    name: str
    aliases: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    area_id: str | None = None
    device_id: str | None = None
    domain: str | None = None
    platform: str | None = None
    device_class: str | None = None
    entity_category: str | None = None
    unit_of_measurement: str | None = None
    supported_features: int = 0
    current_state: str | None = None
    disabled: bool = False
    hidden: bool = False


@dataclass(frozen=True, kw_only=True)
class HomeSemanticSource:
    """Normalized inputs for building a home semantic index."""

    floors: Sequence[FloorSource] = ()
    areas: Sequence[AreaSource] = ()
    devices: Sequence[DeviceSource] = ()
    entities: Sequence[EntitySource] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
