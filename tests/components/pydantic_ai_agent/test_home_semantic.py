"""Tests for the local Home Semantic Index."""

from collections.abc import Awaitable, Callable
from datetime import timedelta

from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er, label_registry as lr
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)

from custom_components.pydantic_ai_agent.home_semantic.builder import (
    async_build_home_semantic_index,
    build_home_semantic_index,
)
from custom_components.pydantic_ai_agent.home_semantic.diagnostics import (
    semantic_index_diagnostics,
    semantic_manager_diagnostics,
)
from custom_components.pydantic_ai_agent.home_semantic.index import HomeSemanticIndex
from custom_components.pydantic_ai_agent.home_semantic.manager import (
    HomeSemanticIndexManager,
)
from custom_components.pydantic_ai_agent.home_semantic.models import (
    AreaSource,
    DeviceSource,
    EntitySource,
    FloorSource,
    HomeSemanticSource,
)
from tests.components.pydantic_ai_agent.support.builders import workspace_entry


def _index_with_light(entity_id: str = "light.test_light") -> HomeSemanticIndex:
    """Return a small semantic index for manager tests."""
    return build_home_semantic_index(
        HomeSemanticSource(
            areas=(AreaSource(area_id="test_area", name="Test Area"),),
            entities=(
                EntitySource(
                    entity_id=entity_id,
                    name="Test Light",
                    area_id="test_area",
                    domain="light",
                ),
            ),
        )
    )


def _entry(hass: HomeAssistant) -> MockConfigEntry:
    """Return a workspace entry registered with hass."""
    entry = workspace_entry(title="Workspace")
    entry.add_to_hass(hass)
    return entry


async def _fire_time(hass: HomeAssistant, seconds: float) -> None:
    """Fire time changed and drain pending tasks."""
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=seconds))
    await hass.async_block_till_done()
    await hass.async_block_till_done()


def _counting_builder(
    calls: list[str], index: HomeSemanticIndex | None = None
) -> Callable[[], Awaitable[HomeSemanticIndex]]:
    """Return a build callback that records calls."""

    async def build() -> HomeSemanticIndex:
        calls.append("build")
        return index or _index_with_light()

    return build


async def test_manager_start_does_not_build_immediately(
    hass: HomeAssistant,
) -> None:
    """Test startup only schedules initial work and does not block setup."""
    calls: list[str] = []
    manager = HomeSemanticIndexManager(
        hass,
        _entry(hass),
        build_index=_counting_builder(calls),
        initial_delay_seconds=60,
        initial_jitter_seconds=0,
    )

    manager.async_start()
    await hass.async_block_till_done()

    assert calls == []
    assert manager.index is None
    diagnostics = manager.diagnostics()
    assert diagnostics["status"] == "loading"
    assert diagnostics["scheduled"] is True

    manager.async_stop()


async def test_manager_delayed_initial_build_swaps_index(
    hass: HomeAssistant,
) -> None:
    """Test delayed initial refresh builds and atomically exposes the index."""
    calls: list[str] = []
    manager = HomeSemanticIndexManager(
        hass,
        _entry(hass),
        build_index=_counting_builder(calls),
        initial_delay_seconds=1,
        initial_jitter_seconds=0,
    )

    manager.async_start()
    await _fire_time(hass, 2)
    await _fire_time(hass, 0.1)

    assert calls == ["build"]
    assert manager.index is not None
    assert manager.generation == 1
    assert manager.status == "ready"

    manager.async_stop()


async def test_manager_coalesces_refresh_requests(
    hass: HomeAssistant,
) -> None:
    """Test event storms schedule one refresh instead of one per event."""
    calls: list[str] = []
    manager = HomeSemanticIndexManager(
        hass,
        _entry(hass),
        build_index=_counting_builder(calls),
        initial_delay_seconds=600,
        initial_jitter_seconds=0,
    )

    manager.async_start()
    manager.async_request_refresh(
        reason="state_changed", entity_id="light.one", delay=0
    )
    manager.async_request_refresh(
        reason="state_changed", entity_id="light.two", delay=0
    )
    manager.async_request_refresh(
        reason="entity_registry_updated", structural=True, delay=0
    )
    await _fire_time(hass, 1)

    assert calls == ["build"]
    assert manager.generation == 1

    manager.async_stop()


async def test_manager_periodic_rescan_catches_state_only_entities(
    hass: HomeAssistant,
) -> None:
    """Test periodic rescans pick up state-only entities created later."""
    entry = _entry(hass)
    calls: list[str] = []
    manager = HomeSemanticIndexManager(
        hass,
        entry,
        initial_delay_seconds=600,
        initial_jitter_seconds=0,
        periodic_seconds=1,
        build_index=_counting_builder(
            calls, _index_with_light("light.late_yaml_light")
        ),
    )
    manager.async_start()
    hass.states.async_set(
        "light.late_yaml_light",
        "off",
        {"friendly_name": "Late YAML Light"},
    )

    await _fire_time(hass, 2)

    assert calls == ["build"]
    assert manager.index is not None
    assert "light.late_yaml_light" in manager.index.documents_by_entity_id
    assert manager.last_refresh_reason == "periodic_rescan"

    manager.async_stop()


async def test_manager_ignores_high_churn_sensor_value_updates(
    hass: HomeAssistant,
) -> None:
    """Test sensor value-only changes do not schedule semantic rebuilds."""
    manager = HomeSemanticIndexManager(
        hass,
        _entry(hass),
        build_index=_counting_builder([], _index_with_light()),
        initial_delay_seconds=60,
        debounce_seconds=0,
    )
    manager.index = build_home_semantic_index(
        HomeSemanticSource(
            entities=(
                EntitySource(
                    entity_id="sensor.temperature",
                    name="Temperature",
                    domain="sensor",
                    current_state="20",
                ),
            )
        )
    )

    hass.states.async_set("sensor.temperature", "20")
    await hass.async_block_till_done()
    manager.async_start()
    hass.states.async_set("sensor.temperature", "21")
    await hass.async_block_till_done()
    await _fire_time(hass, 1)

    assert manager.diagnostics()["dirty_entity_count"] == 0
    assert manager.generation == 0
    manager.async_stop()


async def test_manager_stop_cancels_scheduled_initial_build(
    hass: HomeAssistant,
) -> None:
    """Test unload cleanup cancels queued refresh work."""
    calls: list[str] = []
    manager = HomeSemanticIndexManager(
        hass,
        _entry(hass),
        build_index=_counting_builder(calls),
        initial_delay_seconds=1,
        initial_jitter_seconds=0,
    )

    manager.async_start()
    manager.async_stop()
    await _fire_time(hass, 2)

    assert calls == []
    assert manager.status == "stopped"


async def test_manager_diagnostics_are_aggregate(
    hass: HomeAssistant,
) -> None:
    """Test manager diagnostics expose status and counts, not documents."""
    calls: list[str] = []
    manager = HomeSemanticIndexManager(
        hass,
        _entry(hass),
        build_index=_counting_builder(calls),
        initial_delay_seconds=1,
        initial_jitter_seconds=0,
    )

    manager.async_start()
    await _fire_time(hass, 2)

    diagnostics = semantic_manager_diagnostics(manager)

    assert diagnostics["loaded"] is True
    assert diagnostics["ready"] is True
    assert diagnostics["generation"] == 1
    assert diagnostics["document_counts"] == {
        "area": 1,
        "capability": 1,
        "entity": 1,
    }
    assert "documents" not in diagnostics
    assert "Test Light" not in str(diagnostics)

    manager.async_stop()


def test_builds_area_capability_with_group_preferred_target() -> None:
    """Test semantic documents prefer grouped controls for area capabilities."""
    index = build_home_semantic_index(
        HomeSemanticSource(
            floors=(FloorSource(floor_id="upstairs", name="Upstairs"),),
            areas=(
                AreaSource(
                    area_id="bedroom",
                    name="Bedroom",
                    aliases=("main bedroom", "our room"),
                    floor_id="upstairs",
                ),
            ),
            entities=(
                EntitySource(
                    entity_id="light.bedroom_lights",
                    name="Bedroom Lights",
                    area_id="bedroom",
                    domain="light",
                    platform="group",
                ),
                EntitySource(
                    entity_id="light.jane_nightstand",
                    name="Jane Nightstand",
                    area_id="bedroom",
                    domain="light",
                    platform="hue",
                ),
            ),
        )
    )

    capability = index.documents_by_id["capability:bedroom:lights"]
    results = index.search("bedroom lights", action="turn_off")

    assert capability.target_entity_id == "light.bedroom_lights"
    assert results[0].document.entity_id == "light.bedroom_lights"
    assert results[0].document.rank.group is True


def test_unknown_query_returns_no_results() -> None:
    """Test action compatibility alone does not create unrelated matches."""
    index = build_home_semantic_index(
        HomeSemanticSource(
            areas=(AreaSource(area_id="bedroom", name="Bedroom"),),
            entities=(
                EntitySource(
                    entity_id="light.bedroom_lights",
                    name="Bedroom Lights",
                    area_id="bedroom",
                    domain="light",
                    platform="group",
                ),
            ),
        )
    )

    assert index.search("zqxj unknown phrase", action="turn_off") == []


def test_current_state_is_not_searchable_for_target_resolution() -> None:
    """Test raw state tokens do not create unrelated target matches."""
    index = build_home_semantic_index(
        HomeSemanticSource(
            entities=(
                EntitySource(
                    entity_id="light.office_lamp",
                    name="Office Lamp",
                    domain="light",
                    current_state="off",
                ),
            ),
        )
    )

    assert index.search("zqxj off", action="turn_off") == []


async def test_state_only_entities_are_indexed(hass: HomeAssistant) -> None:
    """Test entities present only in the state machine are indexed."""
    hass.states.async_set(
        "light.yaml_light",
        "off",
        {"friendly_name": "YAML Light"},
    )

    index = await async_build_home_semantic_index(hass)

    assert index.documents_by_entity_id["light.yaml_light"].name == "YAML Light"
    assert index.search("yaml light", action="turn_off")[0].document.entity_id == (
        "light.yaml_light"
    )


async def test_registry_label_ids_are_indexed_as_current_names(
    hass: HomeAssistant,
) -> None:
    """Test semantic text uses current HA label names, not stable label IDs."""
    label = lr.async_get(hass).async_create("Holiday Mode")
    entity_registry = er.async_get(hass)
    entry = entity_registry.async_get_or_create(
        "light",
        "test",
        "labeled-light",
        original_name="Labeled Light",
    )
    entity_registry.async_update_entity(entry.entity_id, labels={label.label_id})

    index = await async_build_home_semantic_index(hass)

    assert index.search("holiday mode", action="turn_off")[0].document.entity_id == (
        entry.entity_id
    )


def test_builder_uses_device_area_for_entities_without_area() -> None:
    """Test device area membership contributes to entity capability scopes."""
    index = build_home_semantic_index(
        HomeSemanticSource(
            areas=(AreaSource(area_id="office", name="Office"),),
            devices=(
                DeviceSource(
                    device_id="lamp-device", name="Desk Lamp", area_id="office"
                ),
            ),
            entities=(
                EntitySource(
                    entity_id="light.desk_lamp",
                    name="Desk Lamp",
                    device_id="lamp-device",
                    domain="light",
                    platform="zha",
                ),
            ),
        )
    )

    capability = index.documents_by_id["capability:office:lights"]
    entity = index.documents_by_entity_id["light.desk_lamp"]

    assert capability.target_entity_id == "light.desk_lamp"
    assert entity.area_id == "office"


def test_builder_does_not_prefer_one_of_many_ungrouped_targets() -> None:
    """Test multi-entity capabilities require a group or explicit entity."""
    index = build_home_semantic_index(
        HomeSemanticSource(
            areas=(AreaSource(area_id="bedroom", name="Bedroom"),),
            entities=(
                EntitySource(
                    entity_id="light.bedroom_ceiling",
                    name="Bedroom Ceiling",
                    area_id="bedroom",
                    domain="light",
                ),
                EntitySource(
                    entity_id="light.bedroom_lamp",
                    name="Bedroom Lamp",
                    area_id="bedroom",
                    domain="light",
                ),
            ),
        )
    )

    capability = index.documents_by_id["capability:bedroom:lights"]

    assert capability.target_entity_id is None


def test_search_penalizes_diagnostic_high_churn_entities() -> None:
    """Test diagnostics do not outrank normal user-facing controls."""
    index = build_home_semantic_index(
        HomeSemanticSource(
            areas=(AreaSource(area_id="bedroom", name="Bedroom"),),
            entities=(
                EntitySource(
                    entity_id="switch.bedroom_fan",
                    name="Bedroom Fan",
                    area_id="bedroom",
                    domain="switch",
                    platform="zha",
                ),
                EntitySource(
                    entity_id="sensor.bedroom_fan_signal",
                    name="Bedroom Fan Signal",
                    area_id="bedroom",
                    domain="sensor",
                    platform="zha",
                    entity_category="diagnostic",
                ),
            ),
        )
    )

    results = index.search("bedroom fan", action="turn_off")

    assert results[0].document.entity_id == "switch.bedroom_fan"


def test_semantic_index_diagnostics_are_aggregate_and_json_safe() -> None:
    """Test semantic diagnostics expose counts without raw registry payloads."""
    index = build_home_semantic_index(
        HomeSemanticSource(
            areas=(AreaSource(area_id="bedroom", name="Bedroom"),),
            entities=(
                EntitySource(
                    entity_id="light.bedroom_lights",
                    name="Bedroom Lights",
                    area_id="bedroom",
                    domain="light",
                    platform="group",
                ),
            ),
        )
    )

    diagnostics = semantic_index_diagnostics(index)

    assert diagnostics["loaded"] is True
    assert diagnostics["document_counts"] == {
        "area": 1,
        "capability": 1,
        "group": 1,
    }
    assert diagnostics["domain_counts"] == {"light": 1}
    assert diagnostics["capability_counts"] == {"lights": 2}
