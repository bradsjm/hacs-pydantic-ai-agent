"""Tests for the semantic Home Assistant LLM API."""

from typing import Any, cast
from unittest.mock import AsyncMock, patch

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_expose_entity
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import Context, HomeAssistant, ServiceCall
from homeassistant.helpers import llm
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.pydantic_ai_agent import (
    SERVICE_GET_AGENT_RUN_DIAGNOSTICS,
    WorkspaceRuntimeData,
    async_setup,
    async_setup_entry,
)
from custom_components.pydantic_ai_agent.const import DOMAIN
from custom_components.pydantic_ai_agent.home_semantic import (
    HomeSemanticAPI,
    HomeSemanticIndexManager,
    semantic_api_id,
)
from custom_components.pydantic_ai_agent.home_semantic.builder import (
    build_home_semantic_index,
)
from custom_components.pydantic_ai_agent.home_semantic.services import (
    SERVICE_BENCHMARK_HOME_SEMANTIC_RESOLUTION,
    SERVICE_GET_HOME_SEMANTIC_DOCUMENT,
    SERVICE_GET_HOME_SEMANTIC_CONTEXT,
    SERVICE_GET_HOME_SEMANTIC_SUMMARY,
    SERVICE_PLAN_HOME_SEMANTIC_CONTROL,
    SERVICE_REFRESH_HOME_SEMANTIC_INDEX,
    SERVICE_RESOLVE_HOME_SEMANTIC_TARGET,
    SERVICE_TRACE_HOME_SEMANTIC_RESOLUTION,
)
from custom_components.pydantic_ai_agent.home_semantic.models import (
    AreaSource,
    EntitySource,
    HomeSemanticSource,
)
from custom_components.pydantic_ai_agent.ha_toolset import tools_from_llm_api
from custom_components.pydantic_ai_agent.model_profiles import model_profile_ref
from tests.components.pydantic_ai_agent.support.builders import (
    conversation_subentry_data,
    provider_subentry_data,
    workspace_entry,
)


def _workspace_with_manager(
    hass: HomeAssistant, index_source: HomeSemanticSource
) -> tuple[MockConfigEntry, HomeSemanticAPI]:
    """Return a workspace entry with a ready semantic index manager."""
    entry = workspace_entry(title="Workspace")
    entry.add_to_hass(hass)
    manager = HomeSemanticIndexManager(hass, cast(Any, entry))
    manager.index = build_home_semantic_index(index_source)
    entry.runtime_data = WorkspaceRuntimeData(
        workspace_name="Workspace",
        home_semantic=manager,
    )
    return entry, HomeSemanticAPI(hass, cast(Any, entry))


def _llm_context() -> llm.LLMContext:
    """Return a conversation LLM context for semantic API tests."""
    return llm.LLMContext(
        platform=DOMAIN,
        context=Context(),
        language="en",
        assistant=conversation.DOMAIN,
        device_id=None,
    )


async def _call_tool(
    api: HomeSemanticAPI,
    tool_name: str,
    tool_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call one semantic API tool by name."""
    api_instance = await api.async_get_api_instance(_llm_context())
    result = await api_instance.async_call_tool(
        llm.ToolInput(
            tool_name=tool_name,
            tool_args={} if tool_args is None else tool_args,
            id="test-call",
        )
    )
    return cast(dict[str, Any], result)


async def _call_service(
    hass: HomeAssistant, service: str, data: dict[str, Any]
) -> dict[str, Any]:
    """Call one semantic response service."""
    result = await hass.services.async_call(
        DOMAIN,
        service,
        data,
        blocking=True,
        return_response=True,
    )
    return cast(dict[str, Any], result)


async def _expose(hass: HomeAssistant, *entity_ids: str) -> None:
    """Expose entities to the conversation assistant."""
    for entity_id in entity_ids:
        async_expose_entity(hass, conversation.DOMAIN, entity_id, True)


async def _hide(hass: HomeAssistant, *entity_ids: str) -> None:
    """Hide entities from the conversation assistant."""
    for entity_id in entity_ids:
        async_expose_entity(hass, conversation.DOMAIN, entity_id, False)


async def test_setup_registers_entry_scoped_semantic_api(
    hass: HomeAssistant,
) -> None:
    """Test setup registers and unload callbacks unregister the semantic API."""
    profile_ref = model_profile_ref("provider-1", "profile-1")
    entry = workspace_entry(
        (provider_subentry_data(subentry_id="provider-1", discovered=True),)
    )
    entry.add_to_hass(hass)

    with (
        patch(
            "custom_components.pydantic_ai_agent.async_probe_model",
            new_callable=AsyncMock,
        ),
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new_callable=AsyncMock,
        ),
    ):
        assert await async_setup_entry(hass, cast(Any, entry))

    api_ids = {api.id for api in llm.async_get_apis(hass)}
    assert profile_ref in entry.runtime_data.model_profiles
    assert semantic_api_id(entry.entry_id) in api_ids

    await entry._async_process_on_unload(hass)  # noqa: SLF001

    api_ids = {api.id for api in llm.async_get_apis(hass)}
    assert semantic_api_id(entry.entry_id) not in api_ids


async def test_semantic_api_exposes_four_tools(hass: HomeAssistant) -> None:
    """Test the semantic API exposes the expected compact tool surface."""
    _, api = _workspace_with_manager(hass, HomeSemanticSource())
    api_instance = await api.async_get_api_instance(_llm_context())

    assert {tool.name for tool in api_instance.tools} == {
        "control_home",
        "get_home_context",
        "get_home_summary",
        "resolve_home_target",
    }


async def test_setup_registers_home_semantic_response_services(
    hass: HomeAssistant,
) -> None:
    """Test async setup registers read-only semantic response services."""
    assert await async_setup(hass, {})

    assert hass.services.has_service(DOMAIN, SERVICE_GET_HOME_SEMANTIC_SUMMARY)
    assert hass.services.has_service(DOMAIN, SERVICE_RESOLVE_HOME_SEMANTIC_TARGET)
    assert hass.services.has_service(DOMAIN, SERVICE_GET_HOME_SEMANTIC_CONTEXT)
    assert hass.services.has_service(DOMAIN, SERVICE_REFRESH_HOME_SEMANTIC_INDEX)
    assert hass.services.has_service(DOMAIN, SERVICE_TRACE_HOME_SEMANTIC_RESOLUTION)
    assert hass.services.has_service(DOMAIN, SERVICE_PLAN_HOME_SEMANTIC_CONTROL)
    assert hass.services.has_service(DOMAIN, SERVICE_GET_HOME_SEMANTIC_DOCUMENT)
    assert hass.services.has_service(DOMAIN, SERVICE_BENCHMARK_HOME_SEMANTIC_RESOLUTION)
    assert hass.services.has_service(DOMAIN, SERVICE_GET_AGENT_RUN_DIAGNOSTICS)


async def test_home_semantic_summary_service_filters_exposure(
    hass: HomeAssistant,
) -> None:
    """Test summary response service returns only assistant-exposed targets."""
    hass.states.async_set("light.kitchen", "on")
    hass.states.async_set("light.hidden", "on")
    await _expose(hass, "light.kitchen")
    await _hide(hass, "light.hidden")
    entry, _api = _workspace_with_manager(
        hass,
        HomeSemanticSource(
            areas=(AreaSource(area_id="kitchen", name="Kitchen"),),
            entities=(
                EntitySource(
                    entity_id="light.kitchen",
                    name="Kitchen Light",
                    area_id="kitchen",
                    domain="light",
                ),
                EntitySource(
                    entity_id="light.hidden",
                    name="Hidden Light",
                    area_id="kitchen",
                    domain="light",
                ),
            ),
        ),
    )
    assert await async_setup(hass, {})

    response = await _call_service(
        hass,
        SERVICE_GET_HOME_SEMANTIC_SUMMARY,
        {"config_entry_id": entry.entry_id},
    )

    assert response["success"] is True
    assert response["ready"] is True
    assert response["domains"] == {"light": 1}
    assert response["errors"] == []


async def test_home_semantic_services_resolve_and_context(
    hass: HomeAssistant,
) -> None:
    """Test semantic response services share LLM API resolution behavior."""
    hass.states.async_set("light.kitchen", "off")
    hass.states.async_set("light.bedroom", "on")
    await _expose(hass, "light.kitchen", "light.bedroom")
    entry, _api = _workspace_with_manager(
        hass,
        HomeSemanticSource(
            areas=(
                AreaSource(area_id="kitchen", name="Kitchen"),
                AreaSource(area_id="bedroom", name="Bedroom"),
            ),
            entities=(
                EntitySource(
                    entity_id="light.kitchen",
                    name="Kitchen Light",
                    area_id="kitchen",
                    domain="light",
                ),
                EntitySource(
                    entity_id="light.bedroom",
                    name="Bedroom Light",
                    area_id="bedroom",
                    domain="light",
                ),
            ),
        ),
    )
    assert await async_setup(hass, {})

    resolved = await _call_service(
        hass,
        SERVICE_RESOLVE_HOME_SEMANTIC_TARGET,
        {
            "config_entry_id": entry.entry_id,
            "phrase": "kitchen light",
            "action": "turn_on",
        },
    )
    context = await _call_service(
        hass,
        SERVICE_GET_HOME_SEMANTIC_CONTEXT,
        {"config_entry_id": entry.entry_id, "area_id": "kitchen"},
    )

    assert resolved["success"] is True
    assert resolved["target"]["entity_id"] == "light.kitchen"
    assert [entity["entity_id"] for entity in context["entities"]] == [
        "light.kitchen"
    ]


async def test_trace_home_semantic_resolution_reports_rejections(
    hass: HomeAssistant,
) -> None:
    """Test trace service returns ranked candidates and rejection reasons."""
    hass.states.async_set("light.kitchen", "off")
    hass.states.async_set("light.hidden", "off")
    await _expose(hass, "light.kitchen")
    await _hide(hass, "light.hidden")
    entry, _api = _workspace_with_manager(
        hass,
        HomeSemanticSource(
            areas=(AreaSource(area_id="kitchen", name="Kitchen"),),
            entities=(
                EntitySource(
                    entity_id="light.kitchen",
                    name="Kitchen Light",
                    area_id="kitchen",
                    domain="light",
                ),
                EntitySource(
                    entity_id="light.hidden",
                    name="Kitchen Hidden Light",
                    area_id="kitchen",
                    domain="light",
                ),
            ),
        ),
    )
    assert await async_setup(hass, {})

    response = await _call_service(
        hass,
        SERVICE_TRACE_HOME_SEMANTIC_RESOLUTION,
        {
            "config_entry_id": entry.entry_id,
            "phrase": "kitchen light",
            "action": "turn_on",
        },
    )

    assert response["success"] is True
    assert response["selected"]["entity_id"] == "light.kitchen"
    assert "light.hidden" not in str(response["candidates"])
    assert "not_exposed" not in str(response["candidates"])


async def test_plan_home_semantic_control_does_not_call_services(
    hass: HomeAssistant,
) -> None:
    """Test dry-run planning returns service calls without executing them."""
    calls: list[ServiceCall] = []

    async def async_record_call(call: ServiceCall) -> None:
        calls.append(call)

    hass.services.async_register("light", "turn_off", async_record_call)
    hass.states.async_set("light.kitchen", "on")
    await _expose(hass, "light.kitchen")
    entry, _api = _workspace_with_manager(
        hass,
        HomeSemanticSource(
            entities=(
                EntitySource(
                    entity_id="light.kitchen",
                    name="Kitchen Light",
                    domain="light",
                ),
            ),
        ),
    )
    assert await async_setup(hass, {})

    response = await _call_service(
        hass,
        SERVICE_PLAN_HOME_SEMANTIC_CONTROL,
        {
            "config_entry_id": entry.entry_id,
            "action": "turn_off",
            "phrase": "kitchen light",
        },
    )

    assert response["allowed"] is True
    assert response["calls"] == [
        {
            "domain": "light",
            "service": "turn_off",
            "target": {ATTR_ENTITY_ID: "light.kitchen"},
        }
    ]
    assert calls == []


async def test_home_semantic_document_and_benchmark_services(
    hass: HomeAssistant,
) -> None:
    """Test compact document lookup and batch benchmark services."""
    hass.states.async_set("light.kitchen", "off")
    await _expose(hass, "light.kitchen")
    entry, _api = _workspace_with_manager(
        hass,
        HomeSemanticSource(
            areas=(AreaSource(area_id="kitchen", name="Kitchen"),),
            entities=(
                EntitySource(
                    entity_id="light.kitchen",
                    name="Kitchen Light",
                    area_id="kitchen",
                    domain="light",
                ),
            ),
        ),
    )
    assert await async_setup(hass, {})

    document = await _call_service(
        hass,
        SERVICE_GET_HOME_SEMANTIC_DOCUMENT,
        {"config_entry_id": entry.entry_id, "entity_id": "light.kitchen"},
    )
    benchmark = await _call_service(
        hass,
        SERVICE_BENCHMARK_HOME_SEMANTIC_RESOLUTION,
        {
            "config_entry_id": entry.entry_id,
            "cases": [
                {
                    "phrase": "kitchen light",
                    "action": "turn_on",
                    "expected_entity_id": "light.kitchen",
                }
            ],
        },
    )

    assert document["document"]["document_id"] == "entity:light.kitchen"
    assert document["document"]["exposed"] is True
    assert benchmark["success"] is True
    assert benchmark["aggregate"] == {"case_count": 1, "passed": 1, "failed": 0}


async def test_home_semantic_document_rejects_unexposed_entity(
    hass: HomeAssistant,
) -> None:
    """Test document lookup does not expose hidden entity details."""
    hass.states.async_set("light.hidden", "off")
    await _hide(hass, "light.hidden")
    entry, _api = _workspace_with_manager(
        hass,
        HomeSemanticSource(
            entities=(
                EntitySource(
                    entity_id="light.hidden",
                    name="Hidden Light",
                    domain="light",
                ),
            ),
        ),
    )
    assert await async_setup(hass, {})

    document = await _call_service(
        hass,
        SERVICE_GET_HOME_SEMANTIC_DOCUMENT,
        {"config_entry_id": entry.entry_id, "entity_id": "light.hidden"},
    )

    assert document["success"] is False
    assert document["errors"][0]["code"] == "not_exposed"
    assert document["document"] is None


async def test_home_semantic_document_rejects_unexposed_area_capability(
    hass: HomeAssistant,
) -> None:
    """Test non-entity document lookup does not expose hidden capabilities."""
    hass.states.async_set("light.hidden", "off")
    await _hide(hass, "light.hidden")
    entry, _api = _workspace_with_manager(
        hass,
        HomeSemanticSource(
            areas=(AreaSource(area_id="kitchen", name="Kitchen"),),
            entities=(
                EntitySource(
                    entity_id="light.hidden",
                    name="Hidden Light",
                    area_id="kitchen",
                    domain="light",
                ),
            ),
        ),
    )
    assert await async_setup(hass, {})

    document = await _call_service(
        hass,
        SERVICE_GET_HOME_SEMANTIC_DOCUMENT,
        {"config_entry_id": entry.entry_id, "document_id": "area:kitchen"},
    )

    assert document["success"] is False
    assert document["errors"][0]["code"] == "not_exposed"
    assert document["document"] is None


async def test_home_semantic_document_filters_hidden_only_area_details(
    hass: HomeAssistant,
) -> None:
    """Test area documents derive tokens and relationships from exposed scope."""
    hass.states.async_set("light.kitchen", "off")
    hass.states.async_set("switch.hidden", "off")
    await _expose(hass, "light.kitchen")
    await _hide(hass, "switch.hidden")
    entry, _api = _workspace_with_manager(
        hass,
        HomeSemanticSource(
            areas=(AreaSource(area_id="kitchen", name="Kitchen"),),
            entities=(
                EntitySource(
                    entity_id="light.kitchen",
                    name="Kitchen Light",
                    area_id="kitchen",
                    domain="light",
                ),
                EntitySource(
                    entity_id="switch.hidden",
                    name="Hidden Switch",
                    area_id="kitchen",
                    domain="switch",
                ),
            ),
        ),
    )
    assert await async_setup(hass, {})

    document = await _call_service(
        hass,
        SERVICE_GET_HOME_SEMANTIC_DOCUMENT,
        {"config_entry_id": entry.entry_id, "document_id": "area:kitchen"},
    )

    assert document["success"] is True
    assert "switch" not in str(document["document"])
    assert document["document"]["capabilities"] == [
        {
            "capability": "lights",
            "entity_count": 1,
            "preferred_target": "light.kitchen",
        }
    ]


async def test_get_agent_run_diagnostics_service_sections(
    hass: HomeAssistant,
) -> None:
    """Test targeted latest-run diagnostics service returns compact sections."""
    subentry_id = "agent-1"
    entry = workspace_entry(
        (
            conversation_subentry_data(
                model_profile_ref("provider-1", "profile-1"),
                subentry_id=subentry_id,
            ),
        ),
        title="Workspace",
    )
    entry.add_to_hass(hass)
    entry.runtime_data = WorkspaceRuntimeData(
        workspace_name="Workspace",
        latest_run_diagnostics={
            subentry_id: {
                "run_id": "run-1",
                "status": "success",
                "started_at": "2026-01-01T00:00:00+00:00",
                "finished_at": "2026-01-01T00:00:01+00:00",
                "duration_ms": 1000,
                "timeline_event_count": 3,
                "timeline": [
                    {
                        "seq": 1,
                        "phase": "input",
                        "event": "messages_prepared",
                        "data": {"llm_tool_definitions": [{"name": "tool"}]},
                    },
                    {"seq": 2, "phase": "tool_call", "event": "call_started"},
                    {"seq": 3, "phase": "tool_call", "event": "call_finished"},
                ],
                "summary": {
                    "output": "ok",
                    "usage": {"requests": 1},
                    "model_profile": "Primary",
                },
            }
        },
    )
    assert await async_setup(hass, {})

    summary = await _call_service(
        hass,
        SERVICE_GET_AGENT_RUN_DIAGNOSTICS,
        {"config_entry_id": entry.entry_id, "subentry_id": subentry_id},
    )
    timeline = await _call_service(
        hass,
        SERVICE_GET_AGENT_RUN_DIAGNOSTICS,
        {
            "config_entry_id": entry.entry_id,
            "subentry_id": subentry_id,
            "section": "timeline",
            "offset": 1,
            "limit": 1,
        },
    )

    assert summary["run"]["summary"]["usage"] == {"requests": 1}
    assert summary["run"]["timeline"]["total_count"] == 3
    assert timeline["run"]["timeline"]["items"] == [
        {"seq": 2, "phase": "tool_call", "event": "call_started"}
    ]


async def test_home_semantic_context_service_reports_errors(
    hass: HomeAssistant,
) -> None:
    """Test context response service returns stable semantic errors."""
    entry = workspace_entry(title="Workspace")
    entry.add_to_hass(hass)
    entry.runtime_data = WorkspaceRuntimeData(
        workspace_name="Workspace",
        home_semantic=HomeSemanticIndexManager(hass, cast(Any, entry)),
    )
    assert await async_setup(hass, {})

    missing_scope = await _call_service(
        hass,
        SERVICE_GET_HOME_SEMANTIC_CONTEXT,
        {"config_entry_id": entry.entry_id},
    )
    not_ready = await _call_service(
        hass,
        SERVICE_GET_HOME_SEMANTIC_CONTEXT,
        {"config_entry_id": entry.entry_id, "phrase": "kitchen"},
    )

    assert missing_scope["success"] is False
    assert missing_scope["errors"][0]["code"] == "scope_required"
    assert not_ready["success"] is False
    assert not_ready["errors"][0]["code"] == "index_not_ready"


async def test_summary_and_resolution_respect_exposed_entities(
    hass: HomeAssistant,
) -> None:
    """Test retrieval tools only return exposed semantic targets."""
    hass.states.async_set("light.bedroom_lights", "on")
    hass.states.async_set("light.secret_lamp", "on")
    await _expose(hass, "light.bedroom_lights")
    await _hide(hass, "light.secret_lamp")
    _, api = _workspace_with_manager(
        hass,
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
                EntitySource(
                    entity_id="light.secret_lamp",
                    name="Secret Lamp",
                    area_id="bedroom",
                    domain="light",
                    platform="hue",
                ),
            ),
        ),
    )

    summary = await _call_tool(api, "get_home_summary")
    resolved = await _call_tool(
        api,
        "resolve_home_target",
        {"phrase": "bedroom lights", "action": "turn_off"},
    )
    secret = await _call_tool(
        api,
        "resolve_home_target",
        {"phrase": "secret lamp", "action": "turn_off"},
    )

    assert summary["domains"] == {"light": 1}
    assert resolved["entity_id"] == "light.bedroom_lights"
    assert secret["code"] == "not_found"


async def test_get_home_context_requires_scope_and_filters_exposure(
    hass: HomeAssistant,
) -> None:
    """Test scoped context never defaults to all entities."""
    hass.states.async_set("light.kitchen", "off")
    hass.states.async_set("light.hidden", "on")
    await _expose(hass, "light.kitchen")
    await _hide(hass, "light.hidden")
    _, api = _workspace_with_manager(
        hass,
        HomeSemanticSource(
            entities=(
                EntitySource(
                    entity_id="light.kitchen",
                    name="Kitchen Light",
                    domain="light",
                ),
                EntitySource(
                    entity_id="light.hidden",
                    name="Hidden Light",
                    domain="light",
                ),
            ),
        ),
    )

    missing_scope = await _call_tool(api, "get_home_context")
    context = await _call_tool(
        api,
        "get_home_context",
        {"entity_ids": ["light.kitchen", "light.hidden"]},
    )

    assert missing_scope["code"] == "scope_required"
    assert context["entities"] == [
        {
            "entity_id": "light.kitchen",
            "domain": "light",
            "name": "Kitchen Light",
            "state": "off",
            "capability": "lights",
        }
    ]


async def test_get_home_context_phrase_filters_specific_scope(
    hass: HomeAssistant,
) -> None:
    """Test phrase context does not include unrelated capability matches."""
    hass.states.async_set("light.bedroom", "off")
    hass.states.async_set("light.kitchen", "on")
    await _expose(hass, "light.bedroom", "light.kitchen")
    _, api = _workspace_with_manager(
        hass,
        HomeSemanticSource(
            areas=(
                AreaSource(area_id="bedroom", name="Bedroom"),
                AreaSource(area_id="kitchen", name="Kitchen"),
            ),
            entities=(
                EntitySource(
                    entity_id="light.bedroom",
                    name="Bedroom Light",
                    area_id="bedroom",
                    domain="light",
                ),
                EntitySource(
                    entity_id="light.kitchen",
                    name="Kitchen Light",
                    area_id="kitchen",
                    domain="light",
                ),
            ),
        ),
    )

    context = await _call_tool(
        api,
        "get_home_context",
        {"phrase": "bedroom lights"},
    )

    assert [entity["entity_id"] for entity in context["entities"]] == ["light.bedroom"]


async def test_get_home_context_reports_index_not_ready_for_semantic_scopes(
    hass: HomeAssistant,
) -> None:
    """Test semantic scopes do not look like empty results during warmup."""
    entry = workspace_entry(title="Workspace")
    entry.add_to_hass(hass)
    entry.runtime_data = WorkspaceRuntimeData(
        workspace_name="Workspace",
        home_semantic=HomeSemanticIndexManager(hass, cast(Any, entry)),
    )
    api = HomeSemanticAPI(hass, cast(Any, entry))

    domain_context = await _call_tool(
        api,
        "get_home_context",
        {"domain": "light"},
    )
    area_context = await _call_tool(
        api,
        "get_home_context",
        {"area_id": "bedroom"},
    )

    assert domain_context["code"] == "index_not_ready"
    assert area_context["code"] == "index_not_ready"


async def test_control_home_calls_constrained_light_service(
    hass: HomeAssistant,
) -> None:
    """Test control_home executes a constrained exposed light action."""
    calls: list[ServiceCall] = []

    async def async_record_call(call: ServiceCall) -> None:
        calls.append(call)

    hass.services.async_register("light", "turn_off", async_record_call)
    hass.states.async_set("light.bedroom_lights", "on")
    await _expose(hass, "light.bedroom_lights")
    _, api = _workspace_with_manager(
        hass,
        HomeSemanticSource(
            entities=(
                EntitySource(
                    entity_id="light.bedroom_lights",
                    name="Bedroom Lights",
                    domain="light",
                    platform="group",
                ),
            ),
        ),
    )

    result = await _call_tool(
        api,
        "control_home",
        {"action": "turn_off", "entity_id": "light.bedroom_lights"},
    )

    assert result == {
        "status": "ok",
        "action": "turn_off",
        "domain": "light",
        "service": "turn_off",
        "target": {ATTR_ENTITY_ID: "light.bedroom_lights"},
    }
    assert calls[0].data == {ATTR_ENTITY_ID: "light.bedroom_lights"}


async def test_control_home_rejects_generic_capability_only_phrase(
    hass: HomeAssistant,
) -> None:
    """Test phrase control requires a specific target match."""
    calls: list[ServiceCall] = []

    async def async_record_call(call: ServiceCall) -> None:
        calls.append(call)

    hass.services.async_register("light", "turn_off", async_record_call)
    hass.states.async_set("light.bedroom_lights", "on")
    await _expose(hass, "light.bedroom_lights")
    _, api = _workspace_with_manager(
        hass,
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
        ),
    )

    result = await _call_tool(
        api,
        "control_home",
        {"action": "turn_off", "phrase": "attic lights"},
    )

    assert result["code"] == "not_found"
    assert calls == []


async def test_control_home_accepts_common_phrase_prepositions(
    hass: HomeAssistant,
) -> None:
    """Test phrase control ignores common prepositions for target matching."""
    calls: list[ServiceCall] = []

    async def async_record_call(call: ServiceCall) -> None:
        calls.append(call)

    hass.services.async_register("light", "turn_off", async_record_call)
    hass.states.async_set("light.bedroom_lights", "on")
    await _expose(hass, "light.bedroom_lights")
    _, api = _workspace_with_manager(
        hass,
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
        ),
    )

    result = await _call_tool(
        api,
        "control_home",
        {"action": "turn_off", "phrase": "lights in bedroom"},
    )

    assert result["status"] == "ok"
    assert calls[0].data == {ATTR_ENTITY_ID: "light.bedroom_lights"}


async def test_control_home_accepts_area_entity_phrase(
    hass: HomeAssistant,
) -> None:
    """Test entity documents include area names for specific phrases."""
    calls: list[ServiceCall] = []

    async def async_record_call(call: ServiceCall) -> None:
        calls.append(call)

    hass.services.async_register("light", "turn_off", async_record_call)
    hass.states.async_set("light.bedroom_ceiling", "on")
    await _expose(hass, "light.bedroom_ceiling")
    _, api = _workspace_with_manager(
        hass,
        HomeSemanticSource(
            areas=(AreaSource(area_id="bedroom", name="Bedroom"),),
            entities=(
                EntitySource(
                    entity_id="light.bedroom_ceiling",
                    name="Ceiling",
                    area_id="bedroom",
                    domain="light",
                ),
            ),
        ),
    )

    result = await _call_tool(
        api,
        "control_home",
        {"action": "turn_off", "phrase": "turn off bedroom ceiling"},
    )

    assert result["status"] == "ok"
    assert calls[0].data == {ATTR_ENTITY_ID: "light.bedroom_ceiling"}


async def test_control_home_rejects_ambiguous_ungrouped_area_capability(
    hass: HomeAssistant,
) -> None:
    """Test area capability phrases do not pick one ungrouped target."""
    calls: list[ServiceCall] = []

    async def async_record_call(call: ServiceCall) -> None:
        calls.append(call)

    hass.services.async_register("light", "turn_off", async_record_call)
    hass.states.async_set("light.bedroom_ceiling", "on")
    hass.states.async_set("light.bedroom_lamp", "on")
    await _expose(hass, "light.bedroom_ceiling", "light.bedroom_lamp")
    _, api = _workspace_with_manager(
        hass,
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
        ),
    )

    result = await _call_tool(
        api,
        "control_home",
        {"action": "turn_off", "phrase": "bedroom lights"},
    )

    assert result["code"] == "ambiguous_target"
    assert calls == []


async def test_control_home_accepts_action_words_in_phrase(
    hass: HomeAssistant,
) -> None:
    """Test phrase confidence ignores action wording when target is specific."""
    calls: list[ServiceCall] = []

    async def async_record_call(call: ServiceCall) -> None:
        calls.append(call)

    hass.services.async_register("light", "turn_off", async_record_call)
    hass.states.async_set("light.kitchen", "on")
    await _expose(hass, "light.kitchen")
    _, api = _workspace_with_manager(
        hass,
        HomeSemanticSource(
            entities=(
                EntitySource(
                    entity_id="light.kitchen",
                    name="Kitchen Light",
                    domain="light",
                ),
            ),
        ),
    )

    result = await _call_tool(
        api,
        "control_home",
        {"action": "turn_off", "phrase": "turn off kitchen light"},
    )

    assert result["status"] == "ok"
    assert calls[0].data == {ATTR_ENTITY_ID: "light.kitchen"}


async def test_control_home_skips_action_incompatible_phrase_targets(
    hass: HomeAssistant,
) -> None:
    """Test phrase resolution filters targets incompatible with the action."""
    calls: list[ServiceCall] = []

    async def async_record_call(call: ServiceCall) -> None:
        calls.append(call)

    hass.services.async_register("scene", "turn_on", async_record_call)
    hass.states.async_set("scene.movie_scene", "off")
    await _expose(hass, "scene.movie_scene")
    _, api = _workspace_with_manager(
        hass,
        HomeSemanticSource(
            entities=(
                EntitySource(
                    entity_id="scene.movie_scene",
                    name="Movie Scene",
                    domain="scene",
                ),
            ),
        ),
    )

    result = await _call_tool(
        api,
        "control_home",
        {"action": "turn_off", "phrase": "movie scene"},
    )

    assert result["code"] == "not_found"
    assert calls == []


async def test_resolve_home_target_rejects_generic_phrase_only(
    hass: HomeAssistant,
) -> None:
    """Test semantic resolution requires a specific target match."""
    hass.states.async_set("light.bedroom_lights", "on")
    await _expose(hass, "light.bedroom_lights")
    _, api = _workspace_with_manager(
        hass,
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
        ),
    )

    result = await _call_tool(
        api,
        "resolve_home_target",
        {"action": "turn_off", "phrase": "attic lights"},
    )

    assert result["code"] == "not_found"


async def test_control_home_rejects_unexposed_target(hass: HomeAssistant) -> None:
    """Test control_home rejects unexposed explicit targets."""
    hass.states.async_set("switch.secret", "on")
    await _hide(hass, "switch.secret")
    _, api = _workspace_with_manager(
        hass,
        HomeSemanticSource(
            entities=(
                EntitySource(
                    entity_id="switch.secret",
                    name="Secret Switch",
                    domain="switch",
                ),
            ),
        ),
    )

    result = await _call_tool(
        api,
        "control_home",
        {"action": "turn_off", "entity_id": "switch.secret"},
    )

    assert result["code"] == "not_exposed"


async def test_control_home_expands_exposed_group_members(
    hass: HomeAssistant,
) -> None:
    """Test HA groups expand to exposed supported members only."""
    calls: list[ServiceCall] = []

    async def async_record_call(call: ServiceCall) -> None:
        calls.append(call)

    hass.services.async_register("light", "turn_off", async_record_call)
    hass.states.async_set(
        "group.downstairs",
        "on",
        {ATTR_ENTITY_ID: ("light.kitchen", "light.unexposed")},
    )
    hass.states.async_set("light.kitchen", "on")
    hass.states.async_set("light.unexposed", "on")
    await _expose(hass, "group.downstairs", "light.kitchen")
    await _hide(hass, "light.unexposed")
    _, api = _workspace_with_manager(
        hass,
        HomeSemanticSource(
            entities=(
                EntitySource(
                    entity_id="group.downstairs",
                    name="Downstairs",
                    domain="group",
                ),
            ),
        ),
    )

    result = await _call_tool(
        api,
        "control_home",
        {"action": "turn_off", "entity_id": "group.downstairs"},
    )

    assert result["calls"] == [
        {
            "domain": "light",
            "service": "turn_off",
            "target": {ATTR_ENTITY_ID: ["light.kitchen"]},
        }
    ]
    assert calls[0].data == {ATTR_ENTITY_ID: ["light.kitchen"]}


async def test_control_home_expands_mixed_domain_groups(
    hass: HomeAssistant,
) -> None:
    """Test HA groups can control exposed members across supported domains."""
    calls: list[ServiceCall] = []

    async def async_record_call(call: ServiceCall) -> None:
        calls.append(call)

    hass.services.async_register("light", "turn_off", async_record_call)
    hass.services.async_register("switch", "turn_off", async_record_call)
    hass.states.async_set(
        "group.downstairs",
        "on",
        {ATTR_ENTITY_ID: ("light.kitchen", "switch.fan", "light.unexposed")},
    )
    hass.states.async_set("light.kitchen", "on")
    hass.states.async_set("switch.fan", "on")
    hass.states.async_set("light.unexposed", "on")
    await _expose(hass, "group.downstairs", "light.kitchen", "switch.fan")
    await _hide(hass, "light.unexposed")
    _, api = _workspace_with_manager(
        hass,
        HomeSemanticSource(
            entities=(
                EntitySource(
                    entity_id="group.downstairs",
                    name="Downstairs",
                    domain="group",
                ),
            ),
        ),
    )

    result = await _call_tool(
        api,
        "control_home",
        {"action": "turn_off", "entity_id": "group.downstairs"},
    )

    assert result["calls"] == [
        {
            "domain": "light",
            "service": "turn_off",
            "target": {ATTR_ENTITY_ID: ["light.kitchen"]},
        },
        {
            "domain": "switch",
            "service": "turn_off",
            "target": {ATTR_ENTITY_ID: ["switch.fan"]},
        },
    ]
    assert [call.domain for call in calls] == ["light", "switch"]
    assert [call.data for call in calls] == [
        {ATTR_ENTITY_ID: ["light.kitchen"]},
        {ATTR_ENTITY_ID: ["switch.fan"]},
    ]


async def test_semantic_api_tools_convert_to_pydantic_ai_tools(
    hass: HomeAssistant,
) -> None:
    """Test existing HA tool adapter can consume the semantic API."""
    hass.states.async_set("light.kitchen", "on")
    await _expose(hass, "light.kitchen")
    _, api = _workspace_with_manager(
        hass,
        HomeSemanticSource(
            entities=(
                EntitySource(
                    entity_id="light.kitchen",
                    name="Kitchen Light",
                    domain="light",
                ),
            ),
        ),
    )
    api_instance = await api.async_get_api_instance(_llm_context())

    tools = tools_from_llm_api(api_instance)

    assert {tool.name for tool in tools} == {
        "control_home",
        "get_home_context",
        "get_home_summary",
        "resolve_home_target",
    }
