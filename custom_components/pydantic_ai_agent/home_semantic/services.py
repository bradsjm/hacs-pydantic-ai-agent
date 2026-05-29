"""Response services for the Home Semantic Index."""

from typing import Any
from time import monotonic

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
import voluptuous as vol

from ..const import DOMAIN
from .manager import HomeSemanticIndexManager
from .query import (
    DEFAULT_CONTEXT_LIMIT,
    SUPPORTED_ACTIONS,
    default_assistant_id,
    entity_document,
    error,
    get_home_context,
    get_home_summary,
    is_exposed,
    normalize_tokens,
    plan_home_control,
    resolve_home_target,
    supported_entity,
    trace_home_resolution,
)

SERVICE_REFRESH_HOME_SEMANTIC_INDEX = "refresh_home_semantic_index"
SERVICE_TRACE_HOME_SEMANTIC_RESOLUTION = "trace_home_semantic_resolution"
SERVICE_PLAN_HOME_SEMANTIC_CONTROL = "plan_home_semantic_control"
SERVICE_GET_HOME_SEMANTIC_DOCUMENT = "get_home_semantic_document"
SERVICE_BENCHMARK_HOME_SEMANTIC_RESOLUTION = "benchmark_home_semantic_resolution"
SERVICE_GET_HOME_SEMANTIC_SUMMARY = "get_home_semantic_summary"
SERVICE_RESOLVE_HOME_SEMANTIC_TARGET = "resolve_home_semantic_target"
SERVICE_GET_HOME_SEMANTIC_CONTEXT = "get_home_semantic_context"

ATTR_ASSISTANT_ID = "assistant_id"
ATTR_CONFIG_ENTRY_ID = "config_entry_id"
ATTR_ACTION = "action"
ATTR_AREA_ID = "area_id"
ATTR_DOMAIN = "domain"
ATTR_DOCUMENT_ID = "document_id"
ATTR_ENTITY_IDS = "entity_ids"
ATTR_ENTITY_ID = "entity_id"
ATTR_LIMIT = "limit"
ATTR_PHRASE = "phrase"
ATTR_REASON = "reason"
ATTR_WAIT = "wait"
ATTR_CASES = "cases"
ATTR_EXPECTED_TARGET = "expected_target"
ATTR_EXPECTED_ENTITY_ID = "expected_entity_id"

_MAX_TRACE_LIMIT = 50

_SUMMARY_SERVICE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CONFIG_ENTRY_ID): str,
        vol.Optional(ATTR_ASSISTANT_ID): str,
    }
)
_RESOLVE_SERVICE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CONFIG_ENTRY_ID): str,
        vol.Optional(ATTR_ASSISTANT_ID): str,
        vol.Required(ATTR_PHRASE): str,
        vol.Optional(ATTR_ACTION): vol.In(SUPPORTED_ACTIONS),
    }
)
_CONTEXT_SERVICE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CONFIG_ENTRY_ID): str,
        vol.Optional(ATTR_ASSISTANT_ID): str,
        vol.Optional(ATTR_ENTITY_IDS): [str],
        vol.Optional(ATTR_PHRASE): str,
        vol.Optional(ATTR_DOMAIN): str,
        vol.Optional(ATTR_AREA_ID): str,
        vol.Optional(ATTR_LIMIT, default=DEFAULT_CONTEXT_LIMIT): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=100)
        ),
    }
)
_REFRESH_SERVICE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CONFIG_ENTRY_ID): str,
        vol.Optional(ATTR_REASON, default="service_request"): str,
        vol.Optional(ATTR_WAIT, default=True): bool,
    }
)
_TRACE_SERVICE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CONFIG_ENTRY_ID): str,
        vol.Optional(ATTR_ASSISTANT_ID): str,
        vol.Required(ATTR_PHRASE): str,
        vol.Optional(ATTR_ACTION): vol.In(SUPPORTED_ACTIONS),
        vol.Optional(ATTR_AREA_ID): str,
        vol.Optional(ATTR_DOMAIN): str,
        vol.Optional(ATTR_LIMIT, default=10): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=_MAX_TRACE_LIMIT)
        ),
    }
)
_PLAN_SERVICE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CONFIG_ENTRY_ID): str,
        vol.Optional(ATTR_ASSISTANT_ID): str,
        vol.Required(ATTR_ACTION): vol.In(SUPPORTED_ACTIONS),
        vol.Optional(ATTR_PHRASE): str,
        vol.Optional(ATTR_ENTITY_ID): str,
    }
)
_DOCUMENT_SERVICE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CONFIG_ENTRY_ID): str,
        vol.Optional(ATTR_ASSISTANT_ID): str,
        vol.Optional(ATTR_ENTITY_ID): str,
        vol.Optional(ATTR_DOCUMENT_ID): str,
    }
)
_BENCHMARK_CASE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_PHRASE): str,
        vol.Optional(ATTR_ACTION): vol.In(SUPPORTED_ACTIONS),
        vol.Optional(ATTR_EXPECTED_TARGET): str,
        vol.Optional(ATTR_EXPECTED_ENTITY_ID): str,
    }
)
_BENCHMARK_SERVICE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CONFIG_ENTRY_ID): str,
        vol.Optional(ATTR_ASSISTANT_ID): str,
        vol.Required(ATTR_CASES): [_BENCHMARK_CASE_SCHEMA],
        vol.Optional(ATTR_LIMIT, default=20): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=100)
        ),
    }
)


async def async_setup_services(hass: HomeAssistant) -> None:
    """Register home semantic response services."""

    async def async_refresh_index(call: ServiceCall) -> dict[str, Any]:
        """Handle the waitable semantic index refresh response service."""
        entry, manager, service_error = _service_target(
            hass, call.data[ATTR_CONFIG_ENTRY_ID]
        )
        if service_error is not None:
            return service_error
        assert entry is not None and manager is not None
        refresh = await manager.async_refresh_now(
            reason=call.data[ATTR_REASON],
            wait=call.data[ATTR_WAIT],
        )
        return {
            **_response_base(
                entry,
                manager,
                success=refresh["ready"] is True,
                errors=[] if refresh["ready"] is True else [_index_not_ready_error()],
            ),
            "refresh": refresh,
        }

    async def async_trace_resolution(call: ServiceCall) -> dict[str, Any]:
        """Handle traceable home semantic resolution."""
        entry, manager, service_error = _service_target(
            hass, call.data[ATTR_CONFIG_ENTRY_ID]
        )
        if service_error is not None:
            return service_error
        result = trace_home_resolution(
            hass,
            manager,
            assistant_id=call.data.get(ATTR_ASSISTANT_ID),
            phrase=call.data[ATTR_PHRASE],
            action=call.data.get(ATTR_ACTION),
            area_id=call.data.get(ATTR_AREA_ID),
            domain=call.data.get(ATTR_DOMAIN),
            limit=call.data[ATTR_LIMIT],
        )
        if result.get("status") == "error":
            return {
                **_response_base(
                    entry,
                    manager,
                    assistant_id=call.data.get(ATTR_ASSISTANT_ID),
                    success=False,
                    errors=[result],
                ),
                "selected": None,
                "candidates": [],
            }
        return {
            **_response_base(
                entry,
                manager,
                assistant_id=call.data.get(ATTR_ASSISTANT_ID),
                success=result["selected"] is not None,
                errors=[] if result["selected"] is not None else [
                    error("not_found", "No exposed supported target matched.")
                ],
            ),
            **result,
        }

    async def async_plan_control(call: ServiceCall) -> dict[str, Any]:
        """Handle dry-run semantic control planning."""
        entry, manager, service_error = _service_target(
            hass, call.data[ATTR_CONFIG_ENTRY_ID]
        )
        if service_error is not None:
            return service_error
        plan = plan_home_control(
            hass,
            manager,
            assistant_id=call.data.get(ATTR_ASSISTANT_ID),
            action=call.data[ATTR_ACTION],
            phrase=call.data.get(ATTR_PHRASE),
            entity_id=call.data.get(ATTR_ENTITY_ID),
        )
        if isinstance(plan, dict):
            return {
                **_response_base(
                    entry,
                    manager,
                    assistant_id=call.data.get(ATTR_ASSISTANT_ID),
                    success=False,
                    errors=[plan],
                ),
                "allowed": False,
                "reason": plan["code"],
                "target": None,
                "calls": [],
            }
        return {
            **_response_base(
                entry,
                manager,
                assistant_id=call.data.get(ATTR_ASSISTANT_ID),
                success=True,
                errors=[],
            ),
            **plan.as_dict(),
        }

    async def async_get_document(call: ServiceCall) -> dict[str, Any]:
        """Handle compact semantic document lookup."""
        entry, manager, service_error = _service_target(
            hass, call.data[ATTR_CONFIG_ENTRY_ID]
        )
        if service_error is not None:
            return service_error
        assert entry is not None and manager is not None
        document = _document_for_call(
            manager,
            entity_id=call.data.get(ATTR_ENTITY_ID),
            document_id=call.data.get(ATTR_DOCUMENT_ID),
        )
        if isinstance(document, dict):
            return {
                **_response_base(
                    entry,
                    manager,
                    assistant_id=call.data.get(ATTR_ASSISTANT_ID),
                    success=False,
                    errors=[document],
                ),
                "document": None,
            }
        assistant = default_assistant_id(call.data.get(ATTR_ASSISTANT_ID))
        if not _document_is_visible(hass, manager, assistant, document):
            service_error = error("not_exposed", "Semantic document is not exposed.")
            return {
                **_response_base(
                    entry,
                    manager,
                    assistant_id=call.data.get(ATTR_ASSISTANT_ID),
                    success=False,
                    errors=[service_error],
                ),
                "document": None,
            }
        return {
            **_response_base(
                entry,
                manager,
                assistant_id=call.data.get(ATTR_ASSISTANT_ID),
                success=True,
                errors=[],
            ),
            "document": _compact_document(hass, manager, assistant, document),
        }

    async def async_benchmark_resolution(call: ServiceCall) -> dict[str, Any]:
        """Handle small semantic resolution benchmark batches."""
        entry, manager, service_error = _service_target(
            hass, call.data[ATTR_CONFIG_ENTRY_ID]
        )
        if service_error is not None:
            return service_error
        cases: list[dict[str, Any]] = []
        passed = 0
        failed = 0
        for case in call.data[ATTR_CASES][: call.data[ATTR_LIMIT]]:
            started = monotonic()
            result = resolve_home_target(
                hass,
                manager,
                assistant_id=call.data.get(ATTR_ASSISTANT_ID),
                phrase=case[ATTR_PHRASE],
                action=case.get(ATTR_ACTION),
            )
            expected = case.get(ATTR_EXPECTED_ENTITY_ID) or case.get(
                ATTR_EXPECTED_TARGET
            )
            selected_entity_id = None if isinstance(result, dict) else result.entity_id
            matched = expected is None or selected_entity_id == expected
            passed += int(matched)
            failed += int(not matched)
            cases.append(
                {
                    "phrase": case[ATTR_PHRASE],
                    "action": case.get(ATTR_ACTION),
                    "expected_entity_id": expected,
                    "selected_entity_id": selected_entity_id,
                    "confidence": None
                    if isinstance(result, dict)
                    else round(result.confidence, 2),
                    "passed": matched,
                    "latency_ms": round((monotonic() - started) * 1000, 3),
                    "error": result if isinstance(result, dict) else None,
                }
            )
        return {
            **_response_base(
                entry,
                manager,
                assistant_id=call.data.get(ATTR_ASSISTANT_ID),
                success=failed == 0,
                errors=[],
            ),
            "cases": cases,
            "aggregate": {
                "case_count": len(cases),
                "passed": passed,
                "failed": failed,
            },
        }

    async def async_get_summary(call: ServiceCall) -> dict[str, Any]:
        """Handle the home semantic summary response service."""
        entry, manager, service_error = _service_target(
            hass, call.data[ATTR_CONFIG_ENTRY_ID]
        )
        if service_error is not None:
            return service_error
        result = get_home_summary(
            hass,
            manager,
            assistant_id=call.data.get(ATTR_ASSISTANT_ID),
        )
        errors = [] if result["ready"] else [_index_not_ready_error()]
        return {
            **_response_base(
                entry,
                manager,
                assistant_id=call.data.get(ATTR_ASSISTANT_ID),
                success=result["ready"],
                errors=errors,
            ),
            "areas": result["areas"],
            "domains": result["domains"],
        }

    async def async_resolve_target(call: ServiceCall) -> dict[str, Any]:
        """Handle the home semantic target resolution response service."""
        entry, manager, service_error = _service_target(
            hass, call.data[ATTR_CONFIG_ENTRY_ID]
        )
        if service_error is not None:
            return service_error
        result = resolve_home_target(
            hass,
            manager,
            assistant_id=call.data.get(ATTR_ASSISTANT_ID),
            phrase=call.data[ATTR_PHRASE],
            action=call.data.get(ATTR_ACTION),
        )
        if isinstance(result, dict):
            return {
                **_response_base(
                    entry,
                    manager,
                    assistant_id=call.data.get(ATTR_ASSISTANT_ID),
                    success=False,
                    errors=[result],
                ),
                "target": None,
            }
        return {
            **_response_base(
                entry,
                manager,
                assistant_id=call.data.get(ATTR_ASSISTANT_ID),
                success=True,
                errors=[],
            ),
            "target": {
                "target_type": "entity",
                "entity_id": result.entity_id,
                "confidence": round(result.confidence, 2),
                "reason": result.reason,
                "alternatives": result.alternatives,
            },
        }

    async def async_get_context(call: ServiceCall) -> dict[str, Any]:
        """Handle the home semantic context response service."""
        entry, manager, service_error = _service_target(
            hass, call.data[ATTR_CONFIG_ENTRY_ID]
        )
        if service_error is not None:
            return service_error
        result = get_home_context(
            hass,
            manager,
            assistant_id=call.data.get(ATTR_ASSISTANT_ID),
            entity_ids=call.data.get(ATTR_ENTITY_IDS),
            phrase=call.data.get(ATTR_PHRASE),
            domain=call.data.get(ATTR_DOMAIN),
            area_id=call.data.get(ATTR_AREA_ID),
            limit=call.data[ATTR_LIMIT],
        )
        if result.get("status") == "error":
            return {
                **_response_base(
                    entry,
                    manager,
                    assistant_id=call.data.get(ATTR_ASSISTANT_ID),
                    success=False,
                    errors=[result],
                ),
                "entities": [],
            }
        return {
            **_response_base(
                entry,
                manager,
                assistant_id=call.data.get(ATTR_ASSISTANT_ID),
                success=True,
                errors=[],
            ),
            "entities": result["entities"],
        }

    hass.services.async_register(
        DOMAIN,
        SERVICE_REFRESH_HOME_SEMANTIC_INDEX,
        async_refresh_index,
        schema=_REFRESH_SERVICE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_TRACE_HOME_SEMANTIC_RESOLUTION,
        async_trace_resolution,
        schema=_TRACE_SERVICE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_PLAN_HOME_SEMANTIC_CONTROL,
        async_plan_control,
        schema=_PLAN_SERVICE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_HOME_SEMANTIC_DOCUMENT,
        async_get_document,
        schema=_DOCUMENT_SERVICE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_BENCHMARK_HOME_SEMANTIC_RESOLUTION,
        async_benchmark_resolution,
        schema=_BENCHMARK_SERVICE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_HOME_SEMANTIC_SUMMARY,
        async_get_summary,
        schema=_SUMMARY_SERVICE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RESOLVE_HOME_SEMANTIC_TARGET,
        async_resolve_target,
        schema=_RESOLVE_SERVICE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_HOME_SEMANTIC_CONTEXT,
        async_get_context,
        schema=_CONTEXT_SERVICE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )


def _service_target(
    hass: HomeAssistant, entry_id: str
) -> tuple[
    ConfigEntry[Any] | None, HomeSemanticIndexManager | None, dict[str, Any] | None
]:
    """Return the entry and semantic manager for a response service call."""
    entry = hass.config_entries.async_get_entry(entry_id)
    if entry is None or entry.domain != DOMAIN:
        return None, None, _error_response(
            "config_entry_not_found", "Pydantic AI Agent config entry was not found."
        )
    runtime_data = getattr(entry, "runtime_data", None)
    manager = getattr(runtime_data, "home_semantic", None)
    if manager is None:
        return entry, None, _error_response(
            "entry_not_loaded", "Home Semantic Index is not loaded.", entry=entry
        )
    return entry, manager, None


def _response_base(
    entry: ConfigEntry[Any] | None,
    manager: HomeSemanticIndexManager | None,
    *,
    assistant_id: str | None = None,
    success: bool,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return common JSON-serializable response fields."""
    return {
        "success": success,
        "ready": manager is not None and manager.index is not None,
        "status": "entry_not_loaded" if manager is None else manager.status,
        "generation": None if manager is None else manager.generation,
        "config_entry_id": None if entry is None else entry.entry_id,
        "assistant_id": assistant_id or conversation.DOMAIN,
        "errors": errors,
    }


def _error_response(
    code: str,
    message: str,
    *,
    entry: ConfigEntry[Any] | None = None,
) -> dict[str, Any]:
    """Return a response-service error payload."""
    return {
        **_response_base(entry, None, success=False, errors=[error(code, message)]),
        "areas": [],
        "domains": {},
        "target": None,
        "entities": [],
    }


def _index_not_ready_error() -> dict[str, Any]:
    """Return the standard not-ready semantic index error."""
    return error("index_not_ready", "Semantic home index is still warming up.")


def _document_for_call(
    manager: HomeSemanticIndexManager,
    *,
    entity_id: str | None,
    document_id: str | None,
) -> Any:
    """Return the requested document or an error payload."""
    if entity_id is None and document_id is None:
        return error("target_required", "Provide entity_id or document_id.")
    if entity_id is not None and document_id is not None:
        return error("target_ambiguous", "Provide only one of entity_id or document_id.")
    if manager.index is None:
        return _index_not_ready_error()
    if entity_id is not None:
        document = entity_document(manager.index, entity_id)
    else:
        document = manager.index.documents_by_id.get(str(document_id))
    if document is None:
        return error("not_found", "Semantic document was not found.")
    return document


def _compact_document(
    hass: HomeAssistant,
    manager: HomeSemanticIndexManager,
    assistant_id: str,
    document: Any,
) -> dict[str, Any]:
    """Return compact JSON-safe document data for diagnostics."""
    relationships = []
    if manager.index is not None:
        relationships = [
            edge.as_dict()
            for edge in manager.index.edges
            if edge.source_id == document.document_id or edge.target_id == document.document_id
            if _edge_is_exposed(hass, manager, assistant_id, edge.source_id)
            and _edge_is_exposed(hass, manager, assistant_id, edge.target_id)
        ][:20]
    exposed = (
        document.entity_id is not None
        and hass.states.get(document.entity_id) is not None
        and is_exposed(hass, assistant_id, document.entity_id)
    )
    control_supported = (
        document.entity_id is not None
        and supported_entity(hass, document.entity_id, for_control=True)
    )
    return {
        "document_id": document.document_id,
        "kind": document.document_type,
        "name": document.name,
        "area_id": document.area_id,
        "domain": document.domain,
        "entity_id": document.entity_id,
        "target_entity_id": document.target_entity_id
        if document.target_entity_id is not None
        and is_exposed(hass, assistant_id, document.target_entity_id)
        else None,
        "capabilities": _compact_capabilities(hass, manager, assistant_id, document),
        "capability": document.capability,
        "aliases": list(document.aliases),
        "tokens": _safe_document_tokens(hass, manager, assistant_id, document),
        "relationships": relationships,
        "exposed": exposed,
        "control_supported": control_supported,
        "preferred_target": document.rank.preferred_target,
        "group": document.rank.group,
    }


def _edge_is_exposed(
    hass: HomeAssistant,
    manager: HomeSemanticIndexManager,
    assistant_id: str,
    document_id: str,
) -> bool:
    """Return whether an edge endpoint can be exposed in document diagnostics."""
    if manager.index is None:
        return False
    document = manager.index.documents_by_id.get(document_id)
    if document is None or document.entity_id is None:
        return document is not None and _document_is_visible(
            hass, manager, assistant_id, document
        )
    return is_exposed(hass, assistant_id, document.entity_id)


def _document_is_visible(
    hass: HomeAssistant,
    manager: HomeSemanticIndexManager,
    assistant_id: str,
    document: Any,
) -> bool:
    """Return whether a document can be exposed through response services."""
    if manager.index is None:
        return False
    if document.entity_id is not None:
        return is_exposed(hass, assistant_id, document.entity_id)
    if document.target_entity_id is not None and is_exposed(
        hass, assistant_id, document.target_entity_id
    ):
        return True
    return any(
        _document_matches_scope(document, candidate)
        and candidate.entity_id is not None
        and is_exposed(hass, assistant_id, candidate.entity_id)
        for candidate in manager.index.documents
    )


def _document_matches_scope(document: Any, candidate: Any) -> bool:
    """Return whether an entity document belongs to another document's scope."""
    if candidate.entity_id is None:
        return False
    if document.document_type == "area":
        return candidate.area_id == document.area_id
    if document.document_type == "device":
        return candidate.device_id == document.device_id
    if document.document_type == "capability":
        return (
            candidate.area_id == document.area_id
            and candidate.capability == document.capability
        )
    if document.document_type == "floor":
        return candidate.floor_id == document.floor_id
    return False


def _compact_capabilities(
    hass: HomeAssistant,
    manager: HomeSemanticIndexManager,
    assistant_id: str,
    document: Any,
) -> list[dict[str, Any]]:
    """Return capability counts derived only from exposed entity documents."""
    if manager.index is None:
        return []
    counts: dict[str, int] = {}
    preferred: dict[str, str] = {}
    for candidate in manager.index.documents:
        if (
            candidate.entity_id is None
            or candidate.capability is None
            or not _document_matches_scope(document, candidate)
            or not is_exposed(hass, assistant_id, candidate.entity_id)
        ):
            continue
        counts[candidate.capability] = counts.get(candidate.capability, 0) + 1
        if candidate.rank.preferred_target:
            preferred[candidate.capability] = candidate.entity_id
    return [
        {
            "capability": capability,
            "entity_count": count,
            **(
                {"preferred_target": preferred[capability]}
                if capability in preferred
                else {}
            ),
        }
        for capability, count in sorted(counts.items())
    ]


def _safe_document_tokens(
    hass: HomeAssistant,
    manager: HomeSemanticIndexManager,
    assistant_id: str,
    document: Any,
) -> list[str]:
    """Return document tokens without hidden-only capabilities."""
    if document.entity_id is not None:
        parts = document.searchable_parts()
    else:
        capabilities = [
            capability["capability"]
            for capability in _compact_capabilities(
                hass, manager, assistant_id, document
            )
        ]
        parts = (
            document.name,
            document.document_type,
            document.floor_id,
            document.area_id,
            document.device_id,
            document.domain,
            document.capability,
            *document.aliases,
            *document.labels,
            *capabilities,
        )
    return sorted(set(normalize_tokens(" ".join(part for part in parts if part))))[:50]
