"""Shared query helpers for the Home Semantic Index."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from homeassistant.components import conversation
from homeassistant.components.homeassistant import async_should_expose
from homeassistant.core import HomeAssistant

from .index import HomeSemanticIndex, normalize_tokens
from .manager import HomeSemanticIndexManager
from .models import HomeSemanticDocument

_MAX_SUMMARY_AREAS = 20
DEFAULT_CONTEXT_LIMIT = 20
_MAX_ALTERNATIVES = 5
SUPPORTED_CONTROL_DOMAINS = {"group", "light", "scene", "script", "switch"}
_TOGGLE_DOMAINS = {"light", "switch"}
_ACTIVATE_DOMAINS = {"scene", "script"}
SUPPORTED_ACTIONS = ("turn_on", "turn_off", "toggle", "activate")
_GENERIC_CONTROL_TOKENS = {
    "a",
    "an",
    "and",
    "activate",
    "all",
    "control",
    "entity",
    "for",
    "group",
    "groups",
    "in",
    "light",
    "lights",
    "of",
    "off",
    "on",
    "please",
    "scene",
    "scenes",
    "script",
    "scripts",
    "switch",
    "switches",
    "to",
    "the",
    "toggle",
    "turn",
}


@dataclass(frozen=True, kw_only=True)
class ResolvedHomeTarget:
    """A resolved exposed control target."""

    entity_id: str
    document: HomeSemanticDocument | None
    confidence: float
    reason: str
    alternatives: list[dict[str, Any]]


def default_assistant_id(assistant_id: str | None = None) -> str:
    """Return the assistant id used for HA exposure checks."""
    return assistant_id or conversation.DOMAIN


def get_home_summary(
    hass: HomeAssistant,
    manager: HomeSemanticIndexManager | None,
    *,
    assistant_id: str | None = None,
) -> dict[str, Any]:
    """Return exposed semantic home summary."""
    index = _index(manager)
    if index is None:
        return {"ready": False, "areas": [], "domains": {}}
    assistant = default_assistant_id(assistant_id)
    domain_counts: Counter[str] = Counter()
    area_capabilities: dict[str, Counter[str]] = defaultdict(Counter)
    preferred: dict[str, list[dict[str, str]]] = defaultdict(list)
    for document in index.documents:
        if document.entity_id is None:
            continue
        if not is_exposed(hass, assistant, document.entity_id):
            continue
        if document.domain not in SUPPORTED_CONTROL_DOMAINS:
            continue
        domain_counts[document.domain or "unknown"] += 1
        area_id = document.area_id or "unknown"
        if document.capability is not None:
            area_capabilities[area_id][document.capability] += 1
        if document.rank.preferred_target:
            preferred[area_id].append(
                {"entity_id": document.entity_id, "name": document.name}
            )
    areas = [
        {
            "area_id": area_id,
            "capabilities": dict(sorted(capabilities.items())),
            "preferred_targets": preferred[area_id][:_MAX_ALTERNATIVES],
        }
        for area_id, capabilities in sorted(area_capabilities.items())[
            :_MAX_SUMMARY_AREAS
        ]
    ]
    return {
        "ready": True,
        "areas": areas,
        "domains": dict(sorted(domain_counts.items())),
    }


def resolve_home_target(
    hass: HomeAssistant,
    manager: HomeSemanticIndexManager | None,
    *,
    assistant_id: str | None = None,
    phrase: str | None = None,
    entity_id: str | None = None,
    action: str | None = None,
) -> ResolvedHomeTarget | dict[str, Any]:
    """Resolve a phrase or explicit entity id to one exposed target."""
    assistant = default_assistant_id(assistant_id)
    index = _index(manager)
    if entity_id is not None:
        document = entity_document(index, entity_id)
        if not is_exposed(hass, assistant, entity_id):
            return error("not_exposed", "Target entity is not exposed.")
        if not supported_entity(hass, entity_id, for_control=True):
            return error("unsupported_domain", "Target domain is not supported.")
        return ResolvedHomeTarget(
            entity_id=entity_id,
            document=document,
            confidence=1.0,
            reason="Explicit exposed entity target",
            alternatives=[],
        )
    if not phrase:
        return error("target_required", "Provide entity_id or phrase.")
    if index is None:
        return error("index_not_ready", "Semantic home index is still warming up.")
    candidates: list[ResolvedHomeTarget] = []
    alternatives: list[dict[str, Any]] = []
    for result in index.search(
        phrase,
        action=action,
        document_types=("capability", "entity", "group"),
        limit=100,
    ):
        target_entity_id = result.document.entity_id or result.document.target_entity_id
        if target_entity_id is None:
            continue
        state = hass.states.get(target_entity_id)
        if state is None:
            continue
        if not is_exposed(hass, assistant, target_entity_id):
            continue
        if not supported_entity(hass, target_entity_id, for_control=True):
            continue
        if (
            action is not None
            and state.domain != "group"
            and service_for_action(state.domain, action)[0] == ""
        ):
            continue
        if not phrase_matches_specific_target(phrase, result.document):
            continue
        confidence = min(0.99, max(0.1, result.score / 60))
        confidence = max(confidence, 0.7)
        compact = {
            "entity_id": target_entity_id,
            "name": result.document.name,
            "confidence": round(confidence, 2),
            "reason": ",".join(result.reasons),
        }
        alternatives.append(compact)
        candidates.append(
            ResolvedHomeTarget(
                entity_id=target_entity_id,
                document=result.document,
                confidence=confidence,
                reason=compact["reason"],
                alternatives=[],
            )
        )
    if not candidates:
        return error("not_found", "No exposed supported target matched.")
    best = candidates[0]
    ambiguous_matches = {
        candidate.entity_id
        for candidate in candidates
        if candidate.confidence == best.confidence
    }
    if len(ambiguous_matches) > 1:
        return error(
            "ambiguous_target",
            "Multiple exposed targets matched the phrase.",
            alternatives=alternatives[:_MAX_ALTERNATIVES],
        )
    best_alternatives = [
        item
        for item in alternatives[1 : _MAX_ALTERNATIVES + 1]
        if item["entity_id"] != best.entity_id
    ]
    return ResolvedHomeTarget(
        entity_id=best.entity_id,
        document=best.document,
        confidence=best.confidence,
        reason=best.reason,
        alternatives=best_alternatives,
    )


def get_home_context(
    hass: HomeAssistant,
    manager: HomeSemanticIndexManager | None,
    *,
    assistant_id: str | None = None,
    entity_ids: list[str] | None = None,
    phrase: str | None = None,
    domain: str | None = None,
    area_id: str | None = None,
    limit: int = DEFAULT_CONTEXT_LIMIT,
) -> dict[str, Any]:
    """Return compact scoped home context."""
    if not any((entity_ids, phrase, domain, area_id)):
        return error("scope_required", "Provide entity_ids, phrase, domain, or area_id.")
    index = _index(manager)
    resolved_entity_ids: list[str] = list(entity_ids or [])
    if index is None and any((phrase, domain, area_id)):
        return error("index_not_ready", "Semantic home index is still warming up.")
    if phrase and index is not None:
        for result in index.search(phrase, limit=100):
            if not phrase_matches_specific_target(phrase, result.document):
                continue
            result_entity_id = result.document.entity_id or result.document.target_entity_id
            if result_entity_id is not None:
                resolved_entity_ids.append(result_entity_id)
    if index is not None and domain:
        resolved_entity_ids.extend(
            document.entity_id
            for document in index.documents
            if document.entity_id is not None and document.domain == domain
        )
    if index is not None and area_id:
        resolved_entity_ids.extend(
            document.entity_id
            for document in index.documents
            if document.entity_id is not None and document.area_id == area_id
        )
    assistant = default_assistant_id(assistant_id)
    seen: set[str] = set()
    entities: list[dict[str, Any]] = []
    for resolved_entity_id in resolved_entity_ids:
        if resolved_entity_id in seen:
            continue
        seen.add(resolved_entity_id)
        compact = compact_entity(hass, manager, assistant, resolved_entity_id)
        if compact is not None:
            entities.append(compact)
        if len(entities) >= limit:
            break
    return {"status": "ok", "entities": entities}


def is_exposed(hass: HomeAssistant, assistant_id: str, entity_id: str) -> bool:
    """Return whether an entity is exposed to this assistant context."""
    return hass.states.get(entity_id) is not None and async_should_expose(
        hass, assistant_id, entity_id
    )


def entity_document(
    index: HomeSemanticIndex | None, entity_id: str
) -> HomeSemanticDocument | None:
    """Return the indexed document for an entity id."""
    if index is None:
        return None
    return index.documents_by_entity_id.get(entity_id)


def compact_entity(
    hass: HomeAssistant,
    manager: HomeSemanticIndexManager | None,
    assistant_id: str,
    entity_id: str,
    document: HomeSemanticDocument | None = None,
) -> dict[str, Any] | None:
    """Return compact exposed live state for one entity."""
    if not is_exposed(hass, assistant_id, entity_id):
        return None
    state = hass.states.get(entity_id)
    if state is None:
        return None
    doc = document or entity_document(_index(manager), entity_id)
    data: dict[str, Any] = {
        "entity_id": entity_id,
        "domain": state.domain,
        "name": doc.name if doc is not None else state.name,
        "state": state.state,
    }
    if doc is not None:
        if doc.area_id is not None:
            data["area_id"] = doc.area_id
        if doc.capability is not None:
            data["capability"] = doc.capability
        if doc.device_class is not None:
            data["device_class"] = doc.device_class
        if doc.unit_of_measurement is not None:
            data["unit_of_measurement"] = doc.unit_of_measurement
    return data


def supported_entity(
    hass: HomeAssistant, entity_id: str, *, for_control: bool = False
) -> bool:
    """Return whether an entity is in the semantic API support boundary."""
    state = hass.states.get(entity_id)
    if state is None:
        return False
    if state.domain not in SUPPORTED_CONTROL_DOMAINS:
        return not for_control
    return True


def service_for_action(domain: str, action: str) -> tuple[str, str]:
    """Return the HA service domain/name for a constrained action."""
    if action in {"turn_on", "turn_off", "toggle"} and domain in _TOGGLE_DOMAINS:
        return domain, action
    if action == "activate" and domain in _ACTIVATE_DOMAINS:
        return domain, "turn_on"
    return "", ""


def phrase_matches_specific_target(
    phrase: str, document: HomeSemanticDocument | None
) -> bool:
    """Return whether phrase contains target-specific tokens, not only capability words."""
    if document is None:
        return False
    salient_tokens = set(normalize_tokens(phrase)) - _GENERIC_CONTROL_TOKENS
    if not salient_tokens:
        return False
    document_tokens: set[str] = set()
    for part in document.searchable_parts():
        document_tokens.update(normalize_tokens(part))
    return salient_tokens <= document_tokens


def error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    """Return a stable compact semantic error payload."""
    return {"status": "error", "code": code, "message": message, **extra}


def _index(manager: HomeSemanticIndexManager | None) -> HomeSemanticIndex | None:
    """Return the current semantic index if it is ready."""
    return None if manager is None else manager.index
