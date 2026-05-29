"""Shared query helpers for the Home Semantic Index."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite
from time import monotonic
from typing import Any

from homeassistant.components import conversation
from homeassistant.components.homeassistant import async_should_expose
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant

from .actions import (
    ACTION_SET_TEMPERATURE,
    SUPPORTED_ACTIONS,
    SUPPORTED_CONTROL_DOMAINS,
    live_control_allowed,
    service_for_action,
    state_supports_action,
)
from .index import HomeSemanticIndex, normalize_tokens
from .manager import HomeSemanticIndexManager
from .models import HomeSemanticDocument

_MAX_SUMMARY_AREAS = 20
DEFAULT_CONTEXT_LIMIT = 20
_MAX_ALTERNATIVES = 5
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
    "close",
    "cover",
    "covers",
    "lock",
    "locks",
    "open",
    "set",
    "switch",
    "switches",
    "temperature",
    "to",
    "the",
    "toggle",
    "turn",
    "unlock",
}


@dataclass(frozen=True, kw_only=True)
class ResolvedHomeTarget:
    """A resolved exposed control target."""

    entity_id: str
    document: HomeSemanticDocument | None
    confidence: float
    reason: str
    alternatives: list[dict[str, Any]]


@dataclass(frozen=True, kw_only=True)
class PlannedServiceCall:
    """One HA service call approved by semantic control planning."""

    domain: str
    service: str
    target: dict[str, Any]
    data: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable call data."""
        result: dict[str, Any] = {
            "domain": self.domain,
            "service": self.service,
            "target": self.target,
        }
        if self.data is not None:
            result["data"] = self.data
        return result


@dataclass(frozen=True, kw_only=True)
class HomeControlPlan:
    """Approved dry-run plan for a constrained semantic control."""

    action: str
    target: ResolvedHomeTarget
    calls: list[PlannedServiceCall]
    live_executable: bool = True
    execution_policy: str = "live_allowed"
    group_expansion: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable plan data."""
        data: dict[str, Any] = {
            "status": "ok",
            "allowed": True,
            "reason": "approved",
            "action": self.action,
            "live_executable": self.live_executable,
            "execution_policy": self.execution_policy,
            "target": compact_resolved_target(self.target),
            "calls": [call.as_dict() for call in self.calls],
        }
        if self.group_expansion is not None:
            data["group_expansion"] = self.group_expansion
        return data


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
    record_ambiguity: bool = False,
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
        state = hass.states.get(entity_id)
        if (
            action is not None
            and state is not None
            and state.domain != "group"
            and not state_supports_action(state, action)
        ):
            return error("unsupported_action", "Action is not supported for target.")
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
            and not state_supports_action(state, action)
        ):
            continue
        if not phrase_matches_specific_target(phrase, result.document):
            continue
        confidence = min(0.99, max(0.1, result.score / 60))
        confidence = max(confidence, 0.7)
        candidates.append(
            ResolvedHomeTarget(
                entity_id=target_entity_id,
                document=result.document,
                confidence=confidence,
                reason=",".join(result.reasons),
                alternatives=[],
            )
        )
    if not candidates:
        return error("not_found", "No exposed supported target matched.")
    candidates = _apply_memory_ranking(
        manager,
        phrase=phrase,
        action=action,
        candidates=candidates,
    )
    alternatives = [
        {
            "entity_id": candidate.entity_id,
            "name": candidate.document.name if candidate.document is not None else candidate.entity_id,
            "confidence": round(candidate.confidence, 2),
            "reason": candidate.reason,
        }
        for candidate in candidates
    ]
    best = candidates[0]
    ambiguous_matches = {
        candidate.entity_id
        for candidate in candidates
        if candidate.confidence == best.confidence
    }
    if len(ambiguous_matches) > 1:
        if record_ambiguity and manager is not None:
            manager.memory.record_ambiguity(
                phrase=phrase,
                candidate_entity_ids=ambiguous_matches,
            )
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


def trace_home_resolution(
    hass: HomeAssistant,
    manager: HomeSemanticIndexManager | None,
    *,
    assistant_id: str | None = None,
    phrase: str,
    action: str | None = None,
    area_id: str | None = None,
    domain: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Return trace-capable ranking output for one semantic phrase."""
    started = monotonic()
    index = _index(manager)
    if index is None:
        return error("index_not_ready", "Semantic home index is still warming up.")
    assistant = default_assistant_id(assistant_id)
    candidates: list[dict[str, Any]] = []
    accepted: list[ResolvedHomeTarget] = []
    for result in index.search(
        phrase,
        action=action,
        document_types=("capability", "entity", "group"),
        limit=100,
    ):
        target_entity_id = result.document.entity_id or result.document.target_entity_id
        rejection_reasons: list[str] = []
        state = hass.states.get(target_entity_id) if target_entity_id is not None else None
        if target_entity_id is None:
            continue
        if state is None:
            rejection_reasons.append("unavailable")
        if area_id is not None and result.document.area_id != area_id:
            rejection_reasons.append("wrong_area")
        if domain is not None and state is not None and state.domain != domain:
            rejection_reasons.append("unsupported_domain")
        not_exposed = target_entity_id is not None and not is_exposed(
            hass, assistant, target_entity_id
        )
        if not_exposed:
            rejection_reasons.append("not_exposed")
            continue
        if target_entity_id is not None and not supported_entity(
            hass, target_entity_id, for_control=True
        ):
            rejection_reasons.append("unsupported_domain")
        if (
            action is not None
            and state is not None
            and state.domain != "group"
            and not state_supports_action(state, action)
        ):
            rejection_reasons.append("unsupported_action")
        if not phrase_matches_specific_target(phrase, result.document):
            rejection_reasons.append("generic_phrase")
        confidence = min(0.99, max(0.1, result.score / 60))
        confidence = max(confidence, 0.7)
        candidate = {
            "document_id": result.document.document_id,
            "entity_id": target_entity_id,
            "name": result.document.name,
            "domain": None if state is None else state.domain,
            "area_id": result.document.area_id,
            "score": round(result.score, 3),
            "confidence": round(confidence, 2),
            "matched_tokens": matched_tokens(phrase, result.document),
            "reasons": list(result.reasons),
            "rejection_reasons": rejection_reasons,
        }
        candidates.append(candidate)
        if target_entity_id is not None and not rejection_reasons:
            accepted.append(
                ResolvedHomeTarget(
                    entity_id=target_entity_id,
                    document=result.document,
                    confidence=confidence,
                    reason=",".join(result.reasons),
                    alternatives=[],
                )
            )
    selected: dict[str, Any] | None = None
    if accepted:
        accepted = _apply_memory_ranking(
            manager,
            phrase=phrase,
            action=action,
            candidates=accepted,
        )
        selected = compact_resolved_target(accepted[0])
    return {
        "status": "ok",
        "selected": selected,
        "candidates": candidates[:limit],
        "candidate_count": len(candidates),
        "latency_ms": round((monotonic() - started) * 1000, 3),
    }


def plan_home_control(
    hass: HomeAssistant,
    manager: HomeSemanticIndexManager | None,
    *,
    assistant_id: str | None = None,
    action: str,
    phrase: str | None = None,
    entity_id: str | None = None,
    temperature: float | None = None,
    record_ambiguity: bool = False,
) -> HomeControlPlan | dict[str, Any]:
    """Plan constrained exposed home controls without executing them."""
    if action not in SUPPORTED_ACTIONS:
        return error("unsupported_action", "Action is not supported.")
    if action == ACTION_SET_TEMPERATURE and (
        temperature is None or not isfinite(temperature)
    ):
        return error(
            "unsupported_action_parameters",
            "Action requires a finite target temperature.",
        )
    target = resolve_home_target(
        hass,
        manager,
        assistant_id=assistant_id,
        entity_id=entity_id,
        phrase=phrase,
        action=action,
        record_ambiguity=record_ambiguity,
    )
    if isinstance(target, dict):
        return target
    if phrase is not None and target.confidence < 0.45:
        return error(
            "ambiguous_target",
            "Target confidence is too low for automatic control.",
            alternatives=target.alternatives,
        )
    state = hass.states.get(target.entity_id)
    if state is None:
        return error("not_found", "Target entity is unavailable.")
    assistant = default_assistant_id(assistant_id)
    if state.domain == "group":
        expanded = expand_group_control_calls(hass, assistant, target.entity_id, action)
        if isinstance(expanded, dict):
            return expanded
        calls = [
            PlannedServiceCall(
                domain=service_domain,
                service=service_name,
                target={ATTR_ENTITY_ID: entity_ids},
                data={"temperature": temperature}
                if action == ACTION_SET_TEMPERATURE
                else None,
            )
            for service_domain, service_name, entity_ids in expanded
        ]
        live_executable = all(live_control_allowed(call.domain, action) for call in calls)
        return HomeControlPlan(
            action=action,
            target=target,
            calls=calls,
            live_executable=live_executable,
            execution_policy="live_allowed" if live_executable else "plan_only",
            group_expansion={
                "group_entity_id": target.entity_id,
                "member_entity_ids": [
                    member
                    for _service_domain, _service_name, entity_ids in expanded
                    for member in entity_ids
                ],
            },
        )
    service_domain, service_name = service_for_action(state.domain, action)
    if service_domain == "":
        return error("unsupported_domain", "Action is not supported for target.")
    live_executable = live_control_allowed(service_domain, action)
    data = {"temperature": temperature} if action == ACTION_SET_TEMPERATURE else None
    return HomeControlPlan(
        action=action,
        target=target,
        calls=[
            PlannedServiceCall(
                domain=service_domain,
                service=service_name,
                target={ATTR_ENTITY_ID: target.entity_id},
                data=data,
            )
        ],
        live_executable=live_executable,
        execution_policy="live_allowed" if live_executable else "plan_only",
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
    domain = domain or None
    area_id = area_id or None
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
    if index is not None and (domain or area_id):
        scoped_entity_ids = [
            document.entity_id
            for document in index.documents
            if document.entity_id is not None
            and (domain is None or document.domain == domain)
            and (area_id is None or document.area_id == area_id)
        ]
        if resolved_entity_ids:
            scoped_entity_id_set = set(scoped_entity_ids)
            resolved_entity_ids = [
                resolved_entity_id
                for resolved_entity_id in resolved_entity_ids
                if resolved_entity_id in scoped_entity_id_set
            ]
        else:
            resolved_entity_ids.extend(scoped_entity_ids)
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


def expand_group_control_calls(
    hass: HomeAssistant,
    assistant_id: str,
    entity_id: str,
    action: str,
) -> list[tuple[str, str, list[str]]] | dict[str, Any]:
    """Expand an exposed HA group into exposed supported member targets."""
    group_state = hass.states.get(entity_id)
    if group_state is None:
        return error("not_found", "Group entity is unavailable.")
    raw_members = group_state.attributes.get(ATTR_ENTITY_ID, [])
    if not isinstance(raw_members, list | tuple):
        return error("unsupported_domain", "Group has no entity members.")
    members = [member for member in raw_members if isinstance(member, str)]
    targets_by_service: dict[tuple[str, str], list[str]] = defaultdict(list)
    for member in members:
        state = hass.states.get(member)
        if state is None or not is_exposed(hass, assistant_id, member):
            continue
        if not state_supports_action(state, action):
            continue
        service_domain, service_name = service_for_action(state.domain, action)
        if service_domain == "":
            continue
        targets_by_service[(service_domain, service_name)].append(member)
    if not targets_by_service:
        return error("not_exposed", "Group has no exposed supported members.")
    return [
        (service_domain, service_name, entity_ids)
        for (service_domain, service_name), entity_ids in targets_by_service.items()
    ]


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


def compact_resolved_target(target: ResolvedHomeTarget) -> dict[str, Any]:
    """Return compact JSON-safe target data."""
    return {
        "target_type": "entity",
        "entity_id": target.entity_id,
        "confidence": round(target.confidence, 2),
        "reason": target.reason,
        "alternatives": target.alternatives,
    }


def matched_tokens(phrase: str, document: HomeSemanticDocument) -> list[str]:
    """Return normalized query tokens that matched document text."""
    query_tokens = set(normalize_tokens(phrase))
    document_tokens: set[str] = set()
    for part in document.searchable_parts():
        document_tokens.update(normalize_tokens(part))
    return sorted(query_tokens & document_tokens)


def _apply_memory_ranking(
    manager: HomeSemanticIndexManager | None,
    *,
    phrase: str,
    action: str | None,
    candidates: list[ResolvedHomeTarget],
) -> list[ResolvedHomeTarget]:
    """Apply post-filter memory boosts without reviving rejected entities."""
    if manager is None:
        return candidates
    adjustments = manager.memory.ranking_adjustments(
        phrase=phrase,
        action=action,
        area_id=None,
        domain=None,
        candidate_entity_ids=[candidate.entity_id for candidate in candidates],
    )
    if not adjustments:
        return candidates
    adjusted: list[ResolvedHomeTarget] = []
    for candidate in candidates:
        adjustment, reasons = adjustments.get(candidate.entity_id, (0.0, ()))
        adjusted.append(
            ResolvedHomeTarget(
                entity_id=candidate.entity_id,
                document=candidate.document,
                confidence=max(0.0, min(0.99, candidate.confidence + adjustment)),
                reason=",".join((candidate.reason, *reasons)),
                alternatives=candidate.alternatives,
            )
        )
    adjusted.sort(key=lambda candidate: (candidate.confidence, candidate.entity_id), reverse=True)
    return adjusted


def error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    """Return a stable compact semantic error payload."""
    return {"status": "error", "code": code, "message": message, **extra}


def _index(manager: HomeSemanticIndexManager | None) -> HomeSemanticIndex | None:
    """Return the current semantic index if it is ready."""
    return None if manager is None else manager.index
