"""Home Assistant LLM API backed by the local Home Semantic Index."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from homeassistant.components import conversation
from homeassistant.components.homeassistant import async_should_expose
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
import voluptuous as vol

from .index import HomeSemanticIndex, normalize_tokens
from .manager import HomeSemanticIndexManager
from .models import HomeSemanticDocument

_API_ID_PREFIX = "pydantic_ai_agent_home_"
_MAX_SUMMARY_AREAS = 20
_MAX_CONTEXT_ENTITIES = 20
_MAX_ALTERNATIVES = 5
_SUPPORTED_CONTROL_DOMAINS = {"group", "light", "scene", "script", "switch"}
_TOGGLE_DOMAINS = {"light", "switch"}
_ACTIVATE_DOMAINS = {"scene", "script"}
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


def semantic_api_id(entry_id: str) -> str:
    """Return the entry-scoped semantic Home Assistant LLM API id."""
    return f"{_API_ID_PREFIX}{entry_id}"


class HomeSemanticAPI(llm.API):
    """Entry-scoped semantic home API for Home Assistant LLM tools."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry[Any]) -> None:
        """Initialize the semantic API for one workspace entry."""
        super().__init__(
            hass=hass,
            id=semantic_api_id(entry.entry_id),
            name=f"{entry.title} home",
        )
        self.entry = entry

    async def async_get_api_instance(
        self, llm_context: llm.LLMContext
    ) -> llm.APIInstance:
        """Return an API instance with compact semantic home tools."""
        tools: list[llm.Tool] = [
            GetHomeSummaryTool(self.entry),
            ResolveHomeTargetTool(self.entry),
            GetHomeContextTool(self.entry),
            ControlHomeTool(self.entry),
        ]
        return llm.APIInstance(
            api=self,
            api_prompt=(
                "Use the semantic home tools to resolve areas, devices, groups, "
                "and safe controls. Prefer grouped targets. Never assume access "
                "to entities that are not returned by these tools."
            ),
            llm_context=llm_context,
            tools=tools,
        )


@dataclass(frozen=True, kw_only=True)
class _ResolvedTarget:
    """A resolved exposed control target."""

    entity_id: str
    document: HomeSemanticDocument | None
    confidence: float
    reason: str
    alternatives: list[dict[str, Any]]


class _SemanticTool(llm.Tool):
    """Base class for entry-scoped semantic tools."""

    def __init__(self, entry: ConfigEntry[Any]) -> None:
        """Initialize the tool for one workspace entry."""
        self.entry = entry

    def _manager(self) -> HomeSemanticIndexManager | None:
        """Return the entry semantic manager if it is available."""
        runtime_data = getattr(self.entry, "runtime_data", None)
        return getattr(runtime_data, "home_semantic", None)

    def _index(self) -> HomeSemanticIndex | None:
        """Return the current semantic index if it is ready."""
        manager = self._manager()
        return None if manager is None else manager.index

    def _assistant(self, llm_context: llm.LLMContext) -> str:
        """Return the assistant id used for HA exposure checks."""
        return llm_context.assistant or conversation.DOMAIN

    def _is_exposed(
        self, hass: HomeAssistant, llm_context: llm.LLMContext, entity_id: str
    ) -> bool:
        """Return whether an entity is exposed to this LLM context."""
        return hass.states.get(entity_id) is not None and async_should_expose(
            hass, self._assistant(llm_context), entity_id
        )

    def _entity_document(
        self, index: HomeSemanticIndex | None, entity_id: str
    ) -> HomeSemanticDocument | None:
        """Return the indexed document for an entity id."""
        if index is None:
            return None
        return index.documents_by_entity_id.get(entity_id)

    def _compact_entity(
        self,
        hass: HomeAssistant,
        llm_context: llm.LLMContext,
        entity_id: str,
        document: HomeSemanticDocument | None = None,
    ) -> dict[str, Any] | None:
        """Return compact exposed live state for one entity."""
        if not self._is_exposed(hass, llm_context, entity_id):
            return None
        state = hass.states.get(entity_id)
        if state is None:
            return None
        doc = document or self._entity_document(self._index(), entity_id)
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

    def _supported_entity(
        self, hass: HomeAssistant, entity_id: str, *, for_control: bool = False
    ) -> bool:
        """Return whether an entity is in the semantic API support boundary."""
        state = hass.states.get(entity_id)
        if state is None:
            return False
        if state.domain not in _SUPPORTED_CONTROL_DOMAINS:
            return not for_control
        return True

    def _resolve_target(
        self,
        hass: HomeAssistant,
        llm_context: llm.LLMContext,
        *,
        phrase: str | None = None,
        entity_id: str | None = None,
        action: str | None = None,
    ) -> _ResolvedTarget | dict[str, Any]:
        """Resolve a phrase or explicit entity id to one exposed target."""
        index = self._index()
        if entity_id is not None:
            document = self._entity_document(index, entity_id)
            if not self._is_exposed(hass, llm_context, entity_id):
                return _error("not_exposed", "Target entity is not exposed.")
            if not self._supported_entity(hass, entity_id, for_control=True):
                return _error("unsupported_domain", "Target domain is not supported.")
            return _ResolvedTarget(
                entity_id=entity_id,
                document=document,
                confidence=1.0,
                reason="Explicit exposed entity target",
                alternatives=[],
            )
        if not phrase:
            return _error("target_required", "Provide entity_id or phrase.")
        if index is None:
            return _error("index_not_ready", "Semantic home index is still warming up.")
        candidates: list[_ResolvedTarget] = []
        alternatives: list[dict[str, Any]] = []
        for result in index.search(
            phrase,
            action=action,
            document_types=("capability", "entity", "group"),
            limit=100,
        ):
            target_entity_id = (
                result.document.entity_id or result.document.target_entity_id
            )
            if target_entity_id is None:
                continue
            state = hass.states.get(target_entity_id)
            if state is None:
                continue
            if not self._is_exposed(hass, llm_context, target_entity_id):
                continue
            if not self._supported_entity(hass, target_entity_id, for_control=True):
                continue
            if (
                action is not None
                and state.domain != "group"
                and _service_for_action(state.domain, action)[0] == ""
            ):
                continue
            if not _phrase_matches_specific_target(phrase, result.document):
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
                _ResolvedTarget(
                    entity_id=target_entity_id,
                    document=result.document,
                    confidence=confidence,
                    reason=compact["reason"],
                    alternatives=[],
                )
            )
        if not candidates:
            return _error("not_found", "No exposed supported target matched.")
        best = candidates[0]
        ambiguous_matches = {
            candidate.entity_id
            for candidate in candidates
            if candidate.confidence == best.confidence
        }
        if len(ambiguous_matches) > 1:
            return _error(
                "ambiguous_target",
                "Multiple exposed targets matched the phrase.",
                alternatives=alternatives[:_MAX_ALTERNATIVES],
            )
        best_alternatives = [
            item
            for item in alternatives[1 : _MAX_ALTERNATIVES + 1]
            if item["entity_id"] != best.entity_id
        ]
        return _ResolvedTarget(
            entity_id=best.entity_id,
            document=best.document,
            confidence=best.confidence,
            reason=best.reason,
            alternatives=best_alternatives,
        )


class GetHomeSummaryTool(_SemanticTool):
    """Return a compact summary of exposed semantic home controls."""

    name = "get_home_summary"
    description = "Return compact exposed areas, capabilities, and control counts."
    parameters = vol.Schema({})

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict[str, Any]:
        """Return exposed semantic home summary."""
        index = self._index()
        if index is None:
            return {"ready": False, "areas": [], "domains": {}}
        domain_counts: Counter[str] = Counter()
        area_capabilities: dict[str, Counter[str]] = defaultdict(Counter)
        preferred: dict[str, list[dict[str, str]]] = defaultdict(list)
        for document in index.documents:
            if document.entity_id is None:
                continue
            if not self._is_exposed(hass, llm_context, document.entity_id):
                continue
            if document.domain not in _SUPPORTED_CONTROL_DOMAINS:
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


class ResolveHomeTargetTool(_SemanticTool):
    """Resolve a natural-language phrase into one exposed home target."""

    name = "resolve_home_target"
    description = "Resolve a home phrase into the best exposed semantic target."
    parameters = vol.Schema(
        {
            vol.Required("phrase"): str,
            vol.Optional("action"): vol.In(
                ["turn_on", "turn_off", "toggle", "activate"]
            ),
        }
    )

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict[str, Any]:
        """Resolve a phrase to an exposed target."""
        target = self._resolve_target(
            hass,
            llm_context,
            phrase=tool_input.tool_args["phrase"],
            action=tool_input.tool_args.get("action"),
        )
        if isinstance(target, dict):
            return target
        return {
            "status": "ok",
            "confidence": round(target.confidence, 2),
            "target_type": "entity",
            "entity_id": target.entity_id,
            "reason": target.reason,
            "alternatives": target.alternatives,
        }


class GetHomeContextTool(_SemanticTool):
    """Return compact live state for an explicit exposed scope."""

    name = "get_home_context"
    description = (
        "Return compact live state for explicit exposed entity or query scope."
    )
    parameters = vol.Schema(
        {
            vol.Optional("entity_ids"): [str],
            vol.Optional("phrase"): str,
            vol.Optional("domain"): str,
            vol.Optional("area_id"): str,
        }
    )

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict[str, Any]:
        """Return compact scoped home context."""
        args = tool_input.tool_args
        if not any(
            key in args for key in ("entity_ids", "phrase", "domain", "area_id")
        ):
            return _error(
                "scope_required", "Provide entity_ids, phrase, domain, or area_id."
            )
        index = self._index()
        entity_ids: list[str] = []
        if ids := args.get("entity_ids"):
            entity_ids.extend(ids)
        if index is None and any(
            key in args for key in ("phrase", "domain", "area_id")
        ):
            return _error("index_not_ready", "Semantic home index is still warming up.")
        if (phrase := args.get("phrase")) and index is not None:
            for result in index.search(phrase, limit=100):
                if not _phrase_matches_specific_target(phrase, result.document):
                    continue
                entity_id = (
                    result.document.entity_id or result.document.target_entity_id
                )
                if entity_id is not None:
                    entity_ids.append(entity_id)
        if index is not None and (domain := args.get("domain")):
            entity_ids.extend(
                document.entity_id
                for document in index.documents
                if document.entity_id is not None and document.domain == domain
            )
        if index is not None and (area_id := args.get("area_id")):
            entity_ids.extend(
                document.entity_id
                for document in index.documents
                if document.entity_id is not None and document.area_id == area_id
            )
        seen: set[str] = set()
        entities: list[dict[str, Any]] = []
        for entity_id in entity_ids:
            if entity_id in seen:
                continue
            seen.add(entity_id)
            compact = self._compact_entity(hass, llm_context, entity_id)
            if compact is not None:
                entities.append(compact)
            if len(entities) >= _MAX_CONTEXT_ENTITIES:
                break
        return {"status": "ok", "entities": entities}


class ControlHomeTool(_SemanticTool):
    """Execute constrained exposed home controls."""

    name = "control_home"
    description = (
        "Execute safe exposed light, switch, scene, script, or group controls."
    )
    parameters = vol.Schema(
        {
            vol.Required("action"): vol.In(
                ["turn_on", "turn_off", "toggle", "activate"]
            ),
            vol.Optional("entity_id"): str,
            vol.Optional("phrase"): str,
        }
    )

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict[str, Any]:
        """Execute a constrained exposed home control action."""
        action = tool_input.tool_args["action"]
        target = self._resolve_target(
            hass,
            llm_context,
            entity_id=tool_input.tool_args.get("entity_id"),
            phrase=tool_input.tool_args.get("phrase"),
            action=action,
        )
        if isinstance(target, dict):
            return target
        if phrase := tool_input.tool_args.get("phrase"):
            if not _phrase_matches_specific_target(phrase, target.document):
                return _error(
                    "ambiguous_target",
                    "Phrase did not match a specific exposed target.",
                    alternatives=target.alternatives,
                )
        if tool_input.tool_args.get("phrase") and target.confidence < 0.45:
            return _error(
                "ambiguous_target",
                "Target confidence is too low for automatic control.",
                alternatives=target.alternatives,
            )
        state = hass.states.get(target.entity_id)
        if state is None:
            return _error("not_found", "Target entity is unavailable.")
        service_domain: str
        service_name: str
        service_target: dict[str, Any]
        if state.domain == "group":
            expanded = self._expand_group(hass, llm_context, target.entity_id, action)
            if isinstance(expanded, dict):
                return expanded
            for service_domain, service_name, entity_ids in expanded:
                await hass.services.async_call(
                    service_domain,
                    service_name,
                    {ATTR_ENTITY_ID: entity_ids},
                    blocking=True,
                    context=llm_context.context,
                )
            return {
                "status": "ok",
                "action": action,
                "calls": [
                    {
                        "domain": service_domain,
                        "service": service_name,
                        "target": {ATTR_ENTITY_ID: entity_ids},
                    }
                    for service_domain, service_name, entity_ids in expanded
                ],
            }
        else:
            service_domain, service_name = _service_for_action(state.domain, action)
            if service_domain == "":
                return _error(
                    "unsupported_domain", "Action is not supported for target."
                )
            service_target = {ATTR_ENTITY_ID: target.entity_id}
        await hass.services.async_call(
            service_domain,
            service_name,
            service_target,
            blocking=True,
            context=llm_context.context,
        )
        return {
            "status": "ok",
            "action": action,
            "domain": service_domain,
            "service": service_name,
            "target": service_target,
        }

    def _expand_group(
        self,
        hass: HomeAssistant,
        llm_context: llm.LLMContext,
        entity_id: str,
        action: str,
    ) -> list[tuple[str, str, list[str]]] | dict[str, Any]:
        """Expand an exposed HA group into exposed supported member targets."""
        group_state = hass.states.get(entity_id)
        if group_state is None:
            return _error("not_found", "Group entity is unavailable.")
        raw_members = group_state.attributes.get(ATTR_ENTITY_ID, [])
        if not isinstance(raw_members, list | tuple):
            return _error("unsupported_domain", "Group has no entity members.")
        members = [member for member in raw_members if isinstance(member, str)]
        targets_by_service: dict[tuple[str, str], list[str]] = defaultdict(list)
        for member in members:
            state = hass.states.get(member)
            if state is None or not self._is_exposed(hass, llm_context, member):
                continue
            service_domain, service_name = _service_for_action(state.domain, action)
            if service_domain == "":
                continue
            targets_by_service[(service_domain, service_name)].append(member)
        if not targets_by_service:
            return _error("not_exposed", "Group has no exposed supported members.")
        return [
            (service_domain, service_name, entity_ids)
            for (service_domain, service_name), entity_ids in targets_by_service.items()
        ]


def _service_for_action(domain: str, action: str) -> tuple[str, str]:
    """Return the HA service domain/name for a constrained action."""
    if action in {"turn_on", "turn_off", "toggle"} and domain in _TOGGLE_DOMAINS:
        return domain, action
    if action == "activate" and domain in _ACTIVATE_DOMAINS:
        return domain, "turn_on"
    return "", ""


def _phrase_matches_specific_target(
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


def _error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    """Return a stable compact tool error payload."""
    return {"status": "error", "code": code, "message": message, **extra}
