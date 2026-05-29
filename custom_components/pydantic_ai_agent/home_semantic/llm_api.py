"""Home Assistant LLM API backed by the local Home Semantic Index."""

from collections import defaultdict
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
import voluptuous as vol

from .manager import HomeSemanticIndexManager
from .query import (
    default_assistant_id,
    error as _error,
    get_home_context,
    get_home_summary,
    is_exposed,
    phrase_matches_specific_target as _phrase_matches_specific_target,
    resolve_home_target,
    service_for_action as _service_for_action,
)

_API_ID_PREFIX = "pydantic_ai_agent_home_"


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


class _SemanticTool(llm.Tool):
    """Base class for entry-scoped semantic tools."""

    def __init__(self, entry: ConfigEntry[Any]) -> None:
        """Initialize the tool for one workspace entry."""
        self.entry = entry

    def _manager(self) -> HomeSemanticIndexManager | None:
        """Return the entry semantic manager if it is available."""
        runtime_data = getattr(self.entry, "runtime_data", None)
        return getattr(runtime_data, "home_semantic", None)

    def _assistant(self, llm_context: llm.LLMContext) -> str:
        """Return the assistant id used for HA exposure checks."""
        return default_assistant_id(llm_context.assistant)

    def _is_exposed(
        self, hass: HomeAssistant, llm_context: llm.LLMContext, entity_id: str
    ) -> bool:
        """Return whether an entity is exposed to this LLM context."""
        return is_exposed(hass, self._assistant(llm_context), entity_id)


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
        return get_home_summary(
            hass,
            self._manager(),
            assistant_id=self._assistant(llm_context),
        )


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
        target = resolve_home_target(
            hass,
            self._manager(),
            assistant_id=self._assistant(llm_context),
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
        return get_home_context(
            hass,
            self._manager(),
            assistant_id=self._assistant(llm_context),
            entity_ids=args.get("entity_ids"),
            phrase=args.get("phrase"),
            domain=args.get("domain"),
            area_id=args.get("area_id"),
        )


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
        target = resolve_home_target(
            hass,
            self._manager(),
            assistant_id=self._assistant(llm_context),
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
