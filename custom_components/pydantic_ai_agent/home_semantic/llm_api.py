"""Home Assistant LLM API backed by the local Home Semantic Index."""

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
import voluptuous as vol

from .manager import HomeSemanticIndexManager
from .query import (
    default_assistant_id,
    get_home_context,
    get_home_summary,
    is_exposed,
    plan_home_control,
    resolve_home_target,
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
        plan = plan_home_control(
            hass,
            self._manager(),
            assistant_id=self._assistant(llm_context),
            entity_id=tool_input.tool_args.get("entity_id"),
            phrase=tool_input.tool_args.get("phrase"),
            action=action,
        )
        if isinstance(plan, dict):
            return plan
        for call in plan.calls:
            await hass.services.async_call(
                call.domain,
                call.service,
                call.target,
                blocking=True,
                context=llm_context.context,
            )
        if len(plan.calls) > 1 or plan.group_expansion is not None:
            return {
                "status": "ok",
                "action": action,
                "calls": [call.as_dict() for call in plan.calls],
            }
        call = plan.calls[0]
        return {
            "status": "ok",
            "action": action,
            "domain": call.domain,
            "service": call.service,
            "target": call.target,
        }
