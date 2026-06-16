"""Home Assistant service that returns targeted latest-run diagnostics."""

from collections.abc import Mapping
from typing import Any

import voluptuous as vol
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import ServiceValidationError

from ..const import DOMAIN
from ..runtime.types import PydanticAIAgentConfigEntry

SERVICE_GET_AGENT_RUN_DIAGNOSTICS = "get_agent_run_diagnostics"
ATTR_CONFIG_ENTRY_ID = "config_entry_id"
ATTR_SUBENTRY_ID = "subentry_id"
ATTR_SECTION = "section"
ATTR_OFFSET = "offset"
ATTR_LIMIT = "limit"

_RUN_DIAGNOSTIC_SECTIONS = (
    "summary",
    "usage",
    "tool_definitions",
    "tool_calls",
    "model_profile",
    "output",
    "errors",
    "timeline",
)

_RUN_DIAGNOSTICS_SERVICE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CONFIG_ENTRY_ID): str,
        vol.Required(ATTR_SUBENTRY_ID): str,
        vol.Optional(ATTR_SECTION): vol.In(_RUN_DIAGNOSTIC_SECTIONS),
        vol.Optional(ATTR_OFFSET, default=0): vol.All(
            vol.Coerce(int), vol.Range(min=0)
        ),
        vol.Optional(ATTR_LIMIT, default=25): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=100)
        ),
    }
)


def async_register_run_diagnostics_service(hass: HomeAssistant) -> None:
    """Register the get_agent_run_diagnostics service."""

    async def async_get_agent_run_diagnostics(call: ServiceCall) -> dict[str, Any]:
        return _agent_run_diagnostics_service(hass, call)

    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_AGENT_RUN_DIAGNOSTICS,
        async_get_agent_run_diagnostics,
        schema=_RUN_DIAGNOSTICS_SERVICE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )


def _config_entry_for_service(
    hass: HomeAssistant, entry_id: str
) -> PydanticAIAgentConfigEntry:
    """Return a config entry for a response service."""
    entry = hass.config_entries.async_get_entry(entry_id)
    if entry is None or entry.domain != DOMAIN:
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="config_entry_not_found",
            translation_placeholders={"config_entry_id": entry_id},
        )
    return entry


def _agent_run_diagnostics_service(
    hass: HomeAssistant,
    call: ServiceCall,
) -> dict[str, Any]:
    """Return compact latest-run diagnostics for one agent subentry."""
    entry = _config_entry_for_service(hass, call.data[ATTR_CONFIG_ENTRY_ID])
    subentry_id = call.data[ATTR_SUBENTRY_ID]
    if subentry_id not in entry.subentries:
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="subentry_not_found",
            translation_placeholders={
                "config_entry_id": entry.entry_id,
                "subentry_id": subentry_id,
            },
        )
    runtime_data = getattr(entry, "runtime_data", None)
    diagnostics = (
        None
        if runtime_data is None
        else runtime_data.latest_run_diagnostics.get(subentry_id)
    )
    base: dict[str, Any] = {
        "success": True,
        "config_entry_id": entry.entry_id,
        "subentry_id": subentry_id,
        "section": call.data.get(ATTR_SECTION),
        "errors": [],
    }
    if diagnostics is None:
        return {**base, "status": "no_run_yet", "run": None}
    return {
        **base,
        "status": diagnostics.get("status"),
        "run": _agent_run_diagnostics_section(
            diagnostics,
            section=call.data.get(ATTR_SECTION),
            offset=call.data[ATTR_OFFSET],
            limit=call.data[ATTR_LIMIT],
        ),
    }


def _agent_run_diagnostics_section(
    diagnostics: Mapping[str, Any], *, section: str | None, offset: int, limit: int
) -> dict[str, Any]:
    """Return one compact section from a bounded run diagnostics payload."""
    summary = _mapping(diagnostics.get("summary"))
    timeline = diagnostics.get("timeline")
    if section == "timeline":
        return {"timeline": _timeline_page(timeline, offset=offset, limit=limit)}
    if section == "summary":
        return {"summary": summary}
    if section == "usage":
        return {"usage": summary.get("usage")}
    if section == "model_profile":
        return {"model_profile": summary.get("model_profile")}
    if section == "output":
        return {"output": summary.get("output")}
    if section == "errors":
        return {"errors": _run_errors(summary)}
    if section == "tool_definitions":
        return {"tool_definitions": _tool_definitions(timeline)}
    if section == "tool_calls":
        return {"tool_calls": _tool_calls(timeline, offset=offset, limit=limit)}
    timeline_page = _timeline_page(timeline, offset=0, limit=0)
    return {
        "run_id": diagnostics.get("run_id"),
        "status": diagnostics.get("status"),
        "started_at": diagnostics.get("started_at"),
        "finished_at": diagnostics.get("finished_at"),
        "duration_ms": diagnostics.get("duration_ms"),
        "conversation_id": diagnostics.get("conversation_id"),
        "summary": {
            "model_profile": summary.get("model_profile"),
            "usage": summary.get("usage"),
            "duration": summary.get("duration"),
            "error": summary.get("error"),
            "failure": summary.get("failure"),
        },
        "timeline": {
            "total_count": timeline_page["total_count"],
            "available_count": timeline_page["available_count"],
            "omitted_middle_count": timeline_page.get("omitted_middle_count", 0),
        },
    }


def _mapping(value: object) -> Mapping[str, Any]:
    """Return value when it is a mapping, otherwise an empty mapping."""
    return value if isinstance(value, Mapping) else {}


def _timeline_events(timeline: object) -> tuple[list[dict[str, Any]], int, int]:
    """Return available timeline events, total count, and omitted middle count."""
    if isinstance(timeline, list):
        events = [event for event in timeline if isinstance(event, dict)]
        return events, len(events), 0
    if isinstance(timeline, Mapping):
        head = timeline.get("head", [])
        tail = timeline.get("tail", [])
        events = [event for event in [*head, *tail] if isinstance(event, dict)]
        return (
            events,
            int(timeline.get("total_count", len(events))),
            int(timeline.get("omitted_middle_count", 0)),
        )
    return [], 0, 0


def _timeline_page(timeline: object, *, offset: int, limit: int) -> dict[str, Any]:
    """Return a compact page over the bounded available timeline."""
    events, total_count, omitted_middle_count = _timeline_events(timeline)
    page = events[offset : offset + limit] if limit else []
    return {
        "items": page,
        "offset": offset,
        "limit": limit,
        "total_count": total_count,
        "available_count": len(events),
        "has_more_available": offset + len(page) < len(events),
        "omitted_middle_count": omitted_middle_count,
    }


def _tool_definitions(timeline: object) -> list[Any]:
    """Return model-visible HA tool definitions from the latest run."""
    events, _total_count, _omitted_middle_count = _timeline_events(timeline)
    for event in events:
        if event.get("event") != "messages_prepared":
            continue
        data = _mapping(event.get("data"))
        definitions = data.get("llm_tool_definitions")
        return definitions if isinstance(definitions, list) else []
    return []


def _tool_calls(timeline: object, *, offset: int, limit: int) -> dict[str, Any]:
    """Return a page of HA LLM API tool call diagnostics."""
    events, _total_count, _omitted_middle_count = _timeline_events(timeline)
    calls = [event for event in events if event.get("phase") == "tool_call"]
    page = calls[offset : offset + limit]
    return {
        "items": page,
        "offset": offset,
        "limit": limit,
        "total_count": len(calls),
        "has_more": offset + len(page) < len(calls),
    }


def _run_errors(summary: Mapping[str, Any]) -> list[Any]:
    """Return compact run error/failure summary values."""
    errors = []
    if error := summary.get("error"):
        errors.append(error)
    if failure := summary.get("failure"):
        errors.append(failure)
    return errors
