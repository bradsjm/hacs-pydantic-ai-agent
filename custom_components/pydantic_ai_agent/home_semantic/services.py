"""Response services for the Home Semantic Index."""

from typing import Any

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
import voluptuous as vol

from ..const import DOMAIN
from .manager import HomeSemanticIndexManager
from .query import (
    DEFAULT_CONTEXT_LIMIT,
    SUPPORTED_ACTIONS,
    error,
    get_home_context,
    get_home_summary,
    resolve_home_target,
)

SERVICE_GET_HOME_SEMANTIC_SUMMARY = "get_home_semantic_summary"
SERVICE_RESOLVE_HOME_SEMANTIC_TARGET = "resolve_home_semantic_target"
SERVICE_GET_HOME_SEMANTIC_CONTEXT = "get_home_semantic_context"

ATTR_ASSISTANT_ID = "assistant_id"
ATTR_CONFIG_ENTRY_ID = "config_entry_id"
ATTR_ACTION = "action"
ATTR_AREA_ID = "area_id"
ATTR_DOMAIN = "domain"
ATTR_ENTITY_IDS = "entity_ids"
ATTR_LIMIT = "limit"
ATTR_PHRASE = "phrase"

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


async def async_setup_services(hass: HomeAssistant) -> None:
    """Register home semantic response services."""

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
