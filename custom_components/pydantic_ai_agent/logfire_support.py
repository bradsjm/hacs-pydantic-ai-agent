"""Logfire support for Pydantic AI Agent."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import logging
from typing import Any

from pydantic_ai import Agent

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant

from .const import (
    CONF_LOGFIRE_INCLUDE_CONTENT,
    CONF_LOGFIRE_TOKEN,
    CONF_MCP_SERVER_IDS,
    CONF_OUTPUT_MODE,
    CONF_PROVIDER_MODE,
    DOMAIN,
)
from .repairs import (
    async_create_logfire_token_conflict_issue,
    async_delete_logfire_token_conflict_issue,
)
from .structured_output import structured_output_mode

_LOGGER = logging.getLogger(__name__)
_LOGFIRE_STATE_KEY = "logfire"


@dataclass(slots=True)
class LogfireState:
    """Process-global Logfire configuration owned by Home Assistant state."""

    configured_token: str | None = None
    configured_include_content: bool = False


def _logfire_state(hass: HomeAssistant) -> LogfireState:
    """Return integration-global Logfire state."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    state = domain_data.get(_LOGFIRE_STATE_KEY)
    if state is None:
        state = domain_data[_LOGFIRE_STATE_KEY] = LogfireState()
    return state


def configure_logfire(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Configure process-global Logfire once for a config entry."""
    token = _entry_logfire_token(entry)
    if token is None:
        async_delete_logfire_token_conflict_issue(hass, entry)
        return False

    include_content = bool(entry.data.get(CONF_LOGFIRE_INCLUDE_CONTENT, False))
    state = _logfire_state(hass)
    if state.configured_token is None:
        try:
            import logfire

            logfire.configure(
                send_to_logfire=True,
                token=token,
                service_name=DOMAIN,
                console=False,
                inspect_arguments=False,
            )
        except Exception:
            _LOGGER.exception(
                "Failed to configure Logfire for Pydantic AI Agent entry %s",
                entry.entry_id,
            )
            return False
        state.configured_token = token
        state.configured_include_content = include_content
        async_delete_logfire_token_conflict_issue(hass, entry)
        return True

    if token == state.configured_token:
        async_delete_logfire_token_conflict_issue(hass, entry)
        return True

    _LOGGER.warning(
        "Logfire is already configured by another Pydantic AI Agent entry; "
        "Logfire is disabled for entry %s",
        entry.entry_id,
    )
    async_create_logfire_token_conflict_issue(hass, entry)
    return False


def logfire_enabled(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Return if Logfire is actively emitting traces for this entry."""
    return logfire_active_for_entry(hass, entry)


def logfire_active_for_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Return if Logfire should emit traces for this entry."""
    token = _entry_logfire_token(entry)
    return token is not None and token == _logfire_state(hass).configured_token


def logfire_include_content(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Return if Logfire should include prompt and completion content."""
    if not logfire_active_for_entry(hass, entry):
        return False
    return _logfire_state(hass).configured_include_content


def logfire_token_conflict(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Return if the entry has a token that conflicts with active Logfire."""
    token = _entry_logfire_token(entry)
    configured_token = _logfire_state(hass).configured_token
    return (
        token is not None
        and configured_token is not None
        and token != configured_token
    )


def instrument_agent(
    hass: HomeAssistant, entry: ConfigEntry, agent: Agent[Any, Any]
) -> None:
    """Instrument one Pydantic AI agent when this entry owns active Logfire."""
    if not logfire_active_for_entry(hass, entry):
        return

    try:
        import logfire

        logfire.instrument_pydantic_ai(
            agent,
            include_content=logfire_include_content(hass, entry),
        )
    except Exception:
        _LOGGER.exception(
            "Failed to instrument Pydantic AI agent with Logfire for entry %s",
            entry.entry_id,
        )


@contextmanager
def agent_run_span(
    hass: HomeAssistant,
    entry: ConfigEntry,
    subentry: ConfigSubentry,
    *,
    entity_id: str,
    conversation_id: str | None,
    model_name: str,
) -> Iterator[None]:
    """Wrap one Pydantic AI run with safe Home Assistant trace metadata."""
    if not logfire_active_for_entry(hass, entry):
        with nullcontext():
            yield
        return

    try:
        import logfire

        span = logfire.span(
            "Run Pydantic AI agent",
            **_span_attributes(
                hass, entry, subentry, entity_id, conversation_id, model_name
            ),
        )
        span.__enter__()
    except Exception:
        _LOGGER.exception(
            "Failed to start Logfire span for Pydantic AI Agent entry %s",
            entry.entry_id,
        )
        yield
        return

    exc_info: tuple[type[BaseException] | None, BaseException | None, object] = (
        None,
        None,
        None,
    )
    try:
        yield
    except BaseException as err:
        exc_info = (type(err), err, err.__traceback__)
        raise
    finally:
        try:
            span.__exit__(*exc_info)
        except Exception:
            _LOGGER.exception(
                "Failed to finish Logfire span for Pydantic AI Agent entry %s",
                entry.entry_id,
            )


def _entry_logfire_token(entry: ConfigEntry) -> str | None:
    """Return a normalized Logfire token from entry data."""
    token = entry.data.get(CONF_LOGFIRE_TOKEN)
    if not isinstance(token, str):
        return None
    token = token.strip()
    return token or None


def _span_attributes(
    hass: HomeAssistant,
    entry: ConfigEntry,
    subentry: ConfigSubentry,
    entity_id: str,
    conversation_id: str | None,
    model_name: str,
) -> Mapping[str, Any]:
    """Return low-risk trace attributes for Home Assistant context."""
    llm_api_ids = subentry.data.get(CONF_LLM_HASS_API)
    mcp_server_ids = subentry.data.get(CONF_MCP_SERVER_IDS)
    if not isinstance(llm_api_ids, list):
        llm_api_ids = []
    if not isinstance(mcp_server_ids, list):
        mcp_server_ids = []
    return {
        "ha.domain": DOMAIN,
        "ha.version": getattr(hass, "version", None),
        "ha.entry_id": entry.entry_id,
        "ha.entry_title": entry.title,
        "ha.subentry_id": subentry.subentry_id,
        "ha.subentry_title": subentry.title,
        "ha.subentry_type": subentry.subentry_type,
        "ha.entity_id": entity_id,
        "ha.conversation_id": conversation_id,
        "ha.provider_mode": entry.data.get(CONF_PROVIDER_MODE),
        "ha.model": model_name,
        "ha.output_mode": structured_output_mode(subentry.data.get(CONF_OUTPUT_MODE)),
        "ha.ha_tools_enabled": bool(llm_api_ids),
        "ha.llm_api_ids": llm_api_ids,
        "ha.mcp_server_count": len(mcp_server_ids),
        "ha.logfire_include_content": logfire_include_content(hass, entry),
    }
