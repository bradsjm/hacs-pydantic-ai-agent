"""Logfire support for Pydantic AI Agent."""

import asyncio
import json
import logging
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from homeassistant.config_entries import ConfigEntry, ConfigEntryState, ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, __version__
from homeassistant.core import HomeAssistant
from pydantic_ai import Agent

from .const import (
    CONF_AGENT_NAME,
    CONF_AI_TASK_NAME,
    CONF_LOGFIRE_INCLUDE_CONTENT,
    CONF_LOGFIRE_TOKEN,
    CONF_OUTPUT_MODE,
    CONF_TODO_LIST_ENTITY_ID,
    CONF_VIRTUAL_WORKSPACE_ENABLED,
    CONF_WEB_FETCH_ENABLED,
    DOMAIN,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE_GEMINI,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
)
from .model_profiles import ModelProfile
from .repair_issues import (
    async_create_logfire_token_conflict_issue,
    async_delete_logfire_token_conflict_issue,
)
from .structured_output import structured_output_mode

_LOGGER = logging.getLogger(__name__)
_LOGFIRE_STATE_KEY = "logfire"


def _load_integration_version() -> str | None:
    """Return the packaged integration version once from manifest.json."""
    try:
        manifest = json.loads(
            Path(__file__).with_name("manifest.json").read_text(encoding="utf-8")
        )
    except Exception:
        return None
    version = manifest.get("version")
    if not isinstance(version, str):
        return None
    version = version.strip()
    return version or None


_INTEGRATION_VERSION = _load_integration_version()


@dataclass(slots=True)
class LogfireState:
    """Process-global Logfire configuration owned by Home Assistant state."""

    configured_token: str | None = None
    configured_include_content: bool = False
    owner_include_content_by_entry_id: dict[str, bool] = field(default_factory=dict)
    configure_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _logfire_state(hass: HomeAssistant) -> LogfireState:
    """Return integration-global Logfire state."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    state = domain_data.get(_LOGFIRE_STATE_KEY)
    if state is None:
        state = domain_data[_LOGFIRE_STATE_KEY] = LogfireState()
    return state


async def async_configure_logfire(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Configure process-global Logfire once for a config entry."""
    token = _entry_logfire_token(entry)
    if token is None:
        await async_release_logfire(hass, entry)
        async_delete_logfire_token_conflict_issue(hass, entry)
        return False

    include_content = bool(entry.data.get(CONF_LOGFIRE_INCLUDE_CONTENT, False))
    state = _logfire_state(hass)
    async with state.configure_lock:
        if (
            state.configured_token is not None
            and not state.owner_include_content_by_entry_id
        ):
            state.configured_token = None
            state.configured_include_content = False

        if state.configured_token is None:
            try:
                await hass.async_add_executor_job(_configure_logfire_sync, token)
            except Exception:
                _LOGGER.exception(
                    "Failed to configure Logfire for Pydantic AI Agent entry %s",
                    entry.entry_id,
                )
                return False
            state.configured_token = token
            state.owner_include_content_by_entry_id[entry.entry_id] = include_content
            state.configured_include_content = _configured_include_content(state)
            async_delete_logfire_token_conflict_issue(hass, entry)
            return True

        if token == state.configured_token:
            state.owner_include_content_by_entry_id[entry.entry_id] = include_content
            state.configured_include_content = _configured_include_content(state)
            async_delete_logfire_token_conflict_issue(hass, entry)
            return True

        _LOGGER.warning(
            "Logfire is already configured by another Pydantic AI Agent entry; "
            "Logfire is disabled for entry %s",
            entry.entry_id,
        )
        async_create_logfire_token_conflict_issue(hass, entry)
        return False


async def async_release_logfire(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Release this entry's ownership of the process-global Logfire token."""
    state = _logfire_state(hass)
    async with state.configure_lock:
        state.owner_include_content_by_entry_id.pop(entry.entry_id, None)
        if state.owner_include_content_by_entry_id:
            state.configured_include_content = _configured_include_content(state)
            return
        state.configured_token = None
        state.configured_include_content = False
    await _async_configure_next_logfire_owner(hass, entry.entry_id)


def _configured_include_content(state: LogfireState) -> bool:
    """Return if any active Logfire owner captures prompt/completion content."""
    return any(state.owner_include_content_by_entry_id.values())


async def _async_configure_next_logfire_owner(
    hass: HomeAssistant, released_entry_id: str
) -> None:
    """Promote a loaded conflicting entry after the last owner releases Logfire."""
    promoted_token: str | None = None
    for candidate in hass.config_entries.async_entries(DOMAIN):
        if candidate.entry_id == released_entry_id:
            continue
        if candidate.state is not ConfigEntryState.LOADED:
            continue
        token = _entry_logfire_token(candidate)
        if token is None:
            continue
        if promoted_token is not None and token != promoted_token:
            continue
        if await async_configure_logfire(hass, candidate):
            promoted_token = token


def _configure_logfire_sync(token: str) -> None:
    """Configure Logfire in an executor because it performs blocking file I/O."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"^handler names should be lower-case, and use underscores "
                r"instead of hyphens: 'LambdaRuntimeClient' => "
                r"'lambdaruntimeclient'$"
            ),
            category=Warning,
            module=r"^passlib\.registry$",
        )
        import logfire

        logfire.configure(
            send_to_logfire=True,
            token=token,
            service_name=DOMAIN,
            console=False,
            inspect_arguments=False,
        )


def logfire_enabled(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Return if Logfire is actively emitting traces for this entry."""
    return logfire_active_for_entry(hass, entry)


def logfire_active_for_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Return if Logfire should emit traces for this entry."""
    token = _entry_logfire_token(entry)
    state = _logfire_state(hass)
    return (
        token is not None
        and token == state.configured_token
        and entry.entry_id in state.owner_include_content_by_entry_id
    )


def logfire_include_content(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Return if Logfire should include prompt and completion content."""
    if not logfire_active_for_entry(hass, entry):
        return False
    state = _logfire_state(hass)
    return state.owner_include_content_by_entry_id.get(entry.entry_id, False)


def logfire_token_conflict(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Return if the entry has a token that conflicts with active Logfire."""
    token = _entry_logfire_token(entry)
    configured_token = _logfire_state(hass).configured_token
    return (
        token is not None and configured_token is not None and token != configured_token
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
    profile: ModelProfile,
    attempt_index: int,
    attempt_count: int,
    entity_id: str,
    conversation_id: str | None,
) -> Iterator[Any | None]:
    """Wrap one Pydantic AI run with safe Home Assistant trace metadata."""
    if not logfire_active_for_entry(hass, entry):
        yield None
        return

    try:
        import logfire

        span = logfire.span(
            _span_title(subentry, profile, attempt_index, attempt_count),
            **_span_attributes(
                hass,
                entry,
                subentry,
                profile,
                entity_id,
                conversation_id,
                attempt_index,
                attempt_count,
            ),
        )
        span.__enter__()
    except Exception:
        _LOGGER.exception(
            "Failed to start Logfire span for Pydantic AI Agent entry %s",
            entry.entry_id,
        )
        yield None
        return

    exc_info: tuple[type[BaseException] | None, BaseException | None, object] = (
        None,
        None,
        None,
    )
    try:
        yield span
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
    profile: ModelProfile,
    entity_id: str,
    conversation_id: str | None,
    attempt_index: int,
    attempt_count: int,
) -> Mapping[str, Any]:
    """Return low-risk trace attributes for Home Assistant context."""
    llm_api_ids = subentry.data.get(CONF_LLM_HASS_API)
    if not isinstance(llm_api_ids, list):
        llm_api_ids = []
    attributes: dict[str, Any] = {
        "ha.domain": DOMAIN,
        "ha.version": __version__,
        "ha.core_version": __version__,
        "ha.entry_id": entry.entry_id,
        "ha.entry_title": entry.title,
        "ha.subentry_id": subentry.subentry_id,
        "ha.subentry_title": subentry.title,
        "ha.subentry_type": subentry.subentry_type,
        "ha.entity_id": entity_id,
        "ha.conversation_id": conversation_id,
        "ha.provider_mode": profile.provider_mode,
        "ha.provider_title": profile.provider_title,
        "ha.provider_subentry_id": profile.provider_subentry_id,
        "ha.model": profile.model_name,
        "ha.model_profile": profile.title,
        "ha.model_profile_ref": profile.ref,
        "ha.profile_id": profile.profile_id,
        "ha.attempt_index": attempt_index,
        "ha.attempt_count": attempt_count,
        "ha.is_fallback_attempt": attempt_index > 0,
        "ha.structured_output_mode": structured_output_mode(
            subentry.data.get(CONF_OUTPUT_MODE)
        ),
        "ha.ha_tools_enabled": bool(llm_api_ids),
        "ha.llm_api_ids": llm_api_ids,
        "ha.logfire_include_content": logfire_include_content(hass, entry),
        "ha.web_fetch_enabled": bool(subentry.data.get(CONF_WEB_FETCH_ENABLED, False)),
        "ha.virtual_workspace_enabled": bool(
            subentry.data.get(CONF_VIRTUAL_WORKSPACE_ENABLED, False)
        ),
        "gen_ai.operation.name": _gen_ai_operation_name(profile.provider_mode),
        "gen_ai.system": _gen_ai_provider_name(profile.provider_mode),
        "gen_ai.provider.name": _gen_ai_provider_name(profile.provider_mode),
        "gen_ai.request.model": profile.model_name,
        "gen_ai.response.model": profile.model_name,
    }
    if _INTEGRATION_VERSION is not None:
        attributes["ha.integration_version"] = _INTEGRATION_VERSION
    if subentry.subentry_type == SUBENTRY_TYPE_CONVERSATION:
        attributes["ha.agent_name"] = str(
            subentry.data.get(CONF_AGENT_NAME, subentry.title)
        )
    if subentry.subentry_type == SUBENTRY_TYPE_AI_TASK:
        attributes["ha.ai_task_name"] = str(
            subentry.data.get(CONF_AI_TASK_NAME, subentry.title)
        )
        attributes["ha.todo_workspace_enabled"] = bool(
            subentry.data.get(CONF_TODO_LIST_ENTITY_ID)
        )
    return attributes


def _span_title(
    subentry: ConfigSubentry,
    profile: ModelProfile,
    attempt_index: int,
    attempt_count: int,
) -> str:
    """Return the Logfire title for one wrapper span."""
    title = f"{subentry.title} → {profile.title}"
    if attempt_index > 0:
        title += f" (fallback {attempt_index + 1}/{attempt_count})"
    return _escape_span_template_text(title)


def _gen_ai_operation_name(provider_mode: str) -> str:
    """Return the GenAI operation name for the active provider mode."""
    if provider_mode == PROVIDER_GOOGLE_GEMINI:
        return "generate_content"
    return "chat"


def _gen_ai_provider_name(provider_mode: str) -> str:
    """Return the provider family for GenAI trace attributes."""
    if provider_mode in {
        PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
    }:
        return "openai"
    if provider_mode == PROVIDER_ANTHROPIC:
        return "anthropic"
    if provider_mode == PROVIDER_GOOGLE_GEMINI:
        return "gcp.gemini"
    return provider_mode


def _escape_span_template_text(value: str) -> str:
    """Escape braces so user-configured titles remain safe Logfire templates."""
    return value.replace("{", "{{").replace("}", "}}")
