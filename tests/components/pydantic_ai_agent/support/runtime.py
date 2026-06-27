"""Runtime-oriented test helpers for Pydantic AI Agent tests."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from custom_components.pydantic_ai_agent import ProviderRuntimeData
from custom_components.pydantic_ai_agent.const import (
    CONF_DEFAULT_MODEL_PROFILE_ID,
    DOMAIN,
)
from custom_components.pydantic_ai_agent.entity import unique_id_for_subentry_entity
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.util import slugify
from pydantic_ai.capabilities import Thinking
from pydantic_ai_summarization import ContextManagerCapability, SlidingWindowCapability
from pytest_homeassistant_custom_component.common import MockConfigEntry

from .builders import (
    ai_task_subentry_data,
    conversation_subentry_data,
    loaded_workspace_entry,
    provider_runtime_data,
    provider_subentry_data,
)


def loaded_conversation_entry(
    *,
    llm_hass_api: Sequence[str] | None = None,
    skills: Sequence[str] | None = None,
    streaming_enabled: bool = True,
    virtual_workspace_enabled: bool = False,
    web_fetch_enabled: bool = False,
    web_search_enabled: bool = False,
    extra_data: Mapping[str, object] | None = None,
    workspace_data: Mapping[str, object] | None = None,
    provider_subentry: dict[str, object] | None = None,
    conversation_subentries: tuple[dict[str, object], ...] | None = None,
    provider_runtime: ProviderRuntimeData | None = None,
) -> MockConfigEntry:
    """Return a workspace entry with conversation subentries and runtime data."""
    provider = provider_subentry or provider_subentry_data()
    provider_id = str(provider["subentry_id"])
    provider_data = provider["data"]
    assert isinstance(provider_data, dict)
    profile_ref = f"{provider_id}:{provider_data[CONF_DEFAULT_MODEL_PROFILE_ID]}"
    conversations = conversation_subentries or (
        conversation_subentry_data(
            profile_ref,
            llm_hass_api=llm_hass_api,
            skills=skills,
            streaming_enabled=streaming_enabled,
            virtual_workspace_enabled=virtual_workspace_enabled,
            web_fetch_enabled=web_fetch_enabled,
            web_search_enabled=web_search_enabled,
            extra_data=extra_data,
        ),
    )
    return loaded_workspace_entry(
        (*conversations, provider),
        data=workspace_data,
        providers={
            provider_id: provider_runtime
            or provider_runtime_data(
                subentry_id=provider_id,
                name=str(provider["title"]),
            )
        },
    )


def loaded_ai_task_entry(
    *,
    include_task_name: bool = True,
    subentry_title: str = "AI task subentry title",
    web_search_enabled: bool = False,
    extra_data: Mapping[str, object] | None = None,
    workspace_data: Mapping[str, object] | None = None,
    provider_subentry: dict[str, object] | None = None,
    ai_task_subentries: tuple[dict[str, object], ...] | None = None,
    provider_runtime: ProviderRuntimeData | None = None,
) -> MockConfigEntry:
    """Return a workspace entry with AI task subentries and runtime data."""
    provider = provider_subentry or provider_subentry_data()
    provider_id = str(provider["subentry_id"])
    provider_data = provider["data"]
    assert isinstance(provider_data, dict)
    profile_ref = f"{provider_id}:{provider_data[CONF_DEFAULT_MODEL_PROFILE_ID]}"
    ai_task_subentries_data = ai_task_subentries or (
        ai_task_subentry_data(
            profile_ref,
            title=subentry_title,
            task_name="Report task" if include_task_name else None,
            web_search_enabled=web_search_enabled,
            extra_data=extra_data,
        ),
    )
    return loaded_workspace_entry(
        (*ai_task_subentries_data, provider),
        data=workspace_data,
        providers={
            provider_id: provider_runtime
            or provider_runtime_data(
                subentry_id=provider_id,
                name=str(provider["title"]),
            )
        },
    )


def first_non_default_conversation_entity_id(
    hass: HomeAssistant, *, suffix: str | None = None
) -> str:
    """Return the first non-default conversation entity id."""
    for state in hass.states.async_all("conversation"):
        if state.entity_id == "conversation.home_assistant":
            continue
        if suffix is not None and not state.entity_id.endswith(suffix):
            continue
        return state.entity_id
    raise AssertionError("No matching conversation entity found")


def only_entity_id(hass: HomeAssistant, domain: str) -> str:
    """Return the single entity id for a domain."""
    entity_ids = [state.entity_id for state in hass.states.async_all(domain)]
    assert len(entity_ids) == 1
    return entity_ids[0]


def state_value(hass: HomeAssistant, entity_id: str) -> str:
    """Return a required entity state value."""
    state = hass.states.get(entity_id)
    assert state is not None
    return state.state


def enable_diagnostic_entities(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    subentry: ConfigSubentry,
    entity_domain: str,
    keys: tuple[str, ...],
    *,
    name: str,
) -> None:
    """Create disabled-by-default diagnostic entities for a subentry."""
    entity_registry = er.async_get(hass)
    for key in keys:
        entity_registry.async_get_or_create(
            entity_domain,
            DOMAIN,
            unique_id_for_subentry_entity(entry, subentry, key),
            suggested_object_id=slugify(f"{name} {key}"),
        )


def assert_has_context_management_capability(capabilities: Sequence[object]) -> None:
    """Assert a capabilities collection contains context management."""
    assert any(
        isinstance(capability, ContextManagerCapability | SlidingWindowCapability)
        for capability in capabilities
    )


def thinking_capabilities(capabilities: Sequence[object]) -> list[Thinking]:
    """Return Thinking capabilities from an Agent constructor call."""
    return [
        capability for capability in capabilities if isinstance(capability, Thinking)
    ]


def diagnostics_subentry(
    diagnostics: Mapping[str, Any],
    *,
    subentry_type: str | None = None,
    subentry_id: str | None = None,
    title: str | None = None,
) -> Mapping[str, Any]:
    """Return a diagnostics subentry matched by stable identifying fields."""
    for subentry in diagnostics["subentries"]:
        summary = subentry.get("configuration_summary", {})
        if subentry_type is not None and summary.get("subentry_type") != subentry_type:
            continue
        if subentry_id is not None and subentry.get("subentry_id") != subentry_id:
            continue
        if title is not None and subentry.get("title") != title:
            continue
        return cast(Mapping[str, Any], subentry)
    raise AssertionError("No matching diagnostics subentry found")
