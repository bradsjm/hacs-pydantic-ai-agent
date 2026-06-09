"""Neutral builders for Pydantic AI Agent tests."""

from collections.abc import Mapping, Sequence
from typing import Any

from custom_components.pydantic_ai_agent import (
    ProviderRuntimeData,
    WorkspaceRuntimeData,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_AI_TASK_NAME,
    CONF_BASE_URL,
    CONF_DEFAULT_MODEL_PROFILE_ID,
    CONF_DESCRIPTION,
    CONF_DISCOVERED,
    CONF_ENABLED,
    CONF_MODEL,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_OUTPUT_MODE,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_METADATA,
    CONF_PROVIDER_MODE,
    CONF_SKILL_CONTENT,
    CONF_SKILL_REFERENCES,
    CONF_SKILLS,
    CONF_TODO_LIST_ENTITY_ID,
    CONF_VIRTUAL_WORKSPACE_ENABLED,
    CONF_WEB_FETCH_ENABLED,
    DOMAIN,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_PROVIDER,
    SUBENTRY_TYPE_SKILL,
)
from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_NAME
from pytest_homeassistant_custom_component.common import MockConfigEntry


def model_profile_data(
    *,
    profile_id: str = "profile-1",
    name: str = "Fast GPT",
    model: str = "gpt-test",
    enabled: bool = True,
    discovered: bool | None = None,
    model_settings: Mapping[str, object] | None = None,
    extra_data: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return a model profile data mapping."""
    data: dict[str, object] = {
        "id": profile_id,
        CONF_NAME: name,
        CONF_MODEL: model,
        CONF_ENABLED: enabled,
    }
    if discovered is not None:
        data[CONF_DISCOVERED] = discovered
    if model_settings is not None:
        data[CONF_MODEL_SETTINGS] = dict(model_settings)
    if extra_data is not None:
        data.update(extra_data)
    return data


def provider_subentry_data(
    *,
    subentry_id: str = "provider-1",
    title: str = "OpenAI-compatible",
    name: str | None = None,
    api_key: str = "sk-test",
    provider_mode: str = PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    profile_id: str = "profile-1",
    profile_name: str = "Fast GPT",
    model: str = "gpt-test",
    model_settings: Mapping[str, object] | None = None,
    model_profiles: Mapping[str, Mapping[str, object]] | None = None,
    default_model_profile_id: str | None = None,
    base_url: str | None = None,
    provider_headers: Mapping[str, str] | None = None,
    provider_extra_body: Mapping[str, Any] | None = None,
    provider_metadata: Mapping[str, object] | None = None,
    discovered: bool | None = None,
) -> dict[str, object]:
    """Return a provider config subentry payload."""
    profiles = (
        {key: dict(value) for key, value in model_profiles.items()}
        if model_profiles is not None
        else {
            profile_id: model_profile_data(
                profile_id=profile_id,
                name=profile_name,
                model=model,
                discovered=discovered,
                model_settings=model_settings,
            )
        }
    )
    default_profile_id = default_model_profile_id or next(iter(profiles))
    data: dict[str, object] = {
        CONF_NAME: title if name is None else name,
        CONF_PROVIDER_MODE: provider_mode,
        CONF_API_KEY: api_key,
        CONF_MODEL_PROFILES: profiles,
        CONF_DEFAULT_MODEL_PROFILE_ID: default_profile_id,
    }
    if base_url is not None:
        data[CONF_BASE_URL] = base_url
    if provider_headers is not None:
        data[CONF_PROVIDER_HEADERS] = dict(provider_headers)
    if provider_extra_body is not None:
        data[CONF_PROVIDER_EXTRA_BODY] = dict(provider_extra_body)
    if provider_metadata is not None:
        data[CONF_PROVIDER_METADATA] = dict(provider_metadata)
    return {
        "subentry_id": subentry_id,
        "subentry_type": SUBENTRY_TYPE_PROVIDER,
        "title": title,
        "unique_id": None,
        "data": data,
    }


def conversation_subentry_data(
    profile_ref: str,
    *,
    subentry_id: str | None = None,
    title: str = "Kitchen Agent",
    agent_name: str | None = "Kitchen Agent",
    llm_hass_api: Sequence[str] | None = None,
    skills: Sequence[str] | None = None,
    virtual_workspace_enabled: bool = False,
    web_fetch_enabled: bool = False,
    extra_data: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return a conversation config subentry payload."""
    data: dict[str, object] = {CONF_PRIMARY_MODEL_REF: profile_ref}
    if agent_name is not None:
        data[CONF_AGENT_NAME] = agent_name
    if llm_hass_api is not None:
        data["llm_hass_api"] = list(llm_hass_api)
    if skills is not None:
        data[CONF_SKILLS] = list(skills)
    if web_fetch_enabled:
        data[CONF_WEB_FETCH_ENABLED] = True
    if virtual_workspace_enabled:
        data[CONF_VIRTUAL_WORKSPACE_ENABLED] = True
    if extra_data is not None:
        data.update(extra_data)
    payload: dict[str, object] = {
        "subentry_type": SUBENTRY_TYPE_CONVERSATION,
        "title": title,
        "unique_id": None,
        "data": data,
    }
    if subentry_id is not None:
        payload["subentry_id"] = subentry_id
    return payload


def ai_task_subentry_data(
    profile_ref: str,
    *,
    subentry_id: str | None = None,
    title: str = "Report task",
    task_name: str | None = None,
    output_mode: str | None = None,
    skills: Sequence[str] | None = None,
    virtual_workspace_enabled: bool = False,
    web_fetch_enabled: bool = False,
    todo_workspace_entity_id: str | None = None,
    extra_data: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return an AI task config subentry payload."""
    data: dict[str, object] = {CONF_PRIMARY_MODEL_REF: profile_ref}
    if task_name is not None:
        data[CONF_AI_TASK_NAME] = task_name
    if output_mode is not None:
        data[CONF_OUTPUT_MODE] = output_mode
    if skills is not None:
        data[CONF_SKILLS] = list(skills)
    if web_fetch_enabled:
        data[CONF_WEB_FETCH_ENABLED] = True
    if virtual_workspace_enabled:
        data[CONF_VIRTUAL_WORKSPACE_ENABLED] = True
    if todo_workspace_entity_id is not None:
        data[CONF_TODO_LIST_ENTITY_ID] = todo_workspace_entity_id
    if extra_data is not None:
        data.update(extra_data)
    payload: dict[str, object] = {
        "subentry_type": SUBENTRY_TYPE_AI_TASK,
        "title": title,
        "unique_id": None,
        "data": data,
    }
    if subentry_id is not None:
        payload["subentry_id"] = subentry_id
    return payload


def skill_subentry_data(
    *,
    subentry_id: str = "skill-1",
    title: str = "Skill",
    description: str | None = None,
    content: str = "Use this skill.",
    references: Sequence[str] | None = None,
) -> dict[str, object]:
    """Return a native Skill config subentry payload."""
    data: dict[str, object] = {CONF_NAME: title, CONF_SKILL_CONTENT: content}
    if description is not None:
        data[CONF_DESCRIPTION] = description
    if references is not None:
        data[CONF_SKILL_REFERENCES] = list(references)
    return {
        "subentry_id": subentry_id,
        "subentry_type": SUBENTRY_TYPE_SKILL,
        "title": title,
        "unique_id": None,
        "data": data,
    }


def workspace_entry(
    subentries_data: tuple[dict[str, object], ...] = (),
    *,
    title: str = "Workspace",
    data: Mapping[str, object] | None = None,
    options: Mapping[str, object] | None = None,
) -> MockConfigEntry:
    """Return a workspace config entry."""
    entry_data: dict[str, object] = {CONF_NAME: title}
    if data is not None:
        entry_data.update(data)
    return MockConfigEntry(
        domain=DOMAIN,
        title=title,
        data=entry_data,
        source=config_entries.SOURCE_USER,
        subentries_data=subentries_data,
        options={} if options is None else dict(options),
        unique_id=None,
        version=2,
        minor_version=1,
    )


def provider_runtime_data(
    *,
    subentry_id: str = "provider-1",
    name: str = "Hosted OpenAI",
    api_key: str = "sk-test",
    provider_mode: str = PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    base_url: str | None = None,
    provider_headers: Mapping[str, str] | None = None,
    provider_extra_body: Mapping[str, Any] | None = None,
) -> ProviderRuntimeData:
    """Return provider runtime data."""
    return ProviderRuntimeData(
        provider_subentry_id=subentry_id,
        name=name,
        api_key=api_key,
        provider_mode=provider_mode,
        base_url=base_url,
        provider_headers={} if provider_headers is None else dict(provider_headers),
        provider_extra_body=(
            {} if provider_extra_body is None else dict(provider_extra_body)
        ),
    )


def workspace_runtime_data(
    *,
    workspace_name: str = "Workspace",
    providers: Mapping[str, ProviderRuntimeData] | None = None,
    logfire_enabled: bool = False,
    logfire_include_content: bool = False,
) -> WorkspaceRuntimeData:
    """Return workspace runtime data."""
    return WorkspaceRuntimeData(
        workspace_name=workspace_name,
        providers={} if providers is None else dict(providers),
        logfire_enabled=logfire_enabled,
        logfire_include_content=logfire_include_content,
    )
