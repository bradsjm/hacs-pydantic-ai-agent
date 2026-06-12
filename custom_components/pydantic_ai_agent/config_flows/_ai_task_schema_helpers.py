"""AI task schema helpers for config flows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import voluptuous as vol
from homeassistant.components.todo.const import DOMAIN as TODO_DOMAIN
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import section
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    BooleanSelector,
    EntitySelector,
    EntitySelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
)
from homeassistant.helpers.typing import VolDictType

from ..const import (
    CONF_AI_TASK_NAME,
    CONF_FALLBACK_MODEL_REFS,
    CONF_LLM_HASS_API,
    CONF_MCP_SERVER_IDS,
    CONF_OUTPUT_MODE,
    CONF_PRIMARY_MODEL_REF,
    CONF_SKILLS,
    CONF_THINKING,
    CONF_TODO_LIST_ENTITY_ID,
    CONF_VIRTUAL_WORKSPACE_ENABLED,
    CONF_WEB_FETCH_ENABLED,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_OUTPUT_MODE,
)
from ..structured_output import (
    structured_output_mode as normalise_structured_output_mode,
)
from ._constants import (
    _OUTPUT_MODE_OPTIONS,
    _SECTION_EXTERNAL_TOOLS,
    _SECTION_FALLBACK_MODELS,
    _SECTION_HASS_CONTROL,
    _SECTION_RUN_SETTINGS,
    _SECTION_SKILLS,
)
from ._profile_helpers import (
    _FALLBACK_MODEL_REF_FIELD,
    _fallback_model_profile_select_options,
    _format_fallback_model_ref_rows,
    _model_profile_select_options,
    _normalise_fallback_model_refs,
    _parse_fallback_model_ref_rows,
    _run_settings_visibility,
)
from ._schema_helpers import _run_settings_schema
from ._settings_parsing import _normalise_run_settings
from .helpers import (
    _flatten_section_data,
    _section_schema_key,
    _single_value_rows_selector,
)
from .mcp_helpers import (
    _append_mcp_server_schema_fields,
    _normalise_mcp_server_selection,
)
from .skill_helpers import _append_skill_schema_fields, _normalise_skill_selection


def _ai_task_data_schema(
    hass: HomeAssistant,
    options: Mapping[str, Any] | None = None,
    entry: ConfigEntry | None = None,
) -> vol.Schema:
    """Return the AI task data subentry schema."""
    options = dict(options or {})
    model_options = _model_profile_select_options(entry)
    fallback_refs = _normalise_fallback_model_refs(
        options.get(CONF_FALLBACK_MODEL_REFS, [])
    )
    if not fallback_refs:
        fallback_refs = _parse_fallback_model_ref_rows(
            options.get(CONF_FALLBACK_MODEL_REFS, [])
        )
    fallback_model_options = _fallback_model_profile_select_options(
        hass, entry, fallback_refs
    )
    hass_apis: list[SelectOptionDict] = []
    valid_api_ids: set[str] = set()
    for api in llm.async_get_apis(hass):
        hass_apis.append(SelectOptionDict(label=api.name, value=api.id))
        valid_api_ids.add(api.id)

    if CONF_LLM_HASS_API in options:
        options[CONF_LLM_HASS_API] = [
            api for api in options[CONF_LLM_HASS_API] if api in valid_api_ids
        ]
    schema: VolDictType = {
        vol.Required(
            CONF_AI_TASK_NAME,
            default=options.get(CONF_AI_TASK_NAME, DEFAULT_AI_TASK_NAME),
        ): TextSelector(TextSelectorConfig()),
    }
    if model_options:
        schema[
            vol.Required(
                CONF_PRIMARY_MODEL_REF,
                default=options.get(CONF_PRIMARY_MODEL_REF, ""),
            )
        ] = SelectSelector(
            SelectSelectorConfig(
                options=model_options,
                mode=SelectSelectorMode.DROPDOWN,
                translation_key=CONF_PRIMARY_MODEL_REF,
            )
        )
        fallback_schema: VolDictType = {}
        fallback_schema[
            vol.Optional(
                CONF_FALLBACK_MODEL_REFS,
                default=_format_fallback_model_ref_rows(
                    options.get(CONF_FALLBACK_MODEL_REFS, [])
                ),
            )
        ] = _single_value_rows_selector(
            _FALLBACK_MODEL_REF_FIELD,
            SelectSelector(
                SelectSelectorConfig(
                    options=fallback_model_options,
                    translation_key=CONF_FALLBACK_MODEL_REFS,
                )
            ),
        )
        schema[_section_schema_key(_SECTION_FALLBACK_MODELS, fallback_schema)] = (
            section(vol.Schema(fallback_schema), {"collapsed": True})
        )
    run_settings_schema = _run_settings_schema(
        options,
        default_max_iterations=30,
        visibility=_run_settings_visibility(
            entry,
            [
                options.get(CONF_PRIMARY_MODEL_REF),
                *fallback_refs,
            ],
        ),
    )
    schema[_section_schema_key(_SECTION_RUN_SETTINGS, run_settings_schema.schema)] = (
        section(run_settings_schema, {"collapsed": True})
    )
    api_schema_key = vol.Optional(CONF_LLM_HASS_API)
    if CONF_LLM_HASS_API in options:
        api_schema_key = vol.Optional(
            CONF_LLM_HASS_API,
            default=options[CONF_LLM_HASS_API],
        )
    hass_control_schema: VolDictType = {}
    hass_control_schema[api_schema_key] = SelectSelector(
        SelectSelectorConfig(options=hass_apis, multiple=True)
    )
    schema[_section_schema_key(_SECTION_HASS_CONTROL, hass_control_schema)] = section(
        vol.Schema(hass_control_schema), {"collapsed": True}
    )
    external_tools_schema: VolDictType = {}
    todo_schema_key = vol.Optional(CONF_TODO_LIST_ENTITY_ID)
    if CONF_TODO_LIST_ENTITY_ID in options:
        todo_schema_key = vol.Optional(
            CONF_TODO_LIST_ENTITY_ID,
            default=options[CONF_TODO_LIST_ENTITY_ID],
        )
    external_tools_schema[todo_schema_key] = EntitySelector(
        EntitySelectorConfig(domain=TODO_DOMAIN)
    )
    external_tools_schema[
        vol.Optional(
            CONF_WEB_FETCH_ENABLED,
            default=bool(options.get(CONF_WEB_FETCH_ENABLED, False)),
        )
    ] = BooleanSelector()
    external_tools_schema[
        vol.Optional(
            CONF_VIRTUAL_WORKSPACE_ENABLED,
            default=options.get(CONF_VIRTUAL_WORKSPACE_ENABLED) is True,
        )
    ] = BooleanSelector()
    _append_mcp_server_schema_fields(external_tools_schema, options, entry)
    schema[
        vol.Required(
            CONF_OUTPUT_MODE,
            default=normalise_structured_output_mode(
                options.get(CONF_OUTPUT_MODE, DEFAULT_OUTPUT_MODE)
            ),
        )
    ] = SelectSelector(
        SelectSelectorConfig(
            options=list(_OUTPUT_MODE_OPTIONS),
            mode=SelectSelectorMode.DROPDOWN,
            translation_key=CONF_OUTPUT_MODE,
        )
    )
    schema[_section_schema_key(_SECTION_EXTERNAL_TOOLS, external_tools_schema)] = (
        section(vol.Schema(external_tools_schema), {"collapsed": True})
    )
    skills_schema: VolDictType = {}
    _append_skill_schema_fields(skills_schema, options, entry)
    schema[_section_schema_key(_SECTION_SKILLS, skills_schema)] = section(
        vol.Schema(skills_schema, extra=vol.REMOVE_EXTRA), {"collapsed": True}
    )
    return vol.Schema(schema)


def _ai_task_data_from_user_input(
    user_input: Mapping[str, Any],
    options: Mapping[str, Any],
    entry: ConfigEntry | None = None,
) -> dict[str, Any]:
    """Return AI task subentry data with a selected structured output mode."""
    user_input = _flatten_section_data(
        user_input,
        (
            _SECTION_EXTERNAL_TOOLS,
            _SECTION_FALLBACK_MODELS,
            _SECTION_HASS_CONTROL,
            _SECTION_RUN_SETTINGS,
            _SECTION_SKILLS,
        ),
    )
    data = dict(user_input)
    data[CONF_FALLBACK_MODEL_REFS] = _parse_fallback_model_ref_rows(
        data.get(CONF_FALLBACK_MODEL_REFS)
    )
    data.setdefault(
        CONF_OUTPUT_MODE,
        normalise_structured_output_mode(options.get(CONF_OUTPUT_MODE)),
    )
    if not data.get(CONF_WEB_FETCH_ENABLED):
        data.pop(CONF_WEB_FETCH_ENABLED, None)
    if data.get(CONF_VIRTUAL_WORKSPACE_ENABLED) is not True:
        data.pop(CONF_VIRTUAL_WORKSPACE_ENABLED, None)
    if not data.get(CONF_FALLBACK_MODEL_REFS):
        data.pop(CONF_FALLBACK_MODEL_REFS, None)
    if not data.get(CONF_TODO_LIST_ENTITY_ID):
        data.pop(CONF_TODO_LIST_ENTITY_ID, None)
    if not data.get(CONF_LLM_HASS_API):
        data.pop(CONF_LLM_HASS_API, None)
    if CONF_SKILLS not in user_input and options.get(CONF_SKILLS):
        data[CONF_SKILLS] = options[CONF_SKILLS]
    if CONF_MCP_SERVER_IDS not in user_input and options.get(CONF_MCP_SERVER_IDS):
        data[CONF_MCP_SERVER_IDS] = options[CONF_MCP_SERVER_IDS]
    _normalise_run_settings(data)
    fallback_refs = data.get(CONF_FALLBACK_MODEL_REFS, [])
    if isinstance(fallback_refs, str) or not isinstance(fallback_refs, list):
        fallback_refs = []
    visibility = _run_settings_visibility(
        entry,
        [data.get(CONF_PRIMARY_MODEL_REF), *fallback_refs],
    )
    if not visibility.supports_thinking or (
        data.get(CONF_THINKING) is False and not visibility.can_disable_thinking
    ):
        data.pop(CONF_THINKING, None)
    _normalise_mcp_server_selection(data)
    _normalise_skill_selection(data)
    return data
