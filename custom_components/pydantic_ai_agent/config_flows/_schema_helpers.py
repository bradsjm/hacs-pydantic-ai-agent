"""Schema builder helpers for config flows."""

from collections.abc import Iterable, Mapping
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import section
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
)
from homeassistant.helpers.typing import VolDictType

from ..const import (
    CONF_AGENT_NAME,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
    CONF_FALLBACK_MODEL_REFS,
    CONF_MCP_SERVER_IDS,
    CONF_MODEL,
    CONF_MODEL_PRICING,
    CONF_MODEL_SETTINGS,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROMPT,
    CONF_SKILLS,
    CONF_STREAMING_ENABLED,
    CONF_THINKING,
    CONF_TOOL_RETRIES,
    CONF_VIRTUAL_WORKSPACE_ENABLED,
    CONF_WEB_FETCH_ENABLED,
    CONF_WEB_SEARCH_ENABLED,
    DEFAULT_AGENT_NAME,
    DEFAULT_TIMEOUT,
    DEFAULT_TOOL_RETRIES,
)
from ._constants import (
    _MODEL_PRICING_CACHE_READ,
    _MODEL_PRICING_INPUT,
    _MODEL_PRICING_OUTPUT,
    _MODEL_SETTING_FREQUENCY_PENALTY,
    _MODEL_SETTING_MAX_ITERATIONS,
    _MODEL_SETTING_MAX_TOKENS,
    _MODEL_SETTING_PRESENCE_PENALTY,
    _MODEL_SETTING_SEED,
    _MODEL_SETTING_TEMPERATURE,
    _MODEL_SETTING_TEMPLATED_EXTRA_BODY,
    _MODEL_SETTING_THINKING,
    _MODEL_SETTING_TIMEOUT,
    _MODEL_SETTING_TOP_P,
    _RUN_SETTING_KEYS,
    _SECTION_ADVANCED_MODEL_SETTINGS,
    _SECTION_EXTERNAL_TOOLS,
    _SECTION_FALLBACK_MODELS,
    _SECTION_HASS_CONTROL,
    _SECTION_RUN_SETTINGS,
    _SECTION_SKILLS,
    _THINKING_OPTIONS,
)
from ._profile_helpers import RunSettingsVisibility, _run_settings_visibility
from ._profile_selection import (
    _deduplicate_fallback_model_refs,
    _fallback_model_profile_select_options,
    _model_profile_select_options,
    _normalise_fallback_model_refs,
)
from ._settings_parsing import (
    _format_templated_extra_body,
    _format_thinking_value,
    _normalise_run_settings,
)
from .helpers import (
    _flatten_section_data,
    _key_value_rows_selector,
    _section_schema_key,
)
from .mcp_helpers import (
    _append_mcp_server_schema_fields,
    _normalise_mcp_server_selection,
)
from .skill_helpers import _append_skill_schema_fields, _normalise_skill_selection

_THINKING_OPTIONS_WITHOUT_DISABLE = tuple(
    option for option in _THINKING_OPTIONS if option != "false"
)


def _agent_form_suggested_values(
    options: Mapping[str, Any], hass: HomeAssistant | None = None
) -> dict[str, Any]:
    """Return per-agent suggested values matching the sectioned form schema."""
    suggested_values = dict(options)
    suggested_values[_SECTION_RUN_SETTINGS] = {
        key: options[key] for key in _RUN_SETTING_KEYS if key in options
    }
    if CONF_LLM_HASS_API in options:
        llm_hass_api = options[CONF_LLM_HASS_API]
        if hass is not None:
            valid_api_ids = {api.id for api in llm.async_get_apis(hass)}
            llm_hass_api = [api for api in llm_hass_api if api in valid_api_ids]
        suggested_values[_SECTION_HASS_CONTROL] = {CONF_LLM_HASS_API: llm_hass_api}
    return suggested_values


def _conversation_schema(
    hass: HomeAssistant,
    options: Mapping[str, Any] | None = None,
    entry: ConfigEntry | None = None,
) -> vol.Schema:
    """Return the conversation subentry schema, pruning unavailable HA APIs."""
    options = dict(options or {})
    model_options = _model_profile_select_options(entry)
    fallback_refs = _deduplicate_fallback_model_refs(
        options.get(CONF_FALLBACK_MODEL_REFS, [])
    )
    primary_model_ref = options.get(CONF_PRIMARY_MODEL_REF)
    if not isinstance(primary_model_ref, str):
        primary_model_ref = None
    fallback_refs = [ref for ref in fallback_refs if ref != primary_model_ref]
    fallback_model_options = _fallback_model_profile_select_options(
        hass, entry, fallback_refs, primary_model_ref
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
            CONF_AGENT_NAME,
            default=options.get(CONF_AGENT_NAME, DEFAULT_AGENT_NAME),
        ): str,
        vol.Optional(
            CONF_PROMPT,
            description={"suggested_value": options.get(CONF_PROMPT, "")},
        ): TemplateSelector(),
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
                default=fallback_refs,
            )
        ] = SelectSelector(
            SelectSelectorConfig(
                options=fallback_model_options,
                mode=SelectSelectorMode.DROPDOWN,
                multiple=True,
                translation_key=CONF_FALLBACK_MODEL_REFS,
            )
        )
        schema[_section_schema_key(_SECTION_FALLBACK_MODELS, fallback_schema)] = (
            section(vol.Schema(fallback_schema), {"collapsed": True})
        )
    run_settings_schema = _run_settings_schema(
        options,
        default_max_iterations=10,
        visibility=_run_settings_visibility(
            entry,
            [
                options.get(CONF_PRIMARY_MODEL_REF),
                *fallback_refs,
            ],
        ),
    )
    run_settings_fields = dict(run_settings_schema.schema)
    run_settings_fields[
        vol.Optional(
            CONF_STREAMING_ENABLED,
            default=options.get(CONF_STREAMING_ENABLED, True),
        )
    ] = BooleanSelector()
    schema[_section_schema_key(_SECTION_RUN_SETTINGS, run_settings_fields)] = section(
        vol.Schema(run_settings_fields), {"collapsed": True}
    )
    api_schema_key = vol.Optional(CONF_LLM_HASS_API)
    if CONF_LLM_HASS_API in options:
        api_schema_key = vol.Optional(
            CONF_LLM_HASS_API,
            default=options[CONF_LLM_HASS_API],
        )
    hass_control_schema: VolDictType = {}
    hass_control_schema[api_schema_key] = SelectSelector(
        SelectSelectorConfig(
            options=hass_apis,
            mode=SelectSelectorMode.DROPDOWN,
            multiple=True,
        )
    )
    schema[_section_schema_key(_SECTION_HASS_CONTROL, hass_control_schema)] = section(
        vol.Schema(hass_control_schema), {"collapsed": True}
    )
    external_tools_schema: VolDictType = {}
    external_tools_schema[
        vol.Optional(
            CONF_WEB_FETCH_ENABLED,
            default=bool(options.get(CONF_WEB_FETCH_ENABLED, False)),
        )
    ] = BooleanSelector()
    external_tools_schema[
        vol.Optional(
            CONF_WEB_SEARCH_ENABLED,
            default=bool(options.get(CONF_WEB_SEARCH_ENABLED, False)),
        )
    ] = BooleanSelector()
    external_tools_schema[
        vol.Optional(
            CONF_VIRTUAL_WORKSPACE_ENABLED,
            default=options.get(CONF_VIRTUAL_WORKSPACE_ENABLED) is True,
        )
    ] = BooleanSelector()
    _append_mcp_server_schema_fields(external_tools_schema, options, entry)
    schema[_section_schema_key(_SECTION_EXTERNAL_TOOLS, external_tools_schema)] = (
        section(vol.Schema(external_tools_schema), {"collapsed": True})
    )
    skills_schema: VolDictType = {}
    _append_skill_schema_fields(skills_schema, options, entry)
    schema[_section_schema_key(_SECTION_SKILLS, skills_schema)] = section(
        vol.Schema(skills_schema, extra=vol.REMOVE_EXTRA), {"collapsed": True}
    )
    return vol.Schema(schema)


def _model_profile_schema(
    options: Mapping[str, Any] | None = None,
    model_names: Iterable[str] | None = None,
) -> vol.Schema:
    """Return the model profile subentry schema."""
    options = dict(options or {})
    model_settings = options.get(CONF_MODEL_SETTINGS, {})
    if not isinstance(model_settings, Mapping):
        model_settings = {}
    model_schema_key = vol.Required(
        CONF_MODEL,
        default=options.get(CONF_MODEL, ""),
    )
    if model_names:
        model_options = sorted(set(model_names))
        existing_model = options.get(CONF_MODEL)
        if isinstance(existing_model, str) and existing_model not in model_options:
            model_options.insert(0, existing_model)
        default_model = options.get(CONF_MODEL, model_options[0])
        model_schema_key = vol.Required(CONF_MODEL, default=default_model)
        model_selector = SelectSelector(
            SelectSelectorConfig(
                options=model_options,
                mode=SelectSelectorMode.DROPDOWN,
            )
        )
    else:
        model_selector = TextSelector(TextSelectorConfig())
    advanced_model_settings_schema = _model_settings_schema(options)
    return vol.Schema(
        {
            model_schema_key: model_selector,
            vol.Required(
                CONF_NAME,
                default=options.get(CONF_NAME, options.get(CONF_MODEL, "")),
            ): TextSelector(TextSelectorConfig()),
            vol.Optional(
                _MODEL_SETTING_TEMPERATURE,
                description={
                    "suggested_value": model_settings.get(_MODEL_SETTING_TEMPERATURE)
                },
            ): NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=0.1)
            ),
            _section_schema_key(
                _SECTION_ADVANCED_MODEL_SETTINGS,
                advanced_model_settings_schema.schema,
            ): section(advanced_model_settings_schema, {"collapsed": True}),
        }
    )


def _model_settings_schema(options: Mapping[str, Any] | None = None) -> vol.Schema:
    """Return the advanced model settings schema."""
    options = dict(options or {})
    model_settings = options.get(CONF_MODEL_SETTINGS, {})
    if not isinstance(model_settings, Mapping):
        model_settings = {}
    return vol.Schema(
        {
            vol.Optional(
                _MODEL_SETTING_TOP_P,
                description={
                    "suggested_value": model_settings.get(_MODEL_SETTING_TOP_P)
                },
            ): NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=0.1)
            ),
            vol.Optional(
                _MODEL_SETTING_SEED,
                description={
                    "suggested_value": model_settings.get(_MODEL_SETTING_SEED)
                },
            ): NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=1)
            ),
            vol.Optional(
                _MODEL_SETTING_PRESENCE_PENALTY,
                description={
                    "suggested_value": model_settings.get(
                        _MODEL_SETTING_PRESENCE_PENALTY
                    )
                },
            ): NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=0.1)
            ),
            vol.Optional(
                _MODEL_SETTING_FREQUENCY_PENALTY,
                description={
                    "suggested_value": model_settings.get(
                        _MODEL_SETTING_FREQUENCY_PENALTY
                    )
                },
            ): NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=0.1)
            ),
            vol.Optional(
                _MODEL_SETTING_TEMPLATED_EXTRA_BODY,
                default=_format_templated_extra_body(
                    model_settings.get(_MODEL_SETTING_TEMPLATED_EXTRA_BODY)
                ),
            ): _key_value_rows_selector(
                CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE, {"template": None}
            ),
        }
    )


def _run_settings_schema(
    options: Mapping[str, Any] | None = None,
    *,
    default_max_iterations: int,
    visibility: RunSettingsVisibility | None = None,
) -> vol.Schema:
    """Return per-conversation/task run settings schema."""
    options = dict(options or {})
    visibility = visibility or RunSettingsVisibility()
    schema: VolDictType = {
        vol.Optional(
            _MODEL_SETTING_MAX_TOKENS,
            description={"suggested_value": options.get(_MODEL_SETTING_MAX_TOKENS)},
        ): NumberSelector(NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=1)),
        vol.Required(
            _MODEL_SETTING_MAX_ITERATIONS,
            default=options.get(_MODEL_SETTING_MAX_ITERATIONS, default_max_iterations),
        ): NumberSelector(NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=1)),
        vol.Required(
            _MODEL_SETTING_TIMEOUT,
            default=options.get(_MODEL_SETTING_TIMEOUT, DEFAULT_TIMEOUT),
        ): NumberSelector(NumberSelectorConfig(mode=NumberSelectorMode.BOX, step=0.1)),
        vol.Required(
            CONF_TOOL_RETRIES,
            default=options.get(CONF_TOOL_RETRIES, DEFAULT_TOOL_RETRIES),
        ): NumberSelector(
            NumberSelectorConfig(mode=NumberSelectorMode.BOX, min=0, step=1)
        ),
    }
    if visibility.supports_thinking:
        thinking_options = (
            list(_THINKING_OPTIONS)
            if visibility.can_disable_thinking
            else list(_THINKING_OPTIONS_WITHOUT_DISABLE)
        )
        thinking_default = _format_thinking_value(options)
        if thinking_default == "false" and not visibility.can_disable_thinking:
            thinking_default = ""
        schema[
            vol.Optional(
                _MODEL_SETTING_THINKING,
                default=thinking_default,
            )
        ] = SelectSelector(
            SelectSelectorConfig(
                options=thinking_options,
                mode=SelectSelectorMode.DROPDOWN,
                translation_key=_MODEL_SETTING_THINKING,
            )
        )
    return vol.Schema(schema)


def _model_pricing_schema(options: Mapping[str, Any] | None = None) -> vol.Schema:
    """Return the model pricing schema."""
    options = dict(options or {})
    pricing = options.get(CONF_MODEL_PRICING, {})
    if not isinstance(pricing, Mapping):
        pricing = {}
    input_key = vol.Optional(_MODEL_PRICING_INPUT)
    if pricing.get("input") is not None:
        input_key = vol.Optional(_MODEL_PRICING_INPUT, default=pricing["input"])
    output_key = vol.Optional(_MODEL_PRICING_OUTPUT)
    if pricing.get("output") is not None:
        output_key = vol.Optional(_MODEL_PRICING_OUTPUT, default=pricing["output"])
    cache_read_key = vol.Optional(_MODEL_PRICING_CACHE_READ)
    if pricing.get("cache_read") is not None:
        cache_read_key = vol.Optional(
            _MODEL_PRICING_CACHE_READ, default=pricing["cache_read"]
        )
    return vol.Schema(
        {
            input_key: NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step="any")
            ),
            output_key: NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step="any")
            ),
            cache_read_key: NumberSelector(
                NumberSelectorConfig(mode=NumberSelectorMode.BOX, step="any")
            ),
        }
    )


def _conversation_data_from_user_input(
    user_input: Mapping[str, Any],
    options: Mapping[str, Any],
    entry: ConfigEntry | None = None,
) -> dict[str, Any]:
    """Return conversation fields with model profile references."""
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
    data = {key: value for key, value in user_input.items()}
    data[CONF_FALLBACK_MODEL_REFS] = _normalise_fallback_model_refs(
        data.get(CONF_FALLBACK_MODEL_REFS, [])
    )
    if not data.get(CONF_LLM_HASS_API):
        data.pop(CONF_LLM_HASS_API, None)
    if data.get(CONF_STREAMING_ENABLED, True) is not False:
        data.pop(CONF_STREAMING_ENABLED, None)
    _drop_disabled_external_tool_flags(data)
    if data.get(CONF_VIRTUAL_WORKSPACE_ENABLED) is not True:
        data.pop(CONF_VIRTUAL_WORKSPACE_ENABLED, None)
    if not data.get(CONF_FALLBACK_MODEL_REFS):
        data.pop(CONF_FALLBACK_MODEL_REFS, None)
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


def _drop_disabled_external_tool_flags(data: dict[str, Any]) -> None:
    """Remove absent or disabled external tool flags from subentry data."""
    for key in (CONF_WEB_FETCH_ENABLED, CONF_WEB_SEARCH_ENABLED):
        if not data.get(key):
            data.pop(key, None)
