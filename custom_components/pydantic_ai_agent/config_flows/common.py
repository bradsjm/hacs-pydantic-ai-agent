"""Shared config-flow helpers for Pydantic AI Agent.

Re-export shim for all helpers extracted into leaf modules.
"""

# ruff: noqa: F401

from __future__ import annotations

import logging
from collections.abc import AsyncIterable, Callable, Iterable, Mapping
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import (
    SOURCE_USER,
    ConfigEntry,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentry,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.core import callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util
from pydantic_ai import (
    ModelRequest,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.messages import ModelResponseStreamEvent
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings

from ..const import (
    CONF_AGENT_NAME,
    CONF_AI_TASK_NAME,
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_CUSTOM_MODEL_NAMES,
    CONF_DISCOVERED,
    CONF_DISCOVERED_MODELS,
    CONF_DISCOVERED_MODELS_AT,
    CONF_DISCOVERED_MODELS_CACHE_KEY,
    CONF_ENABLED,
    CONF_FALLBACK_MODEL_REFS,
    CONF_MCP_SERVER_IDS,
    CONF_MODEL,
    CONF_MODEL_PRICING,
    CONF_MODEL_PROFILES,
    CONF_MODEL_SETTINGS,
    CONF_NAME,
    CONF_OUTPUT_MODE,
    CONF_PRIMARY_MODEL_REF,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_METADATA,
    CONF_PROVIDER_MODE,
    CONF_TODO_LIST_ENTITY_ID,
    DOMAIN,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE_GEMINI,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    PROVIDER_OPENAI_COMPATIBLE_RESPONSES,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_MCP_SERVER,
    SUBENTRY_TYPE_PROVIDER,
    SUBENTRY_TYPE_SKILL,
    default_conversation_options,
)
from ..model_profiles import (
    configured_model_profile_exists,
    model_profile_display_name,
    model_profile_ref,
    parse_model_profile_ref,
    provider_model_profiles,
    provider_subentries,
)
from ..provider_validation import (
    ProviderValidationError,
    _format_api_error,
    _map_http_error,
    async_list_provider_model_names,
    async_probe_model,
)
from ._ai_task_schema_helpers import (
    _ai_task_data_from_user_input,
    _ai_task_data_schema,
)
from ._constants import (
    _ADVANCED_MODEL_SETTING_KEYS,
    _CONF_MODEL_PROFILE_ID,
    _MAIN_MODEL_SETTING_KEYS,
    _MODEL_LIST_CACHE_TTL,
    _MODEL_PRICING_CACHE_READ,
    _MODEL_PRICING_INPUT,
    _MODEL_PRICING_OUTPUT,
    _MODEL_SETTING_CHAT_TEMPLATE_KWARGS,
    _MODEL_SETTING_EXTRA_BODY,
    _MODEL_SETTING_FREQUENCY_PENALTY,
    _MODEL_SETTING_MAX_ITERATIONS,
    _MODEL_SETTING_MAX_TOKENS,
    _MODEL_SETTING_PARALLEL_TOOL_CALLS,
    _MODEL_SETTING_PRESENCE_PENALTY,
    _MODEL_SETTING_SEED,
    _MODEL_SETTING_TEMPERATURE,
    _MODEL_SETTING_THINKING,
    _MODEL_SETTING_TIMEOUT,
    _MODEL_SETTING_TOP_P,
    _OUTPUT_MODE_OPTIONS,
    _PROVIDER_EXTRA_BODY_MODES,
    _REMOVED_MODEL_SETTING_KEYS,
    _RUN_SETTING_KEYS,
    _SECTION_ADVANCED_MODEL_SETTINGS,
    _SECTION_ADVANCED_OPTIONS,
    _SECTION_CUSTOMIZE_MODEL_LIST,
    _SECTION_EXTERNAL_TOOLS,
    _SECTION_FALLBACK_MODELS,
    _SECTION_HASS_CONTROL,
    _SECTION_MODEL_PRICING,
    _SECTION_RUN_SETTINGS,
    _SECTION_SKILLS,
    _STRUCTURED_PROBE_OUTPUT_NAME,
    _STRUCTURED_PROBE_SCHEMA,
    _THINKING_OPTIONS,
    _TODO_WORKSPACE_REQUIRED_FEATURES,
)
from ._profile_helpers import (
    _classify_existing_provider_profile,
    _fallback_model_profile_select_options,
    _log_provider_validation_failure,
    _model_profile_edit_schema,
    _model_profile_select_options,
    _normalise_fallback_model_refs,
    _normalise_provider_model_profiles,
    _normalised_provider_profile,
    _provider_model_profiles_for_discovery_mode,
    _provider_profile_dependents,
    _provider_profile_options,
    _provider_profile_selector_schema,
    _provider_validation_placeholders,
    _referenced_provider_profile_ids,
    _selected_model_profile_error,
    _selected_model_profile_refs,
    _selected_todo_workspace_error,
)
from ._provider_data import (
    _base_url_endpoint_suffix,
    _cached_provider_model_names,
    _clear_provider_model_cache,
    _dedupe_data,
    _format_custom_model_names,
    _format_http_headers,
    _normalise_base_url,
    _normalise_provider_data,
    _parse_custom_model_names,
    _parse_http_headers,
    _parse_provider_headers,
    _provider_connection_schema,
    _provider_custom_model_names,
    _provider_data_matches,
    _provider_extra_body_supported,
    _provider_model_cache_key,
    _provider_schema,
    _store_provider_model_cache,
    _validate_base_url,
    _validate_provider_data,
)
from ._schema_helpers import (
    _agent_form_suggested_values,
    _conversation_data_from_user_input,
    _conversation_schema,
    _model_pricing_schema,
    _model_profile_schema,
    _model_settings_schema,
    _run_settings_schema,
)
from ._settings_parsing import (
    RunSettingsValidationError,
    _format_chat_template_kwargs,
    _format_key_value_json_setting,
    _format_thinking_value,
    _is_blank,
    _merge_model_pricing,
    _merge_model_settings,
    _model_pricing_from_options,
    _model_profile_data_from_user_input,
    _model_setting_error,
    _model_settings_from_options,
    _normalise_run_setting,
    _normalise_run_settings,
    _parse_chat_template_kwargs,
    _parse_float_setting,
    _parse_int_setting,
    _parse_key_value_json_setting,
    _parse_model_pricing,
    _parse_model_setting_value,
    _parse_model_settings,
    _parse_non_negative_float_setting,
    _parse_non_negative_int_setting,
    _parse_positive_float_setting,
    _parse_positive_int_setting,
    _parse_thinking_setting,
    _pricing_storage_key,
    _store_model_pricing,
    _store_model_settings,
)
from .helpers import _flatten_section_data, _section_schema_key, _sorted_select_options
from .mcp_helpers import (
    _append_mcp_server_schema_fields,
    _format_mcp_headers,
    _mcp_server_data_from_user_input,
    _mcp_server_schema,
    _mcp_server_select_options,
    _mcp_tool_options,
    _mcp_tools_schema,
    _mcp_url_already_configured,
    _mcp_url_identity,
    _mcp_validation_placeholders,
    _normalise_mcp_server_selection,
    _selected_mcp_server_error,
)

_LOGGER = logging.getLogger(__name__)
