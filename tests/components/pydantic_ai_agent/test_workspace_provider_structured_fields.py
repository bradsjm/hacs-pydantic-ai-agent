"""Tests for structured provider connection fields."""

from typing import Any, cast

from custom_components.pydantic_ai_agent.const import (
    CONF_API_KEY,
    CONF_KEY_VALUE_IS_SECRET,
    CONF_KEY_VALUE_JSON_VALUE,
    CONF_KEY_VALUE_KEY,
    CONF_KEY_VALUE_VALUE,
    CONF_NAME,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_MODE,
    CONF_PROVIDER_SECRET_HEADER_KEYS,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_PROVIDER,
)
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from tests.components.pydantic_ai_agent.support.schemas import (
    serialized_section_default as _serialized_section_default,
)
from tests.components.pydantic_ai_agent.test_workspace_provider_reconfigure import (
    _loaded_workspace_entry,
    _provider_subentry_data,
    _subentry_configure_result,
    _subentry_init_result,
)


def _section_field_default(
    data_schema: Any, section_name: str, field_name: str
) -> object:
    """Return a nested section field default from a flow schema."""
    for section_key, section_value in data_schema.schema.items():
        if section_key.schema != section_name:
            continue
        for field_key in section_value.schema.schema:
            if field_key.schema == field_name:
                return field_key.default()
    raise AssertionError(f"Section field {section_name}.{field_name} not found")


async def test_provider_edit_connection_structured_fields_round_trip(
    hass: HomeAssistant,
) -> None:
    """Test provider structured rows render and save as stored dict values."""
    provider_data = _provider_subentry_data()
    provider_config = cast(dict[str, Any], provider_data["data"])
    provider_config[CONF_PROVIDER_HEADERS] = {"Authorization": "Bearer old"}
    provider_config[CONF_PROVIDER_SECRET_HEADER_KEYS] = ["Authorization"]
    provider_config[CONF_PROVIDER_EXTRA_BODY] = {"service_tier": "default"}
    entry = await _loaded_workspace_entry(hass, (provider_data,))
    provider_subentry = next(iter(entry.subentries.values()))

    result = await _subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": provider_subentry.subentry_id,
        },
    )
    result = await _subentry_configure_result(
        hass, result["flow_id"], {"next_step_id": "edit_connection"}
    )

    assert _serialized_section_default(result["data_schema"], "advanced_options") == {}
    assert _section_field_default(
        result["data_schema"], "advanced_options", CONF_PROVIDER_HEADERS
    ) == [
        {
            CONF_KEY_VALUE_KEY: "Authorization",
            CONF_KEY_VALUE_VALUE: "Bearer old",
            CONF_KEY_VALUE_IS_SECRET: True,
        }
    ]
    assert _section_field_default(
        result["data_schema"], "advanced_options", CONF_PROVIDER_EXTRA_BODY
    ) == [{CONF_KEY_VALUE_KEY: "service_tier", CONF_KEY_VALUE_JSON_VALUE: '"default"'}]

    result = await _subentry_configure_result(
        hass,
        result["flow_id"],
        {
            CONF_NAME: "OpenAI-compatible",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "sk-updated",
            "advanced_options": {
                CONF_PROVIDER_HEADERS: [
                    {
                        CONF_KEY_VALUE_KEY: "Authorization",
                        CONF_KEY_VALUE_VALUE: "Bearer new",
                        CONF_KEY_VALUE_IS_SECRET: True,
                    },
                    {
                        CONF_KEY_VALUE_KEY: "X-Trace",
                        CONF_KEY_VALUE_VALUE: "trace-1",
                        CONF_KEY_VALUE_IS_SECRET: False,
                    },
                ],
                CONF_PROVIDER_EXTRA_BODY: [
                    {
                        CONF_KEY_VALUE_KEY: "service_tier",
                        CONF_KEY_VALUE_JSON_VALUE: '"flex"',
                    }
                ],
            },
        },
    )

    assert result["type"] is FlowResultType.ABORT
    updated_subentry = entry.subentries[provider_subentry.subentry_id]
    assert updated_subentry.data[CONF_PROVIDER_HEADERS] == {
        "Authorization": "Bearer new",
        "X-Trace": "trace-1",
    }
    assert updated_subentry.data[CONF_PROVIDER_SECRET_HEADER_KEYS] == ["Authorization"]
    assert updated_subentry.data[CONF_PROVIDER_EXTRA_BODY] == {"service_tier": "flex"}

    reopened_result = await _subentry_init_result(
        hass,
        (entry.entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": provider_subentry.subentry_id,
        },
    )
    reopened_result = await _subentry_configure_result(
        hass, reopened_result["flow_id"], {"next_step_id": "edit_connection"}
    )

    assert _section_field_default(
        reopened_result["data_schema"], "advanced_options", CONF_PROVIDER_HEADERS
    ) == [
        {
            CONF_KEY_VALUE_KEY: "Authorization",
            CONF_KEY_VALUE_VALUE: "Bearer new",
            CONF_KEY_VALUE_IS_SECRET: True,
        },
        {
            CONF_KEY_VALUE_KEY: "X-Trace",
            CONF_KEY_VALUE_VALUE: "trace-1",
            CONF_KEY_VALUE_IS_SECRET: False,
        },
    ]
