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
from custom_components.pydantic_ai_agent.runtime.header_metadata import (
    HEADER_VALUE_REDACTED,
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


def _provider_subentry_data_with_structured_fields() -> dict[str, object]:
    """Return a provider subentry with stored structured connection fields."""
    provider_data = _provider_subentry_data()
    provider_config = cast(dict[str, Any], provider_data["data"])
    provider_config[CONF_PROVIDER_HEADERS] = {"Authorization": "Bearer old"}
    provider_config[CONF_PROVIDER_SECRET_HEADER_KEYS] = ["Authorization"]
    provider_config[CONF_PROVIDER_EXTRA_BODY] = {"service_tier": "default"}
    return provider_data


async def _start_provider_edit_connection(
    hass: HomeAssistant,
    entry_id: str,
    subentry_id: str,
) -> dict[str, Any]:
    """Open the provider edit-connection form for a reconfigure flow."""
    result = await _subentry_init_result(
        hass,
        (entry_id, SUBENTRY_TYPE_PROVIDER),
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "subentry_id": subentry_id,
        },
    )
    return await _subentry_configure_result(
        hass, result["flow_id"], {"next_step_id": "edit_connection"}
    )


def _provider_edit_payload(
    advanced_options: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return a provider edit-connection submission."""
    payload: dict[str, object] = {
        CONF_NAME: "OpenAI-compatible",
        CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        CONF_API_KEY: "sk-updated",
    }
    if advanced_options is not None:
        payload["advanced_options"] = advanced_options
    return payload


async def test_provider_edit_connection_structured_fields_round_trip(
    hass: HomeAssistant,
) -> None:
    """Test provider structured rows render and save as stored dict values."""
    provider_data = _provider_subentry_data_with_structured_fields()
    entry = await _loaded_workspace_entry(hass, (provider_data,))
    provider_subentry = next(iter(entry.subentries.values()))

    result = await _start_provider_edit_connection(
        hass, entry.entry_id, provider_subentry.subentry_id
    )

    section_default = _serialized_section_default(
        result["data_schema"], "advanced_options"
    )
    assert section_default[CONF_PROVIDER_HEADERS] == [
        {
            CONF_KEY_VALUE_KEY: "Authorization",
            CONF_KEY_VALUE_VALUE: HEADER_VALUE_REDACTED,
            CONF_KEY_VALUE_IS_SECRET: True,
        }
    ]
    assert section_default[CONF_PROVIDER_EXTRA_BODY] == [
        {CONF_KEY_VALUE_KEY: "service_tier", CONF_KEY_VALUE_JSON_VALUE: '"default"'}
    ]
    assert _section_field_default(
        result["data_schema"], "advanced_options", CONF_PROVIDER_HEADERS
    ) == [
        {
            CONF_KEY_VALUE_KEY: "Authorization",
            CONF_KEY_VALUE_VALUE: HEADER_VALUE_REDACTED,
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
            CONF_KEY_VALUE_VALUE: HEADER_VALUE_REDACTED,
            CONF_KEY_VALUE_IS_SECRET: True,
        },
        {
            CONF_KEY_VALUE_KEY: "X-Trace",
            CONF_KEY_VALUE_VALUE: "trace-1",
            CONF_KEY_VALUE_IS_SECRET: False,
        },
    ]


async def test_provider_edit_connection_preserves_structured_fields_when_omitted(
    hass: HomeAssistant,
) -> None:
    """Test editing only main fields keeps untouched provider structured fields."""
    entry = await _loaded_workspace_entry(
        hass, (_provider_subentry_data_with_structured_fields(),)
    )
    provider_subentry = next(iter(entry.subentries.values()))
    result = await _start_provider_edit_connection(
        hass, entry.entry_id, provider_subentry.subentry_id
    )

    result = await _subentry_configure_result(
        hass,
        result["flow_id"],
        _provider_edit_payload(),
    )

    assert result["type"] is FlowResultType.ABORT
    updated_subentry = entry.subentries[provider_subentry.subentry_id]
    assert updated_subentry.data[CONF_PROVIDER_HEADERS] == {
        "Authorization": "Bearer old"
    }
    assert updated_subentry.data[CONF_PROVIDER_SECRET_HEADER_KEYS] == ["Authorization"]
    assert updated_subentry.data[CONF_PROVIDER_EXTRA_BODY] == {
        "service_tier": "default"
    }


async def test_provider_edit_connection_redacted_secret_keeps_stored_value(
    hass: HomeAssistant,
) -> None:
    """Test unchanged redacted secret header values keep stored secrets."""
    entry = await _loaded_workspace_entry(
        hass, (_provider_subentry_data_with_structured_fields(),)
    )
    provider_subentry = next(iter(entry.subentries.values()))
    result = await _start_provider_edit_connection(
        hass, entry.entry_id, provider_subentry.subentry_id
    )

    result = await _subentry_configure_result(
        hass,
        result["flow_id"],
        _provider_edit_payload(
            {
                CONF_PROVIDER_HEADERS: [
                    {
                        CONF_KEY_VALUE_KEY: "Authorization",
                        CONF_KEY_VALUE_VALUE: HEADER_VALUE_REDACTED,
                        CONF_KEY_VALUE_IS_SECRET: True,
                    }
                ]
            }
        ),
    )

    assert result["type"] is FlowResultType.ABORT
    updated_subentry = entry.subentries[provider_subentry.subentry_id]
    assert updated_subentry.data[CONF_PROVIDER_HEADERS] == {
        "Authorization": "Bearer old"
    }
    assert updated_subentry.data[CONF_PROVIDER_SECRET_HEADER_KEYS] == ["Authorization"]


async def test_provider_edit_connection_empty_advanced_preserves_structured_fields(
    hass: HomeAssistant,
) -> None:
    """Test an empty advanced section keeps provider structured fields untouched."""
    entry = await _loaded_workspace_entry(
        hass, (_provider_subentry_data_with_structured_fields(),)
    )
    provider_subentry = next(iter(entry.subentries.values()))
    result = await _start_provider_edit_connection(
        hass, entry.entry_id, provider_subentry.subentry_id
    )

    result = await _subentry_configure_result(
        hass,
        result["flow_id"],
        _provider_edit_payload({}),
    )

    assert result["type"] is FlowResultType.ABORT
    updated_subentry = entry.subentries[provider_subentry.subentry_id]
    assert updated_subentry.data[CONF_PROVIDER_HEADERS] == {
        "Authorization": "Bearer old"
    }
    assert updated_subentry.data[CONF_PROVIDER_SECRET_HEADER_KEYS] == ["Authorization"]
    assert updated_subentry.data[CONF_PROVIDER_EXTRA_BODY] == {
        "service_tier": "default"
    }


async def test_provider_edit_connection_clearing_headers_removes_secret_metadata(
    hass: HomeAssistant,
) -> None:
    """Test clearing provider header rows removes stored headers and secrets."""
    entry = await _loaded_workspace_entry(
        hass, (_provider_subentry_data_with_structured_fields(),)
    )
    provider_subentry = next(iter(entry.subentries.values()))
    result = await _start_provider_edit_connection(
        hass, entry.entry_id, provider_subentry.subentry_id
    )

    result = await _subentry_configure_result(
        hass,
        result["flow_id"],
        _provider_edit_payload({CONF_PROVIDER_HEADERS: []}),
    )

    assert result["type"] is FlowResultType.ABORT
    updated_subentry = entry.subentries[provider_subentry.subentry_id]
    assert CONF_PROVIDER_HEADERS not in updated_subentry.data
    assert CONF_PROVIDER_SECRET_HEADER_KEYS not in updated_subentry.data
    assert updated_subentry.data[CONF_PROVIDER_EXTRA_BODY] == {
        "service_tier": "default"
    }


async def test_provider_edit_connection_clearing_extra_body_preserves_headers(
    hass: HomeAssistant,
) -> None:
    """Test clearing provider extra body leaves untouched headers in place."""
    entry = await _loaded_workspace_entry(
        hass, (_provider_subentry_data_with_structured_fields(),)
    )
    provider_subentry = next(iter(entry.subentries.values()))
    result = await _start_provider_edit_connection(
        hass, entry.entry_id, provider_subentry.subentry_id
    )

    result = await _subentry_configure_result(
        hass,
        result["flow_id"],
        _provider_edit_payload({CONF_PROVIDER_EXTRA_BODY: []}),
    )

    assert result["type"] is FlowResultType.ABORT
    updated_subentry = entry.subentries[provider_subentry.subentry_id]
    assert updated_subentry.data[CONF_PROVIDER_HEADERS] == {
        "Authorization": "Bearer old"
    }
    assert updated_subentry.data[CONF_PROVIDER_SECRET_HEADER_KEYS] == ["Authorization"]
    assert CONF_PROVIDER_EXTRA_BODY not in updated_subentry.data


async def test_provider_edit_connection_non_secret_header_clears_secret_keys(
    hass: HomeAssistant,
) -> None:
    """Test provider header edits recompute secret metadata from submitted rows."""
    entry = await _loaded_workspace_entry(
        hass, (_provider_subentry_data_with_structured_fields(),)
    )
    provider_subentry = next(iter(entry.subentries.values()))
    result = await _start_provider_edit_connection(
        hass, entry.entry_id, provider_subentry.subentry_id
    )

    result = await _subentry_configure_result(
        hass,
        result["flow_id"],
        _provider_edit_payload(
            {
                CONF_PROVIDER_HEADERS: [
                    {
                        CONF_KEY_VALUE_KEY: "X-Custom",
                        CONF_KEY_VALUE_VALUE: "non-secret-value",
                        CONF_KEY_VALUE_IS_SECRET: False,
                    }
                ]
            }
        ),
    )

    assert result["type"] is FlowResultType.ABORT
    updated_subentry = entry.subentries[provider_subentry.subentry_id]
    assert updated_subentry.data[CONF_PROVIDER_HEADERS] == {
        "X-Custom": "non-secret-value"
    }
    assert updated_subentry.data[CONF_PROVIDER_SECRET_HEADER_KEYS] == []
    assert updated_subentry.data[CONF_PROVIDER_EXTRA_BODY] == {
        "service_tier": "default"
    }
