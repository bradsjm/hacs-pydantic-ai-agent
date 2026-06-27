from custom_components.pydantic_ai_agent.const import (
    CONF_API_KEY,
    CONF_LOGFIRE_TOKEN,
    CONF_MCP_HEADERS,
    CONF_MCP_SECRET_HEADER_KEYS,
    CONF_PROMPT,
    CONF_PROVIDER_HEADERS,
    CONF_PROVIDER_SECRET_HEADER_KEYS,
)
from custom_components.pydantic_ai_agent.runtime.header_metadata import REDACTED
from custom_components.pydantic_ai_agent.runtime.redaction import (
    TO_REDACT,
    redact_data,
    redaction_keys,
)
import pytest


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (CONF_API_KEY, "api-secret"),
        (CONF_LOGFIRE_TOKEN, "logfire-secret"),
        (CONF_PROMPT, "private instructions"),
        ("Authorization", "Bearer token"),
        ("password", "secret-password"),
        ("headers", {"X-Header": "secret"}),
    ],
)
def test_redact_data_redacts_top_level_sensitive_keys(key: str, value: object) -> None:
    assert redact_data({key: value, "name": "kept"}) == {
        key: REDACTED,
        "name": "kept",
    }


def test_redact_data_recurses_nested_mappings_and_sequences() -> None:
    data = {
        "outer": {
            "safe": "visible",
            "items": [
                {"token": "hidden"},
                ("plain", {"client_secret": "hidden-too"}),
            ],
        },
        "bytes": b"not-a-sequence-for-redaction",
    }

    assert redact_data(data) == {
        "outer": {
            "safe": "visible",
            "items": [
                {"token": REDACTED},
                ["plain", {"client_secret": REDACTED}],
            ],
        },
        "bytes": b"not-a-sequence-for-redaction",
    }


def test_redact_data_preserves_non_sensitive_values_and_string_scalars() -> None:
    data = {
        "message": "authorization token appears inside value but key is safe",
        "count": 3,
        "enabled": True,
        "none": None,
    }

    assert redact_data(data) == data


def test_redact_data_uses_extra_sensitive_keys() -> None:
    assert redact_data(
        {"custom_secret": "hidden", "safe": "visible"},
        extra_sensitive_keys=("custom_secret",),
    ) == {"custom_secret": REDACTED, "safe": "visible"}


def test_redaction_keys_include_defaults_and_extras() -> None:
    keys = redaction_keys(("custom_secret",))

    assert keys >= TO_REDACT
    assert "custom_secret" in keys


def test_redact_data_masks_provider_header_container_secret_values() -> None:
    assert redact_data(
        {
            CONF_PROVIDER_HEADERS: {
                "Authorization": "provider-token",
                "X-Trace": "trace-id",
            },
            CONF_PROVIDER_SECRET_HEADER_KEYS: ["authorization"],
        }
    ) == {
        CONF_PROVIDER_HEADERS: {
            "Authorization": REDACTED,
            "X-Trace": "trace-id",
        },
        CONF_PROVIDER_SECRET_HEADER_KEYS: ["authorization"],
    }


def test_redact_data_masks_mcp_header_container_secret_values() -> None:
    assert redact_data(
        {
            CONF_MCP_HEADERS: {
                "X-API-Key": "mcp-token",
                "X-Trace": "trace-id",
            },
            CONF_MCP_SECRET_HEADER_KEYS: ["x-api-key"],
        }
    ) == {
        CONF_MCP_HEADERS: {
            "X-API-Key": REDACTED,
            "X-Trace": "trace-id",
        },
        CONF_MCP_SECRET_HEADER_KEYS: ["x-api-key"],
    }


def test_redact_data_generic_key_matching_is_case_sensitive() -> None:
    assert redact_data(
        {
            "Authorization": "hidden",
            "AUTHORIZATION": "visible",
            "authorization": "also-hidden",
        }
    ) == {
        "Authorization": REDACTED,
        "AUTHORIZATION": "visible",
        "authorization": REDACTED,
    }
