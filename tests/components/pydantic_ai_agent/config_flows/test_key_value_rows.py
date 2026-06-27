from custom_components.pydantic_ai_agent.config_flows._key_value_rows import (
    _format_key_value_json_rows,
    _format_key_value_text_rows,
    _parse_key_value_json_rows,
    _parse_key_value_text_rows,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_KEY_VALUE_JSON_VALUE,
    CONF_KEY_VALUE_KEY,
    CONF_KEY_VALUE_VALUE,
)
import pytest


def test_format_text_rows_sorts_mapping_keys_and_keeps_string_values() -> None:
    rows = _format_key_value_text_rows({"z": "last", "a": "first", "skip": 1})

    assert rows == [
        {CONF_KEY_VALUE_KEY: "a", CONF_KEY_VALUE_VALUE: "first"},
        {CONF_KEY_VALUE_KEY: "z", CONF_KEY_VALUE_VALUE: "last"},
    ]


def test_format_json_rows_sorts_mapping_keys_and_dumps_values() -> None:
    rows = _format_key_value_json_rows({"z": [2, 1], "a": {"b": True}})

    assert rows == [
        {CONF_KEY_VALUE_KEY: "a", CONF_KEY_VALUE_JSON_VALUE: '{"b": true}'},
        {CONF_KEY_VALUE_KEY: "z", CONF_KEY_VALUE_JSON_VALUE: "[2, 1]"},
    ]


def test_format_list_rows_preserves_valid_selector_rows() -> None:
    assert _format_key_value_text_rows(
        [
            {CONF_KEY_VALUE_KEY: "a", CONF_KEY_VALUE_VALUE: "one"},
            {CONF_KEY_VALUE_KEY: "b", CONF_KEY_VALUE_JSON_VALUE: "{}"},
            "skip",
        ]
    ) == [{CONF_KEY_VALUE_KEY: "a", CONF_KEY_VALUE_VALUE: "one"}]
    assert _format_key_value_json_rows(
        [
            {CONF_KEY_VALUE_KEY: "a", CONF_KEY_VALUE_JSON_VALUE: "{}"},
            {CONF_KEY_VALUE_KEY: "b", CONF_KEY_VALUE_VALUE: "one"},
            "skip",
        ]
    ) == [{CONF_KEY_VALUE_KEY: "a", CONF_KEY_VALUE_JSON_VALUE: "{}"}]


def test_parse_text_rows_skips_blank_rows_and_trims_keys() -> None:
    parsed = _parse_key_value_text_rows(
        [
            {CONF_KEY_VALUE_KEY: "  ", CONF_KEY_VALUE_VALUE: ""},
            {CONF_KEY_VALUE_KEY: " token ", CONF_KEY_VALUE_VALUE: "secret"},
        ]
    )

    assert parsed == {"token": "secret"}


def test_parse_json_rows_decodes_values_and_skips_blank_rows() -> None:
    parsed = _parse_key_value_json_rows(
        [
            {CONF_KEY_VALUE_KEY: "", CONF_KEY_VALUE_JSON_VALUE: None},
            {CONF_KEY_VALUE_KEY: "body", CONF_KEY_VALUE_JSON_VALUE: '{"flag": true}'},
        ]
    )

    assert parsed == {"body": {"flag": True}}


@pytest.mark.parametrize(
    ("parser", "rows", "reason"),
    [
        (
            _parse_key_value_text_rows,
            [
                {CONF_KEY_VALUE_KEY: "dup", CONF_KEY_VALUE_VALUE: "one"},
                {CONF_KEY_VALUE_KEY: "dup", CONF_KEY_VALUE_VALUE: "two"},
            ],
            "duplicate_key",
        ),
        (
            _parse_key_value_json_rows,
            [
                {CONF_KEY_VALUE_KEY: "body", CONF_KEY_VALUE_JSON_VALUE: "{"},
            ],
            "invalid_json",
        ),
        (
            _parse_key_value_text_rows,
            [{CONF_KEY_VALUE_KEY: "missing value"}],
            "invalid_key_value",
        ),
        (
            _parse_key_value_json_rows,
            ["not a row"],
            "invalid_key_value",
        ),
    ],
)
def test_parse_rows_raise_stable_reason_keys(parser, rows, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        parser(rows)


def test_parse_mapping_values() -> None:
    assert _parse_key_value_text_rows({"a": "b"}) == {"a": "b"}
    assert _parse_key_value_json_rows({"a": {"already": "parsed"}}) == {
        "a": {"already": "parsed"}
    }
