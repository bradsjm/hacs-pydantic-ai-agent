from custom_components.pydantic_ai_agent.const import (
    CONF_KEY_VALUE_IS_SECRET,
    CONF_KEY_VALUE_KEY,
    CONF_KEY_VALUE_VALUE,
)
from custom_components.pydantic_ai_agent.runtime.header_metadata import (
    HEADER_VALUE_REDACTED,
    REDACTED,
    format_header_rows,
    mask_secret_header_values,
    normalize_secret_header_keys,
    parse_header_rows,
)
import pytest


def _row(key: str, value: str, is_secret: bool = False) -> dict[str, str | bool]:
    return {
        CONF_KEY_VALUE_KEY: key,
        CONF_KEY_VALUE_VALUE: value,
        CONF_KEY_VALUE_IS_SECRET: is_secret,
    }


def test_format_header_rows_formats_list_rows_and_skips_malformed_items() -> None:
    rows = [
        _row("X-First", "one"),
        {CONF_KEY_VALUE_KEY: "Authorization", CONF_KEY_VALUE_VALUE: "secret"},
        {
            CONF_KEY_VALUE_KEY: "X-Explicit",
            CONF_KEY_VALUE_VALUE: "shown",
            CONF_KEY_VALUE_IS_SECRET: False,
        },
        {CONF_KEY_VALUE_KEY: "X-Bad"},
        "not-a-row",
    ]

    assert format_header_rows(rows, ["authorization", "x-explicit"]) == [
        _row("X-First", "one"),
        _row("Authorization", HEADER_VALUE_REDACTED, True),
        _row("X-Explicit", "shown", False),
    ]


def test_format_header_rows_formats_mapping_rows_sorted_by_key() -> None:
    assert format_header_rows(
        {"X-Zeta": "z", "Authorization": "secret", "X-Alpha": "a", "X-Number": 1},
        ["authorization"],
    ) == [
        _row("Authorization", HEADER_VALUE_REDACTED, True),
        _row("X-Alpha", "a"),
        _row("X-Zeta", "z"),
    ]


@pytest.mark.parametrize("value", [None, "", object(), 12])
def test_format_header_rows_returns_empty_list_for_invalid_input(value: object) -> None:
    assert format_header_rows(value) == []


@pytest.mark.parametrize("value", [None, ""])
def test_parse_header_rows_returns_empty_for_blank_values(value: object) -> None:
    assert parse_header_rows(value) == ({}, [])


def test_parse_header_rows_accepts_mapping() -> None:
    assert parse_header_rows({"X-Header": "value"}) == ({"X-Header": "value"}, [])


def test_parse_header_rows_restores_redacted_mapping_values_from_previous_secret() -> (
    None
):
    assert parse_header_rows(
        {"Authorization": HEADER_VALUE_REDACTED, "X-Trace": "trace"},
        previous_headers={"authorization": "old-token"},
        previous_secret_header_keys=["Authorization"],
    ) == ({"Authorization": "old-token", "X-Trace": "trace"}, [])


def test_parse_header_rows_accepts_list_rows_and_tracks_secret_keys() -> None:
    rows = [
        _row(" X-Trimmed ", "value"),
        _row("Authorization", "secret", True),
        {CONF_KEY_VALUE_KEY: "", CONF_KEY_VALUE_VALUE: ""},
    ]

    assert parse_header_rows(rows) == (
        {"X-Trimmed": "value", "Authorization": "secret"},
        ["Authorization"],
    )


def test_parse_header_rows_restores_redacted_list_secret_from_previous_header() -> None:
    assert parse_header_rows(
        [_row("Authorization", HEADER_VALUE_REDACTED, True)],
        previous_headers={"Authorization": "old-token"},
        previous_secret_header_keys=["authorization"],
    ) == ({"Authorization": "old-token"}, ["Authorization"])


@pytest.mark.parametrize(
    ("value", "error_key"),
    [
        ({"X-Header": 1}, "invalid_key_value"),
        (object(), "invalid_key_value"),
        (["not-a-row"], "invalid_key_value"),
        ([{CONF_KEY_VALUE_KEY: "X", CONF_KEY_VALUE_VALUE: 1}], "invalid_key_value"),
        ([_row("X", "one"), _row("X", "two")], "duplicate_key"),
        (
            [{CONF_KEY_VALUE_KEY: "", CONF_KEY_VALUE_VALUE: "value"}],
            "invalid_key_value",
        ),
        (
            [_row("X", "value", True) | {CONF_KEY_VALUE_IS_SECRET: "yes"}],
            "invalid_key_value",
        ),
    ],
)
def test_parse_header_rows_rejects_invalid_input(value: object, error_key: str) -> None:
    with pytest.raises(ValueError, match=error_key):
        parse_header_rows(value)


def test_parse_header_rows_rejects_unmatched_redacted_secret_when_previous_exists() -> (
    None
):
    with pytest.raises(ValueError, match="invalid_key_value"):
        parse_header_rows(
            [_row("X-New", HEADER_VALUE_REDACTED, True)],
            previous_headers={"Authorization": "old-token"},
            previous_secret_header_keys=["Authorization"],
        )


def test_parse_header_rows_preserves_unmatched_redacted_secret_without_previous() -> (
    None
):
    assert parse_header_rows([_row("X-New", HEADER_VALUE_REDACTED, True)]) == (
        {"X-New": HEADER_VALUE_REDACTED},
        ["X-New"],
    )


def test_normalize_secret_header_keys_filters_current_mapping_and_preserves_order() -> (
    None
):
    assert normalize_secret_header_keys(
        {
            "X-First": "one",
            "X-Second": "two",
            "X-Third": 3,
            4: "ignored",
        },
        ["x-second", "x-first", "x-third"],
    ) == ["X-First", "X-Second"]


@pytest.mark.parametrize("headers", [None, [], "not-a-mapping"])
def test_normalize_secret_header_keys_returns_empty_for_non_mapping(
    headers: object,
) -> None:
    assert normalize_secret_header_keys(headers, ["authorization"]) == []


def test_mask_secret_header_values_masks_case_insensitively_and_drops_non_string_keys() -> (
    None
):
    assert mask_secret_header_values(
        {
            "Authorization": "token",
            "X-Trace": "trace",
            1: "ignored",
        },
        ["authorization"],
    ) == {"Authorization": REDACTED, "X-Trace": "trace"}


@pytest.mark.parametrize("headers", [None, [], "not-a-mapping"])
def test_mask_secret_header_values_returns_non_mapping_unchanged(
    headers: object,
) -> None:
    assert mask_secret_header_values(headers, ["authorization"]) is headers
