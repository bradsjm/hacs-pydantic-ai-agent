from custom_components.pydantic_ai_agent.config_flows import (
    _settings_parsing as parsing,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_CHAT_TEMPLATE_KWARG_KEY,
    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE,
    CONF_KEY_VALUE_JSON_VALUE,
    CONF_KEY_VALUE_KEY,
)
import pytest


@pytest.mark.parametrize(
    "parser",
    [
        parsing._parse_int_setting,
        parsing._parse_positive_int_setting,
        parsing._parse_non_negative_int_setting,
    ],
)
@pytest.mark.parametrize("value", [True, False, 1.25, "1.25", object()])
def test_integer_parsers_reject_bools_and_non_integral_values(parser, value) -> None:
    try:
        parser(value)
    except ValueError:
        return
    pytest.fail("expected ValueError")


@pytest.mark.parametrize(
    ("parser", "accepted", "rejected"),
    [
        (parsing._parse_positive_int_setting, "1", "0"),
        (parsing._parse_non_negative_int_setting, "0", "-1"),
        (parsing._parse_positive_float_setting, "0.5", "0"),
        (parsing._parse_non_negative_float_setting, "0", "-0.1"),
    ],
)
def test_positive_and_non_negative_parsers_enforce_bounds(
    parser, accepted: str, rejected: str
) -> None:
    assert parser(accepted) in {0, 0.5, 1}
    try:
        parser(rejected)
    except ValueError:
        return
    pytest.fail("expected ValueError")


def test_parse_key_value_json_setting_from_text() -> None:
    parsed = parsing._parse_key_value_json_setting(
        'beta: true\nsettings: {"nested": [1, 2]}'
    )

    assert parsed == {"beta": True, "settings": {"nested": [1, 2]}}


def test_parse_key_value_json_setting_from_list_rows() -> None:
    parsed = parsing._parse_key_value_json_setting(
        [{CONF_KEY_VALUE_KEY: "metadata", CONF_KEY_VALUE_JSON_VALUE: '{"ok": true}'}]
    )

    assert parsed == {"metadata": {"ok": True}}


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("missing-separator", "invalid_key_value"),
        ("body: {", "invalid_json"),
        ("dup: 1\ndup: 2", "duplicate_key"),
        ([], ""),
    ],
)
def test_parse_key_value_json_setting_errors(value, reason: str) -> None:
    if not reason:
        assert parsing._parse_key_value_json_setting(value) == {}
        return
    with pytest.raises(ValueError, match=reason):
        parsing._parse_key_value_json_setting(value)


@pytest.mark.parametrize(
    ("value", "parsed"),
    [("none", "none"), ("low", "low"), ("medium", "medium"), ("xhigh", "xhigh")],
)
def test_parse_thinking_setting_accepts_supported_values(value: str, parsed) -> None:
    assert parsing._parse_thinking_setting(value) == parsed


@pytest.mark.parametrize(
    "value", [True, "", "true", "false", "minimal", "yes", "TRUE", "auto"]
)
def test_parse_thinking_setting_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError, match="invalid thinking setting"):
        parsing._parse_thinking_setting(value)


def test_parse_templated_extra_body_accepts_json_serializable_template(hass) -> None:
    rows = parsing._parse_templated_extra_body(
        hass,
        [
            {
                CONF_CHAT_TEMPLATE_KWARG_KEY: "metadata.mode",
                CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ 1 + 1 }}",
            }
        ],
    )

    assert rows == [
        {
            CONF_CHAT_TEMPLATE_KWARG_KEY: "metadata.mode",
            CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "{{ 1 + 1 }}",
        }
    ]


def test_parse_templated_extra_body_rejects_duplicate_keys(hass) -> None:
    with pytest.raises(ValueError, match="duplicate_key"):
        parsing._parse_templated_extra_body(
            hass,
            [
                {
                    CONF_CHAT_TEMPLATE_KWARG_KEY: "metadata.mode",
                    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "1",
                },
                {
                    CONF_CHAT_TEMPLATE_KWARG_KEY: "metadata.mode",
                    CONF_CHAT_TEMPLATE_KWARG_VALUE_TEMPLATE: "2",
                },
            ],
        )
