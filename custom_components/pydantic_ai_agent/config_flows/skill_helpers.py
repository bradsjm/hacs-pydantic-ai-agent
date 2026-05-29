"""Native Skill config-flow helpers."""

from collections.abc import Iterable, Mapping
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
)
from homeassistant.helpers.typing import VolDictType

from ..const import (
    CONF_DESCRIPTION,
    CONF_SKILL_CONTENT,
    CONF_SKILL_REFERENCES,
    CONF_SKILLS,
    DEFAULT_SKILL_NAME,
    SUBENTRY_TYPE_SKILL,
)
from .helpers import _sorted_select_options

_SKILL_NAME_MAX_LENGTH = 80
_SKILL_DESCRIPTION_MAX_LENGTH = 500
_SKILL_CONTENT_MAX_LENGTH = 20000


class SkillDataValidationError(ValueError):
    """Error raised when native workspace Skill form data is invalid."""

    def __init__(self, errors: dict[str, str]) -> None:
        """Initialize the error with Home Assistant form error keys."""
        super().__init__("invalid_skill")
        self.errors = errors


def _normalise_selected_skill_ids(raw_skill_ids: object) -> list[str]:
    """Return selected Skill subentry IDs in storage order without duplicates."""
    if isinstance(raw_skill_ids, str):
        raw_values: Iterable[object] = (raw_skill_ids,)
    elif isinstance(raw_skill_ids, Iterable):
        raw_values = raw_skill_ids
    else:
        return []
    skill_ids: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        if not isinstance(raw_value, str):
            continue
        skill_id = raw_value.strip()
        if not skill_id or skill_id in seen:
            continue
        seen.add(skill_id)
        skill_ids.append(skill_id)
    return skill_ids


def _skill_select_options(
    entry: ConfigEntry | None, selected_skill_ids: object = None
) -> list[SelectOptionDict]:
    """Return workspace Skill subentries as select options."""
    if entry is None:
        return []
    options = _sorted_select_options(
        [
            SelectOptionDict(label=subentry.title, value=subentry.subentry_id)
            for subentry in entry.subentries.values()
            if subentry.subentry_type == SUBENTRY_TYPE_SKILL
        ]
    )
    configured_ids = {str(option["value"]) for option in options if "value" in option}
    for skill_id in _normalise_selected_skill_ids(selected_skill_ids):
        if skill_id not in configured_ids:
            options.append(
                SelectOptionDict(label=f"Unavailable / {skill_id}", value=skill_id)
            )
    return options


def _append_skill_schema_fields(
    schema: VolDictType,
    options: Mapping[str, Any],
    entry: ConfigEntry | None,
) -> None:
    """Append per-agent workspace Skill selection controls to a subentry form."""
    skill_options = _skill_select_options(entry, options.get(CONF_SKILLS))
    if not skill_options:
        return
    schema[
        vol.Optional(
            CONF_SKILLS,
            default=_normalise_selected_skill_ids(options.get(CONF_SKILLS)),
        )
    ] = SelectSelector(SelectSelectorConfig(options=skill_options, multiple=True))


def _selected_skill_error(entry: ConfigEntry, data: Mapping[str, Any]) -> str | None:
    """Return a form error for selected Skills that no longer exist."""
    for skill_id in _normalise_selected_skill_ids(data.get(CONF_SKILLS)):
        subentry = entry.subentries.get(skill_id)
        if subentry is None or subentry.subentry_type != SUBENTRY_TYPE_SKILL:
            return "skill_not_found"
    return None


def _normalise_skill_selection(data: dict[str, Any]) -> None:
    """Store only current native Skill subentry IDs on agents and tasks."""
    data.pop("enable_skills", None)
    data.pop("skills_folder", None)
    data.pop("enable_skill_script_execution", None)
    skill_ids = _normalise_selected_skill_ids(data.get(CONF_SKILLS))
    if skill_ids:
        data[CONF_SKILLS] = skill_ids
    else:
        data.pop(CONF_SKILLS, None)


def _skill_schema(options: Mapping[str, Any] | None = None) -> vol.Schema:
    """Return the native workspace Skill subentry schema."""
    options = dict(options or {})
    return vol.Schema(
        {
            vol.Required(
                CONF_NAME,
                default=options.get(CONF_NAME, DEFAULT_SKILL_NAME),
            ): TextSelector(TextSelectorConfig()),
            vol.Optional(
                CONF_DESCRIPTION,
                default=options.get(CONF_DESCRIPTION, ""),
            ): TextSelector(TextSelectorConfig(multiline=True)),
            vol.Required(
                CONF_SKILL_CONTENT,
                default=options.get(CONF_SKILL_CONTENT, ""),
            ): TemplateSelector(),
        }
    )


def _skill_data_from_user_input(user_input: Mapping[str, Any]) -> dict[str, Any]:
    """Return normalized native workspace Skill data."""
    name = str(user_input[CONF_NAME]).strip()
    description = str(user_input.get(CONF_DESCRIPTION, "")).strip()
    content = str(user_input.get(CONF_SKILL_CONTENT, "")).strip()
    errors: dict[str, str] = {}
    if not name:
        errors[CONF_NAME] = "required"
    elif len(name) > _SKILL_NAME_MAX_LENGTH:
        errors[CONF_NAME] = "string_too_long"
    if len(description) > _SKILL_DESCRIPTION_MAX_LENGTH:
        errors[CONF_DESCRIPTION] = "string_too_long"
    if not content:
        errors[CONF_SKILL_CONTENT] = "required"
    elif len(content) > _SKILL_CONTENT_MAX_LENGTH:
        errors[CONF_SKILL_CONTENT] = "string_too_long"
    if errors:
        raise SkillDataValidationError(errors)
    data: dict[str, Any] = {
        CONF_NAME: name,
        CONF_SKILL_CONTENT: content,
        CONF_SKILL_REFERENCES: [],
    }
    if description:
        data[CONF_DESCRIPTION] = description
    return data
