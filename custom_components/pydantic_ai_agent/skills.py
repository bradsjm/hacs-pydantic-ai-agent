"""Pydantic AI Skills discovery and runtime helpers."""

from dataclasses import dataclass
import logging
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any

from homeassistant.core import HomeAssistant

from .const import (
    CONF_ENABLE_SKILL_SCRIPT_EXECUTION,
    CONF_ENABLE_SKILLS,
    CONF_SKILLS_FOLDER,
    DEFAULT_SKILLS_FOLDER,
)

_LOGGER = logging.getLogger(__name__)
_RUN_SKILL_SCRIPT_TOOL = "run_skill_script"


@dataclass(frozen=True, kw_only=True)
class AvailableSkill:
    """Selectable skill metadata discovered from the provider skills folder."""

    name: str
    description: str
    has_scripts: bool


def skills_folder_path(hass: HomeAssistant, folder: object | None) -> Path:
    """Return the filesystem path for a configured skills folder."""
    configured = str(folder or DEFAULT_SKILLS_FOLDER).strip() or DEFAULT_SKILLS_FOLDER
    skills_root = Path(hass.config.path("skills")).resolve()
    if configured == DEFAULT_SKILLS_FOLDER:
        path = skills_root
    elif configured.startswith(f"{DEFAULT_SKILLS_FOLDER}/"):
        path = Path(hass.config.path(configured.removeprefix("/config/")))
    else:
        raw_path = Path(configured)
        if raw_path.is_absolute():
            path = raw_path
        else:
            path = Path(hass.config.path(configured))
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(skills_root)
    except ValueError as err:
        raise ValueError("Skills folder must stay inside /config/skills") from err
    return resolved


async def async_available_skills(
    hass: HomeAssistant, entry_data: Mapping[str, Any]
) -> list[AvailableSkill]:
    """Return skills currently available from the provider skills folder."""
    try:
        folder = skills_folder_path(hass, entry_data.get(CONF_SKILLS_FOLDER))
    except ValueError as err:
        _LOGGER.warning("Ignoring invalid Pydantic AI skills folder: %s", err)
        return []
    enable_scripts = bool(entry_data.get(CONF_ENABLE_SKILL_SCRIPT_EXECUTION, False))
    return await hass.async_add_executor_job(
        _discover_available_skills, folder, enable_scripts
    )


async def async_skills_capabilities(
    hass: HomeAssistant,
    skill_settings: Mapping[str, Any],
    selected_skill_names: object,
) -> list[Any]:
    """Return a SkillsCapability for selected skills that still exist."""
    if not skill_settings.get(CONF_ENABLE_SKILLS, False):
        return []
    if not selected_skill_names:
        return []
    if isinstance(selected_skill_names, str):
        selected = {selected_skill_names}
    elif isinstance(selected_skill_names, Iterable):
        selected = {str(name) for name in selected_skill_names if str(name)}
    else:
        return []
    if not selected:
        return []

    try:
        folder = skills_folder_path(hass, skill_settings.get(CONF_SKILLS_FOLDER))
    except ValueError as err:
        _LOGGER.warning("Ignoring invalid Pydantic AI skills folder: %s", err)
        return []
    enable_scripts = bool(skill_settings.get(CONF_ENABLE_SKILL_SCRIPT_EXECUTION, False))
    return await hass.async_add_executor_job(
        _build_skills_capabilities,
        folder,
        selected,
        enable_scripts,
    )


def selected_available_skill_names(
    configured: object, available: list[AvailableSkill]
) -> list[str]:
    """Return configured skill names that are still available."""
    available_names = {skill.name for skill in available}
    if not configured:
        return []
    if isinstance(configured, str):
        return [configured] if configured in available_names else []
    if not isinstance(configured, Iterable):
        return []
    return [name for name in configured if name in available_names]


def _discover_available_skills(
    folder: Path, enable_scripts: bool
) -> list[AvailableSkill]:
    """Discover available skills in a worker thread."""
    return [
        AvailableSkill(
            name=skill.name,
            description=getattr(skill, "description", "") or skill.name,
            has_scripts=_skill_has_scripts(skill),
        )
        for skill in _discover_skills(folder)
        if enable_scripts or not _skill_has_scripts(skill)
    ]


def _build_skills_capabilities(
    folder: Path, selected: set[str], enable_scripts: bool
) -> list[Any]:
    """Build SkillsCapability objects in a worker thread."""
    try:
        from pydantic_ai_skills import SkillsCapability
    except ImportError as err:
        _LOGGER.warning("Pydantic AI skills support is unavailable: %s", err)
        return []

    skills = [
        skill
        for skill in _discover_skills(folder)
        if skill.name in selected and (enable_scripts or not _skill_has_scripts(skill))
    ]
    if not skills:
        return []
    exclude_tools = None if enable_scripts else {_RUN_SKILL_SCRIPT_TOOL}
    return [SkillsCapability(skills=skills, exclude_tools=exclude_tools)]


def _discover_skills(folder: Path) -> list[Any]:
    """Return discoverable skills, ignoring missing folders and invalid skills."""
    try:
        from pydantic_ai_skills import discover_skills
    except ImportError as err:
        _LOGGER.warning("Pydantic AI skills support is unavailable: %s", err)
        return []

    try:
        return list(discover_skills(folder, validate=False))
    except (OSError, ValueError) as err:
        _LOGGER.warning(
            "Failed to discover all Pydantic AI skills from %s: %s", folder, err
        )

    skills: list[Any] = []
    seen: set[str] = set()
    try:
        children = list(folder.iterdir())
    except OSError as err:
        _LOGGER.warning("Failed to list Pydantic AI skills folder %s: %s", folder, err)
        return []
    for child in children:
        if not child.is_dir():
            continue
        try:
            child_skills = discover_skills(child, validate=False)
        except (OSError, ValueError) as err:
            _LOGGER.warning("Ignoring invalid Pydantic AI skill at %s: %s", child, err)
            continue
        for skill in child_skills:
            if skill.name in seen:
                continue
            seen.add(skill.name)
            skills.append(skill)
    return skills


def _skill_has_scripts(skill: Any) -> bool:
    """Return if a skill declares script metadata."""
    scripts = getattr(skill, "scripts", None)
    if scripts is None:
        return False
    return bool(scripts)
