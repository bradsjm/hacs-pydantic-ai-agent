"""Test Pydantic AI skills helpers."""

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from homeassistant import config_entries
from homeassistant.core import HomeAssistant

from custom_components.pydantic_ai_agent.const import (
    CONF_ENABLE_SKILL_SCRIPT_EXECUTION,
    CONF_SKILLS_FOLDER,
    DOMAIN,
)
from custom_components.pydantic_ai_agent.skills import (
    AvailableSkill,
    _build_skills_capabilities,
    _discover_available_skills,
    _discover_skills,
    selected_available_skill_names,
    skills_folder_path,
)


def _skill(name: str, *, scripts: object | None = None) -> SimpleNamespace:
    """Return a fake discovered skill."""
    return SimpleNamespace(name=name, description=f"{name} description", scripts=scripts)


def test_skills_folder_path_stays_under_config_skills(hass: HomeAssistant) -> None:
    """Test configured skills folders are constrained to /config/skills."""
    skills_root = Path(hass.config.path("skills")).resolve()

    assert skills_folder_path(hass, None) == skills_root
    assert skills_folder_path(hass, "/config/skills/custom") == skills_root / "custom"
    assert skills_folder_path(hass, "skills/custom") == skills_root / "custom"

    for invalid in ["/config", "/tmp/skills", ".", "../../skills"]:
        with pytest.raises(ValueError, match="inside /config/skills"):
            skills_folder_path(hass, invalid)


def test_selected_available_skill_names_filters_stale_values() -> None:
    """Test configured skills are kept only when still discoverable."""
    available = [
        AvailableSkill(name="alpha", description="Alpha", has_scripts=False),
        AvailableSkill(name="beta", description="Beta", has_scripts=False),
    ]

    assert selected_available_skill_names("alpha", available) == ["alpha"]
    assert selected_available_skill_names("stale", available) == []
    assert selected_available_skill_names(["beta", "stale", "alpha"], available) == [
        "beta",
        "alpha",
    ]
    assert selected_available_skill_names(None, available) == []
    assert selected_available_skill_names(123, available) == []


def test_discover_available_skills_hides_script_skills_until_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test selectable skills exclude script-capable skills by default."""
    monkeypatch.setattr(
        "custom_components.pydantic_ai_agent.skills._discover_skills",
        lambda _folder: [_skill("plain"), _skill("scripted", scripts={"run": {}})],
    )

    assert _discover_available_skills(tmp_path, enable_scripts=False) == [
        AvailableSkill(
            name="plain", description="plain description", has_scripts=False
        )
    ]
    assert [skill.name for skill in _discover_available_skills(tmp_path, True)] == [
        "plain",
        "scripted",
    ]


def test_build_skills_capabilities_filters_selection_and_script_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test runtime skills capability uses selected safe skills only."""

    class SkillsCapability:
        def __init__(self, *, skills: list[object], exclude_tools: set[str] | None) -> None:
            self.skills = skills
            self.exclude_tools = exclude_tools

    monkeypatch.setitem(
        sys.modules,
        "pydantic_ai_skills",
        SimpleNamespace(SkillsCapability=SkillsCapability),
    )
    monkeypatch.setattr(
        "custom_components.pydantic_ai_agent.skills._discover_skills",
        lambda _folder: [_skill("plain"), _skill("scripted", scripts={"run": {}})],
    )

    capabilities = _build_skills_capabilities(
        tmp_path, {"plain", "scripted"}, enable_scripts=False
    )
    assert len(capabilities) == 1
    assert [skill.name for skill in capabilities[0].skills] == ["plain"]
    assert capabilities[0].exclude_tools == {"run_skill_script"}

    capabilities = _build_skills_capabilities(
        tmp_path, {"scripted"}, enable_scripts=True
    )
    assert [skill.name for skill in capabilities[0].skills] == ["scripted"]
    assert capabilities[0].exclude_tools is None


def test_discover_skills_falls_back_to_valid_child_folders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test invalid top-level discovery falls back to per-folder discovery."""
    valid = tmp_path / "valid"
    invalid = tmp_path / "invalid"
    valid.mkdir()
    invalid.mkdir()
    calls: list[Path] = []

    def discover_skills(folder: Path, *, validate: bool) -> list[SimpleNamespace]:
        assert validate is False
        calls.append(folder)
        if folder == tmp_path or folder == invalid:
            raise ValueError("bad skill")
        return [_skill("alpha"), _skill("alpha")]

    monkeypatch.setitem(
        sys.modules,
        "pydantic_ai_skills",
        SimpleNamespace(discover_skills=discover_skills),
    )

    assert [skill.name for skill in _discover_skills(tmp_path)] == ["alpha"]
    assert calls[0] == tmp_path
    assert set(calls[1:]) == {valid, invalid}


async def test_async_skills_helpers_return_empty_for_invalid_folder(
    hass: HomeAssistant,
) -> None:
    """Test invalid stored skills folders fail closed."""
    from custom_components.pydantic_ai_agent.skills import (
        async_available_skills,
        async_skills_capabilities,
    )

    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Hosted OpenAI",
        data={
            CONF_SKILLS_FOLDER: "/tmp/skills",
            CONF_ENABLE_SKILL_SCRIPT_EXECUTION: True,
        },
        source=config_entries.SOURCE_USER,
        options={},
        unique_id=None,
    )

    assert await async_available_skills(hass, entry.data) == []
    assert await async_skills_capabilities(hass, entry, ["alpha"]) == []
