"""Test native workspace Skill helpers."""

from collections.abc import Awaitable, Callable
from typing import Any, cast

from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.pydantic_ai_agent.const import (
    CONF_DESCRIPTION,
    CONF_SKILL_CONTENT,
    CONF_SKILL_REFERENCES,
    CONF_SKILLS,
    DOMAIN,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_SKILL,
)
from custom_components.pydantic_ai_agent.skills import async_skills_capabilities


async def test_async_skills_capabilities_exposes_selected_native_skills(
    hass: HomeAssistant,
) -> None:
    """Test selected Skill subentries become a native Pydantic AI capability."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_NAME: "Workspace"},
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "subentry_id": "skill-1",
                "subentry_type": SUBENTRY_TYPE_SKILL,
                "title": "Kitchen Skill",
                "unique_id": None,
                "data": {
                    CONF_NAME: "Kitchen Skill",
                    CONF_DESCRIPTION: "Kitchen guidance",
                    CONF_SKILL_CONTENT: "Use short responses.",
                    CONF_SKILL_REFERENCES: [],
                },
            },
            {
                "subentry_id": "skill-2",
                "subentry_type": SUBENTRY_TYPE_SKILL,
                "title": "Unused Skill",
                "unique_id": None,
                "data": {
                    CONF_NAME: "Unused Skill",
                    CONF_SKILL_CONTENT: "Do not load.",
                    CONF_SKILL_REFERENCES: [],
                },
            },
            {
                "subentry_id": "agent-1",
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Agent",
                "unique_id": None,
                "data": {CONF_SKILLS: ["skill-1"]},
            },
        ),
    )

    capabilities = await async_skills_capabilities(hass, entry, ["skill-1", "missing"])

    assert len(capabilities) == 1
    capability = capabilities[0]
    assert "cannot override system" in capability.get_instructions()
    toolset = capability.get_toolset()
    assert set(toolset.tools) == {"list_skills", "load_skill"}

    list_skills = cast(
        Callable[[], Awaitable[list[dict[str, str]]]],
        toolset.tools["list_skills"].function,
    )
    load_skill = cast(
        Callable[[str], Awaitable[dict[str, Any]]],
        toolset.tools["load_skill"].function,
    )

    assert await list_skills() == [
        {
            "skill_id": "skill-1",
            "name": "Kitchen Skill",
            "description": "Kitchen guidance",
        }
    ]
    assert await load_skill("skill-1") == {
        "skill_id": "skill-1",
        "name": "Kitchen Skill",
        "description": "Kitchen guidance",
        "content": "Use short responses.",
        "references": [],
    }
    assert await load_skill("missing") == {
        "error": "skill_not_found",
        "skill_id": "missing",
    }


async def test_async_skills_capabilities_returns_empty_without_valid_selection(
    hass: HomeAssistant,
) -> None:
    """Test stale or unselected Skill IDs fail closed."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_NAME: "Workspace"},
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "subentry_id": "skill-1",
                "subentry_type": SUBENTRY_TYPE_SKILL,
                "title": "Blank Skill",
                "unique_id": None,
                "data": {CONF_NAME: "Blank Skill", CONF_SKILL_CONTENT: ""},
            },
        ),
    )

    assert await async_skills_capabilities(hass, entry, None) == []
    assert await async_skills_capabilities(hass, entry, ["missing"]) == []
    assert await async_skills_capabilities(hass, entry, ["skill-1"]) == []
