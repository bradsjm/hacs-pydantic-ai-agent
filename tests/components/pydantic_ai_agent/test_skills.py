"""Test native workspace Skill helpers."""

from collections.abc import Awaitable, Callable
from typing import Any, cast

from custom_components.pydantic_ai_agent.agent.skills import async_skills_capabilities
from custom_components.pydantic_ai_agent.const import (
    CONF_SKILLS,
)
from homeassistant.core import HomeAssistant
from tests.components.pydantic_ai_agent.support.builders import (
    conversation_subentry_data,
    skill_subentry_data,
    workspace_entry,
)


async def test_async_skills_capabilities_exposes_selected_native_skills(
    hass: HomeAssistant,
) -> None:
    """Test selected Skill subentries become a native Pydantic AI capability."""
    entry = workspace_entry(
        (
            skill_subentry_data(
                subentry_id="skill-1",
                title="Kitchen Skill",
                description="Kitchen guidance",
                content="Use short responses.",
                references=[],
            ),
            skill_subentry_data(
                subentry_id="skill-2",
                title="Unused Skill",
                content="Do not load.",
                references=[],
            ),
            conversation_subentry_data(
                "provider-1:profile-1",
                subentry_id="agent-1",
                title="Agent",
                agent_name=None,
                extra_data={CONF_SKILLS: ["skill-1"]},
            ),
        )
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
    entry = workspace_entry(
        (
            skill_subentry_data(
                subentry_id="skill-1",
                title="Blank Skill",
                content="",
            ),
        )
    )

    assert await async_skills_capabilities(hass, entry, None) == []
    assert await async_skills_capabilities(hass, entry, ["missing"]) == []
    assert await async_skills_capabilities(hass, entry, ["skill-1"]) == []
