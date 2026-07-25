"""Tests for native workspace Skill helpers."""

from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any, cast

from custom_components.pydantic_ai_agent.agent.skills import (
    NativeSkill,
    NativeSkillsCapability,
    async_skills_capabilities,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_DESCRIPTION,
    CONF_SKILL_CONTENT,
    CONF_SKILL_REFERENCES,
    SUBENTRY_TYPE_MCP_SERVER,
    SUBENTRY_TYPE_SKILL,
)
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant


def test_native_skill_metadata_is_serializable_summary() -> None:
    """Skill metadata exposes only the list-tool summary fields."""
    skill = NativeSkill(
        skill_id="skill-1",
        name="Lighting",
        description="Lighting guidance",
        content="Use warm lights.",
        references=[{"title": "Docs"}],
    )

    assert skill.metadata() == {
        "skill_id": "skill-1",
        "name": "Lighting",
        "description": "Lighting guidance",
    }


async def test_native_skills_capability_lists_and_loads_skills() -> None:
    """The native Skill toolset lists metadata and loads full selected content."""
    capability = NativeSkillsCapability(
        [
            NativeSkill(
                skill_id="skill-1",
                name="Lighting",
                description="Lighting guidance",
                content="Use warm lights.",
                references=[{"title": "Docs"}],
            )
        ]
    )
    tools = capability.get_toolset().tools
    list_skills = cast(
        Callable[[], Awaitable[list[dict[str, str]]]], tools["list_skills"].function
    )

    assert await list_skills() == [
        {
            "skill_id": "skill-1",
            "name": "Lighting",
            "description": "Lighting guidance",
        }
    ]
    assert await tools["load_skill"].function("skill-1") == {
        "skill_id": "skill-1",
        "name": "Lighting",
        "description": "Lighting guidance",
        "content": "Use warm lights.",
        "references": [{"title": "Docs"}],
    }
    assert await tools["load_skill"].function("missing") == {
        "error": "skill_not_found",
        "skill_id": "missing",
    }


async def test_async_skills_capabilities_filters_selected_skill_subentries(
    hass: HomeAssistant,
    make_subentry: Callable[..., Any],
) -> None:
    """Only selected valid Skill subentries become native Skill capabilities."""
    skill = make_subentry(
        subentry_type=SUBENTRY_TYPE_SKILL,
        title="Fallback Name",
        data={
            CONF_NAME: " Lighting ",
            CONF_DESCRIPTION: " Guidance ",
            CONF_SKILL_CONTENT: " Use warm lights. ",
            CONF_SKILL_REFERENCES: [{"title": "Docs"}, "ignored"],
        },
    )
    empty_skill = make_subentry(
        subentry_type=SUBENTRY_TYPE_SKILL,
        data={CONF_SKILL_CONTENT: "  "},
    )
    not_a_skill = make_subentry(
        subentry_type=SUBENTRY_TYPE_MCP_SERVER,
        data={CONF_SKILL_CONTENT: "ignored"},
    )
    entry = SimpleNamespace(
        subentries={
            "skill-1": skill,
            "empty": empty_skill,
            "mcp-1": not_a_skill,
        }
    )

    capabilities = await async_skills_capabilities(
        hass,
        entry,
        ["skill-1", "missing", "mcp-1", "empty", "skill-1"],
    )

    assert len(capabilities) == 1
    tools = capabilities[0].get_toolset().tools
    list_skills = cast(
        Callable[[], Awaitable[list[dict[str, str]]]], tools["list_skills"].function
    )
    assert await list_skills() == [
        {
            "skill_id": "skill-1",
            "name": "Lighting",
            "description": "Guidance",
        }
    ]
    loaded_skill = await tools["load_skill"].function("skill-1")
    assert loaded_skill["references"] == [{"title": "Docs"}]


async def test_async_skills_capabilities_returns_empty_for_no_valid_selection(
    hass: HomeAssistant,
) -> None:
    """Invalid selection input or missing selected skills produces no capability."""
    assert (
        await async_skills_capabilities(hass, SimpleNamespace(subentries={}), 123) == []
    )
    assert (
        await async_skills_capabilities(
            hass,
            SimpleNamespace(subentries={}),
            ["missing"],
        )
        == []
    )
