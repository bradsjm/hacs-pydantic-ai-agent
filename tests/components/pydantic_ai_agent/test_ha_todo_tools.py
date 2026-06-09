"""Test Home Assistant todo workspace tools."""

from typing import Any

from custom_components.pydantic_ai_agent.ha_todo_tools import TodoWorkspace
from homeassistant.components.todo import (
    DOMAIN as TODO_DOMAIN,
)
from homeassistant.components.todo import (
    TodoItemStatus,
    TodoServices,
)
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse


async def test_prepare_run_clears_all_items_by_uid(hass: HomeAssistant) -> None:
    """Test run preparation reads all statuses and removes returned UIDs."""
    calls: list[dict[str, Any]] = []

    async def get_items(call: ServiceCall) -> dict[str, Any]:
        calls.append({"service": TodoServices.GET_ITEMS, "data": dict(call.data)})
        return {
            "todo.ai_workspace": {
                "items": [
                    {"uid": "1", "status": "needs_action", "summary": "Plan"},
                    {"uid": "2", "status": "completed", "summary": "Done"},
                ]
            }
        }

    async def remove_item(call: ServiceCall) -> None:
        calls.append({"service": TodoServices.REMOVE_ITEM, "data": dict(call.data)})

    hass.services.async_register(
        TODO_DOMAIN,
        TodoServices.GET_ITEMS,
        get_items,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(TODO_DOMAIN, TodoServices.REMOVE_ITEM, remove_item)

    result = await TodoWorkspace(hass, "todo.ai_workspace").prepare_run()

    assert result == "Cleared 2 item(s) from todo.ai_workspace."
    assert calls == [
        {
            "service": TodoServices.GET_ITEMS,
            "data": {
                ATTR_ENTITY_ID: "todo.ai_workspace",
                "status": [TodoItemStatus.NEEDS_ACTION, TodoItemStatus.COMPLETED],
            },
        },
        {
            "service": TodoServices.REMOVE_ITEM,
            "data": {ATTR_ENTITY_ID: "todo.ai_workspace", "item": ["1", "2"]},
        },
    ]


async def test_complete_item_appends_note_then_marks_completed(
    hass: HomeAssistant,
) -> None:
    """Test completion writes a note before setting completed status."""
    calls: list[dict[str, Any]] = []

    async def get_items(call: ServiceCall) -> dict[str, Any]:
        calls.append({"service": TodoServices.GET_ITEMS, "data": dict(call.data)})
        return {
            "todo.ai_workspace": {
                "items": [
                    {
                        "uid": "1",
                        "status": "needs_action",
                        "summary": "Implement feature",
                        "description": "Existing details",
                    }
                ]
            }
        }

    async def update_item(call: ServiceCall) -> None:
        calls.append({"service": TodoServices.UPDATE_ITEM, "data": dict(call.data)})

    hass.services.async_register(
        TODO_DOMAIN,
        TodoServices.GET_ITEMS,
        get_items,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(TODO_DOMAIN, TodoServices.UPDATE_ITEM, update_item)

    result = await TodoWorkspace(hass, "todo.ai_workspace").complete_item(
        "1", "Verified focused tests pass"
    )

    update_call = calls[1]
    assert update_call["service"] == TodoServices.UPDATE_ITEM
    assert update_call["data"]["item"] == "1"
    assert update_call["data"]["status"] == TodoItemStatus.COMPLETED
    assert "Existing details" in update_call["data"]["description"]
    assert "Verified focused tests pass" in update_call["data"]["description"]
    assert result.startswith("Completed todo item 1 with note timestamp")


async def test_complete_item_rejects_blank_note(hass: HomeAssistant) -> None:
    """Test completion requires an explanatory note."""
    result = await TodoWorkspace(hass, "todo.ai_workspace").complete_item("1", " ")

    assert result == "Error: completion_note is required to complete an item."
