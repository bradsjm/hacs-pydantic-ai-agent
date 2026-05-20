"""Home Assistant todo workspace tools for AI tasks."""

from dataclasses import dataclass
from typing import Any
import asyncio
import weakref

from pydantic_ai import FunctionToolset

from homeassistant.components.todo import (
    ATTR_DESCRIPTION,
    ATTR_ITEM,
    ATTR_RENAME,
    ATTR_STATUS,
    DOMAIN as TODO_DOMAIN,
    TodoItemStatus,
    TodoServices,
)
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

_TODO_COMPLETE_SEPARATOR = "---"

_READ_DESCRIPTION = """Read the current AI Task todo workspace.

Use this when you need to inspect existing task IDs, choose the next pending item,
or confirm that all work is complete. Do not call this repeatedly when the returned
summary already says all tasks are completed; finish the AI Task response instead.

The returned UID values are required for update, complete, and remove operations.
"""

_ADD_DESCRIPTION = """Add one task to the AI Task todo workspace.

Use this for non-trivial multi-step work, when the user asks for multiple tasks,
or when new instructions require tracking additional work. Add clear imperative
task summaries such as "Validate todo config flow". Do not use this for a single
straightforward conversational reply.
"""

_UPDATE_DESCRIPTION = """Rename or clarify one existing workspace task by UID.

Use this when an existing task summary needs to be corrected or made more precise.
Do not use this to mark a task completed; use ha_todo_complete_item so completion
always includes a completion note.
"""

_COMPLETE_DESCRIPTION = """Complete one workspace task by UID with a required completion note.

Use this immediately after finishing a task; do not batch completions. Never mark
a task completed while implementation is partial, validation is failing, or the
work is only planned. The completion_note must describe what was completed and
how it was verified.
"""

_REMOVE_DESCRIPTION = """Remove one or more irrelevant workspace tasks by UID.

Use this only when tasks no longer apply or were created by mistake. Do not remove
tasks to hide unfinished work; keep pending tasks visible until completed or until
you can explain why they are irrelevant.
"""


@dataclass(slots=True)
class TodoWorkspace:
    """Bound Home Assistant todo list used as an AI Task workspace."""

    hass: HomeAssistant
    entity_id: str

    async def prepare_run(self) -> str:
        """Clear all existing todo items before an AI Task run."""
        items = await self._items()
        uids = [item["uid"] for item in items if item.get("uid")]
        if uids:
            await self._remove_uids(uids)
        return f"Cleared {len(uids)} item(s) from {self.entity_id}."

    async def read_items(self) -> str:
        """Return a compact current todo workspace report."""
        return self._format_items(await self._items())

    async def add_item(self, summary: str, description: str | None = None) -> str:
        """Add a needs_action item and return the refreshed workspace."""
        summary = summary.strip()
        if not summary:
            return "Error: summary is required."
        data: dict[str, Any] = {ATTR_ITEM: summary}
        if description:
            data[ATTR_DESCRIPTION] = description
        try:
            await self._call_service(TodoServices.ADD_ITEM, data)
        except HomeAssistantError as err:
            return f"Error adding todo item: {err}"
        return "Added todo item.\n" + await self.read_items()

    async def update_item(self, uid: str, summary: str | None = None) -> str:
        """Update a todo item by UID and return the refreshed workspace."""
        uid = uid.strip()
        if not uid:
            return "Error: uid is required."
        if summary is None or not summary.strip():
            return "Error: summary is required."
        try:
            await self._call_service(
                TodoServices.UPDATE_ITEM,
                {ATTR_ITEM: uid, ATTR_RENAME: summary.strip()},
            )
        except HomeAssistantError as err:
            return f"Error updating todo item {uid}: {err}"
        return f"Updated todo item {uid}.\n" + await self.read_items()

    async def complete_item(self, uid: str, completion_note: str) -> str:
        """Append a completion note and mark the item completed."""
        uid = uid.strip()
        completion_note = completion_note.strip()
        if not uid:
            return "Error: uid is required."
        if not completion_note:
            return "Error: completion_note is required to complete an item."
        items = await self._items()
        item = next(
            (candidate for candidate in items if candidate.get("uid") == uid), None
        )
        if item is None:
            return f"Error: todo item {uid} was not found."
        timestamp = dt_util.utcnow().isoformat()
        current_description = item.get("description", "")
        note = f"{_TODO_COMPLETE_SEPARATOR}\n{timestamp}: {completion_note}"
        description = (
            f"{current_description.rstrip()}\n{note}" if current_description else note
        )
        try:
            await self._call_service(
                TodoServices.UPDATE_ITEM,
                {
                    ATTR_ITEM: uid,
                    ATTR_DESCRIPTION: description,
                    ATTR_STATUS: TodoItemStatus.COMPLETED,
                },
            )
        except HomeAssistantError as err:
            return f"Error completing todo item {uid}: {err}"
        return (
            f"Completed todo item {uid} with note timestamp {timestamp}.\n"
            + await self.read_items()
        )

    async def remove_items(self, uids: list[str]) -> str:
        """Remove todo items by UID and return the refreshed workspace."""
        clean_uids = [uid.strip() for uid in uids if uid.strip()]
        if not clean_uids:
            return "Error: at least one uid is required."
        try:
            await self._remove_uids(clean_uids)
        except HomeAssistantError as err:
            return f"Error removing todo items: {err}"
        return (
            f"Removed todo item(s): {', '.join(clean_uids)}.\n"
            + await self.read_items()
        )

    def toolset(self) -> FunctionToolset[None]:
        """Return Pydantic AI tools bound to this workspace."""
        toolset = FunctionToolset[None](sequential=True)
        toolset.add_function(
            self.read_items,
            name="ha_todo_read_items",
            description=_READ_DESCRIPTION,
        )
        toolset.add_function(
            self.add_item,
            name="ha_todo_add_item",
            description=_ADD_DESCRIPTION,
        )
        toolset.add_function(
            self.update_item,
            name="ha_todo_update_item",
            description=_UPDATE_DESCRIPTION,
        )
        toolset.add_function(
            self.complete_item,
            name="ha_todo_complete_item",
            description=_COMPLETE_DESCRIPTION,
        )
        toolset.add_function(
            self.remove_items,
            name="ha_todo_remove_items",
            description=_REMOVE_DESCRIPTION,
        )
        return toolset

    def instructions(self, initial_state: str) -> str:
        """Return the todo workspace behavioral contract for this run."""
        return f"""You have access to Home Assistant todo workspace tools for this AI Task run:
- ha_todo_read_items: view current tasks with UIDs and status
- ha_todo_add_item: add one needs_action task
- ha_todo_update_item: rename or clarify one task by UID
- ha_todo_complete_item: complete one task by UID with required completion_note
- ha_todo_remove_items: remove irrelevant tasks by UID

The configured todo list {self.entity_id} was cleared before this run. These tools affect only that todo list. Treat it as a temporary scratch workspace; do not imply persistence across runs.

Rules:
1. Use the todo tools for complex, multi-step, or non-trivial work; skip them for a single straightforward response.
2. Break complex work into smaller imperative task summaries.
3. Keep exactly one task conceptually in progress at a time, even though Home Assistant only stores needs_action and completed statuses.
4. Mark tasks completed immediately after finishing; do not batch completions.
5. Never complete a task if implementation is partial, validation is failing, or work is only planned.
6. Use UIDs from ha_todo_read_items for update, complete, and remove.
7. Completing an item requires a completion_note explaining what was done and how it was verified.
8. After completing or removing an item, inspect the returned summary to choose the next task.
9. The final AI Task output must still satisfy the requested structured output schema.

## Current Todo Workspace
{initial_state}
"""

    async def _items(self) -> list[dict[str, str]]:
        """Fetch all todo items from the configured list."""
        response = await self._call_service(
            TodoServices.GET_ITEMS,
            {ATTR_STATUS: [TodoItemStatus.NEEDS_ACTION, TodoItemStatus.COMPLETED]},
            return_response=True,
        )
        if not isinstance(response, dict):
            raise HomeAssistantError("Todo get_items returned an invalid response")
        entity_response = response.get(self.entity_id, {})
        if not isinstance(entity_response, dict):
            raise HomeAssistantError(
                "Todo get_items returned an invalid entity response"
            )
        items = entity_response.get("items", [])
        if not isinstance(items, list):
            raise HomeAssistantError(
                "Todo get_items returned an invalid items response"
            )
        return [item for item in items if isinstance(item, dict)]

    async def _remove_uids(self, uids: list[str]) -> None:
        """Remove todo items by UID."""
        await self._call_service(TodoServices.REMOVE_ITEM, {ATTR_ITEM: uids})

    async def _call_service(
        self,
        service: str,
        data: dict[str, Any],
        *,
        return_response: bool = False,
    ) -> Any:
        """Call a Home Assistant todo service for this workspace."""
        return await self.hass.services.async_call(
            TODO_DOMAIN,
            service,
            data,
            target={ATTR_ENTITY_ID: self.entity_id},
            blocking=True,
            return_response=return_response,
        )

    def _format_items(self, items: list[dict[str, str]]) -> str:
        """Return a compact text summary of todo items."""
        completed = [
            item for item in items if item.get("status") == TodoItemStatus.COMPLETED
        ]
        pending = [
            item for item in items if item.get("status") != TodoItemStatus.COMPLETED
        ]
        lines = [
            f"Summary: {len(completed)} completed, 0 in progress, {len(pending)} pending"
        ]
        if not items:
            lines.append("No todo items exist yet.")
            return "\n".join(lines)
        for heading, group in (("Pending", pending), ("Completed", completed)):
            if not group:
                continue
            lines.append(f"\n{heading}:")
            for item in group:
                lines.append(self._format_item(item))
        if pending:
            return "\n".join(lines)
        lines.append(
            "All tasks are completed. Do not call ha_todo_read_items again; "
            "finish the AI Task response."
        )
        return "\n".join(lines)

    @staticmethod
    def _format_item(item: dict[str, str]) -> str:
        """Return one item as text."""
        uid = item.get("uid", "<missing uid>")
        status = item.get("status", "unknown")
        summary = item.get("summary", "")
        description = item.get("description")
        line = f"- [{status}] [{uid}] {summary}"
        if description:
            line += f"\n  description: {description}"
        return line


def todo_workspace_lock_key(entity_id: str) -> str:
    """Return the lock registry key for a todo workspace entity."""
    return f"todo_workspace:{entity_id}"


def todo_workspace_locks(
    hass: HomeAssistant,
) -> weakref.WeakValueDictionary[str, asyncio.Lock]:
    """Return the integration-global todo workspace lock registry."""
    from .const import DOMAIN

    domain_data = hass.data.setdefault(DOMAIN, {})
    locks = domain_data.get("todo_workspace_locks")
    if not isinstance(locks, weakref.WeakValueDictionary):
        locks = domain_data["todo_workspace_locks"] = weakref.WeakValueDictionary()
    return locks


def todo_workspace_lock(hass: HomeAssistant, entity_id: str) -> asyncio.Lock:
    """Return the shared lock for one todo workspace entity."""
    locks = todo_workspace_locks(hass)
    key = todo_workspace_lock_key(entity_id)
    lock = locks.get(key)
    if lock is None:
        lock = locks[key] = asyncio.Lock()
    return lock
