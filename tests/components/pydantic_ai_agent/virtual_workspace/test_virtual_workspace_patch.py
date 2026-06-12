"""Test virtual workspace patch application."""

from custom_components.pydantic_ai_agent.virtual_workspace.patch import apply_patch
from custom_components.pydantic_ai_agent.virtual_workspace.workspace import (
    VirtualWorkspace,
)


def test_apply_patch_adds_updates_deletes_and_moves_files() -> None:
    """Test supported patch operations use direct VFS helpers."""
    workspace = VirtualWorkspace()

    added = apply_patch(
        workspace,
        """*** Begin Patch
*** Add File: notes.txt
+hello
+world
*** End Patch""",
    )
    updated = apply_patch(
        workspace,
        """*** Begin Patch
*** Update File: notes.txt
@@
-hello
+hi
 world
*** End Patch""",
        confirm=True,
    )
    moved = apply_patch(
        workspace,
        """*** Begin Patch
*** Update File: notes.txt
*** Move to: moved.txt
*** End Patch""",
        confirm=True,
    )
    moved_content = workspace.read_file("moved.txt")["content"]
    deleted = apply_patch(
        workspace,
        """*** Begin Patch
*** Delete File: moved.txt
*** End Patch""",
        confirm=True,
    )

    assert added == {"success": True, "changedFiles": ["/workspace/notes.txt"]}
    assert updated == {"success": True, "changedFiles": ["/workspace/notes.txt"]}
    assert moved_content == "hi\nworld\n"
    assert moved == {
        "success": True,
        "changedFiles": ["/workspace/notes.txt", "/workspace/moved.txt"],
    }
    assert deleted == {"success": True, "changedFiles": ["/workspace/moved.txt"]}
    assert workspace.read_file("moved.txt")["ok"] is False


def test_apply_patch_requires_confirmation_for_destructive_changes() -> None:
    """Test updates/deletes/moves do not run without confirm=true."""
    workspace = VirtualWorkspace()
    workspace.write_file("notes.txt", "hello\n")

    result = apply_patch(
        workspace,
        """*** Begin Patch
*** Update File: notes.txt
@@
-hello
+hi
*** End Patch""",
    )

    assert result["success"] is False
    assert result["changedFiles"] == []
    errors = result.get("errors", [])
    assert errors and "confirm=true" in errors[0]
    assert workspace.read_file("notes.txt")["content"] == "hello\n"


def test_apply_patch_rolls_back_on_failure() -> None:
    """Test a failing operation restores all earlier patch changes."""
    workspace = VirtualWorkspace()

    result = apply_patch(
        workspace,
        """*** Begin Patch
*** Add File: created.txt
+created
*** Update File: missing.txt
@@
-missing
+updated
*** End Patch""",
        confirm=True,
    )

    assert result["success"] is False
    assert result["changedFiles"] == []
    assert workspace.read_file("created.txt")["ok"] is False
