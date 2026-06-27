"""Tests for virtual workspace patch application."""

from custom_components.pydantic_ai_agent.virtual_workspace.patch import apply_patch
from custom_components.pydantic_ai_agent.virtual_workspace.workspace import (
    VirtualWorkspace,
)
import pytest


@pytest.fixture
def workspace() -> VirtualWorkspace:
    """Return a fresh virtual workspace for each patch test."""
    return VirtualWorkspace()


def test_apply_patch_adds_new_file(workspace: VirtualWorkspace) -> None:
    """Add operations create files without confirmation when not replacing."""
    result = apply_patch(
        workspace,
        """*** Begin Patch
*** Add File: notes.txt
+hello
+world
*** End Patch""",
    )

    assert result == {"success": True, "changedFiles": ["/workspace/notes.txt"]}
    assert workspace.read_file("notes.txt")["content"] == "hello\nworld\n"


def test_apply_patch_rejects_invalid_envelope(workspace: VirtualWorkspace) -> None:
    """Patches must use the expected begin/end envelope."""
    result = apply_patch(workspace, "*** Add File: notes.txt\n+hello")

    assert result["success"] is False
    assert result["changedFiles"] == []
    assert "must start" in result["errors"][0]


@pytest.mark.parametrize(
    "patch",
    [
        """*** Begin Patch
*** Update File: notes.txt
@@
-old
+new
*** End Patch""",
        """*** Begin Patch
*** Delete File: notes.txt
*** End Patch""",
    ],
)
def test_update_and_delete_require_confirmation(
    workspace: VirtualWorkspace, patch: str
) -> None:
    """Destructive patch operations are explicitly confirm-gated."""
    workspace.write_file("notes.txt", "old\n")

    result = apply_patch(workspace, patch)

    assert result["success"] is False
    assert result["changedFiles"] == []
    assert "confirm=true" in result["errors"][0]
    assert workspace.read_file("notes.txt")["content"] == "old\n"


def test_apply_patch_updates_moves_and_deletes_with_confirmation(
    workspace: VirtualWorkspace,
) -> None:
    """Confirmed update, move, and delete operations mutate workspace state."""
    workspace.write_file("old.txt", "one\ntwo\n")
    workspace.write_file("delete.txt", "remove me\n")

    result = apply_patch(
        workspace,
        """*** Begin Patch
*** Update File: old.txt
*** Move to: new.txt
@@
 one
-two
+three
*** Delete File: delete.txt
*** End Patch""",
        confirm=True,
    )

    assert result == {
        "success": True,
        "changedFiles": [
            "/workspace/old.txt",
            "/workspace/new.txt",
            "/workspace/delete.txt",
        ],
    }
    assert workspace.read_file("new.txt")["content"] == "one\nthree\n"
    assert workspace.read_file("old.txt")["ok"] is False
    assert workspace.read_file("delete.txt")["ok"] is False


def test_apply_patch_rejects_protected_path_replacement(
    workspace: VirtualWorkspace,
) -> None:
    """Protected virtual root paths cannot be replaced by patch operations."""
    result = apply_patch(
        workspace,
        """*** Begin Patch
*** Add File: /workspace
+not a directory
*** End Patch""",
        confirm=True,
    )

    assert result["success"] is False
    assert "protected workspace paths" in result["errors"][0]
    assert workspace.metadata("/workspace")["type"] == "directory"


def test_apply_patch_rolls_back_when_later_operation_fails(
    workspace: VirtualWorkspace,
) -> None:
    """Patch application is atomic across multiple operations."""
    result = apply_patch(
        workspace,
        """*** Begin Patch
*** Add File: first.txt
+created
*** Delete File: second.txt
+delete must not contain content
*** End Patch""",
        confirm=True,
    )

    assert result["success"] is False
    assert result["changedFiles"] == []
    assert workspace.read_file("first.txt")["ok"] is False
