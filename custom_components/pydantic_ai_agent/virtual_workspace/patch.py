"""Codex-style patch parsing and application for the virtual workspace."""

from dataclasses import dataclass, field
from typing import Literal

from .const import MAX_PATCH_BYTES
from .errors import ConfirmationRequiredError, PatchApplyError, PathValidationError
from .models import PatchResult
from .paths import normalize_vfs_path, protected_replacement_path
from .workspace import VirtualWorkspace

_BEGIN = "*** Begin Patch"
_END = "*** End Patch"
_ADD = "*** Add File: "
_UPDATE = "*** Update File: "
_DELETE = "*** Delete File: "
_MOVE = "*** Move to: "


@dataclass
class _PatchOperation:
    kind: Literal["add", "update", "delete"]
    path: str
    lines: list[str] = field(default_factory=list)
    move_to: str | None = None


def apply_patch(
    workspace: VirtualWorkspace,
    patch: str,
    *,
    confirm: bool = False,
) -> PatchResult:
    """Apply a Codex-style patch atomically to the virtual workspace."""
    snapshot = workspace.snapshot()
    changed_files: list[str] = []
    try:
        if len(patch.encode()) > MAX_PATCH_BYTES:
            raise PatchApplyError("patch exceeds the size limit")
        operations = _parse_patch(patch)
        for operation in operations:
            changed_files.extend(_apply_operation(workspace, operation, confirm=confirm))
        return {"success": True, "changedFiles": changed_files}
    except Exception as err:  # noqa: BLE001 - tool-result boundary must report any patch/VFS failure.
        workspace.restore_snapshot(snapshot)
        return {"success": False, "changedFiles": [], "errors": [str(err)]}


def _parse_patch(patch: str) -> list[_PatchOperation]:
    lines = patch.splitlines()
    if not lines or lines[0] != _BEGIN:
        raise PatchApplyError("patch must start with *** Begin Patch")
    if lines[-1] != _END:
        raise PatchApplyError("patch must end with *** End Patch")
    operations: list[_PatchOperation] = []
    current: _PatchOperation | None = None
    for line in lines[1:-1]:
        current = _process_patch_line(line, current, operations)
    if not operations:
        raise PatchApplyError("patch does not contain file operations")
    return operations


def _process_patch_line(
    line: str,
    current: _PatchOperation | None,
    operations: list[_PatchOperation],
) -> _PatchOperation | None:
    if line.startswith(_ADD):
        return _append_operation(operations, "add", line.removeprefix(_ADD))
    if line.startswith(_UPDATE):
        return _append_operation(operations, "update", line.removeprefix(_UPDATE))
    if line.startswith(_DELETE):
        return _append_operation(operations, "delete", line.removeprefix(_DELETE))
    if line.startswith(_MOVE):
        if current is None or current.kind != "update" or current.lines:
            raise PatchApplyError("Move to must immediately follow an update header")
        current.move_to = line.removeprefix(_MOVE)
        return current
    if current is None:
        raise PatchApplyError("patch content must follow a file header")
    current.lines.append(line)
    return current


def _append_operation(
    operations: list[_PatchOperation],
    kind: Literal["add", "update", "delete"],
    path: str,
) -> _PatchOperation:
    normalized = normalize_vfs_path(path)
    operation = _PatchOperation(kind=kind, path=normalized)
    operations.append(operation)
    return operation


def _apply_operation(
    workspace: VirtualWorkspace,
    operation: _PatchOperation,
    *,
    confirm: bool,
) -> list[str]:
    if operation.kind == "add":
        return _apply_add(workspace, operation, confirm=confirm)
    if operation.kind == "delete":
        return _apply_delete(workspace, operation, confirm=confirm)
    return _apply_update(workspace, operation, confirm=confirm)


def _apply_add(
    workspace: VirtualWorkspace,
    operation: _PatchOperation,
    *,
    confirm: bool,
) -> list[str]:
    exists = workspace.exists(operation.path)
    if exists:
        _require_confirm(confirm, "adding over an existing file requires confirm=true")
    _check_replace_allowed(operation.path)
    content = _added_content(operation.lines)
    result = workspace.write_file(
        operation.path,
        content,
        overwrite=exists,
        confirm=confirm,
    )
    if not result["ok"]:
        raise PatchApplyError(result.get("error", "failed to add file"))
    return [operation.path]


def _apply_delete(
    workspace: VirtualWorkspace,
    operation: _PatchOperation,
    *,
    confirm: bool,
) -> list[str]:
    _require_confirm(confirm, "delete requires confirm=true")
    if operation.lines:
        raise PatchApplyError("delete operations must not contain content")
    result = workspace.remove(operation.path, recursive=True, confirm=True)
    if not result["ok"]:
        raise PatchApplyError(result.get("error", "failed to delete file"))
    return [operation.path]


def _apply_update(
    workspace: VirtualWorkspace,
    operation: _PatchOperation,
    *,
    confirm: bool,
) -> list[str]:
    _require_confirm(confirm, "update and move operations require confirm=true")
    _check_replace_allowed(operation.path)
    changed = [operation.path]
    if operation.lines:
        read = workspace.read_file(operation.path)
        if not read["ok"]:
            raise PatchApplyError(read.get("error", "failed to read file for update"))
        content = _apply_hunks(read["content"], operation.lines)
        result = workspace.write_file(operation.path, content, overwrite=True, confirm=True)
        if not result["ok"]:
            raise PatchApplyError(result.get("error", "failed to update file"))
    elif operation.move_to is None:
        raise PatchApplyError("update operation does not contain a hunk")
    if operation.move_to is not None:
        destination = normalize_vfs_path(operation.move_to)
        _check_replace_allowed(destination)
        move = workspace.move(operation.path, destination, overwrite=False, confirm=True)
        if not move["ok"]:
            raise PatchApplyError(move.get("error", "failed to move file"))
        changed.append(destination)
    return changed


def _added_content(lines: list[str]) -> str:
    content: list[str] = []
    for line in lines:
        if not line.startswith("+"):
            raise PatchApplyError("add file lines must start with +")
        content.append(line[1:])
    return "\n".join(content) + ("\n" if content else "")


def _apply_hunks(original: str, patch_lines: list[str]) -> str:
    source = original.splitlines()
    output: list[str] = []
    index = 0
    saw_hunk_line = False
    for line in patch_lines:
        if line.startswith("@@"):
            continue
        if not line:
            raise PatchApplyError("empty patch lines must include a prefix")
        prefix = line[0]
        text = line[1:]
        if prefix == " ":
            saw_hunk_line = True
            index = _copy_until_match(source, output, index, text)
            output.append(source[index])
            index += 1
            continue
        if prefix == "-":
            saw_hunk_line = True
            index = _copy_until_match(source, output, index, text)
            index += 1
            continue
        if prefix == "+":
            saw_hunk_line = True
            output.append(text)
            continue
        raise PatchApplyError(f"unsupported update line prefix: {prefix}")
    if not saw_hunk_line:
        raise PatchApplyError("update operation does not contain a hunk")
    output.extend(source[index:])
    return "\n".join(output) + ("\n" if original.endswith("\n") else "")


def _copy_until_match(
    source: list[str],
    output: list[str],
    index: int,
    text: str,
) -> int:
    while index < len(source) and source[index] != text:
        output.append(source[index])
        index += 1
    if index >= len(source):
        raise PatchApplyError(f"patch context not found: {text}")
    return index


def _require_confirm(confirm: bool, message: str) -> None:
    if not confirm:
        raise ConfirmationRequiredError(message)


def _check_replace_allowed(path: str) -> None:
    if protected_replacement_path(path):
        raise PathValidationError("protected workspace paths cannot be replaced")
