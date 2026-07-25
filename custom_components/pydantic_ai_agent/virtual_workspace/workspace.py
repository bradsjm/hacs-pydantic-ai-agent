"""Bashkit-backed in-memory virtual workspace."""

import logging
from typing import Any, cast

from bashkit import BashTool

from ._workspace_helpers import (
    bash_error as _bash_error,
)
from ._workspace_helpers import (
    metadata as _metadata,
)
from ._workspace_helpers import (
    parse_cursor as _parse_cursor,
)
from ._workspace_helpers import (
    require_confirmed_overwrite as _require_confirmed_overwrite,
)
from ._workspace_helpers import (
    safe_normalized_path as _safe_normalized_path,
)
from ._workspace_helpers import (
    shell_quote as _shell_quote,
)
from ._workspace_helpers import (
    truncate_text as _truncate_text,
)
from .const import (
    COMMAND_TIMEOUT_SECONDS,
    DEFAULT_DIRECTORY_PAGE_SIZE,
    DEFAULT_WORKING_DIRECTORY,
    MAX_COMMAND_BYTES,
    MAX_DIRECTORY_PAGE_SIZE,
    MAX_READ_BYTES,
    MAX_STDERR_BYTES,
    MAX_STDOUT_BYTES,
    MAX_WORKSPACE_BYTES,
    MAX_WRITE_BYTES,
    VFS_ROOT,
)
from .errors import (
    ConfirmationRequiredError,
    PathValidationError,
    VirtualWorkspaceError,
)
from .models import (
    BashResult,
    CopyMoveResult,
    CreateDirectoryResult,
    DirectoryEntry,
    MetadataResult,
    ReadDirectoryResult,
    ReadFileResult,
    RemoveResult,
    WriteFileResult,
)
from .paths import normalize_vfs_path, protected_replacement_path

_LOGGER = logging.getLogger(__name__)


class VirtualWorkspace:
    """Per-run Bashkit virtual workspace with direct VFS helpers."""

    def __init__(self) -> None:
        """Initialize the in-memory workspace without host mounts or network."""
        self._tool = BashTool(
            max_memory=MAX_WORKSPACE_BYTES,
            timeout_seconds=COMMAND_TIMEOUT_SECONDS,
            network=None,
        )
        self._ensure_workspace()

    async def bash(
        self,
        command: str,
        *,
        working_directory: str | None = None,
    ) -> BashResult:
        """Execute a virtual bash command and restore cwd afterward."""
        snapshot: Any | None = None
        original_cwd = DEFAULT_WORKING_DIRECTORY
        try:
            if not isinstance(command, str) or not command:
                raise VirtualWorkspaceError("command is required")
            if len(command.encode()) > MAX_COMMAND_BYTES:
                raise VirtualWorkspaceError("command exceeds the size limit")
            cwd = normalize_vfs_path(working_directory or DEFAULT_WORKING_DIRECTORY)
            self._require_directory(cwd)
            original_cwd = self._current_cwd()
            snapshot = self.snapshot()
            try:
                result = await self._tool.execute(
                    f"cd {_shell_quote(cwd)}\n{command}",
                )
            except Exception as err:  # noqa: BLE001 - Bashkit command failures are returned as tool errors.
                self.restore_snapshot(snapshot)
                await self._restore_cwd_safely(original_cwd)
                return _bash_error(str(err))
            await self._restore_cwd(original_cwd)
            if self._path_size(VFS_ROOT) > MAX_WORKSPACE_BYTES:
                self.restore_snapshot(snapshot)
                await self._restore_cwd_safely(original_cwd)
                return _bash_error("workspace size limit exceeded")
            stdout, stdout_truncated = _truncate_text(result.stdout, MAX_STDOUT_BYTES)
            stderr, stderr_truncated = _truncate_text(result.stderr, MAX_STDERR_BYTES)
            bash_result: BashResult = {
                "ok": bool(result.success),
                "stdout": stdout,
                "stderr": stderr,
                "exitCode": result.exit_code,
                "stdoutTruncated": stdout_truncated,
                "stderrTruncated": stderr_truncated,
            }
            if result.error:
                bash_result["error"] = str(result.error)
            return bash_result
        except VirtualWorkspaceError as err:
            return _bash_error(str(err))
        except Exception as err:  # noqa: BLE001 - Bashkit surfaces expected command/VFS failures as exceptions.
            if snapshot is not None:
                self.restore_snapshot(snapshot)
                await self._restore_cwd_safely(original_cwd)
            return _bash_error(str(err))

    def read_file(self, path: str) -> ReadFileResult:
        """Read a virtual file."""
        try:
            normalized = normalize_vfs_path(path)
            content = self._tool.read_file(normalized)
            content, truncated = _truncate_text(content, MAX_READ_BYTES)
            return {
                "ok": True,
                "path": normalized,
                "content": content,
                "bytesRead": len(content.encode()),
                "truncated": truncated,
            }
        except Exception as err:  # noqa: BLE001 - virtual file tool returns structured errors for any VFS failure.
            return {
                "ok": False,
                "path": _safe_normalized_path(path),
                "content": "",
                "bytesRead": 0,
                "truncated": False,
                "error": str(err),
            }

    def write_file(
        self,
        path: str,
        content: str,
        *,
        overwrite: bool = False,
        confirm: bool = False,
    ) -> WriteFileResult:
        """Write a virtual file with explicit overwrite confirmation."""
        try:
            normalized = normalize_vfs_path(path)
            if not isinstance(content, str):
                raise VirtualWorkspaceError("content must be a string")
            content_bytes = len(content.encode())
            if content_bytes > MAX_WRITE_BYTES:
                raise VirtualWorkspaceError("content exceeds the write size limit")
            exists = self.exists(normalized)
            if exists:
                _require_confirmed_overwrite(normalized, overwrite, confirm)
            self._ensure_workspace_size_limit(
                added_bytes=content_bytes,
                replaced_path=normalized if exists else None,
            )
            self._tool.write_file(normalized, content)
            return {"ok": True, "path": normalized, "bytesWritten": content_bytes}
        except Exception as err:  # noqa: BLE001 - virtual file tool returns structured errors for any VFS failure.
            return {
                "ok": False,
                "path": _safe_normalized_path(path),
                "bytesWritten": 0,
                "error": str(err),
            }

    def create_directory(
        self,
        path: str,
        *,
        parents: bool = False,
    ) -> CreateDirectoryResult:
        """Create a virtual directory."""
        try:
            normalized = normalize_vfs_path(path)
            existed = self.exists(normalized)
            self._tool.mkdir(normalized, recursive=parents)
            return {"ok": True, "path": normalized, "created": not existed}
        except Exception as err:  # noqa: BLE001 - virtual directory tool returns structured errors for any VFS failure.
            return {
                "ok": False,
                "path": _safe_normalized_path(path),
                "created": False,
                "error": str(err),
            }

    def metadata(self, path: str) -> MetadataResult:
        """Return metadata for a virtual path."""
        try:
            normalized = normalize_vfs_path(path)
            return cast(
                MetadataResult,
                {
                    "ok": True,
                    "path": normalized,
                    **_metadata(self._tool.stat(normalized)),
                },
            )
        except Exception as err:  # noqa: BLE001 - metadata tool returns structured errors for any VFS failure.
            return {
                "ok": False,
                "path": _safe_normalized_path(path),
                "type": "unknown",
                "size": 0,
                "mode": None,
                "created": None,
                "modified": None,
                "error": str(err),
            }

    def read_directory(
        self,
        path: str,
        *,
        cursor: str | None = None,
        limit: int = DEFAULT_DIRECTORY_PAGE_SIZE,
    ) -> ReadDirectoryResult:
        """Read a virtual directory with stable name-sorted pagination."""
        try:
            normalized = normalize_vfs_path(path)
            offset = _parse_cursor(cursor)
            page_size = min(max(int(limit), 1), MAX_DIRECTORY_PAGE_SIZE)
            raw_entries = sorted(
                self._tool.read_dir(normalized),
                key=lambda entry: str(entry.get("name", "")),
            )
            page = raw_entries[offset : offset + page_size]
            entries: list[DirectoryEntry] = []
            for entry in page:
                name = str(entry.get("name", ""))
                metadata = _metadata(entry.get("metadata", {}))
                entries.append(
                    cast(
                        DirectoryEntry,
                        {
                            "name": name,
                            "path": normalize_vfs_path(name, working_directory=normalized),
                            **metadata,
                        },
                    )
                )
            next_offset = offset + page_size
            result: ReadDirectoryResult = {
                "ok": True,
                "path": normalized,
                "entries": entries,
            }
            if next_offset < len(raw_entries):
                result["nextCursor"] = str(next_offset)
            return result
        except Exception as err:  # noqa: BLE001 - directory listing tool returns structured errors for any VFS failure.
            return {
                "ok": False,
                "path": _safe_normalized_path(path),
                "entries": [],
                "error": str(err),
            }

    def remove(
        self,
        path: str,
        *,
        recursive: bool = False,
        confirm: bool = False,
    ) -> RemoveResult:
        """Remove a virtual path after explicit confirmation."""
        try:
            normalized = normalize_vfs_path(path)
            if not confirm:
                raise ConfirmationRequiredError("remove requires confirm=true")
            if protected_replacement_path(normalized):
                raise PathValidationError("protected workspace paths cannot be removed")
            self._tool.remove(normalized, recursive=recursive)
            return {"ok": True, "path": normalized, "removed": True}
        except Exception as err:  # noqa: BLE001 - destructive tool returns structured errors for any VFS failure.
            return {
                "ok": False,
                "path": _safe_normalized_path(path),
                "removed": False,
                "error": str(err),
            }

    def copy(
        self,
        source: str,
        destination: str,
        *,
        overwrite: bool = False,
        confirm: bool = False,
    ) -> CopyMoveResult:
        """Copy a virtual path."""
        try:
            src = normalize_vfs_path(source)
            dest = normalize_vfs_path(destination)
            if src == dest:
                raise VirtualWorkspaceError("source and destination must differ")
            if not self.exists(src):
                raise VirtualWorkspaceError("source does not exist")
            self._reject_descendant_destination(src, dest)
            replaced_path = dest if self.exists(dest) else None
            self._ensure_workspace_size_limit(
                added_bytes=self._path_size(src),
                replaced_path=replaced_path,
            )
            snapshot = self.snapshot()
            try:
                self._prepare_destination(dest, overwrite=overwrite, confirm=confirm)
                self._tool.fs().copy(src, dest)
                self._raise_if_workspace_size_exceeded()
            except Exception:
                self.restore_snapshot(snapshot)
                raise
            return {"ok": True, "source": src, "destination": dest}
        except Exception as err:  # noqa: BLE001 - copy tool returns structured errors for any VFS failure.
            return {
                "ok": False,
                "source": _safe_normalized_path(source),
                "destination": _safe_normalized_path(destination),
                "error": str(err),
            }

    def move(
        self,
        source: str,
        destination: str,
        *,
        overwrite: bool = False,
        confirm: bool = False,
    ) -> CopyMoveResult:
        """Move a virtual path."""
        try:
            src = normalize_vfs_path(source)
            dest = normalize_vfs_path(destination)
            if not confirm:
                raise ConfirmationRequiredError("move requires confirm=true")
            if src == dest:
                raise VirtualWorkspaceError("source and destination must differ")
            if protected_replacement_path(src):
                raise PathValidationError("protected workspace paths cannot be moved")
            if not self.exists(src):
                raise VirtualWorkspaceError("source does not exist")
            self._reject_descendant_destination(src, dest)
            replaced_path = dest if self.exists(dest) else None
            self._ensure_workspace_size_limit(added_bytes=0, replaced_path=replaced_path)
            snapshot = self.snapshot()
            try:
                self._prepare_destination(dest, overwrite=overwrite, confirm=confirm)
                self._tool.fs().rename(src, dest)
                self._raise_if_workspace_size_exceeded()
            except Exception:
                self.restore_snapshot(snapshot)
                raise
            return {"ok": True, "source": src, "destination": dest}
        except Exception as err:  # noqa: BLE001 - move tool returns structured errors for any VFS failure.
            return {
                "ok": False,
                "source": _safe_normalized_path(source),
                "destination": _safe_normalized_path(destination),
                "error": str(err),
            }

    def exists(self, path: str) -> bool:
        """Return whether a normalized virtual path exists."""
        return bool(self._tool.exists(path))

    def snapshot(self) -> bytes:
        """Return a Bashkit filesystem snapshot."""
        return self._tool.snapshot()

    def restore_snapshot(self, snapshot: bytes) -> None:
        """Restore a Bashkit filesystem snapshot."""
        self._tool.restore_snapshot(snapshot)

    def _prepare_destination(
        self,
        destination: str,
        *,
        overwrite: bool,
        confirm: bool,
    ) -> None:
        if not self.exists(destination):
            return
        _require_confirmed_overwrite(destination, overwrite, confirm)
        self._tool.remove(destination, recursive=True)

    def _current_cwd(self) -> str:
        cwd = getattr(self._tool.shell_state(), "cwd", DEFAULT_WORKING_DIRECTORY)
        try:
            return normalize_vfs_path(cwd)
        except PathValidationError:
            return DEFAULT_WORKING_DIRECTORY

    async def _restore_cwd(self, path: str) -> None:
        restore_path = path if self._is_directory(path) else DEFAULT_WORKING_DIRECTORY
        if not self._is_directory(restore_path):
            self._ensure_workspace()
            restore_path = DEFAULT_WORKING_DIRECTORY
        await self._tool.execute(f"cd {_shell_quote(restore_path)}")

    async def _restore_cwd_safely(self, path: str) -> None:
        try:
            await self._restore_cwd(path)
        except Exception:  # noqa: BLE001 - cwd restoration is best-effort cleanup after arbitrary Bashkit failures.
            _LOGGER.warning("Failed to restore virtual workspace cwd", exc_info=True)

    def _require_directory(self, path: str) -> None:
        if not self._is_directory(path):
            raise PathValidationError("workingDirectory must be an existing directory")

    def _is_directory(self, path: str) -> bool:
        try:
            return self.exists(path) and self._tool.stat(path).get("file_type") == "directory"
        except Exception:  # noqa: BLE001 - Bashkit stat failures mean the path is not a usable directory.
            return False

    def _ensure_workspace(self) -> None:
        self._tool.mkdir(DEFAULT_WORKING_DIRECTORY, recursive=True)

    def _reject_descendant_destination(self, source: str, destination: str) -> None:
        if self._is_directory(source) and destination.startswith(f"{source.rstrip('/')}/"):
            raise VirtualWorkspaceError("destination cannot be inside source")

    def _ensure_workspace_size_limit(
        self,
        *,
        added_bytes: int,
        replaced_path: str | None = None,
    ) -> None:
        current_size = self._path_size(VFS_ROOT)
        replaced_size = self._path_size(replaced_path) if replaced_path else 0
        if current_size - replaced_size + added_bytes > MAX_WORKSPACE_BYTES:
            raise VirtualWorkspaceError("workspace size limit exceeded")

    def _raise_if_workspace_size_exceeded(self) -> None:
        if self._path_size(VFS_ROOT) > MAX_WORKSPACE_BYTES:
            raise VirtualWorkspaceError("workspace size limit exceeded")

    def _path_size(self, path: str | None) -> int:
        if path is None or not self.exists(path):
            return 0
        try:
            stat = self._tool.stat(path)
        except Exception:  # noqa: BLE001 - size accounting treats unreadable virtual paths as absent.
            return 0
        if stat.get("file_type") != "directory":
            return int(stat.get("size", 0))
        total = 0
        for entry in self._tool.read_dir(path):
            name = str(entry.get("name", ""))
            if not name:
                continue
            total += self._path_size(normalize_vfs_path(name, working_directory=path))
        return total
