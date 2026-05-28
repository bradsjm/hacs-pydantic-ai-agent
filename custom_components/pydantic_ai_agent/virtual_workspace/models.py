"""JSON-serializable virtual workspace result shapes."""

from typing import NotRequired, TypedDict


class ToolResult(TypedDict):
    """Base tool result."""

    ok: bool
    error: NotRequired[str]


class BashResult(ToolResult):
    """Result from a virtual bash command."""


    stdout: str
    stderr: str
    exitCode: int | None
    stdoutTruncated: bool
    stderrTruncated: bool


class ReadFileResult(ToolResult):
    """Result from reading a virtual file."""

    path: str
    content: str
    bytesRead: int
    truncated: bool


class WriteFileResult(ToolResult):
    """Result from writing a virtual file."""

    path: str
    bytesWritten: int


class DirectoryEntry(TypedDict):
    """Virtual directory entry."""

    name: str
    path: str
    type: str
    size: int
    mode: int | None
    created: str | None
    modified: str | None


class ReadDirectoryResult(ToolResult):
    """Result from reading a virtual directory."""

    path: str
    entries: list[DirectoryEntry]
    nextCursor: NotRequired[str]


class MetadataResult(ToolResult):
    """Virtual file metadata result."""

    path: str
    type: str
    size: int
    mode: int | None
    created: str | None
    modified: str | None


class CreateDirectoryResult(ToolResult):
    """Result from creating a virtual directory."""

    path: str
    created: bool


class RemoveResult(ToolResult):
    """Result from removing a virtual path."""

    path: str
    removed: bool


class CopyMoveResult(ToolResult):
    """Result from copying or moving a virtual path."""

    source: str
    destination: str


class PatchResult(TypedDict):
    """Result from applying a Codex-style patch."""

    success: bool
    changedFiles: list[str]
    errors: NotRequired[list[str]]
