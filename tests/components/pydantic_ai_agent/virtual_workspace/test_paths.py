"""Tests for virtual workspace path normalization."""

from custom_components.pydantic_ai_agent.virtual_workspace.errors import (
    PathValidationError,
)
from custom_components.pydantic_ai_agent.virtual_workspace.paths import (
    normalize_vfs_path,
    protected_replacement_path,
)
import pytest


@pytest.mark.parametrize(
    ("path", "working_directory", "expected"),
    [
        ("file.txt", "/workspace", "/workspace/file.txt"),
        ("./file.txt", "/workspace", "/workspace/file.txt"),
        ("nested/../file.txt", "/workspace", "/workspace/file.txt"),
        ("/workspace/./nested/../file.txt", "/temporary", "/workspace/file.txt"),
        ("../file.txt", "/workspace/nested", "/workspace/file.txt"),
        ("/", "/workspace", "/"),
    ],
)
def test_normalize_vfs_path_lexically_normalizes_inside_root(path: str, working_directory: str, expected: str) -> None:
    """Relative and absolute paths are normalized without touching a real FS."""
    assert normalize_vfs_path(path, working_directory=working_directory) == expected


@pytest.mark.parametrize(
    ("path", "working_directory"),
    [
        ("", "/workspace"),
        ("bad\x00path", "/workspace"),
        ("../../escape", "/workspace"),
        ("file.txt", "workspace"),
        ("/../escape", "/workspace"),
    ],
)
def test_normalize_vfs_path_rejects_invalid_or_escaping_paths(path: str, working_directory: str) -> None:
    """Paths must be non-empty, NUL-free, and remain below the VFS root."""
    with pytest.raises(PathValidationError):
        normalize_vfs_path(path, working_directory=working_directory)


@pytest.mark.parametrize(
    ("path", "expected"),
    [("/", True), ("/workspace", True), ("/workspace/file.txt", False)],
)
def test_protected_replacement_path_marks_only_workspace_roots(path: str, expected: bool) -> None:
    """Only the virtual root and default working directory are protected."""
    assert protected_replacement_path(path) is expected
