"""Test virtual workspace path handling."""

import pytest

from custom_components.pydantic_ai_agent.virtual_workspace.const import (
    DEFAULT_WORKING_DIRECTORY,
    VFS_ROOT,
)
from custom_components.pydantic_ai_agent.virtual_workspace.errors import (
    PathValidationError,
)
from custom_components.pydantic_ai_agent.virtual_workspace.paths import (
    normalize_vfs_path,
    protected_replacement_path,
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("file.txt", "/workspace/file.txt"),
        ("./nested/../file.txt", "/workspace/file.txt"),
        ("../tmp/file.txt", "/tmp/file.txt"),
        ("/workspace/nested", "/workspace/nested"),
        ("/workspace/../tmp", "/tmp"),
    ],
)
def test_normalize_vfs_path(path: str, expected: str) -> None:
    """Test lexical path normalization inside the virtual root."""
    assert normalize_vfs_path(path) == expected


@pytest.mark.parametrize("path", ["", "/..", "../../escape", "bad\x00path"])
def test_normalize_vfs_path_rejects_invalid_paths(path: str) -> None:
    """Test invalid paths fail before reaching Bashkit."""
    with pytest.raises(PathValidationError):
        normalize_vfs_path(path)


@pytest.mark.parametrize("path", [VFS_ROOT, DEFAULT_WORKING_DIRECTORY])
def test_protected_replacement_paths(path: str) -> None:
    """Test protected workspace paths are identified exactly."""
    assert protected_replacement_path(path) is True


def test_nested_workspace_paths_are_not_protected() -> None:
    """Test only the virtual root and workspace root are protected."""
    assert protected_replacement_path("/workspace/file.txt") is False
