"""Lexical virtual filesystem path handling."""

from .const import DEFAULT_WORKING_DIRECTORY, VFS_ROOT
from .errors import PathValidationError

_PROTECTED_REPLACEMENT_PATHS = {VFS_ROOT, DEFAULT_WORKING_DIRECTORY}


def normalize_vfs_path(
    path: str,
    *,
    working_directory: str = DEFAULT_WORKING_DIRECTORY,
) -> str:
    """Normalize a path inside the virtual filesystem."""
    if not isinstance(path, str) or not path:
        raise PathValidationError("path is required")
    if "\x00" in path:
        raise PathValidationError("path must not contain NUL bytes")

    working_directory = _normalize_absolute(working_directory)
    combined = path if path.startswith(VFS_ROOT) else f"{working_directory}/{path}"
    return _normalize_absolute(combined)


def protected_replacement_path(path: str) -> bool:
    """Return whether a path must not be removed or replaced."""
    return path in _PROTECTED_REPLACEMENT_PATHS


def _normalize_absolute(path: str) -> str:
    if not path.startswith(VFS_ROOT):
        raise PathValidationError("working directory must be absolute")
    parts: list[str] = []
    for part in path.split(VFS_ROOT):
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                raise PathValidationError("path escapes the virtual root")
            parts.pop()
            continue
        parts.append(part)
    if not parts:
        return VFS_ROOT
    return f"/{'/'.join(parts)}"
