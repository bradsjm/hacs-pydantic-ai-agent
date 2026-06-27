"""Helper utilities for the virtual workspace implementation."""

from collections.abc import Mapping
from typing import Any

from homeassistant.util import dt as dt_util

from .errors import (
    ConfirmationRequiredError,
    PathValidationError,
    VirtualWorkspaceError,
)
from .models import BashResult
from .paths import normalize_vfs_path, protected_replacement_path


def require_confirmed_overwrite(path: str, overwrite: bool, confirm: bool) -> None:
    """Require explicit confirmation before replacing an existing path."""
    if protected_replacement_path(path):
        raise PathValidationError("protected workspace paths cannot be replaced")
    if not overwrite:
        raise ConfirmationRequiredError(
            "target exists; set overwrite=true to replace it"
        )
    if not confirm:
        raise ConfirmationRequiredError("overwrite requires confirm=true")


def metadata(data: Mapping[str, Any]) -> dict[str, str | int | None]:
    """Return normalized JSON-safe metadata for a filesystem entry."""
    return {
        "type": str(data.get("file_type", "unknown")),
        "size": int(data.get("size", 0)),
        "mode": int(data["mode"]) if data.get("mode") is not None else None,
        "created": timestamp(data.get("created")),
        "modified": timestamp(data.get("modified")),
    }


def timestamp(value: object) -> str | None:
    """Convert numeric timestamps to ISO-8601 strings."""
    if not isinstance(value, int | float):
        return None
    return dt_util.utc_from_timestamp(float(value)).isoformat()


def parse_cursor(cursor: str | None) -> int:
    """Parse a directory cursor into a non-negative offset."""
    if cursor is None or cursor == "":
        return 0
    try:
        offset = int(cursor)
    except ValueError as err:
        raise VirtualWorkspaceError("cursor must be a numeric offset") from err
    if offset < 0:
        raise VirtualWorkspaceError("cursor must be non-negative")
    return offset


def truncate_text(value: str, limit: int) -> tuple[str, bool]:
    """Truncate text by encoded byte size."""
    encoded = value.encode()
    if len(encoded) <= limit:
        return value, False
    return encoded[:limit].decode(errors="replace"), True


def safe_normalized_path(path: object) -> str:
    """Return a normalized path or an empty string on failure."""
    try:
        return normalize_vfs_path(path if isinstance(path, str) else "")
    except PathValidationError:
        return ""


def bash_error(error: str) -> BashResult:
    """Return a structured bash error result."""
    return {
        "ok": False,
        "stdout": "",
        "stderr": "",
        "exitCode": None,
        "stdoutTruncated": False,
        "stderrTruncated": False,
        "error": error,
    }


def shell_quote(value: str) -> str:
    """Single-quote shell text for bashkit commands."""
    return "'" + value.replace("'", "'\\''") + "'"
