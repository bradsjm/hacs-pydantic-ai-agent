"""Per-run virtual workspace toolset."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic_ai.toolsets import AbstractToolset

from ..const import CONF_VIRTUAL_WORKSPACE_ENABLED
from .tools import build_virtual_workspace_toolset
from .workspace import VirtualWorkspace

VIRTUAL_WORKSPACE_INSTRUCTIONS = """A temporary in-memory virtual workspace is available for this run.
Use `/workspace` as the default working directory. Files, directories, shell state, and environment changes persist only during this model run and are discarded after the response. The workspace has no host filesystem mounts and no network access. Destructive file operations require `confirm: true`."""


@dataclass(frozen=True, kw_only=True)
class VirtualWorkspaceParts:
    """Runtime parts for one virtual workspace run."""

    toolsets: tuple[AbstractToolset[Any], ...]
    instructions: str


def virtual_workspace_enabled(data: Mapping[str, Any]) -> bool:
    """Return whether virtual workspace tools are enabled for a subentry."""
    return data.get(CONF_VIRTUAL_WORKSPACE_ENABLED) is True


def virtual_workspace_parts() -> VirtualWorkspaceParts:
    """Create a fresh virtual workspace toolset for one model run."""
    workspace = VirtualWorkspace()
    return VirtualWorkspaceParts(
        toolsets=(build_virtual_workspace_toolset(workspace),),
        instructions=VIRTUAL_WORKSPACE_INSTRUCTIONS,
    )
