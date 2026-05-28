"""Diagnostics helpers for the Home Semantic Index."""

from .index import HomeSemanticIndex
from .manager import HomeSemanticIndexManager


def semantic_index_diagnostics(index: HomeSemanticIndex | None) -> dict[str, object]:
    """Return aggregate, redacted diagnostics for a semantic index."""
    if index is None:
        return {"loaded": False}
    return {"loaded": True, **index.diagnostics_summary()}


def semantic_manager_diagnostics(
    manager: HomeSemanticIndexManager | None,
) -> dict[str, object]:
    """Return aggregate, redacted diagnostics for a semantic index manager."""
    if manager is None:
        return {"loaded": False}
    return manager.diagnostics()
