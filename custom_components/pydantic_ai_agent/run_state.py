"""Pydantic AI run state containers."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .run_failures import _ToolProblem


@dataclass(frozen=True, kw_only=True)
class AgentRunOutcome:
    """Successful agent run data needed for metrics after validation."""

    output: object | None
    usage: Any
    duration: float
    model_profile: str
    model_profile_ref: str
    provider_subentry_id: str
    model_pricing: dict[str, float]
    run_recorder: Any | None = None


@dataclass
class _StreamRunState:
    """Mutable state shared with the ChatLog streaming delta generator."""

    result: Any | None = None
    emitted_deltas: bool = False
    latest_tool_problem: "_ToolProblem | None" = None
