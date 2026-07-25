"""Tool result problem extraction and logging utilities."""

from collections.abc import Mapping, Sequence
import logging

from pydantic_ai.messages import RetryPromptPart, ToolReturnPart

from ..observability.run_failures import _ToolProblem
from ..virtual_workspace.const import TOOL_RETURN_METADATA_SOURCE

_LOGGER = logging.getLogger(__name__)


def _tool_problem_from_part(
    part: ToolReturnPart | RetryPromptPart,
) -> _ToolProblem | None:
    """Return a safe tool problem summary from a Pydantic AI tool result part."""
    if isinstance(part, RetryPromptPart):
        return _ToolProblem(
            tool_name=part.tool_name,
            tool_call_id=part.tool_call_id,
            outcome="retry",
            reason=_safe_tool_result_reason(part.content, getattr(part, "metadata", None)),
        )
    outcome = getattr(part, "outcome", "success")
    reason = _safe_tool_result_reason(part.content, part.metadata)
    if outcome != "success":
        return _ToolProblem(
            tool_name=part.tool_name,
            tool_call_id=part.tool_call_id,
            outcome=outcome,
            reason=reason,
        )
    if isinstance(part.content, Mapping) and part.content.get("success") is False:
        return _ToolProblem(
            tool_name=part.tool_name,
            tool_call_id=part.tool_call_id,
            outcome="failed",
            reason=reason,
        )
    return None


def _safe_tool_result_reason(content: object, metadata: object) -> str | None:
    """Extract a short safe reason from structured tool failure content."""
    if not (isinstance(metadata, Mapping) and metadata.get("source") == TOOL_RETURN_METADATA_SOURCE):
        return None
    reason: object | None = None
    if isinstance(content, Mapping):
        errors = content.get("errors")
        if isinstance(errors, Sequence) and not isinstance(errors, str | bytes):
            reason = next((item for item in errors if isinstance(item, str)), None)
        if reason is None:
            for key in ("error", "message"):
                value = content.get(key)
                if isinstance(value, str):
                    reason = value
                    break
    elif isinstance(content, str):
        reason = content
    if not isinstance(reason, str) or not reason:
        return None
    return reason[:200]


def _log_tool_problem(problem: _ToolProblem) -> None:
    """Log a non-terminal tool problem without exposing tool arguments."""
    _LOGGER.warning(
        'Pydantic AI tool "%s" returned %s for call "%s": %s',
        problem.tool_name or "unknown",
        problem.outcome,
        problem.tool_call_id or "unknown",
        problem.reason or "no safe detail provided",
    )
