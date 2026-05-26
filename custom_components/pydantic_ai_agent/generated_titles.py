"""Default title generation for Pydantic AI Agent config flows."""

from collections.abc import Iterable

from names_generator import generate_name

DEFAULT_AI_TASK_TITLE_SUFFIX = "AI Task"
DEFAULT_AGENT_TITLE_SUFFIX = "Agent"
DEFAULT_SERVICE_TITLE_SUFFIX = "Service"
DEFAULT_WORKSPACE_TITLE_SUFFIX = "Workspace"

_MAX_TITLE_GENERATION_ATTEMPTS = 5


def generated_default_title(
    suffix: str, existing_titles: Iterable[str] = ()
) -> str:
    """Return a capitalized generated title with a fixed suffix."""
    existing = {title.casefold() for title in existing_titles}
    title = ""
    for _ in range(_MAX_TITLE_GENERATION_ATTEMPTS):
        title = _format_title(generate_name(style="capital"), suffix)
        if title.casefold() not in existing:
            return title
    return title or suffix


def _format_title(name: object, suffix: str) -> str:
    """Return a generated name normalized to capitalized words plus suffix."""
    words = str(name or "").replace("_", " ").replace("-", " ").split()
    if not words:
        return suffix
    return f"{' '.join(word.capitalize() for word in words)} {suffix}"
