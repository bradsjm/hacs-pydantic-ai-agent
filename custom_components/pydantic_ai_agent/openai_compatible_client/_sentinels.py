"""OpenAI-style sentinel values."""

from typing import final


@final
class NotGiven:
    """Sentinel used for omitted optional parameters."""

    def __bool__(self) -> bool:
        """Return false for OpenAI SDK compatible truthiness."""
        return False

    def __repr__(self) -> str:
        """Return the SDK-style representation."""
        return "NOT_GIVEN"


@final
class Omit:
    """Sentinel used to explicitly omit a value from serialization."""

    def __bool__(self) -> bool:
        """Return false for OpenAI SDK compatible truthiness."""
        return False

    def __repr__(self) -> str:
        """Return the SDK-style representation."""
        return "omit"


NOT_GIVEN = NotGiven()
omit = Omit()


def is_omitted(value: object) -> bool:
    """Return whether a value should be omitted from request payloads."""
    return isinstance(value, NotGiven | Omit)
