"""Virtual workspace internal exceptions."""


class VirtualWorkspaceError(Exception):
    """Base class for expected virtual workspace errors."""


class PathValidationError(VirtualWorkspaceError):
    """Raised when a virtual path is invalid or unsafe."""


class ConfirmationRequiredError(VirtualWorkspaceError):
    """Raised when a destructive operation requires explicit confirmation."""


class PatchApplyError(VirtualWorkspaceError):
    """Raised when a patch cannot be applied."""
