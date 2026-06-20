"""Integration-owned tool failure exceptions."""

from homeassistant.exceptions import HomeAssistantError


class HAToolRetryExhausted(HomeAssistantError):
    """Raised when a retryable Home Assistant tool failure exhausts retries."""

    def __init__(self, *, tool_name: str, attempts: int, reason: str | None) -> None:
        """Initialize a safe user-facing HA tool retry exhaustion error."""
        self.tool_name = tool_name
        self.attempts = attempts
        self.reason = reason
        super().__init__(
            f'Home Assistant tool "{tool_name}" failed after {attempts} '
            "attempts. Check the requested tool arguments and try again."
        )
