"""Auth-failure helpers used during entity agent runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from homeassistant.core import HomeAssistant
from pydantic_ai.exceptions import ModelHTTPError

from ..models.provider_validation import ProviderValidationError
from ..repair_issues import (
    async_create_provider_auth_issue,
    async_delete_provider_auth_issue,
)

if TYPE_CHECKING:
    from ..runtime.types import PydanticAIAgentConfigEntry


def _join_instructions(*parts: str | None) -> str | None:
    """Join optional instruction blocks for one agent run."""
    instructions = [part.strip() for part in parts if part and part.strip()]
    return "\n\n".join(instructions) if instructions else None


def _has_provider_auth_failure(
    entry: PydanticAIAgentConfigEntry, provider_subentry_id: str
) -> bool:
    """Return if runtime has a current auth issue for a provider."""
    return bool(
        entry.runtime_data.runtime_provider_auth_failures.get(provider_subentry_id)
    )


def _record_runtime_auth_failure(
    entry: PydanticAIAgentConfigEntry,
    profile: Any,  # noqa: ANN401
) -> None:
    """Record a runtime auth issue for one provider/profile pair."""
    failures = entry.runtime_data.runtime_provider_auth_failures.setdefault(
        profile.provider_subentry_id, []
    )
    if profile.ref not in failures:
        failures.append(profile.ref)


def _clear_runtime_auth_failure(
    hass: HomeAssistant,
    entry: PydanticAIAgentConfigEntry,
    profile: Any,  # noqa: ANN401
) -> None:
    """Clear a runtime auth issue for one provider/profile pair when safe."""
    _clear_runtime_auth_failure_for_ref(
        hass, entry, profile.provider_subentry_id, profile.ref
    )


def _clear_runtime_auth_failure_for_ref(
    hass: HomeAssistant,
    entry: PydanticAIAgentConfigEntry,
    provider_subentry_id: str,
    profile_ref: str,
) -> None:
    """Clear a runtime auth issue by provider/profile reference when safe."""
    failures = entry.runtime_data.runtime_provider_auth_failures.get(
        provider_subentry_id
    )
    if failures is not None and profile_ref in failures:
        failures.remove(profile_ref)
        if not failures:
            entry.runtime_data.runtime_provider_auth_failures.pop(
                provider_subentry_id, None
            )
    if not _has_provider_auth_failure(entry, provider_subentry_id):
        async_delete_provider_auth_issue(hass, entry, provider_subentry_id)


def _async_create_runtime_auth_issue(
    hass: HomeAssistant,
    entry: PydanticAIAgentConfigEntry,
    profile: Any,  # noqa: ANN401
    err: BaseException,
    message: str,
) -> bool:
    """Create a provider auth repair issue for runtime credential failures."""
    status_code = _auth_status_code(err)
    if status_code is None:
        return False
    reason = "invalid_auth" if status_code == 401 else "permission_denied"
    _record_runtime_auth_failure(entry, profile)
    async_create_provider_auth_issue(
        hass,
        entry,
        profile.provider_subentry_id,
        profile.provider_title,
        ProviderValidationError(reason, message, status_code),
    )
    return True


def _auth_status_code(err: BaseException) -> int | None:
    """Return 401/403 from a runtime error cause chain when present."""
    seen: set[int] = set()
    current: BaseException | None = err
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ModelHTTPError):
            status_code = getattr(current, "status_code", None)
            if status_code in {401, 403}:
                return cast(int, status_code)
        current = current.__cause__ or current.__context__
    return None
