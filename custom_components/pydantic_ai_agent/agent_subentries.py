"""Shared helpers for subentry-backed agent platforms."""

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigSubentry
from homeassistant.exceptions import HomeAssistantError

from .const import CONF_FALLBACK_MODEL_REFS, CONF_PRIMARY_MODEL_REF

if TYPE_CHECKING:
    from . import PydanticAIAgentConfigEntry

_LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class ValidAgentSubentry[T]:
    """Resolved valid agent subentry data."""

    subentry: ConfigSubentry
    resolved: T


def iter_valid_agent_subentries[T](
    entry: "PydanticAIAgentConfigEntry",
    *,
    subentry_type: str,
    platform: str,
    resolver: Callable[["PydanticAIAgentConfigEntry", ConfigSubentry], T],
) -> Iterator[ValidAgentSubentry[T]]:
    """Yield subentries whose model references resolve for one platform."""
    for subentry in entry.subentries.values():
        if subentry.subentry_type != subentry_type:
            continue
        try:
            yield ValidAgentSubentry(
                subentry=subentry,
                resolved=resolver(entry, subentry),
            )
        except HomeAssistantError as err:
            _log_invalid_agent_subentry(entry, subentry, platform, err)
        except Exception:
            _LOGGER.exception(
                (
                    'Skipping invalid %s subentry "%s" (%s) while setting up '
                    '%s platform for entry "%s"; model_refs=%s'
                ),
                subentry.subentry_type,
                subentry.subentry_id,
                subentry.title,
                platform,
                entry.entry_id,
                _safe_model_refs(subentry.data),
            )


def _log_invalid_agent_subentry(
    entry: "PydanticAIAgentConfigEntry",
    subentry: ConfigSubentry,
    platform: str,
    err: HomeAssistantError,
) -> None:
    """Log safe context for an invalid agent subentry skipped at platform setup."""
    _LOGGER.warning(
        (
            'Skipping invalid %s subentry "%s" (%s) while setting up %s '
            'platform for entry "%s"; model_refs=%s: %s'
        ),
        subentry.subentry_type,
        subentry.subentry_id,
        subentry.title,
        platform,
        entry.entry_id,
        _safe_model_refs(subentry.data),
        err,
    )


def _safe_model_refs(data: Mapping[str, Any]) -> dict[str, str | list[str] | None]:
    """Return only model reference fields safe for diagnostics logs."""
    fallback_refs = data.get(CONF_FALLBACK_MODEL_REFS)
    return {
        CONF_PRIMARY_MODEL_REF: _safe_string(data.get(CONF_PRIMARY_MODEL_REF)),
        CONF_FALLBACK_MODEL_REFS: (
            [item for item in fallback_refs if isinstance(item, str)]
            if isinstance(fallback_refs, Sequence)
            and not isinstance(fallback_refs, str | bytes)
            else None
        ),
    }


def _safe_string(value: object) -> str | None:
    """Return a string value or None for log-safe model references."""
    return value if isinstance(value, str) else None
