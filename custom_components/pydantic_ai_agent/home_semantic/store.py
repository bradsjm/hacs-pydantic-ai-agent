"""Entry-scoped correction and usage memory for semantic home ranking."""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Final

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from ..const import DOMAIN
from .index import normalize_tokens

_STORE_VERSION: Final = 1
_MAX_SUCCESS_COUNT: Final = 20
_EXPLICIT_CORRECTION_BOOST: Final = 0.35
_SUCCESS_BOOST_STEP: Final = 0.01
_MAX_SUCCESS_BOOST: Final = 0.08
_AMBIGUITY_PENALTY: Final = 0.2


class HomeSemanticMemory:
    """Local user-inspectable memory signals for semantic target ranking."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry[Any]) -> None:
        """Initialize entry-owned memory storage."""
        self._hass = hass
        self._entry = entry
        self._store: Store[dict[str, Any]] = Store(
            hass,
            _STORE_VERSION,
            f"{DOMAIN}.home_semantic.{entry.entry_id}",
        )
        self._data = _empty_data()
        self._loaded = False
        self._load_error_type: str | None = None
        self._stale_ignored_count = 0

    async def async_load(self) -> None:
        """Load memory data from Home Assistant storage."""
        try:
            stored = await self._store.async_load()
        except Exception as err:  # noqa: BLE001 - diagnostics expose type only.
            self._load_error_type = type(err).__name__
            self._loaded = True
            return
        self._data = _coerce_data(stored)
        self._loaded = True
        self._load_error_type = None

    async def async_remove(self) -> None:
        """Remove persisted memory for a deleted workspace."""
        await self._store.async_remove()
        self._data = _empty_data()
        self._loaded = False

    @callback
    def add_correction(
        self,
        *,
        phrase: str,
        action: str | None,
        entity_id: str,
        area_id: str | None = None,
        domain: str | None = None,
    ) -> None:
        """Add or update an explicit correction record."""
        normalized_phrase = _normalized_phrase(phrase)
        if not normalized_phrase:
            return
        now = dt_util.utcnow().isoformat()
        records = self._data["corrections"]
        for record in records:
            if (
                record.get("phrase") == normalized_phrase
                and record.get("action") == action
                and record.get("area_id") == area_id
                and record.get("domain") == domain
            ):
                record["entity_id"] = entity_id
                record["updated_at"] = now
                record["uses"] = int(record.get("uses", 0)) + 1
                self._mark_updated(now)
                return
        records.append(
            {
                "kind": "correction",
                "phrase": normalized_phrase,
                "action": action,
                "area_id": area_id,
                "domain": domain,
                "entity_id": entity_id,
                "created_at": now,
                "updated_at": now,
                "uses": 1,
            }
        )
        self._mark_updated(now)

    @callback
    def record_success(self, *, phrase: str | None, action: str, entity_id: str) -> None:
        """Record a compact successful resolution signal."""
        if not phrase:
            return
        phrase_tokens = list(normalize_tokens(phrase))
        if not phrase_tokens:
            return
        now = dt_util.utcnow().isoformat()
        records = self._data["successes"]
        for record in records:
            if (
                record.get("phrase_tokens") == phrase_tokens
                and record.get("action") == action
                and record.get("entity_id") == entity_id
            ):
                record["count"] = min(
                    _MAX_SUCCESS_COUNT,
                    int(record.get("count", 0)) + 1,
                )
                record["last_used_at"] = now
                self._mark_updated(now)
                return
        records.append(
            {
                "kind": "success",
                "phrase_tokens": phrase_tokens,
                "action": action,
                "entity_id": entity_id,
                "count": 1,
                "last_used_at": now,
            }
        )
        self._mark_updated(now)

    @callback
    def record_ambiguity(
        self, *, phrase: str, candidate_entity_ids: Iterable[str]
    ) -> None:
        """Record repeated ambiguity without storing full prompts."""
        phrase_tokens = list(normalize_tokens(phrase))
        candidates = sorted(set(candidate_entity_ids))
        if not phrase_tokens or len(candidates) < 2:
            return
        now = dt_util.utcnow().isoformat()
        records = self._data["ambiguities"]
        for record in records:
            if (
                record.get("phrase_tokens") == phrase_tokens
                and record.get("candidate_entity_ids") == candidates
            ):
                record["count"] = int(record.get("count", 0)) + 1
                record["last_seen_at"] = now
                self._mark_updated(now)
                return
        records.append(
            {
                "kind": "ambiguity",
                "phrase_tokens": phrase_tokens,
                "candidate_entity_ids": candidates,
                "count": 1,
                "last_seen_at": now,
            }
        )
        self._mark_updated(now)

    def ranking_adjustments(
        self,
        *,
        phrase: str,
        action: str | None,
        area_id: str | None,
        domain: str | None,
        candidate_entity_ids: Iterable[str],
    ) -> dict[str, tuple[float, tuple[str, ...]]]:
        """Return bounded memory-derived confidence adjustments per candidate."""
        candidate_ids = set(candidate_entity_ids)
        if not candidate_ids:
            return {}
        normalized_phrase = _normalized_phrase(phrase)
        phrase_tokens = list(normalize_tokens(phrase))
        boosts: dict[str, float] = defaultdict(float)
        reasons: dict[str, list[str]] = defaultdict(list)
        for record in self._data["corrections"]:
            entity_id = _record_entity_id(record)
            if entity_id not in candidate_ids:
                self._stale_ignored_count += int(entity_id is not None)
                continue
            if record.get("phrase") != normalized_phrase:
                continue
            if record.get("action") not in (None, action):
                continue
            if area_id is not None and record.get("area_id") not in (None, area_id):
                continue
            if domain is not None and record.get("domain") not in (None, domain):
                continue
            boosts[entity_id] += _EXPLICIT_CORRECTION_BOOST
            reasons[entity_id].append("memory_correction")
        for record in self._data["successes"]:
            entity_id = _record_entity_id(record)
            if entity_id not in candidate_ids:
                self._stale_ignored_count += int(entity_id is not None)
                continue
            if record.get("phrase_tokens") != phrase_tokens:
                continue
            if record.get("action") != action:
                continue
            boosts[entity_id] += min(
                _MAX_SUCCESS_BOOST,
                int(record.get("count", 0)) * _SUCCESS_BOOST_STEP,
            )
            reasons[entity_id].append("memory_success")
        for record in self._data["ambiguities"]:
            if record.get("phrase_tokens") != phrase_tokens:
                continue
            ambiguous_ids = set(_string_list(record.get("candidate_entity_ids")))
            if len(candidate_ids & ambiguous_ids) < 2:
                continue
            for entity_id in candidate_ids & ambiguous_ids:
                boosts[entity_id] -= _AMBIGUITY_PENALTY
                reasons[entity_id].append("memory_ambiguity_penalty")
        return {
            entity_id: (adjustment, tuple(reasons[entity_id]))
            for entity_id, adjustment in boosts.items()
        }

    def diagnostics(self) -> dict[str, object]:
        """Return aggregate diagnostics without raw correction contents."""
        return {
            "loaded": self._loaded,
            "schema_version": _STORE_VERSION,
            "correction_count": len(self._data["corrections"]),
            "usage_signal_count": len(self._data["successes"]),
            "ambiguity_penalty_count": len(self._data["ambiguities"]),
            "stale_ignored_count": self._stale_ignored_count,
            "last_updated_at": self._data.get("last_updated_at"),
            "load_error_type": self._load_error_type,
        }

    @callback
    def _mark_updated(self, timestamp: str) -> None:
        """Update metadata and persist asynchronously."""
        self._data["last_updated_at"] = timestamp
        self._entry.async_create_background_task(
            self._hass,
            self._store.async_save(self._data),
            name=f"{self._entry.title} Home Semantic Memory save",
        )


def _empty_data() -> dict[str, Any]:
    """Return an empty store payload."""
    return {
        "schema_version": _STORE_VERSION,
        "corrections": [],
        "successes": [],
        "ambiguities": [],
        "last_updated_at": None,
    }


def _coerce_data(stored: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a validated store payload, dropping malformed values."""
    data = _empty_data()
    if not isinstance(stored, Mapping):
        return data
    data["corrections"] = [
        record
        for record in _mapping_list(stored.get("corrections"))
        if _record_entity_id(record) is not None and isinstance(record.get("phrase"), str)
    ]
    data["successes"] = [
        record
        for record in _mapping_list(stored.get("successes"))
        if _record_entity_id(record) is not None
        and _string_list(record.get("phrase_tokens"))
    ]
    data["ambiguities"] = [
        record
        for record in _mapping_list(stored.get("ambiguities"))
        if _string_list(record.get("phrase_tokens"))
        and len(_string_list(record.get("candidate_entity_ids"))) >= 2
    ]
    last_updated = stored.get("last_updated_at")
    data["last_updated_at"] = last_updated if isinstance(last_updated, str) else None
    return data


def _mapping_list(value: object) -> list[dict[str, Any]]:
    """Return mapping items from a stored list."""
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _string_list(value: object) -> list[str]:
    """Return string items from a stored list."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _record_entity_id(record: Mapping[str, Any]) -> str | None:
    """Return a record entity id if valid."""
    entity_id = record.get("entity_id")
    return entity_id if isinstance(entity_id, str) and entity_id else None


def _normalized_phrase(phrase: str) -> str:
    """Return the compact normalized phrase used for explicit corrections."""
    return " ".join(normalize_tokens(phrase))
