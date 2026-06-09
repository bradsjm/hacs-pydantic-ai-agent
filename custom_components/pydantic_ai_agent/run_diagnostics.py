"""Run diagnostics capture and bounding helpers."""

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import UTC
from itertools import islice
from time import perf_counter
from typing import Any, cast
from uuid import uuid4

from homeassistant.util import dt as dt_util

_STRING_EDGE_CHARS = 4096
_SEQUENCE_EDGE_ITEMS = 100
_MAPPING_EDGE_ITEMS = 50
_MAX_DEPTH = 10


class RunDiagnosticsRecorder:
    """Collect one ordered, bounded last-run diagnostics timeline."""

    def __init__(
        self,
        *,
        subentry_id: str,
        subentry_type: str,
        conversation_id: str | None,
    ) -> None:
        """Initialize the recorder."""
        self.run_id = uuid4().hex
        self.subentry_id = subentry_id
        self.subentry_type = subentry_type
        self.conversation_id = conversation_id
        self.started_at = dt_util.utcnow().replace(tzinfo=UTC)
        self._started_monotonic = perf_counter()
        self._last_elapsed_ms = 0.0
        self._sequence = 0
        self._timeline_head: list[dict[str, Any]] = []
        self._timeline_tail: deque[dict[str, Any]] = deque(maxlen=_SEQUENCE_EDGE_ITEMS)

    def record(
        self,
        *,
        phase: str,
        event: str,
        source: str = "integration",
        data: object | None = None,
    ) -> None:
        """Record one ordered timeline event."""
        self._sequence += 1
        now = dt_util.utcnow().replace(tzinfo=UTC)
        elapsed_ms = (perf_counter() - self._started_monotonic) * 1000
        timeline_event = {
            "seq": self._sequence,
            "timestamp": now.isoformat(),
            "elapsed_ms": round(elapsed_ms, 3),
            "delta_ms": round(elapsed_ms - self._last_elapsed_ms, 3),
            "phase": phase,
            "source": source,
            "event": event,
            "data": bound_diagnostics_data(data) if data is not None else {},
        }
        if len(self._timeline_head) < _SEQUENCE_EDGE_ITEMS:
            self._timeline_head.append(timeline_event)
        else:
            self._timeline_tail.append(timeline_event)
        self._last_elapsed_ms = elapsed_ms

    def payload(
        self,
        *,
        status: str,
        summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the bounded diagnostics payload for the run."""
        finished_at = dt_util.utcnow().replace(tzinfo=UTC)
        return cast(
            dict[str, Any],
            bound_diagnostics_data(
                {
                    "schema_version": 1,
                    "run_id": self.run_id,
                    "subentry_id": self.subentry_id,
                    "subentry_type": self.subentry_type,
                    "conversation_id": self.conversation_id,
                    "status": status,
                    "started_at": self.started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "duration_ms": round(
                        (finished_at - self.started_at).total_seconds() * 1000, 3
                    ),
                    "timeline_event_count": self._sequence,
                    "timeline": self._timeline(),
                    "summary": dict(summary or {}),
                }
            ),
        )

    def _timeline(self) -> list[dict[str, Any]] | dict[str, Any]:
        """Return a bounded timeline without retaining all middle events."""
        tail = list(self._timeline_tail)
        if self._sequence <= len(self._timeline_head) + len(tail):
            return self._timeline_head + tail
        return {
            "__diagnostics_bounded__": "sequence",
            "total_count": self._sequence,
            "head": self._timeline_head,
            "tail": tail,
            "omitted_middle_count": self._sequence
            - len(self._timeline_head)
            - len(tail),
        }


def bound_diagnostics_data(value: object, *, _depth: int = 0) -> object:
    """Return a JSON-safe diagnostics value with large content bounded."""
    if _depth > _MAX_DEPTH:
        return {"__diagnostics_bounded__": "max_depth", "type": type(value).__name__}
    if value is None or isinstance(value, str | int | float | bool):
        return _bound_string(value) if isinstance(value, str) else value
    if isinstance(value, bytes | bytearray):
        return _bound_string(value.decode(errors="replace"))
    if isinstance(value, BaseException):
        return {
            "type": type(value).__name__,
            "message": bound_diagnostics_data(str(value), _depth=_depth + 1),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: bound_diagnostics_data(
                getattr(value, field.name), _depth=_depth + 1
            )
            for field in fields(value)
        }
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return bound_diagnostics_data(model_dump(mode="json"), _depth=_depth + 1)
        except Exception:
            pass
    if isinstance(value, Mapping):
        return _bound_mapping(value, _depth=_depth)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return _bound_sequence(value, _depth=_depth)
    return {
        "type": type(value).__name__,
        "repr": _bound_string(repr(value)),
    }


def _bound_string(value: str) -> str | dict[str, Any]:
    """Return a string or a head/tail representation for large strings."""
    if len(value) <= _STRING_EDGE_CHARS * 2:
        return value
    return {
        "__diagnostics_bounded__": "string",
        "original_length": len(value),
        "head": value[:_STRING_EDGE_CHARS],
        "tail": value[-_STRING_EDGE_CHARS:],
        "omitted_chars": len(value) - (_STRING_EDGE_CHARS * 2),
    }


def _bound_sequence(value: Sequence[Any], *, _depth: int) -> object:
    """Return a list or a head/tail representation for large sequences."""
    total = len(value)
    if total <= _SEQUENCE_EDGE_ITEMS * 2:
        return [bound_diagnostics_data(item, _depth=_depth + 1) for item in value]
    return {
        "__diagnostics_bounded__": "sequence",
        "total_count": total,
        "head": [
            bound_diagnostics_data(item, _depth=_depth + 1)
            for item in value[:_SEQUENCE_EDGE_ITEMS]
        ],
        "tail": [
            bound_diagnostics_data(item, _depth=_depth + 1)
            for item in value[-_SEQUENCE_EDGE_ITEMS:]
        ],
        "omitted_middle_count": total - (_SEQUENCE_EDGE_ITEMS * 2),
    }


def _bound_mapping(value: Mapping[Any, Any], *, _depth: int) -> object:
    """Return a mapping or a head/tail representation for large mappings."""
    total = len(value)
    if total <= _MAPPING_EDGE_ITEMS * 2:
        return {
            str(key): bound_diagnostics_data(item, _depth=_depth + 1)
            for key, item in value.items()
        }
    head_items = list(islice(value.items(), _MAPPING_EDGE_ITEMS))
    tail_items = deque(value.items(), maxlen=_MAPPING_EDGE_ITEMS)
    return {
        "__diagnostics_bounded__": "mapping",
        "total_count": total,
        "head": {
            str(key): bound_diagnostics_data(item, _depth=_depth + 1)
            for key, item in head_items
        },
        "tail": {
            str(key): bound_diagnostics_data(item, _depth=_depth + 1)
            for key, item in tail_items
        },
        "omitted_middle_count": total - (_MAPPING_EDGE_ITEMS * 2),
    }
