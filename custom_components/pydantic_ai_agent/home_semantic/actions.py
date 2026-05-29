"""Shared action policy for the Home Semantic Index."""

from typing import Final

from homeassistant.components.climate.const import ClimateEntityFeature
from homeassistant.components.cover import CoverEntityFeature
from homeassistant.core import State

ACTION_TURN_ON: Final = "turn_on"
ACTION_TURN_OFF: Final = "turn_off"
ACTION_TOGGLE: Final = "toggle"
ACTION_ACTIVATE: Final = "activate"
ACTION_OPEN: Final = "open"
ACTION_CLOSE: Final = "close"
ACTION_SET_TEMPERATURE: Final = "set_temperature"
ACTION_LOCK: Final = "lock"
ACTION_UNLOCK: Final = "unlock"

SUPPORTED_ACTIONS: Final = (
    ACTION_TURN_ON,
    ACTION_TURN_OFF,
    ACTION_TOGGLE,
    ACTION_ACTIVATE,
    ACTION_OPEN,
    ACTION_CLOSE,
    ACTION_SET_TEMPERATURE,
    ACTION_LOCK,
    ACTION_UNLOCK,
)

ACTION_CAPABILITIES: Final[dict[str, tuple[str, ...]]] = {
    ACTION_TURN_ON: ("lights", "switches", "fans", "media players"),
    ACTION_TURN_OFF: ("lights", "switches", "fans", "media players"),
    ACTION_TOGGLE: ("lights", "switches"),
    ACTION_OPEN: ("covers",),
    ACTION_CLOSE: ("covers",),
    ACTION_SET_TEMPERATURE: ("climate",),
    ACTION_LOCK: ("locks",),
    ACTION_UNLOCK: ("locks",),
    ACTION_ACTIVATE: ("scenes", "scripts"),
}

SUPPORTED_CONTROL_DOMAINS: Final = {
    "climate",
    "cover",
    "group",
    "light",
    "lock",
    "scene",
    "script",
    "switch",
}

LIVE_CONTROL_ALLOWED: Final[set[tuple[str, str]]] = {
    ("light", ACTION_TURN_ON),
    ("light", ACTION_TURN_OFF),
    ("light", ACTION_TOGGLE),
    ("switch", ACTION_TURN_ON),
    ("switch", ACTION_TURN_OFF),
    ("switch", ACTION_TOGGLE),
    ("scene", ACTION_ACTIVATE),
    ("script", ACTION_ACTIVATE),
    ("cover", ACTION_OPEN),
    ("cover", ACTION_CLOSE),
    ("lock", ACTION_LOCK),
    ("lock", ACTION_UNLOCK),
    ("climate", ACTION_SET_TEMPERATURE),
}

SERVICE_BY_DOMAIN_ACTION: Final[dict[tuple[str, str], tuple[str, str]]] = {
    ("light", ACTION_TURN_ON): ("light", "turn_on"),
    ("light", ACTION_TURN_OFF): ("light", "turn_off"),
    ("light", ACTION_TOGGLE): ("light", "toggle"),
    ("switch", ACTION_TURN_ON): ("switch", "turn_on"),
    ("switch", ACTION_TURN_OFF): ("switch", "turn_off"),
    ("switch", ACTION_TOGGLE): ("switch", "toggle"),
    ("scene", ACTION_ACTIVATE): ("scene", "turn_on"),
    ("script", ACTION_ACTIVATE): ("script", "turn_on"),
    ("cover", ACTION_OPEN): ("cover", "open_cover"),
    ("cover", ACTION_CLOSE): ("cover", "close_cover"),
    ("lock", ACTION_LOCK): ("lock", "lock"),
    ("lock", ACTION_UNLOCK): ("lock", "unlock"),
    ("climate", ACTION_SET_TEMPERATURE): ("climate", "set_temperature"),
}

def service_for_action(domain: str, action: str) -> tuple[str, str]:
    """Return the HA service domain/name for a constrained action."""
    return SERVICE_BY_DOMAIN_ACTION.get((domain, action), ("", ""))


def state_supports_action(state: State, action: str) -> bool:
    """Return whether this state advertises support for the action."""
    if service_for_action(state.domain, action)[0] == "":
        return False
    supported_features = int(state.attributes.get("supported_features", 0))
    if state.domain == "cover" and action == ACTION_OPEN:
        return bool(supported_features & CoverEntityFeature.OPEN)
    if state.domain == "cover" and action == ACTION_CLOSE:
        return bool(supported_features & CoverEntityFeature.CLOSE)
    if state.domain == "climate" and action == ACTION_SET_TEMPERATURE:
        return bool(supported_features & ClimateEntityFeature.TARGET_TEMPERATURE)
    return True


def live_control_allowed(domain: str, action: str) -> bool:
    """Return whether the LLM control tool may execute this action live."""
    return (domain, action) in LIVE_CONTROL_ALLOWED
