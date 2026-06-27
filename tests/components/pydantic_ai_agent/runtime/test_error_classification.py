import errno
from itertools import pairwise
import socket
import ssl

from custom_components.pydantic_ai_agent.runtime.error_classification import (
    connection_failure_message,
    has_connection_failure,
    iter_exception_chain,
)
import httpx
import pytest


def _chain(*errs: BaseException) -> BaseException:
    for parent, child in pairwise(errs):
        parent.__cause__ = child
    return errs[0]


class CustomTimeoutError(TimeoutError):
    pass


class CustomConnectError(httpx.ConnectError):
    pass


@pytest.mark.parametrize(
    ("err", "message"),
    [
        (TimeoutError(), "Request timed out."),
        (httpx.TimeoutException("timeout"), "Request timed out."),
        (socket.gaierror(), "Host not found."),
        (ssl.SSLError(), "TLS error."),
        (OSError(errno.ECONNREFUSED, "refused"), "Connection refused."),
        (OSError(errno.ENETUNREACH, "network"), "Network unreachable."),
        (OSError(errno.EHOSTUNREACH, "host"), "Network unreachable."),
        (httpx.ConnectError("connect"), "Connection failed."),
        (CustomTimeoutError(), "Request timed out."),
        (CustomConnectError("custom connect"), "Connection failed."),
    ],
)
def test_connection_failure_message_classifies_transport_failures(
    err: BaseException, message: str
) -> None:
    assert connection_failure_message(err) == message
    assert has_connection_failure(err) is True


def test_connection_failure_message_returns_none_for_non_failure() -> None:
    err = ValueError("not a connection failure")

    assert connection_failure_message(err) is None
    assert has_connection_failure(err) is False


def test_connection_failure_message_detects_nested_cause_chain() -> None:
    err = _chain(
        RuntimeError("wrapper"),
        ValueError("inner wrapper"),
        httpx.ConnectError("connect"),
    )

    assert connection_failure_message(err) == "Connection failed."


def test_connection_failure_message_cause_wins_over_context() -> None:
    err = RuntimeError("wrapper")
    err.__cause__ = TimeoutError()
    err.__context__ = socket.gaierror()

    assert connection_failure_message(err) == "Request timed out."


def test_iter_exception_chain_follows_context_when_cause_absent() -> None:
    err = RuntimeError("wrapper")
    err.__context__ = socket.gaierror()

    chain = iter_exception_chain(err)

    assert chain == [err, err.__context__]
    assert connection_failure_message(err) == "Host not found."


def test_iter_exception_chain_stops_on_repeated_identity() -> None:
    err = RuntimeError("loop")
    err.__cause__ = err

    assert iter_exception_chain(err) == [err]


def test_iter_exception_chain_long_cycle_remains_finite_and_ordered() -> None:
    errs = [RuntimeError(str(index)) for index in range(20)]
    _chain(*errs)
    errs[-1].__cause__ = errs[5]

    chain = iter_exception_chain(errs[0])

    assert chain[:3] == errs[:3]
    assert len(chain) <= len(errs)
    assert len(chain) == len({id(item) for item in chain})


def test_connection_failure_message_handles_cycle_with_reachable_failure() -> None:
    err = RuntimeError("wrapper")
    failure = OSError(errno.ECONNREFUSED, "refused")
    err.__cause__ = failure
    failure.__cause__ = err

    assert connection_failure_message(err) == "Connection refused."
