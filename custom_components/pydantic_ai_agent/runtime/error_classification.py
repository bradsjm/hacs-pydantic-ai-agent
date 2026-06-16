"""Shared provider/runtime error classification helpers."""

import errno
import socket
import ssl

import httpx


def iter_exception_chain(err: BaseException) -> list[BaseException]:
    """Return an exception and its causes/contexts without looping forever."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = err
    while current is not None and id(current) not in seen and len(chain) < 8:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def connection_failure_message(err: BaseException) -> str | None:
    """Return a user-facing connection error message if one can be identified."""
    for item in iter_exception_chain(err):
        if isinstance(item, TimeoutError | httpx.TimeoutException):
            return "Request timed out."
        if isinstance(item, socket.gaierror):
            return "Host not found."
        if isinstance(item, ssl.SSLError):
            return "TLS error."
        if isinstance(item, OSError):
            if item.errno == errno.ECONNREFUSED:
                return "Connection refused."
            if item.errno in (errno.ENETUNREACH, errno.EHOSTUNREACH):
                return "Network unreachable."
        if isinstance(item, httpx.ConnectError):
            return "Connection failed."
    return None


def has_connection_failure(err: BaseException) -> bool:
    """Return if an exception cause chain indicates transport failure."""
    return connection_failure_message(err) is not None
