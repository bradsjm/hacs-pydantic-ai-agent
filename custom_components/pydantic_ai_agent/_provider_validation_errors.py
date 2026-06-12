"""Private provider-validation error formatting and mapping helpers."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

from ._redaction import redact_data, redaction_keys
from .error_classification import connection_failure_message

_HTTP_STATUS_LABELS = {
    400: "invalid request",
    401: "authentication issue",
    402: "payment issue",
    403: "permission issue",
    404: "model not found",
    408: "timeout",
    409: "conflict",
    422: "validation issue",
    429: "rate limit",
    504: "timeout",
}
_MAX_METADATA_REPR_LENGTH = 1000


@dataclass(slots=True)
class ProviderValidationError(Exception):
    """Provider validation failed with a translation-ready reason."""

    reason: str
    message: str
    status_code: int | None = None


def format_api_error(err: ModelAPIError) -> ProviderValidationError:
    """Map a non-HTTP model API error to a config-flow validation error."""
    if connection_message := connection_failure_message(err):
        reason = (
            "timeout"
            if connection_message == "Request timed out."
            else "cannot_connect"
        )
        return ProviderValidationError(reason, connection_message)
    return ProviderValidationError(
        "provider_error",
        f'The provider returned an API error for model "{err.model_name}".',
    )


def map_http_error(err: ModelHTTPError) -> ProviderValidationError:
    """Map a model HTTP error to a config-flow validation error."""
    status_code = err.status_code
    if status_code == 401:
        reason = "invalid_auth"
    elif status_code == 403:
        reason = "permission_denied"
    elif status_code == 404:
        reason = "invalid_model"
    elif status_code in (408, 504):
        reason = "timeout"
    elif status_code == 429:
        reason = "rate_limited"
    elif status_code == 400:
        reason = "invalid_model"
    else:
        reason = "provider_error"
    return ProviderValidationError(reason, _format_http_error(err), status_code)


def map_structured_http_error(
    err: ModelHTTPError, output_mode: str
) -> ProviderValidationError:
    """Map structured-output probe HTTP errors to capability errors."""
    if err.status_code == 400:
        return ProviderValidationError(
            "unsupported_output_mode",
            (
                f'Model "{err.model_name}" rejected structured output mode '
                f'"{output_mode}". Try a different structured output mode or a '
                "model/provider that supports this mode."
            ),
            err.status_code,
        )
    return map_http_error(err)


def _format_http_error(err: ModelHTTPError) -> str:
    message = (
        f"The provider returned error {err.status_code}"
        f" ({_status_label(err.status_code)})"
        f' for model "{err.model_name}".'
    )
    if isinstance(err.body, Mapping) and (metadata := err.body.get("metadata")):
        message = f"{message} Metadata: {_format_metadata(metadata)}."
    return message


def _format_metadata(metadata: object) -> str:
    redacted = _display_safe_metadata(redact_data(metadata))
    formatted = repr(redacted)
    if len(formatted) > _MAX_METADATA_REPR_LENGTH:
        return f"{formatted[:_MAX_METADATA_REPR_LENGTH]}..."
    return formatted


def _display_safe_metadata(metadata: object) -> object:
    sensitive_keys = redaction_keys()
    if isinstance(metadata, Mapping):
        filtered = {
            key: _display_safe_metadata(value)
            for key, value in metadata.items()
            if key not in sensitive_keys
        }
        return filtered or "**REDACTED**"
    if isinstance(metadata, Sequence) and not isinstance(
        metadata, str | bytes | bytearray
    ):
        return [_display_safe_metadata(value) for value in metadata]
    return metadata


def _status_label(status_code: int) -> str:
    if label := _HTTP_STATUS_LABELS.get(status_code):
        return label
    if 500 <= status_code <= 599:
        return "provider server issue"
    return "HTTP error"
