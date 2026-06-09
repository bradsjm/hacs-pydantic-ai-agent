"""Pydantic AI provider helpers."""

from collections.abc import Mapping
from typing import Any

import httpx
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.helpers.httpx_client import get_async_client
from pydantic_ai.models import Model

from .const import (
    CONF_BASE_URL,
    CONF_PROVIDER_HEADERS,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE_GEMINI,
)
from .openai_compatible_adapter import (
    OpenAICompatibleChatModel,
    OpenAICompatibleProvider,
    OpenAICompatibleResponsesModel,
)
from .openai_compatible_client import AsyncOpenAICompatible


def normalise_base_url(value: object) -> str | None:
    """Return a normalized provider base URL."""
    if not value:
        return None
    return str(value).rstrip("/")


def _provider_base_model_name(provider_mode: str, model_name: str) -> str:
    """Return a model name without an optional matching provider prefix."""
    model_name = model_name.strip()
    if ":" not in model_name:
        return model_name
    prefix, name = model_name.split(":", 1)
    valid_prefixes = {
        PROVIDER_ANTHROPIC: {"anthropic"},
        PROVIDER_GOOGLE_GEMINI: {"google", "google-gla"},
    }.get(provider_mode, set())
    if prefix not in valid_prefixes:
        raise ValueError(
            f"Model prefix {prefix!r} is not valid for provider mode {provider_mode!r}."
        )
    if not name:
        raise ValueError("Model name cannot be empty.")
    return name


def _configured_headers(headers: dict[str, str] | None) -> dict[str, str]:
    """Return configured provider headers without mutating caller-owned data."""
    return dict(headers or {})


def _strip_version_suffix(base_url: str | None, *versions: str) -> str | None:
    """Return a provider base URL without a trailing API version segment."""
    base_url = normalise_base_url(base_url)
    if base_url is None:
        return None
    for version in versions:
        suffix = f"/{version.lstrip('/')}"
        if base_url.endswith(suffix):
            return base_url[: -len(suffix)]
    return base_url


def _anthropic_base_url_for_sdk(base_url: str | None) -> str | None:
    """Return an Anthropic SDK base URL from user input."""
    return _strip_version_suffix(base_url, "v1")


def _google_base_url_for_sdk(base_url: str | None) -> str | None:
    """Return a Google GenAI SDK base URL from user input."""
    return _strip_version_suffix(base_url, "v1beta", "v1")


def anthropic_model(
    hass: HomeAssistant,
    *,
    api_key: str,
    base_url: str | None,
    headers: dict[str, str] | None = None,
    model_name: str,
) -> Model:
    """Build a Pydantic AI Anthropic model."""
    from anthropic import AsyncAnthropic
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    anthropic_client = AsyncAnthropic(
        api_key=api_key,
        base_url=_anthropic_base_url_for_sdk(base_url),
        default_headers=_configured_headers(headers) or None,
        http_client=get_async_client(hass),
    )
    provider = AnthropicProvider(anthropic_client=anthropic_client)
    return AnthropicModel(
        _provider_base_model_name(PROVIDER_ANTHROPIC, model_name), provider=provider
    )


def google_gemini_model(
    hass: HomeAssistant,
    *,
    api_key: str,
    base_url: str | None,
    headers: dict[str, str] | None = None,
    model_name: str,
) -> Model:
    """Build a Pydantic AI Google Gemini model."""
    from google.genai import Client
    from google.genai.types import HttpOptions
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    http_client = get_async_client(hass)
    timeout_seconds = http_client.timeout.read or 10.0
    client = Client(
        vertexai=False,
        api_key=api_key,
        http_options=HttpOptions(
            base_url=_google_base_url_for_sdk(base_url),
            headers=_configured_headers(headers) or None,
            httpx_async_client=http_client,
            timeout=int(timeout_seconds * 1000),
        ),
    )
    provider = GoogleProvider(client=client)
    return GoogleModel(
        _provider_base_model_name(PROVIDER_GOOGLE_GEMINI, model_name), provider=provider
    )


def openai_compatible_completions_model(
    hass: HomeAssistant,
    *,
    api_key: str,
    base_url: str | None,
    headers: dict[str, str] | None = None,
    model_name: str,
) -> Model:
    """Build a Pydantic AI OpenAI-compatible Completions model."""
    provider = OpenAICompatibleProvider(
        api_key=api_key,
        base_url=normalise_base_url(base_url),
        headers=headers,
        name="openai-compatible-completions",
        # Reuse Home Assistant's shared async client for its SSL, proxy, and
        # connection-pooling configuration.
        http_client=get_async_client(hass),
    )
    return OpenAICompatibleChatModel(model_name, provider=provider)


def openai_compatible_responses_model(
    hass: HomeAssistant,
    *,
    api_key: str,
    base_url: str | None,
    headers: dict[str, str] | None = None,
    model_name: str,
) -> Model:
    """Build a Pydantic AI OpenAI-compatible Responses model."""
    provider = OpenAICompatibleProvider(
        api_key=api_key,
        base_url=normalise_base_url(base_url),
        headers=headers,
        name="openai-compatible-responses",
        # Reuse Home Assistant's shared async client for its SSL, proxy, and
        # connection-pooling configuration.
        http_client=get_async_client(hass),
    )
    return OpenAICompatibleResponsesModel(model_name, provider=provider)


def _api_url(base_url: str, path: str) -> str:
    """Return an API URL, preserving callers that configure a versioned base URL."""
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _provider_headers_from_config(data: Mapping[str, Any]) -> dict[str, str]:
    """Return configured provider headers from config entry data."""
    headers = data.get(CONF_PROVIDER_HEADERS)
    return dict(headers) if isinstance(headers, Mapping) else {}


async def list_anthropic_model_names(
    hass: HomeAssistant,
    data: Mapping[str, Any],
    *,
    timeout: float | httpx.Timeout | None,
) -> list[str]:
    """Return model IDs from Anthropic's model listing endpoint."""
    base_url = _anthropic_base_url_for_sdk(data.get(CONF_BASE_URL)) or (
        "https://api.anthropic.com"
    )
    client = get_async_client(hass)
    model_names: list[str] = []
    after_id: str | None = None
    while True:
        params = {"limit": "1000"}
        if after_id is not None:
            params["after_id"] = after_id
        response = await client.get(
            _api_url(base_url, "/v1/models"),
            headers={
                **_provider_headers_from_config(data),
                "anthropic-version": "2023-06-01",
                "x-api-key": data[CONF_API_KEY],
            },
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            return []
        for item in payload.get("data", []):
            if isinstance(item, Mapping) and isinstance(item.get("id"), str):
                model_names.append(item["id"])
        if not payload.get("has_more") or not isinstance(payload.get("last_id"), str):
            break
        after_id = payload["last_id"]
    return sorted(set(model_names))


def _extract_google_model_name(item: object) -> str | None:
    """Extract model name from a Gemini item that supports generateContent."""
    if not isinstance(item, Mapping):
        return None
    methods = item.get("supportedGenerationMethods", [])
    if "generateContent" not in methods:
        return None
    model_id = item.get("baseModelId")
    if not isinstance(model_id, str) or not model_id:
        name = item.get("name")
        if isinstance(name, str):
            model_id = name.removeprefix("models/")
    if isinstance(model_id, str) and model_id:
        return model_id
    return None


async def list_google_gemini_model_names(
    hass: HomeAssistant,
    data: Mapping[str, Any],
    *,
    timeout: float | httpx.Timeout | None,
) -> list[str]:
    """Return text-generation model IDs from the Gemini model listing endpoint."""
    base_url = _google_base_url_for_sdk(data.get(CONF_BASE_URL)) or (
        "https://generativelanguage.googleapis.com"
    )
    client = get_async_client(hass)
    model_names: list[str] = []
    page_token: str | None = None
    while True:
        params = {"pageSize": "1000"}
        if page_token is not None:
            params["pageToken"] = page_token
        response = await client.get(
            _api_url(base_url, "/v1beta/models"),
            headers={
                **_provider_headers_from_config(data),
                "x-goog-api-key": data[CONF_API_KEY],
            },
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            return []
        for item in payload.get("models", []):
            name = _extract_google_model_name(item)
            if name is not None:
                model_names.append(name)
        page_token = payload.get("nextPageToken")
        if not isinstance(page_token, str) or not page_token:
            break
    return sorted(set(model_names))


def openai_compatible_client_from_config(
    hass: HomeAssistant, data: Mapping[str, Any]
) -> AsyncOpenAICompatible:
    """Build a lightweight OpenAI-compatible client from config entry data."""
    headers = data.get(CONF_PROVIDER_HEADERS)
    provider = OpenAICompatibleProvider(
        api_key=data[CONF_API_KEY],
        base_url=normalise_base_url(data.get(CONF_BASE_URL)),
        headers=dict(headers) if isinstance(headers, Mapping) else None,
        http_client=get_async_client(hass),
    )
    return provider.client


def openai_compatible_completions_model_from_config(
    hass: HomeAssistant, data: Mapping[str, Any], model_name: str
) -> Model:
    """Build a Pydantic AI Completions model from config entry data."""
    headers = data.get(CONF_PROVIDER_HEADERS)
    return openai_compatible_completions_model(
        hass,
        api_key=data[CONF_API_KEY],
        base_url=normalise_base_url(data.get(CONF_BASE_URL)),
        headers=dict(headers) if isinstance(headers, Mapping) else None,
        model_name=model_name,
    )


def openai_compatible_responses_model_from_config(
    hass: HomeAssistant, data: Mapping[str, Any], model_name: str
) -> Model:
    """Build a Pydantic AI Responses model from config entry data."""
    headers = data.get(CONF_PROVIDER_HEADERS)
    return openai_compatible_responses_model(
        hass,
        api_key=data[CONF_API_KEY],
        base_url=normalise_base_url(data.get(CONF_BASE_URL)),
        headers=dict(headers) if isinstance(headers, Mapping) else None,
        model_name=model_name,
    )
