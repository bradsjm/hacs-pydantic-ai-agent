"""Tests for bounded provider model-list pagination."""

from typing import Any

from custom_components.pydantic_ai_agent.models import provider
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
import pytest


class _Response:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class _Client:
    def __init__(self, payloads: list[object]) -> None:
        self.payloads = list(payloads)
        self.params: list[dict[str, str]] = []

    async def get(self, _url: str, **kwargs: Any) -> _Response:
        self.params.append(dict(kwargs["params"]))
        return _Response(self.payloads.pop(0))


@pytest.mark.parametrize(
    ("listing", "payloads", "expected", "expected_params"),
    [
        (
            provider.list_anthropic_model_names,
            [
                {"data": [{"id": "b"}], "has_more": True, "last_id": "cursor"},
                {"data": [{"id": "a"}], "has_more": False},
            ],
            ["a", "b"],
            [{"limit": "1000"}, {"limit": "1000", "after_id": "cursor"}],
        ),
        (
            provider.list_google_gemini_model_names,
            [
                {
                    "models": [{"baseModelId": "b", "supportedGenerationMethods": ["generateContent"]}],
                    "nextPageToken": "cursor",
                },
                {"models": [{"baseModelId": "a", "supportedGenerationMethods": ["generateContent"]}]},
            ],
            ["a", "b"],
            [
                {"pageSize": "1000"},
                {"pageSize": "1000", "pageToken": "cursor"},
            ],
        ),
    ],
)
async def test_provider_model_listing_advances_pages(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    listing: Any,
    payloads: list[object],
    expected: list[str],
    expected_params: list[dict[str, str]],
) -> None:
    """Advancing cursors collect sorted unique names over two pages."""
    client = _Client(payloads)
    monkeypatch.setattr(provider, "get_async_client", lambda _: client)

    result = await listing(hass, {CONF_API_KEY: "key"}, request_timeout=None)

    assert result == expected
    assert client.params == expected_params


@pytest.mark.parametrize(
    ("listing", "first", "second"),
    [
        (
            provider.list_anthropic_model_names,
            {"data": [{"id": "one"}], "has_more": True, "last_id": "same"},
            {"data": [{"id": "two"}], "has_more": True, "last_id": "same"},
        ),
        (
            provider.list_google_gemini_model_names,
            {
                "models": [{"baseModelId": "one", "supportedGenerationMethods": ["generateContent"]}],
                "nextPageToken": "same",
            },
            {
                "models": [{"baseModelId": "two", "supportedGenerationMethods": ["generateContent"]}],
                "nextPageToken": "same",
            },
        ),
    ],
)
async def test_provider_model_listing_stops_on_repeated_cursor(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    listing: Any,
    first: dict[str, object],
    second: dict[str, object],
) -> None:
    """Repeated continuation cursors stop pagination without another request."""
    client = _Client([first, second])
    monkeypatch.setattr(provider, "get_async_client", lambda _: client)

    result = await listing(hass, {CONF_API_KEY: "key"}, request_timeout=None)

    assert result == ["one", "two"]


@pytest.mark.parametrize(
    ("listing", "payload"),
    [
        (provider.list_anthropic_model_names, {"data": None, "has_more": False}),
        (provider.list_google_gemini_model_names, {"models": None}),
    ],
)
async def test_provider_model_listing_ignores_malformed_collections(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    listing: Any,
    payload: dict[str, object],
) -> None:
    """Null page collections are treated as empty rather than iterated."""
    monkeypatch.setattr(provider, "get_async_client", lambda _: _Client([payload]))

    assert await listing(hass, {CONF_API_KEY: "key"}, request_timeout=None) == []


async def test_gemini_listing_ignores_null_generation_methods(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed Gemini generation methods do not raise while listing models."""
    monkeypatch.setattr(
        provider,
        "get_async_client",
        lambda _: _Client([{"models": [{"baseModelId": "bad", "supportedGenerationMethods": None}]}]),
    )

    assert await provider.list_google_gemini_model_names(hass, {CONF_API_KEY: "key"}, request_timeout=None) == []


async def test_anthropic_listing_stops_at_page_limit(hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed continuing endpoint cannot trigger more than 100 requests."""
    payloads = [{"data": [{"id": str(index)}], "has_more": True, "last_id": str(index)} for index in range(100)]
    client = _Client(payloads)
    monkeypatch.setattr(provider, "get_async_client", lambda _: client)

    result = await provider.list_anthropic_model_names(hass, {CONF_API_KEY: "key"}, request_timeout=None)

    assert len(client.params) == 100
    assert result == sorted(str(index) for index in range(100))
