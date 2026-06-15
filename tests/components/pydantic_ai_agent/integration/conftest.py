"""Fixtures for provider integration tests."""

import socket
from urllib.parse import urlparse

import pytest
import pytest_socket
from pytest_homeassistant_custom_component import plugins as ha_pytest_plugins

from .config import (
    ModelParam,
    ProviderIntegrationConfig,
    Secret,
)
from .env import env_values, provider_model_params


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize provider integration tests over selected models."""
    if "provider_model" in metafunc.fixturenames:
        metafunc.parametrize("provider_model", provider_model_params(metafunc.config))


@pytest.fixture(name="provider_config")
def fixture_provider_config(provider_model: ModelParam) -> ProviderIntegrationConfig:
    """Return provider integration config or skip with missing variable names."""
    values = env_values()
    if provider_model.skip_reason:
        pytest.skip(provider_model.skip_reason)

    return ProviderIntegrationConfig(
        api_key=Secret(values["OPENAI_API_KEY"]),
        model=provider_model.model,
        base_url=values["OPENAI_BASE_URL"],
    )


@pytest.fixture(autouse=True)
def enable_provider_network(
    provider_config: ProviderIntegrationConfig,
    socket_enabled: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow provider integration tests to resolve provider hostnames."""
    del socket_enabled
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        ha_pytest_plugins._real_getaddrinfo,
    )
    host = urlparse(provider_config.base_url).hostname
    if host is None:
        pytest.skip(
            "OPENAI_BASE_URL must include a hostname for provider integration tests"
        )
    pytest_socket.socket_allow_hosts(
        ["localhost", "127.0.0.1", "::1", host],
        allow_unix_socket=True,
    )
