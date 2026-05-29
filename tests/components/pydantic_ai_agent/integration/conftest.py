"""Fixtures for provider integration tests."""

import socket
from urllib.parse import urlparse

import pytest
import pytest_socket

from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component import plugins as ha_pytest_plugins

from custom_components.pydantic_ai_agent.provider_validation import (
    ProviderValidationError,
    async_probe_model,
)

from .config import (
    MCP_ECHO_URL_ENV,
    PROVIDER_INTEGRATION_TIMEOUT,
    STRUCTURED_OUTPUT_MODES,
    STRUCTURED_OUTPUT_SKIP_REASONS,
    ModelParam,
    ProviderIntegrationConfig,
    Secret,
    StructuredOutputSupport,
)
from .entries import drain_stream_cleanup
from .env import env_values, mcp_echo_url as load_mcp_echo_url, provider_model_params


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


@pytest.fixture(name="mcp_echo_url")
def fixture_mcp_echo_url() -> str:
    """Return the hosted MCP echo server URL for provider integration tests."""
    return load_mcp_echo_url()


@pytest.fixture(autouse=True)
def enable_provider_network(
    provider_config: ProviderIntegrationConfig,
    mcp_echo_url: str,
    socket_enabled: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow provider integration tests to resolve provider and MCP hostnames."""
    del socket_enabled
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        getattr(ha_pytest_plugins, "_real_getaddrinfo"),
    )
    host = urlparse(provider_config.base_url).hostname
    if host is None:
        pytest.skip(
            "OPENAI_BASE_URL must include a hostname for provider integration tests"
        )
    mcp_host = urlparse(mcp_echo_url).hostname
    if mcp_host is None:
        pytest.skip(f"{MCP_ECHO_URL_ENV} must include a hostname")
    pytest_socket.socket_allow_hosts(
        ["localhost", "127.0.0.1", "::1", host, mcp_host],
        allow_unix_socket=True,
    )


@pytest.fixture(name="structured_output_support")
async def fixture_structured_output_support(
    hass: HomeAssistant, provider_config: ProviderIntegrationConfig
) -> StructuredOutputSupport:
    """Return structured output modes supported by the configured model."""
    supported_modes: list[str] = []
    failures: dict[str, ProviderValidationError] = {}
    for output_mode in STRUCTURED_OUTPUT_MODES:
        try:
            await async_probe_model(
                hass,
                provider_config.provider_data,
                provider_config.model,
                {"timeout": PROVIDER_INTEGRATION_TIMEOUT},
                structured_output_mode=output_mode,
            )
        except ProviderValidationError as err:
            if err.reason not in STRUCTURED_OUTPUT_SKIP_REASONS:
                raise
            failures[output_mode] = err
        else:
            supported_modes.append(output_mode)
        finally:
            await drain_stream_cleanup(hass)

    if not supported_modes:
        details = "; ".join(
            f"{mode}: {failures[mode].reason}: {failures[mode].message}"
            for mode in STRUCTURED_OUTPUT_MODES
        )
        pytest.skip(
            "Configured provider integration model does not support any structured "
            f"output mode required by AI task tests: {details}"
        )

    return StructuredOutputSupport(
        supported_modes=tuple(supported_modes), failures=failures
    )
