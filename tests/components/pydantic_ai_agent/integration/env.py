"""Environment loading and model selection for provider integration tests."""

from collections.abc import Mapping
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest
import pytest_socket

from .config import (
    DEFAULT_MODEL_LIMIT,
    ENV_FILE,
    MCP_ECHO_URL,
    MCP_ECHO_URL_ENV,
    MODEL_LIST_TIMEOUT,
    ModelParam,
    REQUIRED_CONNECTION_ENV,
    TRUE_ENV_VALUES,
)


def load_dotenv_values(path: Path = ENV_FILE) -> dict[str, str]:
    """Load simple KEY=VALUE pairs from .env without adding a dependency."""
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def env_values() -> dict[str, str]:
    """Return test configuration from process env and .env."""
    file_values = load_dotenv_values()
    keys = {
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
        "OPENAI_MODELS_URL",
        "OPENAI_MODEL_IDS",
        "OPENAI_TEST_ALL_MODELS",
        "OPENAI_MODEL_INCLUDE",
        "OPENAI_MODEL_EXCLUDE",
        "OPENAI_MODEL_LIMIT",
    }
    return {key: os.environ.get(key, file_values.get(key, "")) for key in keys}


def is_true(value: str) -> bool:
    """Return if an environment flag is enabled."""
    return value.strip().lower() in TRUE_ENV_VALUES


def split_model_ids(value: str) -> list[str]:
    """Return comma-separated model IDs."""
    return [model.strip() for model in value.split(",") if model.strip()]


def model_param_id(model: str) -> str:
    """Return a compact pytest parameter ID for a model name."""
    return model.replace("/", "_").replace(":", "_")[:120]


def models_url(values: Mapping[str, str]) -> str:
    """Return the OpenAI-compatible models endpoint URL."""
    if values["OPENAI_MODELS_URL"]:
        return values["OPENAI_MODELS_URL"].rstrip("/")
    return f"{values['OPENAI_BASE_URL'].rstrip('/')}/models"


def skip_model_param(reason: str) -> object:
    """Return one skipped model parameter."""
    return pytest.param(
        ModelParam(model="", skip_reason=reason),
        id="missing-provider-integration-config",
        marks=pytest.mark.skip(reason=reason),
    )


def limit_model_ids(model_ids: list[str], limit_value: str) -> list[str]:
    """Apply configured model limit. A value of 0 means unlimited."""
    if limit_value:
        try:
            limit = int(limit_value)
        except ValueError:
            return model_ids[:DEFAULT_MODEL_LIMIT]
    else:
        limit = DEFAULT_MODEL_LIMIT
    if limit == 0:
        return model_ids
    return model_ids[: max(limit, 0)]


def filter_model_ids(model_ids: list[str], values: Mapping[str, str]) -> list[str]:
    """Apply include/exclude regex filters and stable sorting."""
    filtered = sorted(dict.fromkeys(model_ids))
    if include := values["OPENAI_MODEL_INCLUDE"]:
        pattern = re.compile(include)
        filtered = [model for model in filtered if pattern.search(model)]
    if exclude := values["OPENAI_MODEL_EXCLUDE"]:
        pattern = re.compile(exclude)
        filtered = [model for model in filtered if not pattern.search(model)]
    return limit_model_ids(filtered, values["OPENAI_MODEL_LIMIT"])


def parse_models_response(data: object) -> list[str]:
    """Return model IDs from an OpenAI-compatible /models response."""
    if not isinstance(data, Mapping) or not isinstance(
        models := data.get("data"), list
    ):
        return []
    model_ids: list[str] = []
    for model in models:
        if isinstance(model, str):
            model_ids.append(model)
        elif isinstance(model, Mapping) and isinstance(
            model_id := model.get("id"), str
        ):
            model_ids.append(model_id)
    return model_ids


def fetch_model_ids(values: Mapping[str, str]) -> list[str]:
    """Fetch model IDs from the configured models endpoint."""
    url = models_url(values)
    host = urlparse(url).hostname
    if host is not None:
        pytest_socket.socket_allow_hosts([host], allow_unix_socket=True)
    headers = {"Authorization": f"Bearer {values['OPENAI_API_KEY']}"}
    with httpx.Client(timeout=MODEL_LIST_TIMEOUT) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        return parse_models_response(response.json())


def provider_model_params(config: pytest.Config) -> list[object]:
    """Return pytest params for selected provider integration models."""
    values = env_values()
    missing = [key for key in REQUIRED_CONNECTION_ENV if not values[key]]
    if missing:
        return [
            skip_model_param(
                "Provider integration tests require these environment values in "
                f".env or the process environment: {', '.join(missing)}"
            )
        ]

    try:
        if explicit_models := split_model_ids(values["OPENAI_MODEL_IDS"]):
            model_ids = filter_model_ids(explicit_models, values)
        elif is_true(values["OPENAI_TEST_ALL_MODELS"]):
            if "not provider_integration" in (config.option.markexpr or ""):
                model_ids = split_model_ids(values["OPENAI_MODEL"])
            else:
                try:
                    model_ids = filter_model_ids(fetch_model_ids(values), values)
                except (httpx.HTTPError, ValueError) as err:
                    return [
                        skip_model_param(
                            "Unable to fetch OpenAI-compatible model list: "
                            f"{type(err).__name__}"
                        )
                    ]
        else:
            model_ids = split_model_ids(values["OPENAI_MODEL"])
    except re.PatternError as err:
        return [
            skip_model_param(f"Invalid provider integration model filter regex: {err}")
        ]

    if not model_ids:
        return [
            skip_model_param(
                "Provider integration tests require OPENAI_MODEL, OPENAI_MODEL_IDS, "
                "or OPENAI_TEST_ALL_MODELS=true with at least one discovered model."
            )
        ]
    return [
        pytest.param(ModelParam(model=model), id=model_param_id(model))
        for model in model_ids
    ]


def mcp_echo_url() -> str:
    """Return the hosted MCP echo server URL for provider integration tests."""
    file_values = load_dotenv_values()
    return os.environ.get(
        MCP_ECHO_URL_ENV, file_values.get(MCP_ECHO_URL_ENV, MCP_ECHO_URL)
    )
