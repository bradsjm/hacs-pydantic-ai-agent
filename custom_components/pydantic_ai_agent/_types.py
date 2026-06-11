"""Shared runtime types for Pydantic AI Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from homeassistant.config_entries import ConfigEntry

from .metrics import MetricsStore


@dataclass(frozen=True, kw_only=True)
class ProviderRuntimeData:
    """Provider runtime data owned by one workspace provider subentry."""

    provider_subentry_id: str
    name: str
    api_key: str
    provider_mode: str
    base_url: str | None
    provider_headers: dict[str, str] = field(default_factory=dict)
    provider_extra_body: dict[str, Any] = field(default_factory=dict)
    client: Any | None = None
    discovered_models: list[str] | None = None


@dataclass(frozen=True, kw_only=True)
class WorkspaceRuntimeData:
    """Workspace data shared by subentry-backed entities."""

    workspace_name: str
    providers: dict[str, ProviderRuntimeData] = field(default_factory=dict)
    model_profiles: dict[str, Any] = field(default_factory=dict)
    metrics: MetricsStore = field(default_factory=MetricsStore)
    latest_stream_traces: dict[str, dict[str, Any]] = field(default_factory=dict)
    latest_run_diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_validation_failures: dict[str, str] = field(default_factory=dict)
    runtime_provider_auth_failures: dict[str, list[str]] = field(default_factory=dict)
    logfire_enabled: bool = False
    logfire_include_content: bool = False


PydanticAIAgentConfigEntry = ConfigEntry[WorkspaceRuntimeData]
