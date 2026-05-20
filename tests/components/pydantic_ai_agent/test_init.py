"""Test setup for Pydantic AI Agent."""

import json
from collections.abc import Callable, Mapping
from pathlib import Path
import sys
import tomllib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

import pytest
from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_NAME, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import (
    ConfigEntryAuthFailed,
    ConfigEntryNotReady,
)
from homeassistant.helpers import issue_registry as ir
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.pydantic_ai_agent import (
    PLATFORMS,
    SERVICE_LIST_MCP_TOOLS,
    SERVICE_REFRESH_MCP_TOOLS,
    async_setup,
    async_setup_entry,
    async_unload_entry,
    async_remove_entry,
)
from custom_components.pydantic_ai_agent.metrics import EVENT_MCP_TOOL_REFRESH_COMPLETED
from custom_components.pydantic_ai_agent.config_flow import ProviderValidationError
from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_FALLBACK_MODEL_SUBENTRY_IDS,
    CONF_LOGFIRE_INCLUDE_CONTENT,
    CONF_LOGFIRE_TOKEN,
    CONF_MCP_URL,
    CONF_MODEL,
    CONF_MODEL_SETTINGS,
    CONF_MODEL_SUBENTRY_ID,
    CONF_OUTPUT_MODE,
    CONF_PROVIDER_MODE,
    DOMAIN,
    OUTPUT_MODE_NATIVE,
    OUTPUT_MODE_TOOL,
    PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_MCP_SERVER,
    SUBENTRY_TYPE_MODEL,
)
from custom_components.pydantic_ai_agent.logfire_support import (
    async_configure_logfire,
    logfire_include_content,
)
from custom_components.pydantic_ai_agent.model_profiles import model_profile_ref
from custom_components.pydantic_ai_agent.repairs import model_validation_issue_id

_REPO_ROOT = Path(__file__).parents[3]
_EXPLICIT_RUNTIME_REQUIREMENTS = {
    "logfire==4.33.0",
    "pydantic-ai-slim==1.97.0",
    "anthropic>=0.97.0",
    "google-genai>=1.70.0",
    "pydantic-ai-skills==0.10.0",
    "tiktoken>=0.12.0",
    "fastmcp-slim[client,server]>=3.3.0",
    "markdownify>=1.2",
}


def _model_subentry(
    subentry_id: str = "model_profile_1",
    *,
    name: str = "Fast GPT",
    model: str = "gpt-test",
    model_settings: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return a model profile config subentry."""
    data: dict[str, object] = {CONF_NAME: name, CONF_MODEL: model}
    if model_settings is not None:
        data[CONF_MODEL_SETTINGS] = dict(model_settings)
    return {
        "subentry_id": subentry_id,
        "data": data,
        "subentry_type": SUBENTRY_TYPE_MODEL,
        "title": name,
        "unique_id": None,
    }


def _conversation_subentry(
    model_subentry_id: str = "model_profile_1",
) -> dict[str, object]:
    """Return a conversation config subentry."""
    return {
        "data": {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_MODEL_SUBENTRY_ID: model_subentry_id,
        },
        "subentry_type": SUBENTRY_TYPE_CONVERSATION,
        "title": "Kitchen Agent",
        "unique_id": None,
    }


def _ai_task_subentry(
    output_mode: str | None = None, model_subentry_id: str = "task_model_profile"
) -> dict[str, object]:
    """Return an AI task data config subentry."""
    data = {CONF_MODEL_SUBENTRY_ID: model_subentry_id}
    if output_mode is not None:
        data[CONF_OUTPUT_MODE] = output_mode
    return {
        "data": data,
        "subentry_type": SUBENTRY_TYPE_AI_TASK,
        "title": "Task Model",
        "unique_id": None,
    }


def _mcp_server_subentry() -> dict[str, object]:
    """Return an MCP server config subentry."""
    return {
        "data": {
            CONF_NAME: "Filesystem MCP",
            CONF_MCP_URL: "https://mcp.example.com/mcp",
        },
        "subentry_type": SUBENTRY_TYPE_MCP_SERVER,
        "title": "Filesystem MCP",
        "unique_id": None,
    }


def _entry(
    subentries_data: tuple[dict[str, object], ...] = (),
    data_extra: dict[str, object] | None = None,
) -> MockConfigEntry:
    """Return a config entry."""
    data: dict[str, object] = {
        CONF_NAME: "Hosted OpenAI",
        CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
        CONF_API_KEY: "sk-test",
    }
    if data_extra is not None:
        data.update(data_extra)
    return MockConfigEntry(
        domain=DOMAIN,
        title="Hosted OpenAI",
        data=data,
        source=config_entries.SOURCE_USER,
        subentries_data=subentries_data,
        options={},
        unique_id=None,
    )


def test_runtime_requirements_are_explicit_for_home_assistant_installer() -> None:
    """Test runtime requirements do not rely on nested extras installation."""
    manifest = json.loads(
        (_REPO_ROOT / "custom_components/pydantic_ai_agent/manifest.json").read_text()
    )
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())

    manifest_requirements = set(manifest["requirements"])
    pyproject_dependencies = set(pyproject["project"]["dependencies"])

    assert _EXPLICIT_RUNTIME_REQUIREMENTS <= manifest_requirements
    assert _EXPLICIT_RUNTIME_REQUIREMENTS <= pyproject_dependencies


async def test_setup_entry_stores_runtime_data(hass: HomeAssistant) -> None:
    """Test setup stores provider runtime data."""
    entry = _entry(
        (
            _model_subentry(),
            _model_subentry(
                "task_model_profile", name="Task Model", model="task-model"
            ),
            _conversation_subentry(),
            _ai_task_subentry(),
            _mcp_server_subentry(),
        )
    )
    entry.add_to_hass(hass)

    with (
        patch(
            "custom_components.pydantic_ai_agent.async_probe_model",
            new_callable=AsyncMock,
        ) as probe_model,
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new_callable=AsyncMock,
        ) as forward_setups,
    ):
        assert await async_setup_entry(hass, entry)
        forward_setups.assert_awaited_once()

    assert entry.runtime_data.provider_mode == PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS
    assert entry.runtime_data.name == "Hosted OpenAI"
    assert entry.runtime_data.api_key == "sk-test"
    assert entry.runtime_data.base_url is None
    assert entry.runtime_data.logfire_enabled is False
    assert entry.runtime_data.logfire_include_content is False
    assert entry.runtime_data.mcp_servers == [
        {
            CONF_NAME: "Filesystem MCP",
            CONF_MCP_URL: "https://mcp.example.com/mcp",
        }
    ]
    probe_model.assert_has_awaits(
        [
            call(hass, entry.data, "gpt-test", {}),
            call(hass, entry.data, "task-model", {}),
            call(
                hass,
                entry.data,
                "task-model",
                {},
                structured_output_mode=OUTPUT_MODE_TOOL,
            ),
        ]
    )
    assert probe_model.await_count == 3


async def test_setup_entry_rejects_unsupported_provider_mode(
    hass: HomeAssistant,
) -> None:
    """Test stale provider modes fail setup instead of being aliased."""
    entry = _entry(data_extra={CONF_PROVIDER_MODE: "openai_compatible"})
    entry.add_to_hass(hass)

    with pytest.raises(ConfigEntryNotReady, match="Unsupported provider mode"):
        await async_setup_entry(hass, entry)


async def test_setup_entry_configures_logfire_before_platform_setup(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test setup configures Logfire after validation and before platforms."""
    events: list[str] = []
    configure = Mock(side_effect=lambda **_kwargs: events.append("configure"))
    monkeypatch.setitem(
        sys.modules,
        "logfire",
        SimpleNamespace(configure=configure),
    )
    entry = _entry(
        (_model_subentry(), _conversation_subentry()),
        {CONF_LOGFIRE_TOKEN: " lf-token ", CONF_LOGFIRE_INCLUDE_CONTENT: True},
    )
    entry.add_to_hass(hass)

    async def probe(*_args: object, **_kwargs: object) -> None:
        events.append("probe")

    async def run_executor(target: Callable[..., object], *args: object) -> object:
        events.append("executor")
        return target(*args)

    with (
        patch(
            "custom_components.pydantic_ai_agent.async_probe_model",
            new_callable=AsyncMock,
            side_effect=probe,
        ),
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new_callable=AsyncMock,
            side_effect=lambda *_args, **_kwargs: events.append("platforms"),
        ),
        patch.object(
            hass,
            "async_add_executor_job",
            new_callable=AsyncMock,
            side_effect=run_executor,
        ) as executor_job,
    ):
        assert await async_setup_entry(hass, entry)

    assert events == ["probe", "executor", "configure", "platforms"]
    executor_job.assert_awaited_once()
    configure.assert_called_once_with(
        send_to_logfire=True,
        token="lf-token",
        service_name=DOMAIN,
        console=False,
        inspect_arguments=False,
    )
    assert entry.runtime_data.logfire_enabled is True
    assert entry.runtime_data.logfire_include_content is True


async def test_setup_entry_does_not_configure_logfire_when_validation_fails(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test invalid entries do not lock Logfire to their token."""
    configure = Mock()
    monkeypatch.setitem(
        sys.modules,
        "logfire",
        SimpleNamespace(configure=configure),
    )
    entry = _entry(
        (_model_subentry(), _conversation_subentry()), {CONF_LOGFIRE_TOKEN: "lf-token"}
    )
    entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
        new_callable=AsyncMock,
        side_effect=ProviderValidationError("invalid_auth", "Invalid API key"),
    ):
        with pytest.raises(ConfigEntryAuthFailed):
            await async_setup_entry(hass, entry)

    configure.assert_not_called()


async def test_setup_entry_logfire_conflict_creates_repair_issue(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a later different Logfire token does not reconfigure Logfire."""
    configure = Mock()
    monkeypatch.setitem(
        sys.modules,
        "logfire",
        SimpleNamespace(configure=configure),
    )
    first_entry = _entry(data_extra={CONF_LOGFIRE_TOKEN: "first-token"})
    assert await async_configure_logfire(hass, first_entry)
    entry = _entry(data_extra={CONF_LOGFIRE_TOKEN: "second-token"})
    entry.add_to_hass(hass)

    with patch.object(
        hass.config_entries,
        "async_forward_entry_setups",
        new_callable=AsyncMock,
    ):
        assert await async_setup_entry(hass, entry)

    configure.assert_called_once()
    assert entry.runtime_data.logfire_enabled is False
    assert entry.runtime_data.logfire_include_content is False
    assert (
        DOMAIN,
        f"logfire_token_conflict_{entry.entry_id}",
    ) in ir.async_get(hass).issues


async def test_setup_entry_logfire_configure_failure_is_non_fatal(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test optional Logfire setup failures do not block entry setup."""
    monkeypatch.setitem(sys.modules, "logfire", SimpleNamespace())
    entry = _entry(data_extra={CONF_LOGFIRE_TOKEN: "lf-token"})
    entry.add_to_hass(hass)

    with patch.object(
        hass.config_entries,
        "async_forward_entry_setups",
        new_callable=AsyncMock,
    ):
        assert await async_setup_entry(hass, entry)

    assert entry.runtime_data.logfire_enabled is False
    assert entry.runtime_data.logfire_include_content is False


async def test_logfire_include_content_uses_first_token_setting(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test same-token entries cannot widen the global content-capture setting."""
    monkeypatch.setitem(
        sys.modules,
        "logfire",
        SimpleNamespace(configure=Mock()),
    )
    first_entry = _entry(data_extra={CONF_LOGFIRE_TOKEN: "lf-token"})
    assert await async_configure_logfire(hass, first_entry)
    entry = _entry(
        data_extra={
            CONF_LOGFIRE_TOKEN: "lf-token",
            CONF_LOGFIRE_INCLUDE_CONTENT: True,
        }
    )

    assert logfire_include_content(hass, entry) is False


async def test_setup_registers_mcp_response_services(hass: HomeAssistant) -> None:
    """Test async setup registers MCP discovery response services."""
    assert await async_setup(hass, {})

    assert hass.services.has_service(DOMAIN, SERVICE_LIST_MCP_TOOLS)
    assert hass.services.has_service(DOMAIN, SERVICE_REFRESH_MCP_TOOLS)


async def test_refresh_mcp_tools_service_returns_discovered_tools(
    hass: HomeAssistant,
) -> None:
    """Test refresh_mcp_tools returns tools for a configured MCP server."""
    entry = _entry((_mcp_server_subentry(),))
    entry.add_to_hass(hass)
    subentry = next(iter(entry.subentries.values()))
    await async_setup(hass, {})
    with patch.object(
        hass.config_entries,
        "async_forward_entry_setups",
        new_callable=AsyncMock,
    ):
        assert await async_setup_entry(hass, entry)
    tools = [
        {
            "server_id": subentry.subentry_id,
            "name": "list_files",
            "schema_hash": "abc123",
        }
    ]
    events: list[dict[str, object]] = []
    hass.bus.async_listen(
        f"{DOMAIN}_{EVENT_MCP_TOOL_REFRESH_COMPLETED}",
        lambda event: events.append(dict(event.data)),
    )

    with patch(
        "custom_components.pydantic_ai_agent.async_refresh_mcp_tools",
        new_callable=AsyncMock,
        return_value=tools,
    ) as refresh_tools:
        response = await hass.services.async_call(
            DOMAIN,
            SERVICE_REFRESH_MCP_TOOLS,
            {
                "config_entry_id": entry.entry_id,
                "mcp_server_id": subentry.subentry_id,
            },
            blocking=True,
            return_response=True,
        )
        await hass.async_block_till_done()

    assert response == {
        "success": True,
        "servers": {subentry.subentry_id: tools},
        "tools": tools,
        "errors": [],
    }
    refresh_tools.assert_awaited_once_with(hass, entry, subentry.subentry_id)
    assert events == [
        {
            "config_entry_id": entry.entry_id,
            "mcp_server_id": subentry.subentry_id,
            "tool_count": 1,
        }
    ]


async def test_setup_entry_validates_ai_task_selected_output_mode(
    hass: HomeAssistant,
) -> None:
    """Test setup validates AI task models with their configured output mode."""
    entry = _entry(
        (
            _model_subentry(
                "task_model_profile", name="Task Model", model="task-model"
            ),
            _ai_task_subentry(OUTPUT_MODE_NATIVE),
        )
    )
    entry.add_to_hass(hass)

    with (
        patch(
            "custom_components.pydantic_ai_agent.async_probe_model",
            new_callable=AsyncMock,
        ) as probe_model,
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new_callable=AsyncMock,
        ),
    ):
        assert await async_setup_entry(hass, entry)

    probe_model.assert_has_awaits(
        [
            call(hass, entry.data, "task-model", {}),
            call(
                hass,
                entry.data,
                "task-model",
                {},
                structured_output_mode=OUTPUT_MODE_NATIVE,
            ),
        ]
    )
    assert probe_model.await_count == 2


def test_platforms_include_conversation_and_ai_task() -> None:
    """Test setup forwards both runtime platforms."""
    assert PLATFORMS == (
        Platform.CONVERSATION,
        Platform.AI_TASK,
        Platform.SENSOR,
        Platform.BINARY_SENSOR,
    )


async def test_setup_entry_validates_each_subentry_model_settings(
    hass: HomeAssistant,
) -> None:
    """Test setup validates each unique model and settings combination."""
    entry = _entry(
        (
            _model_subentry(
                "first_model",
                name="First Model",
                model="shared-model",
                model_settings={"timeout": 20.0},
            ),
            _model_subentry(
                "second_model",
                name="Second Model",
                model="shared-model",
                model_settings={"extra_body": {"service_tier": "flex"}},
            ),
            _model_subentry(
                "duplicate_model",
                name="Duplicate Model",
                model="shared-model",
                model_settings={"extra_body": {"service_tier": "flex"}},
            ),
        )
    )
    entry.add_to_hass(hass)

    with (
        patch(
            "custom_components.pydantic_ai_agent.async_probe_model",
            new_callable=AsyncMock,
        ) as probe_model,
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new_callable=AsyncMock,
        ),
    ):
        assert await async_setup_entry(hass, entry)

    probe_model.assert_has_awaits(
        [
            call(hass, entry.data, "shared-model", {"timeout": 20.0}),
            call(
                hass,
                entry.data,
                "shared-model",
                {"extra_body": {"service_tier": "flex"}},
            ),
        ]
    )
    assert probe_model.await_count == 2


async def test_setup_entry_without_subentries_stores_runtime_data(
    hass: HomeAssistant,
) -> None:
    """Test setup loads provider entries before any subentries exist."""
    entry = _entry()
    entry.add_to_hass(hass)

    with patch.object(
        hass.config_entries,
        "async_forward_entry_setups",
        new_callable=AsyncMock,
    ):
        assert await async_setup_entry(hass, entry)

    assert entry.runtime_data.provider_mode == PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS


async def test_multiple_entries_setup_and_unload_are_isolated(
    hass: HomeAssistant,
) -> None:
    """Test setup and unload operate on the targeted config entry only."""
    first_entry = _entry()
    second_entry = MockConfigEntry(
        domain=DOMAIN,
        title="Other OpenAI",
        data={
            CONF_NAME: "Other OpenAI",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE_COMPLETIONS,
            CONF_API_KEY: "other-key",
        },
        source=config_entries.SOURCE_USER,
        options={},
        unique_id=None,
    )
    first_entry.add_to_hass(hass)
    second_entry.add_to_hass(hass)

    with patch.object(
        hass.config_entries,
        "async_forward_entry_setups",
        new_callable=AsyncMock,
    ) as forward_setups:
        assert await async_setup_entry(hass, first_entry)
        assert await async_setup_entry(hass, second_entry)

    assert first_entry.runtime_data.name == "Hosted OpenAI"
    assert first_entry.runtime_data.api_key == "sk-test"
    assert second_entry.runtime_data.name == "Other OpenAI"
    assert second_entry.runtime_data.api_key == "other-key"
    assert forward_setups.await_args_list[0].args[0] is first_entry
    assert forward_setups.await_args_list[1].args[0] is second_entry

    with patch.object(
        hass.config_entries,
        "async_unload_platforms",
        new_callable=AsyncMock,
        return_value=True,
    ) as unload_platforms:
        assert await async_unload_entry(hass, first_entry)

    unload_platforms.assert_awaited_once()
    assert unload_platforms.await_args is not None
    assert unload_platforms.await_args.args[0] is first_entry
    assert second_entry.runtime_data.name == "Other OpenAI"


@pytest.mark.parametrize(
    ("reason", "exception_type"),
    [
        ("invalid_auth", ConfigEntryAuthFailed),
        ("cannot_connect", ConfigEntryNotReady),
        ("timeout", ConfigEntryNotReady),
        ("invalid_base_url", ConfigEntryNotReady),
        ("rate_limited", ConfigEntryNotReady),
        ("provider_error", ConfigEntryNotReady),
    ],
)
async def test_setup_entry_validation_errors(
    hass: HomeAssistant,
    reason: str,
    exception_type: type[Exception],
) -> None:
    """Test setup maps stored model validation errors to config-entry errors."""
    entry = _entry((_model_subentry(), _conversation_subentry()))
    entry.add_to_hass(hass)

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
        new_callable=AsyncMock,
        side_effect=ProviderValidationError(reason, "validation failed"),
    ):
        with pytest.raises(exception_type):
            await async_setup_entry(hass, entry)


@pytest.mark.parametrize(
    "reason",
    [
        "invalid_model",
        "invalid_provider_config",
        "model_does_not_support_streaming",
        "permission_denied",
    ],
)
async def test_setup_entry_model_errors_keep_entry_reconfigurable(
    hass: HomeAssistant,
    reason: str,
) -> None:
    """Test subentry model errors do not block reconfiguration."""
    entry = _entry((_model_subentry(), _conversation_subentry()))
    entry.add_to_hass(hass)

    with (
        patch(
            "custom_components.pydantic_ai_agent.async_probe_model",
            new_callable=AsyncMock,
            side_effect=ProviderValidationError(reason, "validation failed"),
        ),
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new_callable=AsyncMock,
        ) as forward_setups,
    ):
        assert await async_setup_entry(hass, entry)
        forward_setups.assert_awaited_once()


async def test_setup_entry_model_errors_create_repair_issue(
    hass: HomeAssistant,
) -> None:
    """Test reconfigurable model errors create a repair issue."""
    entry = _entry((_model_subentry(), _conversation_subentry()))
    entry.add_to_hass(hass)

    with (
        patch(
            "custom_components.pydantic_ai_agent.async_probe_model",
            new_callable=AsyncMock,
            side_effect=ProviderValidationError("invalid_model", "model unavailable"),
        ),
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new_callable=AsyncMock,
        ),
    ):
        assert await async_setup_entry(hass, entry)

    issue = ir.async_get(hass).async_get_issue(
        DOMAIN, model_validation_issue_id(entry, "model_profile_1", {})
    )
    assert issue is not None
    assert issue.translation_key == "model_validation_failed"
    assert issue.translation_placeholders == {
        "entry_title": "Hosted OpenAI",
        "model": "gpt-test",
        "reason": "invalid_model",
        "error_message": "model unavailable",
    }


async def test_setup_entry_skips_foreign_fallback_validation_error(
    hass: HomeAssistant,
) -> None:
    """Test foreign fallback validation failures do not block consumer setup."""
    foreign_entry = _entry(
        (_model_subentry("foreign_model", model="foreign-model"),),
        {CONF_NAME: "Fallback OpenAI", CONF_API_KEY: "sk-foreign"},
    )
    foreign_entry.add_to_hass(hass)
    with (
        patch(
            "custom_components.pydantic_ai_agent.async_probe_model",
            new_callable=AsyncMock,
        ),
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new_callable=AsyncMock,
        ),
    ):
        await hass.config_entries.async_setup(foreign_entry.entry_id)
        await hass.async_block_till_done()

    ai_task_subentry = _ai_task_subentry(model_subentry_id="task_model_profile")
    ai_task_data = ai_task_subentry["data"]
    assert isinstance(ai_task_data, dict)
    ai_task_data[CONF_FALLBACK_MODEL_SUBENTRY_IDS] = [
        model_profile_ref(foreign_entry.entry_id, "foreign_model")
    ]
    entry = _entry(
        (
            _model_subentry(
                "task_model_profile", name="Task Model", model="task-model"
            ),
            ai_task_subentry,
        )
    )
    entry.add_to_hass(hass)

    async def probe_model(
        hass: HomeAssistant,
        provider_data: Mapping[str, object],
        model: str,
        model_settings: Mapping[str, object],
        *,
        structured_output_mode: str | None = None,
    ) -> None:
        if provider_data is foreign_entry.data:
            raise ProviderValidationError("invalid_auth", "foreign auth failed")

    with (
        patch(
            "custom_components.pydantic_ai_agent.async_probe_model",
            new_callable=AsyncMock,
            side_effect=probe_model,
        ),
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new_callable=AsyncMock,
        ) as forward_setups,
    ):
        assert await async_setup_entry(hass, entry)

    forward_setups.assert_awaited_once()
    issue = ir.async_get(hass).async_get_issue(
        DOMAIN,
        model_validation_issue_id(
            entry, model_profile_ref(foreign_entry.entry_id, "foreign_model"), {}
        ),
    )
    assert issue is None


async def test_setup_entry_success_clears_model_validation_repair_issue(
    hass: HomeAssistant,
) -> None:
    """Test successful setup clears stale model validation repair issues."""
    entry = _entry((_model_subentry(), _conversation_subentry()))
    entry.add_to_hass(hass)
    ir.async_create_issue(
        hass,
        DOMAIN,
        model_validation_issue_id(entry, "model_profile_1", {}),
        is_fixable=True,
        severity=ir.IssueSeverity.ERROR,
        translation_key="model_validation_failed",
    )

    with (
        patch(
            "custom_components.pydantic_ai_agent.async_probe_model",
            new_callable=AsyncMock,
        ),
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new_callable=AsyncMock,
        ),
    ):
        assert await async_setup_entry(hass, entry)

    assert (
        ir.async_get(hass).async_get_issue(
            DOMAIN, model_validation_issue_id(entry, "model_profile_1", {})
        )
        is None
    )


async def test_setup_entry_model_errors_create_separate_repair_issues_for_settings(
    hass: HomeAssistant,
) -> None:
    """Test identical models with different settings get separate repairs."""
    first_settings = {"timeout": 20.0}
    second_settings = {"extra_body": {"service_tier": "flex"}}
    entry = _entry(
        (
            _model_subentry(
                "first_model",
                name="First Model",
                model="shared-model",
                model_settings=first_settings,
            ),
            _model_subentry(
                "second_model",
                name="Second Model",
                model="shared-model",
                model_settings=second_settings,
            ),
        )
    )
    entry.add_to_hass(hass)

    with (
        patch(
            "custom_components.pydantic_ai_agent.async_probe_model",
            new_callable=AsyncMock,
            side_effect=(
                ProviderValidationError("invalid_model", "model unavailable"),
                ProviderValidationError("permission_denied", "permission denied"),
            ),
        ),
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new_callable=AsyncMock,
        ),
    ):
        assert await async_setup_entry(hass, entry)

    issue_registry = ir.async_get(hass)
    assert (
        issue_registry.async_get_issue(
            DOMAIN, model_validation_issue_id(entry, "first_model", first_settings)
        )
        is not None
    )
    assert (
        issue_registry.async_get_issue(
            DOMAIN, model_validation_issue_id(entry, "second_model", second_settings)
        )
        is not None
    )


async def test_setup_entry_transient_failure_preserves_existing_repair_issue(
    hass: HomeAssistant,
) -> None:
    """Test transient setup failures do not clear unrelated model repairs."""
    entry = _entry(
        (
            _model_subentry("first_model", name="First Model", model="first-model"),
            _model_subentry("bad_model", name="Bad Model", model="bad-model"),
        )
    )
    entry.add_to_hass(hass)
    issue_id = model_validation_issue_id(entry, "bad_model", {})
    ir.async_create_issue(
        hass,
        DOMAIN,
        issue_id,
        is_fixable=True,
        severity=ir.IssueSeverity.ERROR,
        translation_key="model_validation_failed",
    )

    with patch(
        "custom_components.pydantic_ai_agent.async_probe_model",
        new_callable=AsyncMock,
        side_effect=ProviderValidationError("timeout", "provider unavailable"),
    ):
        with pytest.raises(ConfigEntryNotReady):
            await async_setup_entry(hass, entry)

    assert ir.async_get(hass).async_get_issue(DOMAIN, issue_id) is not None


async def test_unload_entry_unloads_platforms(hass: HomeAssistant) -> None:
    """Test unload delegates platform cleanup."""
    entry = _entry()
    entry.add_to_hass(hass)
    issue_id = f"logfire_token_conflict_{entry.entry_id}"
    ir.async_create_issue(
        hass,
        DOMAIN,
        issue_id,
        is_fixable=False,
        severity=ir.IssueSeverity.WARNING,
        translation_key="logfire_token_conflict",
    )

    with patch.object(
        hass.config_entries,
        "async_unload_platforms",
        new_callable=AsyncMock,
        return_value=True,
    ) as unload_platforms:
        assert await async_unload_entry(hass, entry)

    unload_platforms.assert_awaited_once()
    assert ir.async_get(hass).async_get_issue(DOMAIN, issue_id) is None


async def test_remove_entry_cleans_entry_repair_issues(hass: HomeAssistant) -> None:
    """Test permanent removal deletes entry-owned repair issues."""
    entry = _entry((_model_subentry(),))
    entry.add_to_hass(hass)
    logfire_issue_id = f"logfire_token_conflict_{entry.entry_id}"
    model_issue_id = model_validation_issue_id(entry, "model_profile_1", {})
    unrelated_entry = _entry()
    unrelated_model_issue_id = model_validation_issue_id(
        unrelated_entry, "model_profile_1", {}
    )
    for issue_id, translation_key in (
        (logfire_issue_id, "logfire_token_conflict"),
        (model_issue_id, "model_validation_failed"),
        (unrelated_model_issue_id, "model_validation_failed"),
    ):
        ir.async_create_issue(
            hass,
            DOMAIN,
            issue_id,
            is_fixable=True,
            severity=ir.IssueSeverity.WARNING,
            translation_key=translation_key,
        )

    await async_remove_entry(hass, entry)

    issue_registry = ir.async_get(hass)
    assert issue_registry.async_get_issue(DOMAIN, logfire_issue_id) is None
    assert issue_registry.async_get_issue(DOMAIN, model_issue_id) is None
    assert issue_registry.async_get_issue(DOMAIN, unrelated_model_issue_id) is not None
