"""Test setup for Pydantic AI Agent."""

from unittest.mock import AsyncMock, call, patch

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
    async_setup_entry,
    async_unload_entry,
)
from custom_components.pydantic_ai_agent.config_flow import ProviderValidationError
from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_MODEL,
    CONF_MODEL_SETTINGS,
    CONF_OUTPUT_MODE,
    CONF_PROVIDER_MODE,
    DOMAIN,
    OUTPUT_MODE_NATIVE,
    OUTPUT_MODE_TOOL,
    PROVIDER_OPENAI,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
)
from custom_components.pydantic_ai_agent.repairs import model_validation_issue_id


def _conversation_subentry() -> dict[str, object]:
    """Return a conversation config subentry."""
    return {
        "data": {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_MODEL: "gpt-test",
        },
        "subentry_type": SUBENTRY_TYPE_CONVERSATION,
        "title": "Kitchen Agent",
        "unique_id": None,
    }


def _ai_task_subentry(output_mode: str | None = None) -> dict[str, object]:
    """Return an AI task data config subentry."""
    data = {CONF_MODEL: "task-model"}
    if output_mode is not None:
        data[CONF_OUTPUT_MODE] = output_mode
    return {
        "data": data,
        "subentry_type": SUBENTRY_TYPE_AI_TASK,
        "title": "Task Model",
        "unique_id": None,
    }


def _entry(subentries_data: tuple[dict[str, object], ...] = ()) -> MockConfigEntry:
    """Return a config entry."""
    return MockConfigEntry(
        domain=DOMAIN,
        title="Hosted OpenAI",
        data={
            CONF_NAME: "Hosted OpenAI",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI,
            CONF_API_KEY: "sk-test",
        },
        source=config_entries.SOURCE_USER,
        subentries_data=subentries_data,
        options={},
        unique_id=None,
    )


async def test_setup_entry_stores_runtime_data(hass: HomeAssistant) -> None:
    """Test setup stores provider runtime data."""
    entry = _entry((_conversation_subentry(), _ai_task_subentry()))
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

    assert entry.runtime_data.provider_mode == PROVIDER_OPENAI
    assert entry.runtime_data.name == "Hosted OpenAI"
    assert entry.runtime_data.api_key == "sk-test"
    assert entry.runtime_data.base_url is None
    probe_model.assert_has_awaits(
        [
            call(hass, entry.data, "gpt-test", {}),
            call(
                hass,
                entry.data,
                "task-model",
                {},
                structured_output_mode=OUTPUT_MODE_TOOL,
            ),
        ]
    )
    assert probe_model.await_count == 2


async def test_setup_entry_validates_ai_task_selected_output_mode(
    hass: HomeAssistant,
) -> None:
    """Test setup validates AI task models with their configured output mode."""
    entry = _entry((_ai_task_subentry(OUTPUT_MODE_NATIVE),))
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

    probe_model.assert_awaited_once_with(
        hass,
        entry.data,
        "task-model",
        {},
        structured_output_mode=OUTPUT_MODE_NATIVE,
    )


def test_platforms_include_conversation_and_ai_task() -> None:
    """Test setup forwards both runtime platforms."""
    assert PLATFORMS == (Platform.CONVERSATION, Platform.AI_TASK)


async def test_setup_entry_validates_each_subentry_model_settings(
    hass: HomeAssistant,
) -> None:
    """Test setup validates each unique model and settings combination."""
    entry = _entry(
        (
            {
                "data": {
                    CONF_AGENT_NAME: "Kitchen Agent",
                    CONF_MODEL: "shared-model",
                    CONF_MODEL_SETTINGS: {"timeout": 20.0},
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
            },
            {
                "data": {
                    CONF_AGENT_NAME: "Garage Agent",
                    CONF_MODEL: "shared-model",
                    CONF_MODEL_SETTINGS: {"extra_body": {"service_tier": "flex"}},
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Garage Agent",
                "unique_id": None,
            },
            {
                "data": {
                    CONF_AGENT_NAME: "Garage Agent Copy",
                    CONF_MODEL: "shared-model",
                    CONF_MODEL_SETTINGS: {"extra_body": {"service_tier": "flex"}},
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Garage Agent Copy",
                "unique_id": None,
            },
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

    assert entry.runtime_data.provider_mode == PROVIDER_OPENAI


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
            CONF_PROVIDER_MODE: PROVIDER_OPENAI,
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
    entry = _entry((_conversation_subentry(),))
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
    entry = _entry((_conversation_subentry(),))
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
    entry = _entry((_conversation_subentry(),))
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
        DOMAIN, model_validation_issue_id(entry, "gpt-test", {})
    )
    assert issue is not None
    assert issue.translation_key == "model_validation_failed"
    assert issue.translation_placeholders == {
        "entry_title": "Hosted OpenAI",
        "model": "gpt-test",
        "reason": "invalid_model",
        "error_message": "model unavailable",
    }


async def test_setup_entry_success_clears_model_validation_repair_issue(
    hass: HomeAssistant,
) -> None:
    """Test successful setup clears stale model validation repair issues."""
    entry = _entry((_conversation_subentry(),))
    entry.add_to_hass(hass)
    ir.async_create_issue(
        hass,
        DOMAIN,
        model_validation_issue_id(entry, "gpt-test", {}),
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
            DOMAIN, model_validation_issue_id(entry, "gpt-test", {})
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
            {
                "data": {
                    CONF_AGENT_NAME: "Kitchen Agent",
                    CONF_MODEL: "shared-model",
                    CONF_MODEL_SETTINGS: first_settings,
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
            },
            {
                "data": {
                    CONF_AGENT_NAME: "Garage Agent",
                    CONF_MODEL: "shared-model",
                    CONF_MODEL_SETTINGS: second_settings,
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Garage Agent",
                "unique_id": None,
            },
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
            DOMAIN, model_validation_issue_id(entry, "shared-model", first_settings)
        )
        is not None
    )
    assert (
        issue_registry.async_get_issue(
            DOMAIN, model_validation_issue_id(entry, "shared-model", second_settings)
        )
        is not None
    )


async def test_setup_entry_transient_failure_preserves_existing_repair_issue(
    hass: HomeAssistant,
) -> None:
    """Test transient setup failures do not clear unrelated model repairs."""
    entry = _entry(
        (
            {
                "data": {CONF_AGENT_NAME: "First Agent", CONF_MODEL: "first-model"},
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "First Agent",
                "unique_id": None,
            },
            {
                "data": {CONF_AGENT_NAME: "Bad Agent", CONF_MODEL: "bad-model"},
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Bad Agent",
                "unique_id": None,
            },
        )
    )
    entry.add_to_hass(hass)
    issue_id = model_validation_issue_id(entry, "bad-model", {})
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

    with patch.object(
        hass.config_entries,
        "async_unload_platforms",
        new_callable=AsyncMock,
        return_value=True,
    ) as unload_platforms:
        assert await async_unload_entry(hass, entry)

    unload_platforms.assert_awaited_once()
