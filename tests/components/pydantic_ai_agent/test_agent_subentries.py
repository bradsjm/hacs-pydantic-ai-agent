"""Test shared agent subentry setup helpers."""

import logging

from _pytest.logging import LogCaptureFixture
from custom_components.pydantic_ai_agent import PydanticAIAgentConfigEntry
from custom_components.pydantic_ai_agent.agent_subentries import (
    iter_valid_agent_subentries,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_FALLBACK_MODEL_REFS,
    SUBENTRY_TYPE_CONVERSATION,
)
from homeassistant.config_entries import ConfigSubentry
from homeassistant.exceptions import HomeAssistantError

from .support.builders import conversation_subentry_data, workspace_entry


def test_iter_valid_agent_subentries_yields_resolved_subentries() -> None:
    """Test valid subentries are yielded with resolver output."""
    entry = workspace_entry(
        (
            conversation_subentry_data(
                "provider-1:profile-1",
                subentry_id="conversation-1",
            ),
        )
    )

    def resolver(_entry: PydanticAIAgentConfigEntry, subentry: ConfigSubentry) -> str:
        return f"resolved:{subentry.subentry_id}"

    valid = list(
        iter_valid_agent_subentries(
            entry,
            subentry_type=SUBENTRY_TYPE_CONVERSATION,
            platform="conversation",
            resolver=resolver,
        )
    )

    assert len(valid) == 1
    assert valid[0].subentry.subentry_id == "conversation-1"
    assert valid[0].resolved == "resolved:conversation-1"


def test_iter_valid_agent_subentries_logs_safe_invalid_context(
    caplog: LogCaptureFixture,
) -> None:
    """Test expected invalid subentries are skipped with safe log context."""
    entry = workspace_entry(
        (
            conversation_subentry_data(
                "provider-1:missing",
                subentry_id="conversation-1",
                title="Kitchen Agent",
                extra_data={
                    CONF_FALLBACK_MODEL_REFS: ["provider-2:fallback"],
                    "api_key": "sk-secret",
                    "prompt": "private prompt",
                },
            ),
        )
    )

    def resolver(_entry: PydanticAIAgentConfigEntry, _subentry: ConfigSubentry) -> str:
        raise HomeAssistantError("Configured model profile was not found")

    with caplog.at_level(logging.WARNING):
        valid = list(
            iter_valid_agent_subentries(
                entry,
                subentry_type=SUBENTRY_TYPE_CONVERSATION,
                platform="conversation",
                resolver=resolver,
            )
        )

    assert valid == []
    assert "conversation-1" in caplog.text
    assert "Kitchen Agent" in caplog.text
    assert entry.entry_id in caplog.text
    assert "provider-1:missing" in caplog.text
    assert "provider-2:fallback" in caplog.text
    assert "Configured model profile was not found" in caplog.text
    assert "sk-secret" not in caplog.text
    assert "private prompt" not in caplog.text


def test_iter_valid_agent_subentries_logs_unexpected_exception(
    caplog: LogCaptureFixture,
) -> None:
    """Test unexpected resolver failures are logged and skipped."""
    entry = workspace_entry(
        (
            conversation_subentry_data(
                "provider-1:profile-1",
                subentry_id="conversation-1",
            ),
        )
    )

    def resolver(_entry: PydanticAIAgentConfigEntry, _subentry: ConfigSubentry) -> str:
        raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR):
        valid = list(
            iter_valid_agent_subentries(
                entry,
                subentry_type=SUBENTRY_TYPE_CONVERSATION,
                platform="conversation",
                resolver=resolver,
            )
        )

    assert valid == []
    assert "conversation-1" in caplog.text
    assert "provider-1:profile-1" in caplog.text
    assert "Traceback" in caplog.text
