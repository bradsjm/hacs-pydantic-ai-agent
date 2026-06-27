"""Test shared entity runtime helper behavior."""

import errno
import logging
import socket
import ssl
from types import SimpleNamespace
from typing import Any, cast

from custom_components.pydantic_ai_agent.agent._entity_auth import (
    _clear_runtime_auth_failure,
    _has_provider_auth_failure,
    _record_runtime_auth_failure,
)
from custom_components.pydantic_ai_agent.agent.tool_errors import HAToolRetryExhausted
from custom_components.pydantic_ai_agent.ai_task import PydanticAIAgentAITaskEntity
from custom_components.pydantic_ai_agent.binary_sensor import (
    BINARY_SENSOR_DESCRIPTIONS,
    CONFIG_BINARY_SENSOR_DESCRIPTIONS,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_FALLBACK_MODEL_REFS,
    CONF_PROVIDER_EXTRA_BODY,
    CONF_TOOL_RETRIES,
    DEFAULT_TOOL_RETRIES,
    PROVIDER_GOOGLE_GEMINI,
)
from custom_components.pydantic_ai_agent.conversation import (
    PydanticAIConversationEntity,
)
from custom_components.pydantic_ai_agent.entity import PydanticAIBaseLLMEntity
from custom_components.pydantic_ai_agent.models.model_profiles import ModelProfile
from custom_components.pydantic_ai_agent.models.model_request_settings import (
    _model_settings_with_provider_extra_body,
)
from custom_components.pydantic_ai_agent.models.structured_output import (
    structured_agent_output_type,
)
from custom_components.pydantic_ai_agent.observability.metrics import MetricsStore
from custom_components.pydantic_ai_agent.observability.run_failures import (
    _classify_run_failure,
    _has_connection_failure,
    _home_assistant_error,
    _should_fallback,
    _ToolProblem,
)
from custom_components.pydantic_ai_agent.sensor import (
    CONFIG_SENSOR_DESCRIPTIONS,
    SENSOR_DESCRIPTIONS,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
import httpx
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    ModelRetry,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.output import ToolOutput
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits
import pytest
from tests.components.pydantic_ai_agent.support.builders import (
    ai_task_subentry_data,
    conversation_subentry_data,
    provider_runtime_data,
    provider_subentry_data,
    workspace_entry,
    workspace_runtime_data,
)


@pytest.mark.parametrize("status_code", [408, 409, 429, 500, 503])
def test_should_fallback_for_retryable_http_errors(status_code: int) -> None:
    assert _should_fallback(
        ModelHTTPError(status_code=status_code, model_name="gpt-test", body=None)
    )


@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
def test_should_not_fallback_for_non_retryable_http_errors(status_code: int) -> None:
    assert not _should_fallback(
        ModelHTTPError(status_code=status_code, model_name="gpt-test", body=None)
    )


def test_agent_entities_keep_has_entity_name() -> None:
    profile_ref = "provider-1:profile-1"
    entry = workspace_entry(
        (
            provider_subentry_data(subentry_id="provider-1"),
            conversation_subentry_data(profile_ref, subentry_id="conversation-1"),
            ai_task_subentry_data(profile_ref, subentry_id="task-1"),
        )
    )
    entry.runtime_data = workspace_runtime_data(
        providers={"provider-1": provider_runtime_data(subentry_id="provider-1")}
    )
    conv = PydanticAIConversationEntity(
        cast(Any, entry), entry.subentries["conversation-1"]
    )
    task = PydanticAIAgentAITaskEntity(cast(Any, entry), entry.subentries["task-1"])
    assert conv._attr_has_entity_name is True
    assert conv._attr_name is None
    assert task._attr_has_entity_name is True
    assert task._attr_name is None


def test_agent_entities_available_despite_runtime_provider_auth_failures() -> None:
    profile_ref = "provider-1:profile-1"
    entry = workspace_entry(
        (
            provider_subentry_data(subentry_id="provider-1"),
            conversation_subentry_data(profile_ref, subentry_id="conversation-1"),
            ai_task_subentry_data(profile_ref, subentry_id="task-1"),
        )
    )
    runtime_data = workspace_runtime_data(
        providers={"provider-1": provider_runtime_data(subentry_id="provider-1")}
    )
    runtime_data.runtime_provider_auth_failures["provider-1"] = [profile_ref]
    entry.runtime_data = runtime_data
    conv = PydanticAIConversationEntity(
        cast(Any, entry), entry.subentries["conversation-1"]
    )
    task = PydanticAIAgentAITaskEntity(cast(Any, entry), entry.subentries["task-1"])
    assert conv.available is True
    assert task.available is True


def test_agent_entities_require_primary_provider_at_runtime() -> None:
    profile_ref = "provider-1:profile-1"
    entry = workspace_entry(
        (
            provider_subentry_data(subentry_id="provider-1"),
            conversation_subentry_data(profile_ref, subentry_id="conversation-1"),
        )
    )
    entry.runtime_data = workspace_runtime_data()
    with pytest.raises(HomeAssistantError, match="provider was not found"):
        PydanticAIConversationEntity(
            cast(Any, entry), entry.subentries["conversation-1"]
        )


def test_agent_entities_require_primary_profile_even_with_fallback() -> None:
    primary_ref = "provider-1:profile-1"
    fallback_ref = "provider-2:profile-1"
    entry = workspace_entry(
        (
            provider_subentry_data(subentry_id="provider-1"),
            provider_subentry_data(subentry_id="provider-2"),
            conversation_subentry_data(
                primary_ref,
                subentry_id="conversation-1",
                extra_data={CONF_FALLBACK_MODEL_REFS: [fallback_ref]},
            ),
        )
    )
    runtime_data = workspace_runtime_data(
        providers={
            "provider-2": provider_runtime_data(subentry_id="provider-2"),
        }
    )
    entry.runtime_data = runtime_data
    with pytest.raises(HomeAssistantError, match="provider was not found"):
        PydanticAIConversationEntity(
            cast(Any, entry), entry.subentries["conversation-1"]
        )


def test_provider_auth_failure_detection_is_provider_scoped() -> None:
    entry = workspace_entry(
        (
            provider_subentry_data(subentry_id="provider-1"),
            provider_subentry_data(subentry_id="provider-2"),
        )
    )
    entry.runtime_data = workspace_runtime_data(
        providers={
            "provider-1": provider_runtime_data(subentry_id="provider-1"),
            "provider-2": provider_runtime_data(subentry_id="provider-2"),
        }
    )
    entry.runtime_data.runtime_provider_auth_failures["provider-1"] = [
        "provider-1:profile-1"
    ]
    entry.runtime_data.runtime_provider_auth_failures["provider-2"] = [
        "provider-2:profile-1"
    ]

    assert _has_provider_auth_failure(entry, "provider-1") is True
    assert _has_provider_auth_failure(entry, "provider-2") is True
    assert _has_provider_auth_failure(entry, "provider-3") is False


def test_runtime_auth_failure_cleanup_is_profile_scoped(hass: HomeAssistant) -> None:
    entry = workspace_entry((provider_subentry_data(subentry_id="provider-1"),))
    entry.add_to_hass(hass)
    entry.runtime_data = workspace_runtime_data(
        providers={"provider-1": provider_runtime_data(subentry_id="provider-1")}
    )
    failing = ModelProfile(
        ref="provider-1:failing",
        provider_subentry_id="provider-1",
        profile_id="failing",
        title="Failing",
        provider_title="Provider",
        provider_mode="openai_compatible_completions",
        model_name="failing-model",
        model_settings={},
    )
    successful = ModelProfile(
        ref="provider-1:ok",
        provider_subentry_id="provider-1",
        profile_id="ok",
        title="OK",
        provider_title="Provider",
        provider_mode="openai_compatible_completions",
        model_name="ok-model",
        model_settings={},
    )
    _record_runtime_auth_failure(entry, failing)
    _clear_runtime_auth_failure(hass, entry, successful)
    assert entry.runtime_data.runtime_provider_auth_failures == {
        "provider-1": [failing.ref]
    }
    _clear_runtime_auth_failure(hass, entry, failing)
    assert entry.runtime_data.runtime_provider_auth_failures == {}


def test_diagnostic_entity_descriptions_use_translation_keys() -> None:
    descriptions = (
        *SENSOR_DESCRIPTIONS,
        *CONFIG_SENSOR_DESCRIPTIONS,
        *BINARY_SENSOR_DESCRIPTIONS,
        *CONFIG_BINARY_SENSOR_DESCRIPTIONS,
    )
    last_mcp_tool_call = next(
        d for d in SENSOR_DESCRIPTIONS if d.key == "last_mcp_tool_call"
    )

    assert all(d.translation_key == d.key for d in descriptions)
    assert all(not isinstance(d.name, str) for d in descriptions)
    assert last_mcp_tool_call.entity_registry_enabled_default is True


def test_should_fallback_for_timeout_usage_and_transport_api_errors() -> None:
    api_error = ModelAPIError("gpt-test", "request failed")
    api_error.__cause__ = httpx.ConnectError("refused")
    assert _should_fallback(TimeoutError())
    assert _should_fallback(httpx.ReadTimeout("timeout"))
    assert _should_fallback(UsageLimitExceeded("too many requests"))
    assert _should_fallback(api_error)
    assert not _should_fallback(ModelAPIError("gpt-test", "bad request"))


@pytest.mark.parametrize(
    "cause",
    [
        httpx.TimeoutException("timeout"),
        httpx.ConnectError("refused"),
        socket.gaierror(),
        ssl.SSLError("tls"),
        OSError(errno.ECONNREFUSED, "refused"),
        OSError(errno.ENETUNREACH, "unreachable"),
        OSError(errno.EHOSTUNREACH, "host unreachable"),
    ],
)
def test_has_connection_failure_detects_transport_cause(cause: BaseException) -> None:
    err = RuntimeError("wrapper")
    err.__cause__ = RuntimeError("middle")
    err.__cause__.__context__ = cause
    assert _has_connection_failure(err)


def test_has_connection_failure_stops_on_cycles() -> None:
    err = RuntimeError("cycle")
    err.__cause__ = err
    assert not _has_connection_failure(err)


@pytest.mark.parametrize(
    ("err", "message_fragment"),
    [
        (
            ModelHTTPError(status_code=429, model_name="gpt-test", body=None),
            "quota or rate limit",
        ),
        (
            ModelAPIError("gpt-test", "failed"),
            'API error for model "gpt-test"',
        ),
        (
            UnexpectedModelBehavior("bad"),
            "unexpected response",
        ),
        (TimeoutError(), "timed out"),
        (
            UsageLimitExceeded("too many"),
            "configured usage limit",
        ),
        (
            HAToolRetryExhausted(
                tool_name="turn_on", attempts=2, reason="invalid target"
            ),
            "turn_on",
        ),
        (
            NotImplementedError("missing config"),
            "Invalid provider configuration",
        ),
        (UserError("bad config"), "Invalid provider configuration"),
    ],
)
def test_home_assistant_error_maps_runtime_failures(
    err: Exception, message_fragment: str
) -> None:
    result = str(_home_assistant_error(err))
    assert message_fragment in result


def test_home_assistant_error_preserves_existing_ha_errors() -> None:
    err = HomeAssistantError("already HA")
    assert _home_assistant_error(err) is err


def test_classify_run_failure_uses_configured_iteration_limit() -> None:
    failure = _classify_run_failure(
        UsageLimitExceeded("would exceed the request_limit of 24"),
        usage_limits=UsageLimits(request_limit=24),
        partial_response=True,
    )
    assert failure.error_type == "UsageLimitExceeded"
    assert failure.partial_response is True


def test_classify_run_failure_marks_exhausted_ha_tool_retries_actionable() -> None:
    err = HAToolRetryExhausted(tool_name="turn_on", attempts=2, reason="invalid target")
    problem = _ToolProblem(tool_name="turn_on", tool_call_id="tool-1", outcome="retry")
    failure = _classify_run_failure(err, tool_problem=problem)

    assert failure.error_type == "HAToolRetryExhausted"
    assert "unexpected response" not in failure.user_message.lower()
    assert "turn_on" in failure.user_message
    assert failure.tool_problem == problem


def test_classify_run_failure_does_not_scan_unexpected_model_behavior_chain() -> None:
    err = UnexpectedModelBehavior("tool retries exhausted")
    err.__cause__ = ModelRetry(
        'Home Assistant tool "turn_on" failed after retries were exhausted.'
    )

    failure = _classify_run_failure(err)

    assert failure.error_type == "UnexpectedModelBehavior"
    assert "unexpected response" in failure.user_message.lower()
    assert "turn_on" not in failure.user_message
    assert failure.tool_problem is None


def test_record_agent_run_failure_logs_safe_message(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    store = MetricsStore()
    entity = SimpleNamespace(
        hass=hass,
        entry=SimpleNamespace(
            entry_id="entry-1", runtime_data=SimpleNamespace(metrics=store)
        ),
        subentry=SimpleNamespace(subentry_id="subentry-1"),
        entity_id="conversation.test_agent",
    )
    err = ModelHTTPError(status_code=500, model_name="gpt-test", body="raw body")
    with caplog.at_level(logging.ERROR):
        PydanticAIBaseLLMEntity._record_agent_run_failure(
            cast(Any, entity), err, model_profile="GPT Test"
        )
    assert "raw body" not in caplog.text
    assert store.record_for("subentry-1").last_error_type == "ModelHTTPError"


def test_entity_tool_retries_defaults_to_three() -> None:
    entity = SimpleNamespace(subentry=SimpleNamespace(data={}))

    assert (
        PydanticAIBaseLLMEntity._tool_retries(cast(Any, entity)) == DEFAULT_TOOL_RETRIES
    )


def test_entity_tool_retries_honors_explicit_zero() -> None:
    entity = SimpleNamespace(subentry=SimpleNamespace(data={CONF_TOOL_RETRIES: 0}))

    assert PydanticAIBaseLLMEntity._tool_retries(cast(Any, entity)) == 0


def test_structured_output_tool_uses_configured_retry_budget() -> None:
    output_type = structured_agent_output_type(
        output_mode="tool",
        output_name="generated_data",
        json_schema={"type": "object"},
        output_tool_retries=4,
    )

    assert isinstance(output_type, ToolOutput)
    assert cast(ToolOutput, output_type).max_retries == 4


def test_model_settings_with_provider_extra_body_rejects_gemini() -> None:
    profile = ModelProfile(
        ref="p:1",
        provider_subentry_id="p",
        profile_id="1",
        title="Gemini",
        provider_title="P",
        provider_mode=PROVIDER_GOOGLE_GEMINI,
        model_name="gemini-test",
        model_settings={},
    )
    entry = cast(
        Any,
        SimpleNamespace(
            subentries={
                "p": SimpleNamespace(data={CONF_PROVIDER_EXTRA_BODY: {"tier": "flex"}})
            }
        ),
    )
    with pytest.raises(HomeAssistantError, match="OpenAI-compatible and Anthropic"):
        _model_settings_with_provider_extra_body(entry, profile, ModelSettings())
