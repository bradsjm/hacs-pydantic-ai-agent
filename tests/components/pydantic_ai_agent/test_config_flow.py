"""Test the Pydantic AI Agent config flow."""

from collections.abc import AsyncIterator, Generator
from contextlib import asynccontextmanager
import errno
import logging
import socket
import ssl
from unittest.mock import AsyncMock, call, patch

from _pytest.logging import LogCaptureFixture
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
import pytest

from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import llm
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.pydantic_ai_agent.config_flow import (
    ProviderValidationError,
    async_probe_model,
    _conversation_schema,
    _format_api_error,
    _map_http_error,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_BASE_URL,
    CONF_CONFIGURE_ADVANCED_MODEL_SETTINGS,
    CONF_MODEL,
    CONF_MODEL_SETTINGS,
    CONF_PROMPT,
    CONF_PROVIDER_MODE,
    DOMAIN,
    PROVIDER_OPENAI,
    PROVIDER_OPENAI_COMPATIBLE,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
)
from custom_components.pydantic_ai_agent.conversation import (
    PydanticAIConversationEntity,
)


class _SingleEventStream:
    """Async stream with one validation event."""

    def __init__(self) -> None:
        """Initialize the stream."""
        self._yielded = False
        self.events_yielded = 0

    def __aiter__(self) -> "_SingleEventStream":
        """Return the async iterator."""
        return self

    async def __anext__(self) -> object:
        """Return one event, then stop."""
        if self._yielded:
            raise StopAsyncIteration
        self._yielded = True
        self.events_yielded += 1
        return object()


class _FailingStreamContext:
    """Async context manager that fails before streaming starts."""

    async def __aenter__(self) -> object:
        """Raise the streaming failure."""
        raise NotImplementedError("Streamed requests not supported")

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        """Do not suppress exceptions."""
        return False


class _HTTPErrorStreamContext:
    """Async context manager that fails with a provider HTTP error."""

    async def __aenter__(self) -> object:
        """Raise a provider HTTP error."""
        raise ModelHTTPError(status_code=429, model_name="gpt-test", body=None)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        """Do not suppress exceptions."""
        return False


async def _loaded_entry(
    hass: HomeAssistant, subentries_data: tuple[dict[str, object], ...] = ()
) -> MockConfigEntry:
    """Return a loaded provider config entry."""
    entry = MockConfigEntry(
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
    entry.add_to_hass(hass)
    with patch(
        "custom_components.pydantic_ai_agent.async_setup_entry",
        return_value=True,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    return entry


def test_http_error_formats_redacted_compact_metadata() -> None:
    """Test provider HTTP errors redact metadata without SDK wrapper noise."""
    err = ModelHTTPError(
        status_code=402,
        model_name="deepseek/deepseek-v4-flash:free",
        body={
            "message": "Provider returned error",
            "code": 402,
            "metadata": {
                "raw": '{"error":{"type":"insufficient_quota","code":"insufficient_quota","message":"Out of credits. Top up at /dashboard/billing to continue.","request_id":"req_ZOa6wIhd9MgPKabR"}}',
                "provider_name": "Crucible",
                "is_byok": False,
                "access_token": "secret-token",
                "request_headers": {"Authorization": "Bearer nested-secret"},
                "details": {"Authorization": "Bearer nested-secret"},
            },
        },
    )

    result = _map_http_error(err)

    assert result.reason == "provider_error"
    assert result.status_code == 402
    assert result.message.startswith(
        "The provider returned error 402 (payment issue) for model "
        '"deepseek/deepseek-v4-flash:free". Metadata: '
    )
    assert "'provider_name': 'Crucible'" in result.message
    assert "'is_byok': False" in result.message
    assert "'access_token': '**REDACTED**'" in result.message
    assert "'request_headers': '**REDACTED**'" in result.message
    assert "'Authorization': '**REDACTED**'" in result.message
    assert "secret-token" not in result.message
    assert "nested-secret" not in result.message
    assert "status_code:" not in result.message
    assert "model_name:" not in result.message
    assert "body:" not in result.message


@pytest.mark.parametrize(
    ("status_code", "expected_reason", "expected_label"),
    [
        (400, "invalid_model", "invalid request"),
        (401, "invalid_auth", "authentication issue"),
        (403, "permission_denied", "permission issue"),
        (404, "invalid_model", "model not found"),
        (408, "timeout", "timeout"),
        (429, "rate_limited", "rate limit"),
        (500, "provider_error", "provider server issue"),
    ],
)
def test_http_error_status_categories(
    status_code: int, expected_reason: str, expected_label: str
) -> None:
    """Test HTTP status codes map to stable reasons and labels."""
    err = ModelHTTPError(status_code=status_code, model_name="gpt-test", body=None)

    result = _map_http_error(err)

    assert result.reason == expected_reason
    assert result.message == (
        f"The provider returned error {status_code} ({expected_label}) "
        'for model "gpt-test".'
    )


@pytest.mark.parametrize(
    ("cause", "expected_reason", "expected_message"),
    [
        (socket.gaierror(), "cannot_connect", "Host not found."),
        (
            OSError(errno.ECONNREFUSED, "refused"),
            "cannot_connect",
            "Connection refused.",
        ),
        (
            OSError(errno.ENETUNREACH, "unreachable"),
            "cannot_connect",
            "Network unreachable.",
        ),
        (ssl.SSLError("certificate verify failed"), "cannot_connect", "TLS error."),
        (TimeoutError(), "timeout", "Request timed out."),
    ],
)
def test_api_error_connection_categories(
    cause: BaseException, expected_reason: str, expected_message: str
) -> None:
    """Test wrapped connection failures use well-defined messages."""
    err = ModelAPIError("gpt-test", "probe failed")
    err.__cause__ = cause

    result = _format_api_error(err)

    assert result.reason == expected_reason
    assert result.message == expected_message


def test_api_error_fallback_is_concise() -> None:
    """Test non-HTTP API errors avoid raw upstream exception dumps."""
    err = ModelAPIError("gpt-test", "status_code: 500, body: {'huge': 'payload'}")

    result = _format_api_error(err)

    assert result.reason == "provider_error"
    assert result.message == 'The provider returned an API error for model "gpt-test".'


async def test_probe_model_streaming_not_supported_reported(
    hass: HomeAssistant,
) -> None:
    """Test non-streaming models are reported explicitly."""
    data = {
        CONF_NAME: "Hosted OpenAI",
        CONF_PROVIDER_MODE: PROVIDER_OPENAI,
        CONF_API_KEY: "sk-test",
    }

    with (
        patch("custom_components.pydantic_ai_agent.config_flow._openai_chat_model"),
        patch(
            "custom_components.pydantic_ai_agent.config_flow.model_request_stream",
            return_value=_FailingStreamContext(),
        ),
    ):
        with pytest.raises(ProviderValidationError) as exc_info:
            await async_probe_model(hass, data, "gpt-test")

    assert exc_info.value.reason == "model_does_not_support_streaming"
    assert exc_info.value.message == "Streamed requests not supported"


async def test_probe_model_uses_streaming(hass: HomeAssistant) -> None:
    """Test provider validation completes a streaming response."""
    data = {
        CONF_NAME: "Hosted OpenAI",
        CONF_PROVIDER_MODE: PROVIDER_OPENAI,
        CONF_API_KEY: "sk-test",
    }
    stream_result = AsyncMock()
    stream_events = _SingleEventStream()

    @asynccontextmanager
    async def stream(*_: object, **__: object) -> AsyncIterator[_SingleEventStream]:
        yield stream_events

    with (
        patch(
            "custom_components.pydantic_ai_agent.config_flow._openai_chat_model",
            return_value=stream_result,
        ),
        patch(
            "custom_components.pydantic_ai_agent.config_flow.model_request_stream",
            side_effect=stream,
        ) as model_request_stream,
    ):
        await async_probe_model(hass, data, "gpt-test")

    model_request_stream.assert_called_once()
    assert model_request_stream.call_args.args[0] is stream_result
    assert model_request_stream.call_args.kwargs["model_settings"] == {"timeout": 10.0}
    assert stream_events.events_yielded == 1


async def test_probe_model_merges_configured_model_settings(
    hass: HomeAssistant,
) -> None:
    """Test provider validation preserves configured model settings."""
    data = {
        CONF_NAME: "Hosted OpenAI",
        CONF_PROVIDER_MODE: PROVIDER_OPENAI,
        CONF_API_KEY: "sk-test",
    }
    stream_events = _SingleEventStream()

    @asynccontextmanager
    async def stream(*_: object, **__: object) -> AsyncIterator[_SingleEventStream]:
        yield stream_events

    with (
        patch("custom_components.pydantic_ai_agent.config_flow._openai_chat_model"),
        patch(
            "custom_components.pydantic_ai_agent.config_flow.model_request_stream",
            side_effect=stream,
        ) as model_request_stream,
    ):
        await async_probe_model(
            hass,
            data,
            "gpt-test",
            {"temperature": 0.7, "timeout": 30.0},
        )

    assert model_request_stream.call_args.kwargs["model_settings"] == {
        "temperature": 0.7,
        "timeout": 30.0,
    }


async def test_probe_model_openai_compatible_uses_normalized_base_url(
    hass: HomeAssistant,
) -> None:
    """Test OpenAI-compatible validation builds a provider with the base URL."""
    data = {
        CONF_NAME: "Local LLM",
        CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE,
        CONF_API_KEY: "local-key",
        CONF_BASE_URL: "http://localhost:11434/v1/",
    }
    provider = object()
    model = object()
    stream_events = _SingleEventStream()

    @asynccontextmanager
    async def stream(*_: object, **__: object) -> AsyncIterator[_SingleEventStream]:
        yield stream_events

    with (
        patch(
            "pydantic_ai.providers.openai.OpenAIProvider",
            return_value=provider,
        ) as openai_provider,
        patch(
            "pydantic_ai.models.openai.OpenAIChatModel",
            return_value=model,
        ) as openai_chat_model,
        patch(
            "custom_components.pydantic_ai_agent.config_flow.model_request_stream",
            side_effect=stream,
        ) as model_request_stream,
    ):
        await async_probe_model(hass, data, "local-model")

    openai_provider.assert_called_once()
    assert openai_provider.call_args.kwargs["api_key"] == "local-key"
    assert openai_provider.call_args.kwargs["base_url"] == "http://localhost:11434/v1"
    assert openai_provider.call_args.kwargs["http_client"] is not None
    openai_chat_model.assert_called_once_with("local-model", provider=provider)
    assert model_request_stream.call_args.args[0] is model
    assert stream_events.events_yielded == 1


def test_conversation_schema_filters_unavailable_llm_apis(
    hass: HomeAssistant,
) -> None:
    """Test stale LLM API IDs are not preselected in reconfigure forms."""
    data = _conversation_schema(
        hass,
        {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_MODEL: "gpt-test",
            CONF_LLM_HASS_API: ["stale-api", llm.LLM_API_ASSIST],
        },
    )({})

    assert data[CONF_LLM_HASS_API] == [llm.LLM_API_ASSIST]


def test_conversation_schema_does_not_default_existing_disabled_tools(
    hass: HomeAssistant,
) -> None:
    """Test reconfigure forms keep Home Assistant control disabled."""
    data = _conversation_schema(
        hass,
        {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_MODEL: "gpt-test",
        },
    )({})

    assert CONF_LLM_HASS_API not in data


@pytest.fixture
def mock_probe_model() -> Generator[AsyncMock]:
    """Mock provider model probing."""
    with patch(
        "custom_components.pydantic_ai_agent.config_flow.async_probe_model",
        new_callable=AsyncMock,
    ) as flow_mock:
        flow_mock.return_value = None
        yield flow_mock


@pytest.mark.parametrize(
    ("user_input", "expected_data"),
    [
        (
            {
                CONF_NAME: "Hosted OpenAI",
                CONF_PROVIDER_MODE: PROVIDER_OPENAI,
                CONF_API_KEY: "sk-test",
            },
            {
                CONF_NAME: "Hosted OpenAI",
                CONF_PROVIDER_MODE: PROVIDER_OPENAI,
                CONF_API_KEY: "sk-test",
            },
        ),
        (
            {
                CONF_NAME: "Local LLM",
                CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE,
                CONF_API_KEY: "local-key",
                CONF_BASE_URL: "http://localhost:11434/v1/",
            },
            {
                CONF_NAME: "Local LLM",
                CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE,
                CONF_API_KEY: "local-key",
                CONF_BASE_URL: "http://localhost:11434/v1",
            },
        ),
    ],
)
async def test_config_flow_success_creates_service_entry(
    hass: HomeAssistant,
    mock_probe_model: AsyncMock,
    user_input: dict[str, str],
    expected_data: dict[str, str],
) -> None:
    """Test config flow success creates a service entry."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        user_input,
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == expected_data[CONF_NAME]
    assert result["data"] == expected_data
    assert result["subentries"] == ()
    mock_probe_model.assert_not_awaited()


async def test_openai_compatible_config_flow_requires_base_url(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test OpenAI-compatible config flow requires a base URL."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Local LLM",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE,
            CONF_API_KEY: "local-key",
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"
    assert result["errors"] == {"base": "invalid_base_url"}
    mock_probe_model.assert_not_awaited()


async def test_create_conversation_subentry(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test adding a conversation subentry."""
    entry = await _loaded_entry(hass)

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_MODEL: "gpt-test",
            CONF_PROMPT: "Be concise.",
        },
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Kitchen Agent"
    assert result["data"] == {
        CONF_AGENT_NAME: "Kitchen Agent",
        CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
        CONF_MODEL: "gpt-test",
        CONF_PROMPT: "Be concise.",
    }
    mock_probe_model.assert_awaited_once_with(hass, entry.data, "gpt-test", {})


async def test_create_conversation_subentry_without_control(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test adding a conversation subentry without Home Assistant control."""
    entry = await _loaded_entry(hass)

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_AGENT_NAME: "Local Agent",
            CONF_MODEL: "gpt-test",
            CONF_LLM_HASS_API: [],
        },
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert CONF_LLM_HASS_API not in result["data"]
    assert CONF_MODEL_SETTINGS not in result["data"]
    mock_probe_model.assert_awaited_once_with(hass, entry.data, "gpt-test", {})


async def test_create_conversation_subentry_with_main_model_settings(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test main model settings are stored under model_settings."""
    entry = await _loaded_entry(hass)
    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": config_entries.SOURCE_USER},
    )

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_MODEL: "gpt-test",
            "temperature": 0.7,
            "thinking": "high",
        },
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_MODEL_SETTINGS] == {
        "temperature": 0.7,
        "thinking": "high",
    }
    mock_probe_model.assert_awaited_once_with(
        hass,
        entry.data,
        "gpt-test",
        {"temperature": 0.7, "thinking": "high"},
    )


async def test_create_conversation_subentry_with_advanced_model_settings(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test advanced model settings are parsed and stored."""
    entry = await _loaded_entry(hass)
    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_MODEL: "gpt-test",
            "temperature": 0.4,
            "thinking": "true",
            CONF_CONFIGURE_ADVANCED_MODEL_SETTINGS: True,
        },
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "model_settings"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            "max_tokens": 1024,
            "top_p": 0.9,
            "timeout": 30.0,
            "parallel_tool_calls": True,
            "seed": 42,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.3,
            "extra_headers": '{"X-Test": "enabled"}',
            "extra_body": '{"reasoning": {"effort": "high"}}',
        },
    )

    expected_settings = {
        "temperature": 0.4,
        "thinking": True,
        "max_tokens": 1024,
        "top_p": 0.9,
        "timeout": 30.0,
        "parallel_tool_calls": True,
        "seed": 42,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.3,
        "extra_headers": {"X-Test": "enabled"},
        "extra_body": {"reasoning": {"effort": "high"}},
    }
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_MODEL_SETTINGS] == expected_settings
    mock_probe_model.assert_awaited_once_with(
        hass, entry.data, "gpt-test", expected_settings
    )


@pytest.mark.parametrize(
    ("advanced_input", "expected_errors"),
    [
        ({"max_tokens": 0}, {"max_tokens": "invalid_integer"}),
        ({"timeout": -1}, {"timeout": "positive_number"}),
        ({"extra_headers": "not-json"}, {"extra_headers": "invalid_json"}),
        ({"extra_headers": '{"X-Test": 1}'}, {"extra_headers": "invalid_headers"}),
        ({"extra_body": "[]"}, {"extra_body": "invalid_object"}),
    ],
)
async def test_conversation_advanced_model_settings_validation_errors(
    hass: HomeAssistant,
    mock_probe_model: AsyncMock,
    advanced_input: dict[str, object],
    expected_errors: dict[str, str],
) -> None:
    """Test invalid advanced model settings stay on the advanced step."""
    entry = await _loaded_entry(hass)
    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_MODEL: "gpt-test",
            CONF_CONFIGURE_ADVANCED_MODEL_SETTINGS: True,
        },
    )

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], advanced_input
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "model_settings"
    assert result["errors"] == expected_errors
    mock_probe_model.assert_not_awaited()


async def test_conversation_advanced_model_settings_probe_error_stays_on_step(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test probe errors preserve submitted advanced settings."""
    mock_probe_model.side_effect = ProviderValidationError(
        "rate_limited", 'The provider returned error 429 for model "gpt-test".', 429
    )
    entry = await _loaded_entry(hass)
    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_MODEL: "gpt-test",
            CONF_CONFIGURE_ADVANCED_MODEL_SETTINGS: True,
        },
    )

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {"timeout": 30.0},
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "model_settings"
    assert result["errors"] == {"base": "rate_limited"}
    mock_probe_model.assert_awaited_once_with(
        hass, entry.data, "gpt-test", {"timeout": 30.0}
    )


async def test_create_ai_task_data_subentry(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test adding an AI task data subentry."""
    entry = await _loaded_entry(hass)

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_AI_TASK),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {CONF_MODEL: "gpt-test"},
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "gpt-test"
    assert result["data"] == {CONF_MODEL: "gpt-test"}
    mock_probe_model.assert_awaited_once_with(hass, entry.data, "gpt-test")


async def test_reconfigure_conversation_subentry(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test reconfiguring a conversation subentry."""
    entry = await _loaded_entry(
        hass,
        (
            {
                "data": {
                    CONF_AGENT_NAME: "Kitchen Agent",
                    CONF_MODEL: "old-model",
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
            },
        ),
    )
    subentry = next(iter(entry.subentries.values()))
    unique_id = PydanticAIConversationEntity(entry, subentry).unique_id

    result = await entry.start_subentry_reconfigure_flow(hass, subentry.subentry_id)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_MODEL: "new-model",
            CONF_LLM_HASS_API: [],
        },
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert subentry.data[CONF_MODEL] == "new-model"
    assert CONF_LLM_HASS_API not in subentry.data
    assert PydanticAIConversationEntity(entry, subentry).unique_id == unique_id
    mock_probe_model.assert_awaited_once_with(hass, entry.data, "new-model", {})


async def test_reconfigure_conversation_subentry_preserves_advanced_settings(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test skipping advanced settings preserves existing advanced values."""
    entry = await _loaded_entry(
        hass,
        (
            {
                "data": {
                    CONF_AGENT_NAME: "Kitchen Agent",
                    CONF_MODEL: "old-model",
                    CONF_MODEL_SETTINGS: {
                        "temperature": 0.5,
                        "timeout": 30.0,
                        "extra_body": {"service_tier": "flex"},
                    },
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
            },
        ),
    )
    subentry = next(iter(entry.subentries.values()))

    result = await entry.start_subentry_reconfigure_flow(hass, subentry.subentry_id)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_MODEL: "new-model",
            "temperature": 0.8,
            "thinking": "false",
        },
    )

    expected_settings = {
        "temperature": 0.8,
        "thinking": False,
        "timeout": 30.0,
        "extra_body": {"service_tier": "flex"},
    }
    assert result["type"] is FlowResultType.ABORT
    assert subentry.data[CONF_MODEL_SETTINGS] == expected_settings
    mock_probe_model.assert_awaited_once_with(
        hass, entry.data, "new-model", expected_settings
    )


async def test_reconfigure_conversation_subentry_clears_advanced_settings(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test opening advanced settings allows clearing advanced values."""
    entry = await _loaded_entry(
        hass,
        (
            {
                "data": {
                    CONF_AGENT_NAME: "Kitchen Agent",
                    CONF_MODEL: "old-model",
                    CONF_MODEL_SETTINGS: {"timeout": 30.0, "extra_headers": {"X": "Y"}},
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
            },
        ),
    )
    subentry = next(iter(entry.subentries.values()))

    result = await entry.start_subentry_reconfigure_flow(hass, subentry.subentry_id)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_MODEL: "new-model",
            CONF_CONFIGURE_ADVANCED_MODEL_SETTINGS: True,
        },
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {"extra_headers": ""},
    )

    assert result["type"] is FlowResultType.ABORT
    assert CONF_MODEL_SETTINGS not in subentry.data
    mock_probe_model.assert_awaited_once_with(hass, entry.data, "new-model", {})


async def test_reconfigure_conversation_subentry_preserves_parallel_tool_calls(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test advanced reconfigure preserves a stored boolean setting."""
    entry = await _loaded_entry(
        hass,
        (
            {
                "data": {
                    CONF_AGENT_NAME: "Kitchen Agent",
                    CONF_MODEL: "old-model",
                    CONF_MODEL_SETTINGS: {
                        "parallel_tool_calls": True,
                        "timeout": 30.0,
                    },
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
            },
        ),
    )
    subentry = next(iter(entry.subentries.values()))

    result = await entry.start_subentry_reconfigure_flow(hass, subentry.subentry_id)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_MODEL: "new-model",
            CONF_CONFIGURE_ADVANCED_MODEL_SETTINGS: True,
        },
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {"parallel_tool_calls": True},
    )

    expected_settings = {"parallel_tool_calls": True}
    assert result["type"] is FlowResultType.ABORT
    assert subentry.data[CONF_MODEL_SETTINGS] == expected_settings
    mock_probe_model.assert_awaited_once_with(
        hass, entry.data, "new-model", expected_settings
    )


async def test_reconfigure_ai_task_data_subentry(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test reconfiguring an AI task data subentry."""
    entry = await _loaded_entry(
        hass,
        (
            {
                "data": {CONF_MODEL: "old-model"},
                "subentry_type": SUBENTRY_TYPE_AI_TASK,
                "title": "old-model",
                "unique_id": None,
            },
        ),
    )
    subentry = next(iter(entry.subentries.values()))

    result = await entry.start_subentry_reconfigure_flow(hass, subentry.subentry_id)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {CONF_MODEL: "new-model"},
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert subentry.data[CONF_MODEL] == "new-model"
    mock_probe_model.assert_awaited_once_with(hass, entry.data, "new-model")


@pytest.mark.parametrize(
    "subentry_type",
    [SUBENTRY_TYPE_CONVERSATION, SUBENTRY_TYPE_AI_TASK],
)
async def test_subentry_flow_aborts_when_entry_not_loaded(
    hass: HomeAssistant,
    mock_probe_model: AsyncMock,
    subentry_type: str,
) -> None:
    """Test subentry flows require a loaded provider entry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Hosted OpenAI",
        data={
            CONF_NAME: "Hosted OpenAI",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI,
            CONF_API_KEY: "sk-test",
        },
        source=config_entries.SOURCE_USER,
        options={},
        unique_id=None,
    )
    entry.add_to_hass(hass)

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, subentry_type),
        context={"source": config_entries.SOURCE_USER},
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "entry_not_loaded"
    mock_probe_model.assert_not_awaited()


async def test_config_flow_validation_error(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test provider validation errors keep the user on the subentry step."""
    mock_probe_model.side_effect = ProviderValidationError(
        "invalid_model", "The provider rejected the model."
    )
    entry = await _loaded_entry(hass)
    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": config_entries.SOURCE_USER},
    )

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_MODEL: "gpt-test",
            CONF_LLM_HASS_API: [],
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"
    assert result["errors"] == {"base": "invalid_model"}


async def test_conversation_subentry_maps_real_probe_http_error(
    hass: HomeAssistant,
) -> None:
    """Test flow error handling for an HTTP error from the provider probe."""
    entry = await _loaded_entry(hass)
    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": config_entries.SOURCE_USER},
    )

    with (
        patch("custom_components.pydantic_ai_agent.config_flow._openai_chat_model"),
        patch(
            "custom_components.pydantic_ai_agent.config_flow.model_request_stream",
            return_value=_HTTPErrorStreamContext(),
        ),
    ):
        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"],
            {
                CONF_AGENT_NAME: "Kitchen Agent",
                CONF_MODEL: "gpt-test",
                CONF_LLM_HASS_API: [],
            },
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"
    assert result["errors"] == {"base": "rate_limited"}
    assert result["description_placeholders"] == {
        "error_message": (
            'The provider returned error 429 (rate limit) for model "gpt-test".'
        ),
        "status_code": "429",
    }


async def test_config_flow_logs_rate_limit_validation_error(
    hass: HomeAssistant, mock_probe_model: AsyncMock, caplog: LogCaptureFixture
) -> None:
    """Test rate-limited provider validation is logged explicitly."""
    mock_probe_model.side_effect = ProviderValidationError(
        "rate_limited",
        'The provider returned error 429 (rate limit) for model "gpt-test".',
        429,
    )
    caplog.set_level(
        logging.WARNING, logger="custom_components.pydantic_ai_agent.config_flow"
    )
    entry = await _loaded_entry(hass)
    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": config_entries.SOURCE_USER},
    )

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_MODEL: "gpt-test",
            CONF_LLM_HASS_API: [],
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"
    assert result["errors"] == {"base": "rate_limited"}
    warning = next(
        record
        for record in caplog.records
        if record.name == "custom_components.pydantic_ai_agent.config_flow"
    )
    assert warning.levelno == logging.WARNING
    assert (
        warning.message
        == "Provider validation rate limited during conversation subentry for model "
        '"gpt-test": reason=rate_limited status_code=429'
    )
    assert "sk-test" not in warning.message


async def test_config_flow_model_behavior_error_avoids_traceback(
    hass: HomeAssistant, mock_probe_model: AsyncMock, caplog: LogCaptureFixture
) -> None:
    """Test provider response-level probe failures are shown without tracebacks."""
    message = (
        "Model token limit (3) exceeded before any response was generated. "
        "Increase the `max_tokens` model setting, or simplify the prompt to result "
        "in a shorter response that will fit within the limit."
    )
    mock_probe_model.side_effect = ProviderValidationError("provider_error", message)
    caplog.set_level(
        logging.WARNING, logger="custom_components.pydantic_ai_agent.config_flow"
    )
    entry = await _loaded_entry(hass)
    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": config_entries.SOURCE_USER},
    )

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_AGENT_NAME: "Kitchen Agent",
            CONF_MODEL: "gpt-test",
            CONF_LLM_HASS_API: [],
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"
    assert result["errors"] == {"base": "provider_error"}
    assert result["description_placeholders"] == {"error_message": message}
    warning = next(
        record
        for record in caplog.records
        if record.name == "custom_components.pydantic_ai_agent.config_flow"
    )
    assert warning.levelno == logging.WARNING
    assert (
        warning.message
        == "Provider validation failed during conversation subentry for model "
        '"gpt-test": reason=provider_error status_code=None'
    )
    assert warning.exc_info is None
    assert "sk-test" not in warning.message


async def test_duplicate_config_flow_aborts(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test duplicate provider credentials abort."""
    data = {
        CONF_NAME: "Hosted OpenAI",
        CONF_PROVIDER_MODE: PROVIDER_OPENAI,
        CONF_API_KEY: "sk-test",
    }
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Hosted OpenAI",
        data=data,
        source=config_entries.SOURCE_USER,
        options={},
        unique_id=None,
    )
    entry.add_to_hass(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        data | {CONF_NAME: "Different Display Name"},
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"
    mock_probe_model.assert_not_awaited()


async def test_reconfigure_provider_data_updates_entry(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test provider reconfigure updates parent entry data."""
    entry = await _loaded_entry(hass)

    result = await entry.start_reconfigure_flow(hass)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reconfigure"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Local LLM",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE,
            CONF_API_KEY: "local-key",
            CONF_BASE_URL: "http://localhost:11434/v1/",
        },
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert entry.data == {
        CONF_NAME: "Local LLM",
        CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE,
        CONF_API_KEY: "local-key",
        CONF_BASE_URL: "http://localhost:11434/v1",
    }
    mock_probe_model.assert_not_awaited()


async def test_reconfigure_provider_to_openai_drops_base_url(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test OpenAI mode does not keep a stale custom base URL."""
    entry = await _loaded_entry(hass)

    result = await entry.start_reconfigure_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Hosted OpenAI",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI,
            CONF_API_KEY: "sk-test",
            CONF_BASE_URL: "http://localhost:11434/v1/",
        },
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert entry.data == {
        CONF_NAME: "Hosted OpenAI",
        CONF_PROVIDER_MODE: PROVIDER_OPENAI,
        CONF_API_KEY: "sk-test",
    }
    mock_probe_model.assert_not_awaited()


async def test_reconfigure_provider_duplicate_aborts(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test provider reconfigure cannot duplicate another parent entry."""
    await _loaded_entry(hass)
    entry = MockConfigEntry(
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
    entry.add_to_hass(hass)
    with patch(
        "custom_components.pydantic_ai_agent.async_setup_entry",
        return_value=True,
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    result = await entry.start_reconfigure_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Hosted OpenAI",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI,
            CONF_API_KEY: "sk-test",
        },
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"
    assert entry.data[CONF_API_KEY] == "other-key"
    mock_probe_model.assert_not_awaited()


async def test_reconfigure_provider_requires_base_url(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test provider reconfigure validates provider-only fields."""
    entry = await _loaded_entry(hass)

    result = await entry.start_reconfigure_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Local LLM",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE,
            CONF_API_KEY: "local-key",
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reconfigure"
    assert result["errors"] == {"base": "invalid_base_url"}
    assert entry.data[CONF_API_KEY] == "sk-test"
    mock_probe_model.assert_not_awaited()


async def test_reconfigure_provider_validation_error_stays_on_form(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test provider reconfigure validates credentials when models exist."""
    entry = await _loaded_entry(
        hass,
        (
            {
                "data": {
                    CONF_AGENT_NAME: "Kitchen Agent",
                    CONF_MODEL: "old-model",
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
            },
        ),
    )
    mock_probe_model.side_effect = ProviderValidationError(
        "invalid_auth", "The provider rejected the API key."
    )

    result = await entry.start_reconfigure_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Hosted OpenAI",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI,
            CONF_API_KEY: "bad-key",
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reconfigure"
    assert result["errors"] == {"base": "invalid_auth"}
    assert entry.data[CONF_API_KEY] == "sk-test"
    mock_probe_model.assert_awaited_once()


async def test_reconfigure_provider_model_validation_error_stays_on_form(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test provider reconfigure blocks inaccessible existing models."""
    entry = await _loaded_entry(
        hass,
        (
            {
                "data": {
                    CONF_AGENT_NAME: "Kitchen Agent",
                    CONF_MODEL: "old-model",
                    CONF_MODEL_SETTINGS: {"timeout": 20.0},
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
            },
        ),
    )
    mock_probe_model.side_effect = ProviderValidationError(
        "invalid_model", "The provider rejected the model."
    )

    result = await entry.start_reconfigure_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Hosted OpenAI",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI,
            CONF_API_KEY: "new-key",
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reconfigure"
    assert result["errors"] == {"base": "invalid_model"}
    assert entry.data[CONF_API_KEY] == "sk-test"
    mock_probe_model.assert_awaited_once_with(
        hass,
        {
            CONF_NAME: "Hosted OpenAI",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI,
            CONF_API_KEY: "new-key",
        },
        "old-model",
        {"timeout": 20.0},
    )


async def test_reauth_replaces_provider_data_and_preserves_model(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test reauth replaces stale provider data without changing agent models."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Local LLM",
        data={
            CONF_NAME: "Local LLM",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE,
            CONF_API_KEY: "old-key",
            CONF_BASE_URL: "http://localhost:11434/v1",
        },
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "data": {
                    CONF_AGENT_NAME: "Local Agent",
                    CONF_MODEL: "old-model",
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Local Agent",
                "unique_id": None,
            },
            {
                "data": {CONF_MODEL: "task-model"},
                "subentry_type": SUBENTRY_TYPE_AI_TASK,
                "title": "task-model",
                "unique_id": None,
            },
        ),
        options={},
        unique_id=None,
    )
    entry.add_to_hass(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={
            "source": config_entries.SOURCE_REAUTH,
            "entry_id": entry.entry_id,
        },
        data=entry.data,
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Hosted OpenAI",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI,
            CONF_API_KEY: "new-key",
        },
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"
    assert entry.data == {
        CONF_NAME: "Hosted OpenAI",
        CONF_PROVIDER_MODE: PROVIDER_OPENAI,
        CONF_API_KEY: "new-key",
    }
    subentry = next(iter(entry.subentries.values()))
    assert subentry.data[CONF_MODEL] == "old-model"
    mock_probe_model.assert_has_awaits(
        [
            call(
                hass,
                {
                    CONF_NAME: "Hosted OpenAI",
                    CONF_PROVIDER_MODE: PROVIDER_OPENAI,
                    CONF_API_KEY: "new-key",
                },
                "old-model",
                {},
            ),
            call(
                hass,
                {
                    CONF_NAME: "Hosted OpenAI",
                    CONF_PROVIDER_MODE: PROVIDER_OPENAI,
                    CONF_API_KEY: "new-key",
                },
                "task-model",
                {},
            ),
        ]
    )


async def test_reauth_validates_each_subentry_model_settings(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test provider updates validate each subentry's own model settings."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Hosted OpenAI",
        data={
            CONF_NAME: "Hosted OpenAI",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI,
            CONF_API_KEY: "old-key",
        },
        source=config_entries.SOURCE_USER,
        subentries_data=(
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
        ),
        options={},
        unique_id=None,
    )
    entry.add_to_hass(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={
            "source": config_entries.SOURCE_REAUTH,
            "entry_id": entry.entry_id,
        },
        data=entry.data,
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Hosted OpenAI",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI,
            CONF_API_KEY: "new-key",
        },
    )

    assert result["type"] is FlowResultType.ABORT
    mock_probe_model.assert_has_awaits(
        [
            call(
                hass,
                {
                    CONF_NAME: "Hosted OpenAI",
                    CONF_PROVIDER_MODE: PROVIDER_OPENAI,
                    CONF_API_KEY: "new-key",
                },
                "shared-model",
                {"timeout": 20.0},
            ),
            call(
                hass,
                {
                    CONF_NAME: "Hosted OpenAI",
                    CONF_PROVIDER_MODE: PROVIDER_OPENAI,
                    CONF_API_KEY: "new-key",
                },
                "shared-model",
                {"extra_body": {"service_tier": "flex"}},
            ),
        ]
    )
    assert mock_probe_model.await_count == 2


async def test_reauth_validation_error_stays_on_form(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test reauth validates updated provider data before saving it."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Local LLM",
        data={
            CONF_NAME: "Local LLM",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE,
            CONF_API_KEY: "old-key",
            CONF_BASE_URL: "http://localhost:11434/v1",
        },
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "data": {
                    CONF_AGENT_NAME: "Local Agent",
                    CONF_MODEL: "old-model",
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Local Agent",
                "unique_id": None,
            },
            {
                "data": {CONF_MODEL: "task-model"},
                "subentry_type": SUBENTRY_TYPE_AI_TASK,
                "title": "task-model",
                "unique_id": None,
            },
        ),
        options={},
        unique_id=None,
    )
    entry.add_to_hass(hass)
    mock_probe_model.side_effect = [
        None,
        ProviderValidationError("invalid_auth", "The provider rejected the API key."),
    ]

    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={
            "source": config_entries.SOURCE_REAUTH,
            "entry_id": entry.entry_id,
        },
        data=entry.data,
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Hosted OpenAI",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI,
            CONF_API_KEY: "bad-key",
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reauth_confirm"
    assert result["errors"] == {"base": "invalid_auth"}
    assert entry.data[CONF_API_KEY] == "old-key"
    mock_probe_model.assert_has_awaits(
        [
            call(
                hass,
                {
                    CONF_NAME: "Hosted OpenAI",
                    CONF_PROVIDER_MODE: PROVIDER_OPENAI,
                    CONF_API_KEY: "bad-key",
                },
                "old-model",
                {},
            ),
            call(
                hass,
                {
                    CONF_NAME: "Hosted OpenAI",
                    CONF_PROVIDER_MODE: PROVIDER_OPENAI,
                    CONF_API_KEY: "bad-key",
                },
                "task-model",
                {},
            ),
        ]
    )


async def test_reauth_model_validation_error_still_updates_provider_data(
    hass: HomeAssistant, mock_probe_model: AsyncMock
) -> None:
    """Test stale subentry models do not block credential repair."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Hosted OpenAI",
        data={
            CONF_NAME: "Hosted OpenAI",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI,
            CONF_API_KEY: "old-key",
        },
        source=config_entries.SOURCE_USER,
        subentries_data=(
            {
                "data": {
                    CONF_AGENT_NAME: "Kitchen Agent",
                    CONF_MODEL: "removed-model",
                },
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "title": "Kitchen Agent",
                "unique_id": None,
            },
        ),
        options={},
        unique_id=None,
    )
    entry.add_to_hass(hass)
    mock_probe_model.side_effect = ProviderValidationError(
        "invalid_model", "The provider rejected the model."
    )

    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={
            "source": config_entries.SOURCE_REAUTH,
            "entry_id": entry.entry_id,
        },
        data=entry.data,
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Hosted OpenAI",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI,
            CONF_API_KEY: "new-key",
        },
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"
    assert entry.data[CONF_API_KEY] == "new-key"
    mock_probe_model.assert_awaited_once_with(
        hass,
        {
            CONF_NAME: "Hosted OpenAI",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI,
            CONF_API_KEY: "new-key",
        },
        "removed-model",
        {},
    )
