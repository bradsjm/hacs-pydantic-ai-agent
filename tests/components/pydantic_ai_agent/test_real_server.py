"""Real-provider end-to-end tests for Pydantic AI Agent."""

from dataclasses import dataclass, field
import asyncio
import os
from pathlib import Path
import socket
from urllib.parse import urlparse

import pytest
import pytest_socket
from pydantic_ai import (
    ModelRequest,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from pydantic_ai.direct import model_request_stream
from pydantic_ai.settings import ModelSettings
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import llm
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component import plugins as ha_pytest_plugins

from custom_components.pydantic_ai_agent.config_flow import (
    ProviderValidationError,
    async_probe_model,
)
from custom_components.pydantic_ai_agent.const import (
    CONF_AGENT_NAME,
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_MODEL_SETTINGS,
    CONF_PROVIDER_MODE,
    DOMAIN,
    OUTPUT_MODE_TOOL,
    PROVIDER_OPENAI_COMPATIBLE,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
)
from custom_components.pydantic_ai_agent.provider import openai_chat_model_from_config

pytestmark = [pytest.mark.real_server, pytest.mark.usefixtures("socket_enabled")]

_REPO_ROOT = Path(__file__).parents[3]
_ENV_FILE = _REPO_ROOT / ".env"
_REQUIRED_ENV = ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL")
_CONVERSATION_SENTINEL = "PAI_E2E_CONVERSATION_OK"
_AI_TASK_SENTINEL = "PAI_E2E_AI_TASK_OK"
_AI_TASK_STRUCTURED_SENTINEL = "PAI_E2E_AI_TASK_STRUCTURED_OK"
_STREAM_SENTINEL = "PAI_E2E_STREAM_OK"
_TOOL_SENTINEL = "PAI_E2E_TOOL_OK"
_TEST_LLM_API_ID = "pydantic-ai-agent-real-test"
_REAL_SERVER_TIMEOUT = 60.0
_STRUCTURED_OUTPUT_SKIP_REASONS = {"invalid_model", "invalid_provider_config"}


@dataclass(frozen=True, kw_only=True)
class RealServerConfig:
    """Real-provider configuration loaded from the test environment."""

    api_key: str = field(repr=False)
    model: str
    base_url: str

    @property
    def provider_data(self) -> dict[str, str]:
        """Return config-entry provider data for OpenAI-compatible mode."""
        return {
            CONF_NAME: "Real OpenAI-compatible Provider",
            CONF_PROVIDER_MODE: PROVIDER_OPENAI_COMPATIBLE,
            CONF_API_KEY: self.api_key,
            CONF_BASE_URL: self.base_url,
        }


def _load_dotenv_values(path: Path) -> dict[str, str]:
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


class _Secret(str):
    """String value with a redacted repr for pytest failure output."""

    def __repr__(self) -> str:
        """Return a redacted representation."""
        return "'<redacted>'"


@pytest.fixture(name="real_server")
def fixture_real_server() -> RealServerConfig:
    """Return real-server config or skip with missing variable names."""
    file_values = _load_dotenv_values(_ENV_FILE)
    values = {
        key: os.environ.get(key, file_values.get(key, "")) for key in _REQUIRED_ENV
    }
    missing = [key for key, value in values.items() if not value]
    if missing:
        pytest.skip(
            "Real-server tests require these environment values in .env or the "
            f"process environment: {', '.join(missing)}"
        )

    return RealServerConfig(
        api_key=_Secret(values["OPENAI_API_KEY"]),
        model=values["OPENAI_MODEL"],
        base_url=values["OPENAI_BASE_URL"],
    )


@pytest.fixture(autouse=True)
def enable_real_network(
    real_server: RealServerConfig,
    socket_enabled: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow real-server tests to resolve configured provider hostnames."""
    del socket_enabled
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        getattr(ha_pytest_plugins, "_real_getaddrinfo"),
    )
    host = urlparse(real_server.base_url).hostname
    if host is None:
        pytest.skip("OPENAI_BASE_URL must include a hostname for real-server tests")
    pytest_socket.socket_allow_hosts(
        ["localhost", "127.0.0.1", "::1", host], allow_unix_socket=True
    )


@pytest.fixture(name="tool_structured_output")
async def fixture_tool_structured_output(
    hass: HomeAssistant, real_server: RealServerConfig
) -> None:
    """Skip structured AI task tests when the configured model cannot do them."""
    try:
        await async_probe_model(
            hass,
            real_server.provider_data,
            real_server.model,
            {"timeout": _REAL_SERVER_TIMEOUT},
            structured_output_mode=OUTPUT_MODE_TOOL,
        )
    except ProviderValidationError as err:
        if err.reason not in _STRUCTURED_OUTPUT_SKIP_REASONS:
            raise
        pytest.skip(
            "Configured real-server model does not support tool structured "
            f"output required by AI task E2E tests: {err.reason}: {err.message}"
        )


class _EchoTool(llm.Tool):
    """LLM test tool that returns a caller-provided token."""

    name = "pydantic_ai_e2e_echo"
    description = "Return the provided token exactly for E2E verification."
    parameters = vol.Schema({vol.Required("token"): str})

    def __init__(self, calls: list[str]) -> None:
        """Initialize the tool."""
        self._calls = calls

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict[str, str]:
        """Return the token supplied by the model."""
        del hass, llm_context
        token = str(tool_input.tool_args["token"])
        self._calls.append(token)
        return {"token": token}


class _EchoAPI(llm.API):
    """LLM API exposing the E2E echo tool."""

    def __init__(self, hass: HomeAssistant, calls: list[str]) -> None:
        """Initialize the API."""
        super().__init__(hass=hass, id=_TEST_LLM_API_ID, name="Real E2E Tool API")
        self._calls = calls

    async def async_get_api_instance(
        self, llm_context: llm.LLMContext
    ) -> llm.APIInstance:
        """Return the API instance with the test echo tool."""
        return llm.APIInstance(
            self,
            "Use pydantic_ai_e2e_echo when the user asks to call the E2E tool.",
            llm_context,
            [_EchoTool(self._calls)],
        )


def _conversation_subentry(
    real_server: RealServerConfig, llm_hass_api: list[str] | None = None
) -> dict[str, object]:
    """Return a real-server conversation subentry."""
    data: dict[str, object] = {
        CONF_AGENT_NAME: "Real Conversation Agent",
        CONF_MODEL: real_server.model,
        CONF_MODEL_SETTINGS: {"timeout": _REAL_SERVER_TIMEOUT},
    }
    if llm_hass_api is not None:
        data[CONF_LLM_HASS_API] = llm_hass_api

    return {
        "data": data,
        "subentry_type": SUBENTRY_TYPE_CONVERSATION,
        "title": "Real Conversation Agent",
        "unique_id": None,
    }


def _ai_task_subentry(real_server: RealServerConfig) -> dict[str, object]:
    """Return a real-server AI task subentry."""
    return {
        "data": {CONF_MODEL: real_server.model},
        "subentry_type": SUBENTRY_TYPE_AI_TASK,
        "title": "Real AI Task",
        "unique_id": None,
    }


def _entry(
    real_server: RealServerConfig, subentry: dict[str, object]
) -> MockConfigEntry:
    """Return a config entry for a single real-server subentry."""
    return MockConfigEntry(
        domain=DOMAIN,
        title="Real OpenAI-compatible Provider",
        data=real_server.provider_data,
        source=config_entries.SOURCE_USER,
        subentries_data=(subentry,),
        options={},
        unique_id=None,
    )


async def _setup_entry(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Add and load a real-server config entry."""
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED


async def _drain_stream_cleanup(hass: HomeAssistant) -> None:
    """Let async stream finalizers finish before HA cleanup assertions run."""
    await hass.async_block_till_done()
    await asyncio.sleep(0)
    await hass.async_block_till_done()


async def _conversation_entity_id(
    hass: HomeAssistant,
    real_server: RealServerConfig,
    llm_hass_api: list[str] | None = None,
) -> str:
    """Set up a real conversation agent and return its entity ID."""
    await _setup_entry(
        hass,
        _entry(real_server, _conversation_subentry(real_server, llm_hass_api)),
    )
    entity_ids = [
        state.entity_id
        for state in hass.states.async_all("conversation")
        if state.entity_id != "conversation.home_assistant"
    ]
    assert len(entity_ids) == 1
    return entity_ids[0]


async def _ai_task_entity_id(hass: HomeAssistant, real_server: RealServerConfig) -> str:
    """Set up a real AI task entity and return its entity ID."""
    await _setup_entry(hass, _entry(real_server, _ai_task_subentry(real_server)))
    entity_ids = [state.entity_id for state in hass.states.async_all("ai_task")]
    assert len(entity_ids) == 1
    return entity_ids[0]


def _append_text_event(text_parts: list[str], event: object) -> None:
    """Append display text from a Pydantic AI stream event."""
    if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
        text_parts.append(event.part.content)
    elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
        text_parts.append(event.delta.content_delta)
    elif isinstance(event, PartEndEvent) and isinstance(event.part, TextPart):
        text_parts.append(event.part.content)


async def test_real_server_probe_succeeds(
    hass: HomeAssistant, real_server: RealServerConfig
) -> None:
    """Test configured real-server credentials and model pass provider probing."""
    await async_probe_model(
        hass,
        real_server.provider_data,
        real_server.model,
        {"timeout": _REAL_SERVER_TIMEOUT},
    )


async def test_real_server_stream_events_include_text(
    hass: HomeAssistant, real_server: RealServerConfig
) -> None:
    """Test the configured provider emits usable Pydantic AI text stream events."""
    model = openai_chat_model_from_config(
        hass, real_server.provider_data, real_server.model
    )
    text_parts: list[str] = []
    event_count = 0

    async with model_request_stream(
        model,
        [
            ModelRequest.user_text_prompt(
                f"Reply with exactly {_STREAM_SENTINEL}. No punctuation."
            )
        ],
        model_settings=ModelSettings(timeout=_REAL_SERVER_TIMEOUT),
    ) as stream:
        async for event in stream:
            event_count += 1
            _append_text_event(text_parts, event)

    await _drain_stream_cleanup(hass)
    assert event_count > 0
    assert _STREAM_SENTINEL in "".join(text_parts)


async def test_real_server_conversation_plain_response(
    hass: HomeAssistant, real_server: RealServerConfig
) -> None:
    """Test a real provider can answer through the HA conversation API."""
    entity_id = await _conversation_entity_id(hass, real_server)

    result = await conversation.async_converse(
        hass,
        f"Reply with exactly {_CONVERSATION_SENTINEL}. No punctuation.",
        None,
        Context(),
        agent_id=entity_id,
    )

    await _drain_stream_cleanup(hass)
    assert _CONVERSATION_SENTINEL in result.response.speech["plain"]["speech"]


async def test_real_server_conversation_uses_ha_llm_tool(
    hass: HomeAssistant, real_server: RealServerConfig
) -> None:
    """Test a real provider can call a Home Assistant LLM API tool."""
    tool_calls: list[str] = []
    unregister = llm.async_register_api(hass, _EchoAPI(hass, tool_calls))
    entity_id = await _conversation_entity_id(
        hass,
        real_server,
        llm_hass_api=[_TEST_LLM_API_ID],
    )

    try:
        result = await conversation.async_converse(
            hass,
            (
                "Call the pydantic_ai_e2e_echo tool with token "
                f"{_TOOL_SENTINEL}. Then reply with exactly the token returned "
                "by the tool. Do not answer without calling the tool."
            ),
            None,
            Context(),
            agent_id=entity_id,
        )
    finally:
        unregister()

    await _drain_stream_cleanup(hass)
    assert tool_calls == [_TOOL_SENTINEL]
    speech = result.response.speech["plain"]["speech"]
    if "HTTP 400" in speech:
        pytest.skip(
            "Configured real-server provider called the HA tool but rejected "
            "the tool-result follow-up request."
        )
    assert _TOOL_SENTINEL in speech


async def test_real_server_ai_task_plain_generation(
    hass: HomeAssistant, real_server: RealServerConfig
) -> None:
    """Test a real provider can generate plain AI task data."""
    entity_id = await _ai_task_entity_id(hass, real_server)

    result = await ai_task.async_generate_data(
        hass,
        task_name="Real plain task",
        entity_id=entity_id,
        instructions=f"Return exactly {_AI_TASK_SENTINEL}. No punctuation.",
    )

    await _drain_stream_cleanup(hass)
    assert _AI_TASK_SENTINEL in str(result.data)


async def test_real_server_ai_task_structured_generation(
    hass: HomeAssistant,
    real_server: RealServerConfig,
    tool_structured_output: None,
) -> None:
    """Test a real provider can generate schema-validated AI task data."""
    del tool_structured_output
    entity_id = await _ai_task_entity_id(hass, real_server)

    result = await ai_task.async_generate_data(
        hass,
        task_name="Real structured task",
        entity_id=entity_id,
        instructions=(
            f"Generate data where result is exactly {_AI_TASK_STRUCTURED_SENTINEL}."
        ),
        structure=vol.Schema({vol.Required("result"): str}),
    )

    await _drain_stream_cleanup(hass)
    assert isinstance(result.data, dict)
    assert result.data["result"] == _AI_TASK_STRUCTURED_SENTINEL
