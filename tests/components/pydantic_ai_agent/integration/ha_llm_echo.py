"""Home Assistant LLM echo API for provider integration tests."""

import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from .config import TEST_LLM_API_ID


class EchoTool(llm.Tool):
    """LLM test tool that returns a caller-provided token."""

    name = "pydantic_ai_integration_echo"
    description = "Return the provided token exactly for integration verification."
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


class EchoAPI(llm.API):
    """LLM API exposing the integration echo tool."""

    def __init__(self, hass: HomeAssistant, calls: list[str]) -> None:
        """Initialize the API."""
        super().__init__(hass=hass, id=TEST_LLM_API_ID, name="Integration Tool API")
        self._calls = calls

    async def async_get_api_instance(
        self, llm_context: llm.LLMContext
    ) -> llm.APIInstance:
        """Return the API instance with the test echo tool."""
        return llm.APIInstance(
            self,
            "Use pydantic_ai_integration_echo "
            "when the user asks to call the integration tool.",
            llm_context,
            [EchoTool(self._calls)],
        )
