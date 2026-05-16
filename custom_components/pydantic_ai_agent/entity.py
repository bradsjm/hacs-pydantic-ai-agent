"""Shared Pydantic AI entity runtime."""

from collections.abc import Mapping
import logging
from typing import Any

from pydantic_ai.direct import model_request_stream
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm

from . import PydanticAIAgentConfigEntry
from .const import (
    CONF_MODEL,
    CONF_MODEL_SETTINGS,
    CONF_OUTPUT_MODE,
    DEFAULT_TIMEOUT,
    DOMAIN,
    SUBENTRY_TYPE_CONVERSATION,
)
from .ha_toolset import tool_definitions_from_llm_api
from .history import chat_log_content_to_model_messages
from .provider import openai_chat_model
from .structured_output import (
    default_structure_serializer,
    output_tool_names,
    structured_model_request_parameters,
    structured_output_json_schema,
    structured_output_mode,
    structured_output_name,
)
from .stream_adapter import model_stream_to_chat_deltas

_LOGGER = logging.getLogger(__name__)


class PydanticAIBaseLLMEntity:
    """Shared Pydantic AI streaming runtime for subentry-backed entities."""

    entry: PydanticAIAgentConfigEntry
    hass: HomeAssistant
    subentry: ConfigSubentry

    def __init__(
        self,
        entry: PydanticAIAgentConfigEntry,
        subentry: ConfigSubentry,
        *,
        name: str,
    ) -> None:
        """Initialize shared entity metadata."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=name,
            manufacturer="Pydantic AI",
            model=subentry.data[CONF_MODEL],
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure_name: str | None = None,
        structure: vol.Schema | None = None,
        max_iterations: int = 10,
    ) -> None:
        """Stream model responses until Home Assistant tool calls are resolved."""
        runtime_data = self.entry.runtime_data
        model = openai_chat_model(
            self.hass,
            api_key=runtime_data.api_key,
            base_url=runtime_data.base_url,
            model_name=self.subentry.data[CONF_MODEL],
        )
        model_settings = ModelSettings(**self._model_settings())
        # ChatLog needs a stable agent id for deltas, but entity_id can be absent
        # before Home Assistant has fully registered the entity.
        agent_id = getattr(self, "entity_id", None) or getattr(self, "unique_id", None)
        if agent_id is None:
            raise HomeAssistantError("Entity is not ready")

        output_tool_names_for_request: set[str] = set()
        if structure is not None:
            output_mode = structured_output_mode(self.subentry.data.get(CONF_OUTPUT_MODE))
            output_name = self._structured_output_name(chat_log.llm_api, structure_name)
            output_tool_names_for_request = output_tool_names(output_mode, output_name)

        for _iteration in range(max_iterations):
            try:
                # Each pass includes any HA tool results that ChatLog appended
                # during the previous pass, which lets Pydantic AI continue the
                # same tool-calling exchange.
                async with model_request_stream(
                    model,
                    await chat_log_content_to_model_messages(
                        self.hass, chat_log.content
                    ),
                    model_settings=model_settings,
                    model_request_parameters=self._model_request_parameters(
                        chat_log.llm_api,
                        structure_name=structure_name,
                        structure=structure,
                    ),
                ) as stream:
                    async for _content in chat_log.async_add_delta_content_stream(
                        agent_id,
                        model_stream_to_chat_deltas(
                            stream,
                            output_tool_names=output_tool_names_for_request,
                        ),
                    ):
                        pass
            except ModelHTTPError as err:
                # Convert provider/runtime failures into HA-facing errors so
                # conversation and AI task platforms report consistent failures.
                raise HomeAssistantError(_format_http_error(err)) from err
            except ModelAPIError as err:
                raise HomeAssistantError(_format_api_error(err)) from err
            except UnexpectedModelBehavior as err:
                raise HomeAssistantError(
                    "Provider returned an unexpected response"
                ) from err
            except TimeoutError as err:
                raise HomeAssistantError("Provider request timed out") from err
            except (NotImplementedError, UserError) as err:
                raise HomeAssistantError(
                    f"Invalid provider configuration: {err}"
                ) from err

            if not chat_log.unresponded_tool_results:
                return

        _LOGGER.warning(
            "Model exhausted tool iterations for %s with model %s",
            agent_id,
            self.subentry.data[CONF_MODEL],
        )
        raise HomeAssistantError("Model requested tools too many times")

    def _model_settings(self) -> dict[str, Any]:
        """Return subentry model settings with the integration timeout default."""
        settings: Mapping[str, Any] | None = None
        if self.subentry.subentry_type == SUBENTRY_TYPE_CONVERSATION:
            raw_settings = self.subentry.data.get(CONF_MODEL_SETTINGS)
            if isinstance(raw_settings, Mapping):
                settings = raw_settings

        model_settings = dict(settings or {})
        model_settings.setdefault("timeout", DEFAULT_TIMEOUT)
        return model_settings

    def _model_request_parameters(
        self,
        api_instance: llm.APIInstance | None,
        *,
        structure_name: str | None,
        structure: vol.Schema | None,
    ) -> ModelRequestParameters:
        """Return Pydantic AI request parameters for HA tools or AI task output."""
        if structure is None:
            return ModelRequestParameters(
                function_tools=tool_definitions_from_llm_api(api_instance)
            )

        custom_serializer = default_structure_serializer(api_instance)
        output_name = self._structured_output_name(api_instance, structure_name)
        return structured_model_request_parameters(
            function_tools=tool_definitions_from_llm_api(api_instance),
            output_mode=structured_output_mode(self.subentry.data.get(CONF_OUTPUT_MODE)),
            output_name=output_name,
            json_schema=structured_output_json_schema(
                structure,
                custom_serializer=custom_serializer,
            ),
        )

    def _structured_output_name(
        self, api_instance: llm.APIInstance | None, structure_name: str | None
    ) -> str:
        """Return an output name that cannot shadow configured HA tools."""
        return structured_output_name(
            structure_name,
            "generated_data",
            reserved_names=(
                tool.name for tool in tool_definitions_from_llm_api(api_instance)
            ),
        )


def _format_http_error(err: ModelHTTPError) -> str:
    """Return a user-facing provider HTTP error message."""
    return f'The provider returned HTTP {err.status_code} for model "{err.model_name}".'


def _format_api_error(err: ModelAPIError) -> str:
    """Return a user-facing provider API error message."""
    return f'The provider returned an API error for model "{err.model_name}".'
