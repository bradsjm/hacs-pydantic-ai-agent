"""Config subentry flow handlers for native workspace Skills."""

from .common import (
    CONF_NAME,
    SOURCE_USER,
    Any,
    ConfigEntryState,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from .skill_helpers import (
    SkillDataValidationError,
    _skill_data_from_user_input,
    _skill_schema,
)


class SkillSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing native workspace Skill subentries."""

    _options: dict[str, Any]

    @property
    def _is_new(self) -> bool:
        """Return if this flow creates a new subentry."""
        return self.source == SOURCE_USER

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        """Add a Skill subentry."""
        self._options = {}
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        """Reconfigure a Skill subentry."""
        self._options = self._get_reconfigure_subentry().data.copy()
        return await self.async_step_init(user_input)

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        """Manage native Skill options."""
        entry = self._get_entry()
        if entry.state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if user_input is None:
            return self.async_show_form(
                step_id="init",
                data_schema=_skill_schema(self._options),
            )

        try:
            data = _skill_data_from_user_input(user_input)
        except SkillDataValidationError as err:
            return self.async_show_form(
                step_id="init",
                data_schema=_skill_schema(self._options | user_input),
                errors=err.errors,
            )

        if self._is_new:
            return self.async_create_entry(title=data[CONF_NAME], data=data)
        return self.async_update_and_abort(
            entry,
            self._get_reconfigure_subentry(),
            title=data[CONF_NAME],
            data=data,
        )
