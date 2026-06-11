"""Setup-time model validation probes and repair-issue management."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from ._types import PydanticAIAgentConfigEntry
from .const import (
    CONF_ENABLED,
    CONF_FALLBACK_MODEL_REFS,
    CONF_MODEL,
    CONF_MODEL_SETTINGS,
    CONF_OUTPUT_MODE,
    CONF_PRIMARY_MODEL_REF,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_PROVIDER,
)
from .model_profiles import parse_model_profile_ref, provider_model_profiles
from .model_settings import (
    normalise_applied_model_settings,
    validation_probe_model_settings,
)
from .provider_validation import ProviderValidationError, async_probe_model
from .repair_issues import (
    async_create_model_validation_issue,
    async_create_provider_auth_issue,
    async_delete_model_validation_issue,
    async_delete_provider_auth_issue,
    async_delete_stale_model_validation_issues,
    async_delete_stale_provider_auth_issues,
    model_validation_issue_id,
    provider_validation_is_auth_failure,
)
from .structured_output import structured_output_mode

_LOGGER = logging.getLogger(__name__)

_MODEL_VALIDATION_OUTPUT_MODE_KEY = "_pydantic_ai_agent_output_mode"


@dataclass(frozen=True, kw_only=True)
class _ConfiguredModelProbe:
    """One setup-time model validation probe."""

    provider_subentry: ConfigSubentry
    issue_profile_id: str
    failure_keys: tuple[str, ...]
    model: str
    model_settings: dict[str, Any]
    output_mode: str | None


def _configured_subentry_models(
    entry: PydanticAIAgentConfigEntry,
) -> list[_ConfiguredModelProbe]:
    """Return unique model probes needed before the entry can load."""
    models: list[_ConfiguredModelProbe] = []
    seen: dict[tuple[str, str, str, str | None], int] = {}

    for subentry in entry.subentries.values():
        if subentry.subentry_type not in (
            SUBENTRY_TYPE_CONVERSATION,
            SUBENTRY_TYPE_AI_TASK,
        ):
            continue
        if (
            not isinstance(
                primary_ref := subentry.data.get(CONF_PRIMARY_MODEL_REF),
                str,
            )
            or not primary_ref
        ):
            _LOGGER.warning(
                "Skipping legacy %s subentry without model profile: %s",
                subentry.subentry_type,
                subentry.subentry_id,
            )
            continue
        _append_configured_subentry_models(entry, subentry, primary_ref, models, seen)
    return models


def _append_configured_subentry_models(
    entry: PydanticAIAgentConfigEntry,
    subentry: ConfigSubentry,
    primary_ref: str,
    models: list[_ConfiguredModelProbe],
    seen: dict[tuple[str, str, str, str | None], int],
) -> None:
    """Append provider probes referenced by one conversation or AI task subentry."""
    fallback_refs = subentry.data.get(CONF_FALLBACK_MODEL_REFS, [])
    if isinstance(fallback_refs, str) or not isinstance(fallback_refs, list):
        fallback_refs = []
    output_mode = (
        structured_output_mode(subentry.data.get(CONF_OUTPUT_MODE))
        if subentry.subentry_type == SUBENTRY_TYPE_AI_TASK
        else None
    )
    refs = [primary_ref, *[ref for ref in fallback_refs if isinstance(ref, str)]]
    for ref in refs:
        provider_subentry = _provider_subentry_for_profile_ref(entry, subentry, ref)
        if provider_subentry is None:
            continue
        _add_configured_model_probe(
            provider_subentry,
            subentry.subentry_id,
            ref,
            subentry.data,
            output_mode,
            models,
            seen,
        )


def _provider_subentry_for_profile_ref(
    entry: PydanticAIAgentConfigEntry,
    subentry: ConfigSubentry,
    profile_ref: str,
) -> ConfigSubentry | None:
    """Return the provider subentry referenced by one stored model profile ref."""
    try:
        provider_subentry_id, _profile_id = parse_model_profile_ref(profile_ref)
    except HomeAssistantError:
        _LOGGER.warning(
            "Skipping malformed model profile reference %s for subentry %s",
            profile_ref,
            subentry.subentry_id,
        )
        return None
    provider_subentry = entry.subentries.get(provider_subentry_id)
    if (
        provider_subentry is None
        or provider_subentry.subentry_type != SUBENTRY_TYPE_PROVIDER
    ):
        _LOGGER.warning(
            "Skipping stale model profile reference %s for subentry %s",
            profile_ref,
            subentry.subentry_id,
        )
        return None
    return provider_subentry


def _add_configured_model_probe(
    provider_subentry: ConfigSubentry,
    subentry_id: str,
    profile_ref: str,
    subentry_data: Mapping[str, Any],
    output_mode: str | None,
    models: list[_ConfiguredModelProbe],
    seen: dict[tuple[str, str, str, str | None], int],
) -> None:
    """Add or merge one unique configured-model probe for setup validation."""
    provider_subentry_id, profile_id = parse_model_profile_ref(profile_ref)
    if provider_subentry.subentry_id != provider_subentry_id:
        return
    profile = provider_model_profiles(provider_subentry).get(profile_id)
    if profile is None or not bool(profile.get(CONF_ENABLED, False)):
        return
    model = profile.get(CONF_MODEL)
    if not isinstance(model, str) or not model:
        return
    settings = profile.get(CONF_MODEL_SETTINGS)
    model_settings = validation_probe_model_settings(
        settings if isinstance(settings, Mapping) else {}, subentry_data
    )
    dedupe_key = (
        provider_subentry.subentry_id,
        model,
        normalise_applied_model_settings(model_settings),
        output_mode,
    )
    failure_key = f"{subentry_id}:{profile_ref}"
    if dedupe_key in seen:
        index = seen[dedupe_key]
        probe = models[index]
        models[index] = replace(
            probe,
            failure_keys=(*probe.failure_keys, failure_key),
        )
        return
    seen[dedupe_key] = len(models)
    models.append(
        _ConfiguredModelProbe(
            provider_subentry=provider_subentry,
            issue_profile_id=profile_ref,
            failure_keys=(failure_key,),
            model=model,
            model_settings=model_settings,
            output_mode=output_mode,
        )
    )


def _repair_issue_model_settings(
    model_settings: Mapping[str, Any], output_mode: str | None
) -> dict[str, Any]:
    """Return settings material that separates chat and structured probes."""
    if output_mode is None:
        return dict(model_settings)
    return {
        **model_settings,
        _MODEL_VALIDATION_OUTPUT_MODE_KEY: output_mode,
    }


async def _async_validate_configured_models(
    hass: HomeAssistant, entry: PydanticAIAgentConfigEntry
) -> dict[str, str]:
    """Probe configured models and surface provider/profile repairs."""
    current_issue_ids: set[str] = set()
    auth_failure_provider_ids: set[str] = set()
    validation_failures: dict[str, str] = {}
    for probe in _configured_subentry_models(entry):
        repair_settings = _repair_issue_model_settings(
            probe.model_settings, probe.output_mode
        )
        current_issue_ids.add(
            model_validation_issue_id(entry, probe.issue_profile_id, repair_settings)
        )
        try:
            if probe.output_mode is None:
                await async_probe_model(
                    hass,
                    probe.provider_subentry.data,
                    probe.model,
                    probe.model_settings,
                )
            else:
                await async_probe_model(
                    hass,
                    probe.provider_subentry.data,
                    probe.model,
                    probe.model_settings,
                    structured_output_mode=probe.output_mode,
                )
        except ProviderValidationError as err:
            _LOGGER.warning(
                'Provider validation failed during setup for model "%s": '
                "reason=%s status_code=%s",
                probe.model,
                err.reason,
                err.status_code,
            )
            async_create_model_validation_issue(
                hass,
                entry,
                probe.issue_profile_id,
                probe.model,
                repair_settings,
                err,
            )
            validation_failures.update(
                {failure_key: err.reason for failure_key in probe.failure_keys}
            )
            if provider_validation_is_auth_failure(err):
                auth_failure_provider_ids.add(probe.provider_subentry.subentry_id)
                async_create_provider_auth_issue(
                    hass,
                    entry,
                    probe.provider_subentry.subentry_id,
                    probe.provider_subentry.title,
                    err,
                )
            continue
        async_delete_model_validation_issue(
            hass,
            entry,
            probe.issue_profile_id,
            repair_settings,
        )
    async_delete_stale_model_validation_issues(hass, entry, current_issue_ids)
    current_provider_ids = {
        subentry.subentry_id
        for subentry in entry.subentries.values()
        if subentry.subentry_type == SUBENTRY_TYPE_PROVIDER
    }
    for provider_id in current_provider_ids - auth_failure_provider_ids:
        async_delete_provider_auth_issue(hass, entry, provider_id)
    async_delete_stale_provider_auth_issues(
        hass,
        entry,
        current_provider_ids,
    )
    return validation_failures
