"""Normalize models.dev catalog data for the provider setup wizard."""

from collections.abc import Mapping
from urllib.parse import urlparse

from .mapping import supported_drivers_for_provider
from .types import (
    CatalogModelOption,
    CatalogProviderOption,
    CompactCatalog,
    sorted_unique_strings,
)


_ENDPOINT_SUFFIXES = {
    ("audio", "speech"),
    ("audio", "transcriptions"),
    ("audio", "translations"),
    ("batches",),
    ("chat", "completions"),
    ("completions",),
    ("embeddings",),
    ("files",),
    ("fine_tuning", "jobs"),
    ("images", "edits"),
    ("images", "generations"),
    ("images", "variations"),
    ("messages",),
    ("models",),
    ("moderations",),
    ("responses",),
    ("threads",),
}
_ENDPOINT_PATH_ENDINGS = (":generatecontent", ":streamgeneratecontent")


def normalize_catalog(payload: Mapping[str, object]) -> CompactCatalog:
    """Return a compact catalog from a models.dev payload."""
    providers: dict[str, CatalogProviderOption] = {}
    models_by_provider: dict[str, tuple[CatalogModelOption, ...]] = {}
    for raw_provider_id, raw_provider in payload.items():
        if not isinstance(raw_provider_id, str) or not isinstance(raw_provider, Mapping):
            continue
        provider_id = _string(raw_provider.get("id")) or raw_provider_id
        supported_drivers = supported_drivers_for_provider(
            provider_id, raw_provider.get("npm")
        )
        if not supported_drivers:
            continue

        models_payload = raw_provider.get("models")
        if not isinstance(models_payload, Mapping):
            continue
        models = _normalize_models(provider_id, models_payload)
        if not models:
            continue

        providers[provider_id] = CatalogProviderOption(
            id=provider_id,
            name=_string(raw_provider.get("name")) or provider_id,
            doc_url=_string(raw_provider.get("doc")) or "",
            api_key_hints=sorted_unique_strings(raw_provider.get("env", ())),
            default_base_url=_http_url(raw_provider.get("api")),
            supported_drivers=supported_drivers,
            model_count=len(models),
            families=sorted_unique_strings(model.family for model in models),
        )
        models_by_provider[provider_id] = models
    return CompactCatalog(providers=providers, models_by_provider=models_by_provider)


def _normalize_models(
    provider_id: str, models_payload: Mapping[str, object]
) -> tuple[CatalogModelOption, ...]:
    """Return compact models from one provider's raw model payload."""
    models: list[CatalogModelOption] = []
    for raw_model_id, raw_model in models_payload.items():
        if not isinstance(raw_model_id, str) or not isinstance(raw_model, Mapping):
            continue
        model_id = _string(raw_model.get("id")) or raw_model_id
        modalities = raw_model.get("modalities")
        output_modalities: object = ()
        if isinstance(modalities, Mapping):
            output_modalities = modalities.get("output", ())
        limit = raw_model.get("limit")
        context_limit = output_limit = 0
        if isinstance(limit, Mapping):
            context_limit = _int(limit.get("context"))
            output_limit = _int(limit.get("output"))
        models.append(
            CatalogModelOption(
                id=model_id,
                name=_string(raw_model.get("name")) or model_id,
                provider_id=provider_id,
                family=_string(raw_model.get("family")),
                tool_call=raw_model.get("tool_call") is True,
                structured_output=_optional_bool(raw_model.get("structured_output")),
                reasoning=raw_model.get("reasoning") is True,
                attachment=raw_model.get("attachment") is True,
                text_output=_has_text_output(output_modalities),
                context_limit=context_limit,
                output_limit=output_limit,
                status=_string(raw_model.get("status")),
            )
        )
    return tuple(sorted(models, key=lambda model: (model.name.casefold(), model.id)))


def _string(value: object) -> str | None:
    """Return a stripped string or None."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _http_url(value: object) -> str | None:
    """Return an HTTP URL string or None."""
    value = _string(value)
    if value is None or not value.startswith(("https://", "http://")):
        return None
    if _endpoint_suffix(value) is not None:
        return None
    return value.rstrip("/")


def _endpoint_suffix(value: str) -> str | None:
    """Return a forbidden endpoint suffix if the URL points at one."""
    parsed = urlparse(value)
    path = parsed.path.rstrip("/").lower()
    for ending in _ENDPOINT_PATH_ENDINGS:
        if path.endswith(ending):
            return ending.lstrip(":")
    segments = tuple(segment for segment in parsed.path.split("/") if segment)
    lowered = tuple(segment.lower() for segment in segments)
    for suffix in _ENDPOINT_SUFFIXES:
        if len(lowered) >= len(suffix) and lowered[-len(suffix) :] == suffix:
            return "/".join(suffix)
    return None


def _optional_bool(value: object) -> bool | None:
    """Return a bool only when the payload explicitly contains one."""
    return value if isinstance(value, bool) else None


def _has_text_output(output_modalities: object) -> bool:
    """Return if output modalities include text."""
    if not isinstance(output_modalities, list | tuple):
        return False
    return any(value == "text" for value in output_modalities)


def _int(value: object) -> int:
    """Return a non-negative int from payload data."""
    return value if isinstance(value, int) and value >= 0 else 0
