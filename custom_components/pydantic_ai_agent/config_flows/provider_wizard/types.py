"""Compact catalog types for the provider setup wizard."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CatalogProviderOption:
    """A supported catalog provider option."""

    id: str
    name: str
    doc_url: str
    api_key_hints: tuple[str, ...]
    default_base_url: str | None
    supported_drivers: tuple[str, ...]
    model_count: int
    families: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CatalogModelOption:
    """A compact catalog model option."""

    id: str
    name: str
    provider_id: str
    family: str | None
    tool_call: bool
    structured_output: bool | None
    reasoning: bool
    attachment: bool
    text_output: bool
    context_limit: int
    output_limit: int
    status: str | None
    input_price: float | None = None
    output_price: float | None = None
    cache_read_price: float | None = None


@dataclass(frozen=True, slots=True)
class CompactCatalog:
    """Normalized provider catalog used by the wizard."""

    providers: Mapping[str, CatalogProviderOption]
    models_by_provider: Mapping[str, tuple[CatalogModelOption, ...]]

    def sorted_providers(self) -> tuple[CatalogProviderOption, ...]:
        """Return providers sorted for display."""
        return tuple(sorted(self.providers.values(), key=lambda option: option.name))

    def models_for_provider(self, provider_id: str) -> tuple[CatalogModelOption, ...]:
        """Return catalog models for a provider."""
        return self.models_by_provider.get(provider_id, ())


def sorted_unique_strings(values: Iterable[object]) -> tuple[str, ...]:
    """Return normalized non-empty strings in stable display order."""
    parsed = {
        value.strip() for value in values if isinstance(value, str) and value.strip()
    }
    return tuple(sorted(parsed, key=str.casefold))
