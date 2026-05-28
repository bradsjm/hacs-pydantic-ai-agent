"""In-memory symbolic search index for the local home model."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import re

from .models import DocumentType, GraphEdge, HomeSemanticDocument
from .ranker import score_document

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, kw_only=True)
class SearchResult:
    """One ranked home semantic search result."""

    document: HomeSemanticDocument
    score: float
    reasons: tuple[str, ...]


class HomeSemanticIndex:
    """Small entry-scoped symbolic index for home retrieval."""

    def __init__(
        self,
        documents: Sequence[HomeSemanticDocument],
        edges: Sequence[GraphEdge] = (),
    ) -> None:
        """Initialize the symbolic search structures."""
        self.documents = tuple(documents)
        self.edges = tuple(edges)
        self.documents_by_id = {
            document.document_id: document for document in documents
        }
        self.documents_by_entity_id = {
            document.entity_id: document
            for document in documents
            if document.entity_id is not None
        }
        self._token_index: dict[str, set[str]] = defaultdict(set)
        for document in self.documents:
            for token in normalize_tokens(" ".join(document.searchable_parts())):
                self._token_index[token].add(document.document_id)

    def search(
        self,
        phrase: str,
        *,
        action: str | None = None,
        document_types: Iterable[DocumentType] | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Return deterministic local search results for a phrase."""
        query_tokens = frozenset(normalize_tokens(phrase))
        if not query_tokens:
            return []
        allowed_types = set(document_types) if document_types is not None else None
        candidate_ids: set[str] = set()
        for token in query_tokens:
            candidate_ids.update(self._token_index.get(token, ()))
        if not candidate_ids:
            return []
        results: list[SearchResult] = []
        normalized_phrase = " ".join(normalize_tokens(phrase))
        for document_id in candidate_ids:
            document = self.documents_by_id[document_id]
            if (
                allowed_types is not None
                and document.document_type not in allowed_types
            ):
                continue
            score, reasons = score_document(
                document,
                query_tokens,
                normalized_phrase,
                action,
            )
            if score <= 0:
                continue
            results.append(
                SearchResult(document=document, score=score, reasons=reasons)
            )
        results.sort(
            key=lambda result: (
                result.score,
                result.document.rank.preferred_target,
                result.document.rank.group,
                result.document.document_type == "capability",
                result.document.name,
            ),
            reverse=True,
        )
        return results[:limit]

    def diagnostics_summary(self) -> dict[str, object]:
        """Return aggregate, secret-safe diagnostics for the index."""
        document_counts = Counter(document.document_type for document in self.documents)
        domain_counts = Counter(
            document.domain
            for document in self.documents
            if document.domain is not None
        )
        capability_counts = Counter(
            document.capability
            for document in self.documents
            if document.capability is not None
        )
        return {
            "document_count": len(self.documents),
            "edge_count": len(self.edges),
            "document_counts": dict(sorted(document_counts.items())),
            "domain_counts": dict(sorted(domain_counts.items())),
            "capability_counts": dict(sorted(capability_counts.items())),
        }


def normalize_tokens(value: str) -> tuple[str, ...]:
    """Normalize user and registry text into symbolic search tokens."""
    return tuple(_TOKEN_RE.findall(value.replace("_", " ").replace("-", " ").lower()))
