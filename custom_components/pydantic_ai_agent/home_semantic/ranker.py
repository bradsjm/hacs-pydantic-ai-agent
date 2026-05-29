"""Deterministic symbolic ranking for home semantic retrieval."""

from .actions import ACTION_CAPABILITIES
from .models import HomeSemanticDocument


def score_document(
    document: HomeSemanticDocument,
    query_tokens: frozenset[str],
    phrase: str,
    action: str | None = None,
) -> tuple[float, tuple[str, ...]]:
    """Score a document for a local semantic query."""
    matched_reasons: list[str] = []
    score = 0.0
    searchable_text = " ".join(document.searchable_parts()).lower()
    document_tokens = _tokens_from_parts(document.searchable_parts())
    matched_tokens = query_tokens & document_tokens
    if matched_tokens:
        score += len(matched_tokens) * 5
        matched_reasons.append("token_match")
    if phrase and phrase in searchable_text:
        score += 20
        matched_reasons.append("phrase_match")
    if phrase and phrase == document.name.lower():
        score += 30
        matched_reasons.append("exact_name")
    if action is not None and document.capability in ACTION_CAPABILITIES.get(
        action, ()
    ):
        score += 12
        matched_reasons.append("action_capability")
    if document.rank.preferred_target:
        score += 14
        matched_reasons.append("preferred_target")
    if document.rank.group:
        score += 10
        matched_reasons.append("group")
    if document.rank.physical_control:
        score += 4
        matched_reasons.append("physical_control")
    if document.document_type == "capability":
        score += 6
        matched_reasons.append("capability_scope")
    if document.rank.high_churn:
        score -= 8
        matched_reasons.append("high_churn_penalty")
    if document.rank.diagnostic:
        score -= 12
        matched_reasons.append("diagnostic_penalty")
    if document.rank.hidden:
        score -= 20
        matched_reasons.append("hidden_penalty")
    if document.rank.disabled:
        score -= 50
        matched_reasons.append("disabled_penalty")
    return score, tuple(matched_reasons)


def _tokens_from_parts(parts: tuple[str, ...]) -> frozenset[str]:
    """Return normalized token set for text parts."""
    from .index import normalize_tokens

    tokens: set[str] = set()
    for part in parts:
        tokens.update(normalize_tokens(part))
    return frozenset(tokens)
