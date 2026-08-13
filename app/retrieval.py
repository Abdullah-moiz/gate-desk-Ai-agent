"""Hybrid (dense + sparse) retrieval over the two Qdrant collections, with
category-based metadata filtering and per-call logging to Postgres.

Usable standalone (see __main__ below) — the agent (Phase 5) will call
retrieve_policy() / retrieve_precedent() with a real ticket_id once it exists.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    Prefetch,
)

from app.config import CATEGORY_DOC_TYPES, POLICY_COLLECTION, QDRANT_URL, TICKETS_COLLECTION
from app.db import init_schema, log_retrieval
from app.embeddings import dense_embed, sparse_embed_query

_qdrant = QdrantClient(url=QDRANT_URL)
init_schema()


def _hybrid_search(
    collection: str,
    dense_vec: list[float],
    sparse_vec,
    query_filter: Filter | None,
    top_k: int,
) -> list[dict]:
    prefetch_limit = max(top_k * 4, 20)

    hits = _qdrant.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=prefetch_limit, filter=query_filter),
            Prefetch(query=sparse_vec, using="sparse", limit=prefetch_limit, filter=query_filter),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    ).points

    return [{"id": h.id, "score": h.score, "payload": h.payload} for h in hits]


def _dense_top_score(collection: str, dense_vec: list[float], query_filter: Filter | None) -> float:
    """Raw dense cosine similarity of the single best match — unlike the RRF
    fusion score used for ranking, this is bounded and actually reflects
    semantic relevance, which is what Phase 6's confidence gate needs. RRF
    scores are a function of *rank*, so the top hit scores ~1.0 even when
    nothing relevant was retrieved; cosine similarity doesn't have that
    problem."""
    hits = _qdrant.query_points(
        collection_name=collection,
        query=dense_vec,
        using="dense",
        query_filter=query_filter,
        limit=1,
        with_payload=False,
    ).points
    return hits[0].score if hits else 0.0


def _policy_filter(category: str | None) -> Filter | None:
    if category and category in CATEGORY_DOC_TYPES:
        return Filter(must=[FieldCondition(key="doc_type", match=MatchAny(any=CATEGORY_DOC_TYPES[category]))])
    return None


def _precedent_filter(category: str | None) -> Filter | None:
    if category:
        return Filter(must=[FieldCondition(key="category", match=MatchValue(value=category))])
    return None


def retrieve_policy(
    query_text: str, category: str | None = None, top_k: int = 5, ticket_id: str | None = None
) -> list[dict]:
    """Standalone single-collection search — embeds the query itself. Prefer
    retrieve() when you need both collections, since it embeds the query once
    and reuses it for both searches (halves Voyage API calls per ticket)."""
    dense_vec = dense_embed([query_text], input_type="query")[0]
    sparse_vec = sparse_embed_query(query_text)
    results = _hybrid_search(POLICY_COLLECTION, dense_vec, sparse_vec, _policy_filter(category), top_k)
    log_retrieval(ticket_id, POLICY_COLLECTION, query_text, category, results)
    return results


def retrieve_precedent(
    query_text: str, category: str | None = None, top_k: int = 5, ticket_id: str | None = None
) -> list[dict]:
    """Standalone single-collection search — see retrieve_policy() docstring."""
    dense_vec = dense_embed([query_text], input_type="query")[0]
    sparse_vec = sparse_embed_query(query_text)
    results = _hybrid_search(TICKETS_COLLECTION, dense_vec, sparse_vec, _precedent_filter(category), top_k)
    log_retrieval(ticket_id, TICKETS_COLLECTION, query_text, category, results)
    return results


def retrieve(
    query_text: str, category: str | None = None, top_k: int = 5, ticket_id: str | None = None
) -> dict:
    """Embeds the query once (1 Voyage call + 1 local BM25 call) and searches
    both collections with the same vectors — this is what the agent (Phase 5)
    should call per ticket, rather than retrieve_policy()+retrieve_precedent().

    Also returns policy_confidence/precedent_confidence (raw dense cosine
    similarity of the top hit in each collection) for Phase 6's gate."""
    dense_vec = dense_embed([query_text], input_type="query")[0]
    sparse_vec = sparse_embed_query(query_text)
    policy_filter = _policy_filter(category)
    precedent_filter = _precedent_filter(category)

    policy_results = _hybrid_search(POLICY_COLLECTION, dense_vec, sparse_vec, policy_filter, top_k)
    log_retrieval(ticket_id, POLICY_COLLECTION, query_text, category, policy_results)

    precedent_results = _hybrid_search(TICKETS_COLLECTION, dense_vec, sparse_vec, precedent_filter, top_k)
    log_retrieval(ticket_id, TICKETS_COLLECTION, query_text, category, precedent_results)

    return {
        "policy": policy_results,
        "precedent": precedent_results,
        "policy_confidence": _dense_top_score(POLICY_COLLECTION, dense_vec, policy_filter),
        "precedent_confidence": _dense_top_score(TICKETS_COLLECTION, dense_vec, precedent_filter),
    }


if __name__ == "__main__":
    import json
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "can I get a refund for a charge I don't recognize"
    category = sys.argv[2] if len(sys.argv) > 2 else None

    result = retrieve(query, category=category, top_k=3, ticket_id="TEST-STANDALONE")
    print(json.dumps(result, indent=2, default=str))
