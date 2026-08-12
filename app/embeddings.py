"""Dense (Voyage AI) + sparse (local BM25 via fastembed) embedding helpers.

Sparse vectors are computed locally by fastembed — no API call, no cost —
and combined with dense vectors in Qdrant for hybrid retrieval (Phase 3).
"""

import warnings

import voyageai
from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

from app.config import DENSE_MODEL, SPARSE_MODEL, VOYAGE_API_KEY

warnings.filterwarnings("ignore", category=DeprecationWarning, module="fastembed")

BATCH_SIZE = 64

_voyage = voyageai.Client(api_key=VOYAGE_API_KEY)
_bm25 = SparseTextEmbedding(model_name=SPARSE_MODEL)


def dense_embed(texts: list[str], input_type: str) -> list[list[float]]:
    """input_type must be 'document' when embedding content to index, or
    'query' when embedding a search query — Voyage tailors the vector for
    each side of retrieval."""
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        result = _voyage.embed(batch, model=DENSE_MODEL, input_type=input_type)
        embeddings.extend(result.embeddings)
    return embeddings


def sparse_embed_documents(texts: list[str]) -> list[SparseVector]:
    return [
        SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in _bm25.embed(texts)
    ]


def sparse_embed_query(text: str) -> SparseVector:
    e = next(iter(_bm25.query_embed(text)))
    return SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
