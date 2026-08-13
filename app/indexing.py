"""Chunk policy docs + seed tickets, embed (dense via Voyage AI, sparse via
local BM25) and upsert into Qdrant. Idempotent: index_all() recreates both
collections from scratch, so it's safe to re-run after editing
data/policy_docs/*.md or data/tickets_seed.json.

Used by scripts/index.py (manual CLI run) and by app/main.py's startup
check (auto-index on first container boot if the collections are empty —
see needs_indexing()), so the logic lives here once instead of in a script.
"""

import json
import re
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier, PointStruct, SparseVectorParams, VectorParams

from app.config import DENSE_DIM, POLICY_COLLECTION, QDRANT_URL, ROOT, TICKETS_COLLECTION
from app.embeddings import dense_embed, sparse_embed_documents

_UUID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def _uuid5(key: str) -> str:
    return str(uuid.uuid5(_UUID_NAMESPACE, key))


def _qdrant() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def needs_indexing() -> bool:
    """True if either collection is missing or empty — used by app/main.py's
    startup check so a fresh `docker compose up` self-populates Qdrant
    without a manual indexing step."""
    client = _qdrant()
    for name in (POLICY_COLLECTION, TICKETS_COLLECTION):
        if not client.collection_exists(name):
            return True
        if client.get_collection(name).points_count == 0:
            return True
    return False


def chunk_policy_doc(path: Path) -> list[dict]:
    """Split a policy doc into semantic chunks: '##' sections when present,
    otherwise individual bullets / numbered steps / blank-line-separated
    blocks — whichever matches how that doc is actually structured, so a
    rule never gets split apart from its condition."""
    raw_lines = path.read_text().splitlines()
    title = raw_lines[0].lstrip("# ").strip() if raw_lines and raw_lines[0].startswith("#") else path.stem
    body = "\n".join(raw_lines[1:]).strip()
    body_lines = body.splitlines()
    doc_type = path.stem

    chunks: list[tuple[str, str]] = []

    if re.search(r"^## ", body, re.MULTILINE):
        parts = re.split(r"^## ", body, flags=re.MULTILINE)
        preamble, *sections = parts
        if preamble.strip():
            chunks.append(("Overview", preamble.strip()))
        for section in sections:
            header, _, content = section.partition("\n")
            chunks.append((header.strip(), content.strip()))
    elif sum(1 for l in body_lines if l.strip().startswith("- ")) >= 3:
        bullets = [l.strip().lstrip("- ").strip() for l in body_lines if l.strip().startswith("- ")]
        for i, bullet in enumerate(bullets, 1):
            chunks.append((f"Rule {i}", bullet))
    elif re.search(r"^\d+\.\s", body, re.MULTILINE):
        items = re.findall(r"^\d+\.\s.+(?:\n(?!\d+\.).+)*", body, re.MULTILINE)
        for i, item in enumerate(items, 1):
            chunks.append((f"Step {i}", item.strip()))
    else:
        blocks = [b.strip() for b in body.split("\n\n") if b.strip()]
        for i, block in enumerate(blocks, 1):
            chunks.append((f"Item {i}", block))

    return [
        {
            "source": path.name,
            "doc_type": doc_type,
            "section": section,
            "text": f"{title} — {section}\n{content}",
        }
        for section, content in chunks
    ]


def _recreate_collection(client: QdrantClient, name: str) -> None:
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
    )


def index_policy_docs() -> int:
    client = _qdrant()
    docs_dir = ROOT / "data" / "policy_docs"
    all_chunks = []
    for path in sorted(docs_dir.glob("*.md")):
        all_chunks.extend(chunk_policy_doc(path))

    texts = [c["text"] for c in all_chunks]
    dense_vecs = dense_embed(texts, input_type="document")
    sparse_vecs = sparse_embed_documents(texts)

    _recreate_collection(client, POLICY_COLLECTION)
    points = [
        PointStruct(
            id=_uuid5(f"{POLICY_COLLECTION}::{c['source']}::{c['section']}"),
            vector={"dense": dense_vec, "sparse": sparse_vec},
            payload=c,
        )
        for c, dense_vec, sparse_vec in zip(all_chunks, dense_vecs, sparse_vecs)
    ]
    client.upsert(collection_name=POLICY_COLLECTION, points=points)
    return len(points)


def index_resolved_tickets() -> int:
    client = _qdrant()
    tickets = json.loads((ROOT / "data" / "tickets_seed.json").read_text())
    texts = [f"{t['subject']}\n{t['body']}" for t in tickets]

    dense_vecs = dense_embed(texts, input_type="document")
    sparse_vecs = sparse_embed_documents(texts)

    _recreate_collection(client, TICKETS_COLLECTION)
    points = [
        PointStruct(
            id=_uuid5(f"{TICKETS_COLLECTION}::{t['id']}"),
            vector={"dense": dense_vec, "sparse": sparse_vec},
            payload=t,
        )
        for t, dense_vec, sparse_vec in zip(tickets, dense_vecs, sparse_vecs)
    ]
    client.upsert(collection_name=TICKETS_COLLECTION, points=points)
    return len(points)


def index_all() -> None:
    n_policy = index_policy_docs()
    print(f"Indexed {n_policy} policy chunks into '{POLICY_COLLECTION}'")
    n_tickets = index_resolved_tickets()
    print(f"Indexed {n_tickets} resolved tickets into '{TICKETS_COLLECTION}'")
