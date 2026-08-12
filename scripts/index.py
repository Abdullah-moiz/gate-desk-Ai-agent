"""
Chunk policy docs + seed tickets, embed (dense via Voyage AI, sparse via local
BM25) and upsert into Qdrant. Idempotent: each run recreates both collections
from scratch, so it's safe to re-run after editing data/policy_docs/*.md or
data/tickets_seed.json.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier, PointStruct, SparseVectorParams, VectorParams

from app.config import DENSE_DIM, POLICY_COLLECTION, QDRANT_URL, ROOT, TICKETS_COLLECTION
from app.embeddings import dense_embed, sparse_embed_documents

qdrant = QdrantClient(url=QDRANT_URL)


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


def recreate_collection(name: str) -> None:
    if qdrant.collection_exists(name):
        qdrant.delete_collection(name)
    qdrant.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
    )


def index_policy_docs() -> None:
    docs_dir = ROOT / "data" / "policy_docs"
    all_chunks = []
    for path in sorted(docs_dir.glob("*.md")):
        all_chunks.extend(chunk_policy_doc(path))

    print(f"Chunked {len(all_chunks)} sections from {docs_dir}")
    texts = [c["text"] for c in all_chunks]
    dense_vecs = dense_embed(texts, input_type="document")
    sparse_vecs = sparse_embed_documents(texts)

    recreate_collection(POLICY_COLLECTION)
    points = [
        PointStruct(
            id=str(uuid5(f"{POLICY_COLLECTION}::{c['source']}::{c['section']}")),
            vector={"dense": dense_vec, "sparse": sparse_vec},
            payload=c,
        )
        for c, dense_vec, sparse_vec in zip(all_chunks, dense_vecs, sparse_vecs)
    ]
    qdrant.upsert(collection_name=POLICY_COLLECTION, points=points)
    print(f"Indexed {len(points)} policy chunks into '{POLICY_COLLECTION}'")


def index_resolved_tickets() -> None:
    tickets = json.loads((ROOT / "data" / "tickets_seed.json").read_text())
    texts = [f"{t['subject']}\n{t['body']}" for t in tickets]

    print(f"Embedding {len(tickets)} resolved tickets")
    dense_vecs = dense_embed(texts, input_type="document")
    sparse_vecs = sparse_embed_documents(texts)

    recreate_collection(TICKETS_COLLECTION)
    points = [
        PointStruct(
            id=str(uuid5(f"{TICKETS_COLLECTION}::{t['id']}")),
            vector={"dense": dense_vec, "sparse": sparse_vec},
            payload=t,
        )
        for t, dense_vec, sparse_vec in zip(tickets, dense_vecs, sparse_vecs)
    ]
    qdrant.upsert(collection_name=TICKETS_COLLECTION, points=points)
    print(f"Indexed {len(points)} resolved tickets into '{TICKETS_COLLECTION}'")


def uuid5(key: str) -> str:
    import uuid

    namespace = uuid.UUID("12345678-1234-5678-1234-567812345678")
    return str(uuid.uuid5(namespace, key))


if __name__ == "__main__":
    index_policy_docs()
    index_resolved_tickets()
    print("Done.")
