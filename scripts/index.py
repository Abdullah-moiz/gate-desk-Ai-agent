"""
Chunk policy docs + seed tickets, embed via Voyage AI, upsert into Qdrant.
Idempotent: each run recreates both collections from scratch, so it's safe
to re-run after editing data/policy_docs/*.md or data/tickets_seed.json.
"""

import json
import os
import re
import uuid
from pathlib import Path

import voyageai
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

EMBED_MODEL = "voyage-4-lite"
EMBED_DIM = 1024
BATCH_SIZE = 64
NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
qdrant = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))


def deterministic_id(*parts: str) -> str:
    """Stable UUID from content-derived parts so re-running the script upserts
    the same points instead of accumulating duplicates."""
    return str(uuid.uuid5(NAMESPACE, "::".join(parts)))


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


def embed(texts: list[str], input_type: str) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        result = voyage.embed(batch, model=EMBED_MODEL, input_type=input_type)
        embeddings.extend(result.embeddings)
    return embeddings


def recreate_collection(name: str) -> None:
    if qdrant.collection_exists(name):
        qdrant.delete_collection(name)
    qdrant.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )


def index_policy_docs() -> None:
    docs_dir = ROOT / "data" / "policy_docs"
    all_chunks = []
    for path in sorted(docs_dir.glob("*.md")):
        all_chunks.extend(chunk_policy_doc(path))

    print(f"Chunked {len(all_chunks)} sections from {docs_dir}")
    embeddings = embed([c["text"] for c in all_chunks], input_type="document")

    recreate_collection("policy_kb")
    points = [
        PointStruct(
            id=deterministic_id("policy_kb", c["source"], c["section"]),
            vector=vec,
            payload=c,
        )
        for c, vec in zip(all_chunks, embeddings)
    ]
    qdrant.upsert(collection_name="policy_kb", points=points)
    print(f"Indexed {len(points)} policy chunks into 'policy_kb'")


def index_resolved_tickets() -> None:
    tickets = json.loads((ROOT / "data" / "tickets_seed.json").read_text())
    texts = [f"{t['subject']}\n{t['body']}" for t in tickets]

    print(f"Embedding {len(tickets)} resolved tickets")
    embeddings = embed(texts, input_type="document")

    recreate_collection("resolved_tickets")
    points = [
        PointStruct(
            id=deterministic_id("resolved_tickets", t["id"]),
            vector=vec,
            payload=t,
        )
        for t, vec in zip(tickets, embeddings)
    ]
    qdrant.upsert(collection_name="resolved_tickets", points=points)
    print(f"Indexed {len(points)} resolved tickets into 'resolved_tickets'")


if __name__ == "__main__":
    index_policy_docs()
    index_resolved_tickets()
    print("Done.")
