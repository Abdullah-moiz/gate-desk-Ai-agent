import json

import psycopg2

from app.config import DATABASE_URL


def get_conn():
    return psycopg2.connect(DATABASE_URL)


def init_schema() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS retrievals (
                id SERIAL PRIMARY KEY,
                ticket_id TEXT,
                collection_name TEXT NOT NULL,
                query_text TEXT NOT NULL,
                category TEXT,
                result_rank INTEGER NOT NULL,
                point_id TEXT NOT NULL,
                score DOUBLE PRECISION NOT NULL,
                payload JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.commit()


def log_retrieval(
    ticket_id: str | None,
    collection_name: str,
    query_text: str,
    category: str | None,
    results: list[dict],
) -> None:
    if not results:
        return
    with get_conn() as conn, conn.cursor() as cur:
        for rank, r in enumerate(results, start=1):
            cur.execute(
                """
                INSERT INTO retrievals
                    (ticket_id, collection_name, query_text, category, result_rank, point_id, score, payload)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    ticket_id,
                    collection_name,
                    query_text,
                    category,
                    rank,
                    str(r["id"]),
                    r["score"],
                    json.dumps(r["payload"]),
                ),
            )
        conn.commit()
