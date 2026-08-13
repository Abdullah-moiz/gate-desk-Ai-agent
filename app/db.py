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
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                customer_email TEXT,
                category TEXT,
                urgency TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_calls (
                id SERIAL PRIMARY KEY,
                ticket_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                tool_name TEXT NOT NULL,
                arguments JSONB,
                result JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS resolutions (
                ticket_id TEXT PRIMARY KEY,
                outcome TEXT NOT NULL,
                action TEXT,
                arguments JSONB,
                result JSONB,
                trace JSONB,
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


def save_ticket(
    ticket_id: str,
    subject: str,
    body: str,
    customer_email: str | None,
    category: str,
    urgency: str,
) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tickets (ticket_id, subject, body, customer_email, category, urgency)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticket_id) DO UPDATE SET
                subject = EXCLUDED.subject,
                body = EXCLUDED.body,
                customer_email = EXCLUDED.customer_email,
                category = EXCLUDED.category,
                urgency = EXCLUDED.urgency
            """,
            (ticket_id, subject, body, customer_email, category, urgency),
        )
        conn.commit()


def save_tool_call(ticket_id: str, step: int, tool_name: str, arguments: dict, result: dict) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tool_calls (ticket_id, step, tool_name, arguments, result)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (ticket_id, step, tool_name, json.dumps(arguments), json.dumps(result)),
        )
        conn.commit()


def save_resolution(ticket_id: str, resolution: dict, trace: list) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO resolutions (ticket_id, outcome, action, arguments, result, trace)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticket_id) DO UPDATE SET
                outcome = EXCLUDED.outcome,
                action = EXCLUDED.action,
                arguments = EXCLUDED.arguments,
                result = EXCLUDED.result,
                trace = EXCLUDED.trace,
                created_at = now()
            """,
            (
                ticket_id,
                resolution["outcome"],
                resolution["action"],
                json.dumps(resolution["arguments"]),
                json.dumps(resolution["result"]),
                json.dumps(trace, default=str),
            ),
        )
        conn.commit()
