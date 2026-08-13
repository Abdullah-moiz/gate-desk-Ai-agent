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
        # Phase 6 additive migration: the agent's original (pre-gate) tool
        # call, kept alongside the post-gate outcome/action columns above so
        # a gate override never loses the agent's draft. ADD COLUMN IF NOT
        # EXISTS rather than dropping/recreating, unlike the Qdrant
        # collections — this table holds real audit history, not a
        # regenerable index.
        for ddl in (
            "ALTER TABLE resolutions ADD COLUMN IF NOT EXISTS agent_action TEXT",
            "ALTER TABLE resolutions ADD COLUMN IF NOT EXISTS agent_arguments JSONB",
            "ALTER TABLE resolutions ADD COLUMN IF NOT EXISTS agent_result JSONB",
            "ALTER TABLE resolutions ADD COLUMN IF NOT EXISTS agent_confidence DOUBLE PRECISION",
            "ALTER TABLE resolutions ADD COLUMN IF NOT EXISTS policy_confidence DOUBLE PRECISION",
            "ALTER TABLE resolutions ADD COLUMN IF NOT EXISTS precedent_confidence DOUBLE PRECISION",
            "ALTER TABLE resolutions ADD COLUMN IF NOT EXISTS gate_passed BOOLEAN",
            "ALTER TABLE resolutions ADD COLUMN IF NOT EXISTS gate_reason TEXT",
        ):
            cur.execute(ddl)
        conn.commit()


def clear_ticket_trace(ticket_id: str) -> None:
    """Delete any retrievals/tool_calls from a previous run of this ticket_id.

    tickets and resolutions already reflect latest-state-only via
    ON CONFLICT DO UPDATE; retrievals/tool_calls are append-only inserts
    (there can be several tool calls per ticket), so without this a
    re-run of the same ticket_id leaves old and new runs' rows mixed
    together with no way to tell them apart in scripts/trace.py. A
    production system would instead tag rows with a run_id and keep full
    history — overwriting is a deliberate simplification here since each
    ticket in this project is a one-shot, repeatable demo input, not a
    real event stream worth preserving history for.
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM retrievals WHERE ticket_id = %s", (ticket_id,))
        cur.execute("DELETE FROM tool_calls WHERE ticket_id = %s", (ticket_id,))
        conn.commit()


def _rows_as_dicts(cur) -> list[dict]:
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def list_tickets() -> list[dict]:
    """Ticket queue for scripts/trace.py and app/dashboard.py — one row per
    ticket with its latest resolution (if any) joined in."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT t.ticket_id, t.subject, t.category, t.urgency, t.customer_email, t.created_at,
                   r.outcome, r.action
            FROM tickets t
            LEFT JOIN resolutions r ON r.ticket_id = t.ticket_id
            ORDER BY t.created_at DESC
            """
        )
        return _rows_as_dicts(cur)


def get_ticket(ticket_id: str) -> dict | None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT subject, body, customer_email, category, urgency, created_at FROM tickets WHERE ticket_id = %s",
            (ticket_id,),
        )
        rows = _rows_as_dicts(cur)
        return rows[0] if rows else None


def get_retrievals(ticket_id: str) -> list[dict]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT collection_name, result_rank, point_id, score, payload
            FROM retrievals WHERE ticket_id = %s
            ORDER BY collection_name, result_rank
            """,
            (ticket_id,),
        )
        return _rows_as_dicts(cur)


def get_tool_calls(ticket_id: str) -> list[dict]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT step, tool_name, arguments, result FROM tool_calls WHERE ticket_id = %s ORDER BY step",
            (ticket_id,),
        )
        return _rows_as_dicts(cur)


def get_resolution(ticket_id: str) -> dict | None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT outcome, action, arguments, result,
                   agent_action, agent_arguments, agent_result, agent_confidence,
                   policy_confidence, precedent_confidence, gate_passed, gate_reason
            FROM resolutions WHERE ticket_id = %s
            """,
            (ticket_id,),
        )
        rows = _rows_as_dicts(cur)
        return rows[0] if rows else None


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


def save_resolution(
    ticket_id: str,
    final: dict,
    agent_draft: dict,
    agent_confidence: float | None,
    policy_confidence: float,
    precedent_confidence: float,
    gate_passed: bool,
    gate_reason: str,
    trace: list,
) -> None:
    """final = post-gate outcome/action (what actually happened to the
    ticket). agent_draft = what the agent itself chose to do, before the
    gate had a say — identical to final when gate_passed is True."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO resolutions (
                ticket_id, outcome, action, arguments, result, trace,
                agent_action, agent_arguments, agent_result, agent_confidence,
                policy_confidence, precedent_confidence, gate_passed, gate_reason
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticket_id) DO UPDATE SET
                outcome = EXCLUDED.outcome,
                action = EXCLUDED.action,
                arguments = EXCLUDED.arguments,
                result = EXCLUDED.result,
                trace = EXCLUDED.trace,
                agent_action = EXCLUDED.agent_action,
                agent_arguments = EXCLUDED.agent_arguments,
                agent_result = EXCLUDED.agent_result,
                agent_confidence = EXCLUDED.agent_confidence,
                policy_confidence = EXCLUDED.policy_confidence,
                precedent_confidence = EXCLUDED.precedent_confidence,
                gate_passed = EXCLUDED.gate_passed,
                gate_reason = EXCLUDED.gate_reason,
                created_at = now()
            """,
            (
                ticket_id,
                final["outcome"],
                final["action"],
                json.dumps(final["arguments"]),
                json.dumps(final["result"]),
                json.dumps(trace, default=str),
                agent_draft["action"],
                json.dumps(agent_draft["arguments"]),
                json.dumps(agent_draft["result"]),
                agent_confidence,
                policy_confidence,
                precedent_confidence,
                gate_passed,
                gate_reason,
            ),
        )
        conn.commit()
