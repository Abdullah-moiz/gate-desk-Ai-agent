"""Phase 10: the FastAPI service — the real ingestion boundary for this
project. A production deployment would have a real helpdesk (Zendesk,
Intercom, ...) call POST /tickets via webhook; app/dashboard.py is just
one possible client of this API, not a special-cased caller — it talks to
these same endpoints over HTTP, same as anything else would.

This also means the dashboard container needs none of the OpenAI/Voyage/
Qdrant/Postgres credentials — only this service does, which is the actual
reason for the service split in docker-compose.yml, not just "because the
architecture doc said FastAPI."

Auto-indexes Qdrant on startup if the collections are empty, so a fresh
`docker compose up` self-populates without a manual step.

Run with: uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from voyageai.error import RateLimitError

from app import db
from app.agent import handle_ticket
from app.indexing import index_all, needs_indexing


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_schema()
    if needs_indexing():
        print("Qdrant collections empty — indexing policy docs + seed tickets...")
        index_all()
    yield


app = FastAPI(title="GateDesk API", lifespan=lifespan)


class TicketRequest(BaseModel):
    subject: str
    body: str
    customer_email: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/tickets")
def submit_ticket(req: TicketRequest):
    ticket_id = f"LIVE-{uuid.uuid4().hex[:8]}"
    ticket = {"id": ticket_id, "subject": req.subject, "body": req.body, "customer_email": req.customer_email}
    try:
        outcome = handle_ticket(ticket)
    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail=(
                "Voyage AI rate limit hit (3 requests/min on this account tier without a "
                "payment method on file). Wait about 20 seconds and try again."
            ),
        )
    return {"ticket_id": ticket_id, **outcome}


@app.get("/tickets")
def list_tickets():
    return db.list_tickets()


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str):
    ticket = db.get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"No ticket found with id '{ticket_id}'")
    return {
        "ticket": ticket,
        "retrievals": db.get_retrievals(ticket_id),
        "tool_calls": db.get_tool_calls(ticket_id),
        "resolution": db.get_resolution(ticket_id),
    }
