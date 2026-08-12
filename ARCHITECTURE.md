# Support Triage & Auto-Resolution Agent — Architecture Plan

## Goal
A RAG + agentic system that classifies incoming support tickets, retrieves relevant
context, and either auto-resolves the ticket via tool calls or escalates it to a
human — with guardrails, an evaluation harness, and full observability. Built to be
demoable end-to-end via `docker compose up`, and to give a defensible architecture
story for AI developer interviews.

## Final Stack (confirmed, $0 cost)
| Layer | Choice | Why / cost note |
|---|---|---|
| LLM | OpenAI API (`gpt-4o-mini` for dev, swappable) | You already have access; mini model keeps token cost near-zero for a learning project |
| Embeddings | Voyage AI (`voyage-4-lite`) | 200M free tokens/month — a KB of a few hundred docs won't come close |
| Vector store | Qdrant (Docker) | Self-hosted, free, has native hybrid (dense + sparse/BM25) search built in — avoids hand-rolling hybrid retrieval |
| Ticket/audit DB | Postgres (Docker) | Self-hosted, free; gives you a real audit trail table (ticket → decision → tool calls) — more credible in an interview than an in-memory dict |
| Backend | Python + FastAPI | Lightweight, easy to explain line-by-line |
| Orchestration | Raw OpenAI function-calling (no LangChain/LlamaIndex) | You can explain every step yourself in an interview instead of "the framework does it" |
| Demo UI | Streamlit | Fastest way to get a clickable, recordable dashboard |
| Packaging | Docker Compose (single `docker-compose.yml`) | Anyone (or future-you) runs `docker compose up` and it works |

## Data (I will generate this)
- `data/policy_docs/` — refund policy, subscription tiers/pricing, SLA doc, FAQ (5-8 markdown files)
- `data/tickets_seed.json` — ~50 synthetic tickets across categories: billing, account access, bug report, general question
- `data/eval_set.json` — ~20 held-out tickets with a known correct outcome (auto-resolve + expected action, or escalate) — used only for Phase 8 evaluation, never for indexing

---

## Phase 0 — Repo & Environment Scaffolding
1. `docker-compose.yml` with services: `qdrant`, `postgres` (app runs locally during dev, containerized in Phase 10)
2. `.env` for `OPENAI_API_KEY`, `VOYAGE_API_KEY`, DB/Qdrant connection strings — `.env.example` committed, `.env` gitignored
3. Python project scaffold (`uv` or `venv` + `requirements.txt`): `fastapi`, `openai`, `voyageai`, `qdrant-client`, `psycopg2`, `streamlit`
4. `docker compose up -d` → confirm Qdrant (port 6333) and Postgres are reachable

**Deliverable:** empty services running in Docker, health-checked.

## Phase 1 — Knowledge Base & Ticket Data
1. Write policy/FAQ docs (`data/policy_docs/*.md`)
2. Generate `tickets_seed.json` (mix of clean and ambiguous/edge-case tickets on purpose — you want failure cases to discuss later)
3. Generate `eval_set.json` with hand-verified expected outcomes

**Deliverable:** static data files, reviewed by hand so you know ground truth.

## Phase 2 — Ingestion & Indexing Pipeline
1. Chunking function for policy docs (by section/heading, not fixed-size — **decision to justify:** policy docs have semantic sections, arbitrary token-window chunking would split a refund rule from its conditions)
2. Embed chunks via Voyage AI, upsert into Qdrant with metadata: `{source, doc_type, section}`
3. Separately index past-resolved tickets (ticket text + resolution) as a second Qdrant collection
4. Script: `scripts/index.py` — idempotent, re-runnable

**Deliverable:** two Qdrant collections populated (`policy_kb`, `resolved_tickets`).

## Phase 3 — Retrieval Layer
1. Implement hybrid search in Qdrant (dense vector + keyword/sparse) — **decision to justify:** exact terms like plan names, error codes won't reliably surface via pure embeddings
2. Add metadata filtering by ticket category (billing ticket → filter to `doc_type: billing`)
3. Return top-k with scores; log every retrieval (ticket_id → chunks retrieved → scores) to Postgres

**Deliverable:** `retrieve(query, category) -> List[Chunk]` function, unit-testable in isolation.

## Phase 4 — Classification & Routing
1. Lightweight OpenAI call (structured output / JSON mode) to tag: `category`, `urgency`
2. Route determines which tools + which KB filter are exposed to the agent for this ticket — **decision to justify:** don't hand the agent `issue_refund` on a "how do I change my password" ticket; smaller tool surface = fewer failure modes

**Deliverable:** `classify(ticket) -> {category, urgency}`.

## Phase 5 — Agent & Tool Layer
1. Define mock tools as plain Python functions + OpenAI function schemas:
   - `lookup_account(email)` (read-only)
   - `check_subscription_status(account_id)` (read-only)
   - `issue_refund(account_id, amount)` (write, capped — e.g. auto-allowed only under $20)
   - `escalate_to_human(reason)` (always available)
2. Agent loop: classify → retrieve → call OpenAI with tools + retrieved context → execute tool calls → produce final response
3. Log full reasoning trace (messages, tool calls, tool results) to Postgres per ticket

**Deliverable:** `handle_ticket(ticket) -> Resolution`, callable end-to-end.

## Phase 6 — Guardrails / Confidence Gate
1. Define auto-resolve criteria explicitly (not just "trust the LLM"):
   - retrieval top score above threshold, AND
   - tool action (if any) is within the read-only/low-risk allowlist, AND
   - agent's own stated confidence (ask it to self-report) above threshold
2. Anything failing the gate → `escalate_to_human`, ticket queued with the agent's draft response attached for a human to approve/edit
3. This maker/checker split is your #1 interview talking point — write it up explicitly in code comments/README, not just in your head

**Deliverable:** gate function wired into Phase 5's loop; every ticket ends in `auto_resolved` or `escalated` with a reason.

## Phase 7 — Observability
1. Postgres tables: `tickets`, `retrievals`, `tool_calls`, `resolutions` — enough to reconstruct "why did the agent do X" for any ticket after the fact
2. Simple `scripts/trace.py <ticket_id>` to pretty-print the full trace for a ticket — this is what you'll screen-record

**Deliverable:** full audit trail queryable per ticket.

## Phase 8 — Evaluation Harness
1. Run `eval_set.json` through the pipeline, compare against expected outcomes
2. Metrics to compute and record: retrieval precision/recall@k, auto-resolve accuracy, **false-resolve rate** (wrong action taken with high confidence — your headline metric), escalation appropriateness, avg tokens/cost per ticket
3. `scripts/evaluate.py` → prints a small report table

**Deliverable:** a results table you can literally paste into your resume/portfolio README.

## Phase 9 — Demo Dashboard (Streamlit)
1. Ticket queue view (pending / auto-resolved / escalated)
2. Click a ticket → see retrieved chunks, tool calls, final decision, confidence score
3. "Submit new ticket" form to run live during a demo/interview

**Deliverable:** `streamlit run app/dashboard.py`, visually demoable.

## Phase 10 — Full Dockerization
1. Dockerfile for the FastAPI backend, Dockerfile for the Streamlit app
2. Extend `docker-compose.yml`: `qdrant`, `postgres`, `api`, `dashboard`
3. Add a `Makefile` or `docker-compose.yml` `command` that runs indexing (Phase 2) automatically on first boot if collections are empty
4. Verify on a clean machine (or `docker compose down -v && docker compose up`) that one command gets a stranger to a working demo

**Deliverable:** `docker compose up` → dashboard live at `localhost:8501`, nothing installed on host except Docker.

## Phase 11 — Documentation for Interviews
1. `README.md`: architecture diagram (ASCII or draw.io export), setup instructions, screenshot/GIF
2. `DECISIONS.md`: one entry per bolded "decision to justify" above — problem, options considered, choice, tradeoff. This is literally your interview cheat sheet.
3. Record the demo (Streamlit walkthrough + `scripts/trace.py` output for one ticket) once everything above is stable

---

## Suggested order of attack
Phases 0-2 first (get data flowing into Qdrant), then 3-4 (retrieval works standalone,
testable without an agent), then 5-6 together (the agent is only interesting once
the gate exists — build them as one unit), then 7-8 (you need logging before you can
evaluate), then 9-11 last.
