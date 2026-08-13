# Architecture Decisions

One entry per decision worth defending in an interview: the problem, what else was considered, what was chosen, and the tradeoff. Ordered roughly by build phase — cross-references use `[[N]]`.

---

## 1. Structure-aware chunking, not fixed-token windows

**Problem:** How to split policy docs into retrievable chunks.

**Options considered:** Fixed-size token windows (e.g. 200 tokens with overlap) — simple, framework-default; or chunk by the document's own structure.

**Choice:** Chunk by whatever structure a doc actually uses — `##` headers when present, otherwise individual bullets, numbered steps, or blank-line-separated blocks (`app/indexing.py: chunk_policy_doc`).

**Tradeoff:** More code than a fixed-window splitter, and it only works because these docs are short and hand-written. But a fixed-token window would have split rules like *"refunds over $20 must be escalated, unless Enterprise, in which case always escalate"* across chunk boundaries, silently separating a rule from its condition — exactly the kind of bug that's invisible until a bad decision surfaces it.

## 2. Hybrid retrieval: dense + local BM25 sparse, RRF fusion

**Problem:** Pure dense embeddings miss exact-match terms (plan names, error codes) that a support KB is full of.

**Options considered:** Dense-only (simplest); dense + a hosted sparse/keyword model (costs API calls); dense + local BM25.

**Choice:** Dense (Voyage `voyage-4-lite`) + sparse (BM25 via `fastembed`, computed 100% locally, zero API cost) combined with Qdrant's native RRF fusion (`app/retrieval.py`).

**Tradeoff:** One more dependency (`fastembed` pulls in `onnxruntime`) versus a meaningfully better retrieval story for a support-ticket domain full of exact terms, for $0 marginal cost.

## 3. Dense cosine score, not RRF fusion score, for confidence gating — `[[8]]`

**Problem:** The gate needs a number that means "how relevant was the top match," but Qdrant's RRF fusion score is a function of *rank*, not relevance — the #1 result always scores ~1.0 even when nothing retrieved is actually related to the query.

**Options considered:** Use the fusion score directly (simplest, already computed); compute a separate raw dense-only cosine similarity just for gating.

**Choice:** A dedicated dense-only query (`_dense_top_score` in `app/retrieval.py`) reusing the already-computed query vector — no extra embedding API call, just one more local Qdrant query. Empirically calibrated: well-grounded queries scored ~0.63–0.66 cosine, a fully unrelated query scored 0.14 — the 0.35 threshold sits in that gap.

**Tradeoff:** One more Qdrant round-trip per retrieval call, in exchange for a confidence signal that's actually meaningful instead of trivially always-high.

## 4. Category-gated doc-type filtering and tool surface

**Problem:** A misclassified or off-topic ticket shouldn't be able to reach every tool or retrieve from every policy doc.

**Options considered:** Expose all tools/docs to every ticket and trust the agent's judgment; or gate both by the ticket's classified category.

**Choice:** `CATEGORY_DOC_TYPES` and `CATEGORY_TOOLS` in `app/config.py` — e.g. billing tickets never see `send_password_reset`, account tickets never see `issue_refund`.

**Tradeoff:** A genuinely cross-cutting ticket loses access to tools/docs outside its assigned category (mitigated by adding `sla` to both `billing` and `bug` doc-types after eval testing surfaced the gap — see `[[13]]`). In exchange, a classification mistake fails toward "missing a tool," not "handed an inappropriate one."

## 5. `gpt-5-nano` everywhere, deliberately, with the tradeoff documented in code

**Problem:** Which OpenAI model(s) to use for classification and the agent's tool-calling loop.

**Options considered:** `gpt-5-nano` (cheapest, $0.05/$0.40 per 1M tokens); `gpt-5-mini` (5x the cost, still fractions of a cent at this scale).

**Choice:** `gpt-5-nano` for both. Classification scored 16/16 twice, cleanly. The agent loop is less reliable — a 16-ticket eval run showed a 1/16 miss where nano skipped its own `lookup_account` tool and wrote escalation instructions addressed to a human, as if it had forgotten it could act itself. Recurred once more (`E-01`, later run).

**Tradeoff:** Explicitly kept on nano anyway, given the user's actual constraint (zero cost on a learning project, not production). This is the load-bearing reason the confidence gate `[[8]]` exists: it's the backstop for exactly this reliability gap, not a bigger model. Both times nano got confused, the *outcome* was still safe — it escalated rather than acting wrongly. Worth A/B-testing `gpt-5-mini` if this went to production.

## 6. Every resolution action is an explicit tool call, including no-op ones

**Problem:** Some resolutions (acknowledge a compliment, ask a clarifying question) don't have a real external side effect — do they need to be "tools"?

**Options considered:** Let the agent just write a free-text final response for these cases; or model every possible action, including trivial ones, as an explicit tool call.

**Choice:** All 10 terminal actions (`issue_refund` through `close_spam`) are tool calls with JSON-schema arguments (`app/tools.py`), never parsed from prose.

**Tradeoff:** More tool definitions to maintain, but every resolution is now structured and machine-checkable — `scripts/evaluate.py` can compare `action` fields exactly, and the gate `[[8]]` has one uniform interception point instead of needing to parse free text for some cases and inspect tool calls for others.

## 7. Tool-level validation as defense-in-depth, independent of the gate

**Problem:** `issue_refund` needs to enforce the $20 auto-approval cap; separately, `send_password_reset`/`unlock_account`/`issue_refund` were discovered (via `scripts/trace.py`, not the eval score) to blindly trust whatever `account_id` string the agent passed — including, once, the customer's *email* instead of a real ID.

**Options considered:** Trust the LLM's arguments once the gate has approved the action type; or validate arguments inside the tool itself, independent of the gate.

**Choice:** Both. `issue_refund` rejects amounts over the cap; all three account-mutating tools reject unknown `account_id` values (`app/tools.py`), forcing the agent to actually call `lookup_account` first.

**Tradeoff:** This is a second, narrower safety net below the gate — the gate reasons about *which action* and *how confident*, tools validate *whether the arguments are even real*. Neither layer alone would have caught both problems: the eval harness only checks action-type match, never argument correctness, so this bug was invisible to accuracy numbers and only surfaced by actually reading a trace.

## 8. The confidence gate: deterministic rules + risk-tiered thresholds

**Problem:** How to decide whether the agent's chosen tool call is trustworthy enough to count as an auto-resolution, versus needing a human.

**Options considered:** Trust the agent's tool choice outright (what Phase 5 shipped, before the gate existed); a single global confidence threshold; or a tiered system.

**Choice (`app/gate.py`):** Two check types. Deterministic policy rules that no confidence score can override (P0/P1 bugs always escalate). Then calibrated thresholds, tiered by action risk — `issue_refund`/`send_password_reset`/`unlock_account` need both stronger policy grounding (`[[3]]`) and higher self-reported confidence (0.85) than low-risk actions like `acknowledge` or `log_bug` (0.60).

**Tradeoff:** Two more numbers to calibrate and defend (why 0.35, why 0.85 vs 0.60) versus a system that fails toward escalation instead of toward wrong autonomous action — see `[[5]]` for the case this was built for.

## 9. Policy-grounding check restricted to high-risk actions only

**Problem:** The gate initially applied the same 0.35 policy-grounding threshold to every action. First full eval run scored 13/16 — all three misses were low-risk, *correct* actions (acknowledging a compliment, logging a cosmetic bug, offering a documented workaround) that got wrongly escalated.

**Root cause:** These actions don't need policy backing in the first place — nothing in a policy doc says how to respond to "nice redesign!" A weak retrieval score there means "no policy exists for this," not "the agent is ungrounded."

**Choice:** Policy-grounding is now only load-bearing for the three high-risk actions (`app/gate.py`); low-risk actions rely solely on the agent's own confidence threshold.

**Tradeoff:** A more permissive gate for the actions that matter least, in exchange for not over-escalating a large fraction of genuinely safe, correct resolutions. Re-verified: 16/16 after the fix. This is the single clearest "built it, tested it, found it was wrong, fixed the actual mechanism" story in the project.

## 10. Gate overrides preserve the agent's draft, never discard it

**Problem:** When the gate overrides the agent's chosen action, what happens to that original decision?

**Choice:** `resolutions` stores both — `agent_action`/`agent_arguments`/`agent_result` (the draft) alongside `action`/`arguments`/`result` (the final, post-gate outcome), plus `gate_passed`/`gate_reason` (`app/db.py: save_resolution`).

**Tradeoff:** A wider table and a slightly more complex save path, in exchange for the maker/checker split being genuinely auditable — a human reviewing an escalated ticket sees exactly what the agent would have done and why the gate didn't trust it, instead of a bare "escalated."

## 11. Idempotent Qdrant recreation vs. additive Postgres migrations

**Problem:** Both Qdrant collections and Postgres tables need schema evolution as the project grew across phases.

**Choice:** Qdrant collections are dropped and recreated on every `scripts/index.py` run (`app/indexing.py: _recreate_collection`). Postgres tables use `ALTER TABLE ADD COLUMN IF NOT EXISTS` migrations that never drop data (`app/db.py: init_schema`).

**Tradeoff:** Deliberately different regenerability assumptions for the two stores. Qdrant only ever holds a *derived index* of `data/*.json` — dropping and rebuilding it is cheap and correct by construction. Postgres holds real audit history (past ticket resolutions, gate decisions) that a production system couldn't casually discard; a real migration tool (Alembic) would replace the inline DDL at that point, but additive `ALTER TABLE` was enough for this project's scope.

## 12. Ticket traces reflect the latest run only, not full history

**Problem:** `retrievals` and `tool_calls` are append-only inserts (a ticket can have several tool calls). Re-running the same `ticket_id` during development left old and new runs' rows mixed together with no way to tell them apart — `scripts/trace.py` showed 7 duplicate/conflicting entries for a ticket re-run 7 times across testing.

**Choice:** `clear_ticket_trace(ticket_id)` runs at the start of every `handle_ticket()` call, deleting prior rows for that ticket before the new run logs anything (`app/db.py`, called from `app/agent.py`).

**Tradeoff:** Loses the ability to compare how the agent's behavior on the *same* ticket changed across code versions over time — a production system doing that kind of longitudinal analysis would tag rows with a `run_id` and keep full history instead. Deliberate simplification here: every ticket in this project is a one-shot, repeatable demo input, not a real event stream worth preserving history for.

## 13. False-resolve rate as the headline eval metric, not overall accuracy

**Problem:** Which number best represents whether this system is safe to trust.

**Options considered:** Overall outcome accuracy (simple, but treats "wrongly auto-resolved a billing dispute" and "unnecessarily escalated a compliment" as equally bad); false-resolve rate specifically (tickets that should have escalated but didn't).

**Choice:** `scripts/evaluate.py` reports false-resolve rate as the headline number, with over-escalation reported separately as a lower-stakes efficiency metric.

**Tradeoff:** A system that escalates everything scores 0% false-resolve trivially, so this metric alone doesn't prove the system is *useful* — outcome accuracy and action accuracy are reported alongside it for that reason. But if forced to pick one number to defend a support-automation system's safety, this is the one, since it's the actual failure mode that matters: a wrong action taken with unwarranted confidence.

## 14. "Policy grounding hit rate" as a retrieval-quality proxy, not real precision/recall

**Problem:** Textbook retrieval precision/recall@k requires chunk-level relevance judgments (which chunks are *actually* relevant to a given query) that were never hand-labeled for this project.

**Choice:** A proxy metric instead — did the top retrieved policy chunk's `doc_type` match one of the doc-types associated with that ticket's ground-truth category (`CATEGORY_DOC_TYPES`), read straight from the `retrievals` table Phase 3/7 already populate.

**Tradeoff:** This measures "did retrieval pull from the right *document*," not "was this specific *chunk* the best possible match" — a real precision/recall number would need labeled relevance judgments this project doesn't have. Documented as a proxy explicitly, in the script and here, rather than presented as more rigorous than it is.

## 15. API and dashboard as genuinely separate services, not the same code twice

**Problem:** `app/dashboard.py` originally called `app/agent.py`'s `handle_ticket()` directly, in-process — functionally fine for a single-container demo, but not a real service boundary.

**Options considered:** Keep the dashboard importing agent code directly (simpler, fewer moving parts); or put a FastAPI service in front and make the dashboard a pure HTTP client of it.

**Choice:** `app/main.py` is the real ingestion boundary — the endpoint a production helpdesk webhook would call. `app/dashboard.py` talks to it exclusively over HTTP (`requests`), holding zero OpenAI/Voyage/Qdrant/Postgres credentials itself.

**Tradeoff:** An extra network hop and a second Docker service to build/maintain, for a genuine separation of concerns: the `dashboard` container in `docker-compose.yml` needs none of the API keys the `api` container needs, which is a concrete, checkable consequence of the design rather than an assertion about it.

## 16. Auto-index on startup, gated by an actual emptiness check

**Problem:** A fresh `docker compose up` needs Qdrant populated before the agent can retrieve anything, but re-indexing on every boot would be slow and pointless once the collections already exist.

**Choice:** `app/main.py`'s FastAPI `lifespan` calls `needs_indexing()` (`app/indexing.py`) — checks both collections exist *and* have `points_count > 0` — and only runs `index_all()` if that's false.

**Tradeoff:** A slightly slower first boot (indexing takes ~15-20s) versus a genuinely one-command setup story: `docker compose up` alone gets a stranger to a working demo, verified by testing against a fully wiped `docker compose down -v` state.
