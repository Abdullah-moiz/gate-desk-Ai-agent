"""Phase 8: run the full pipeline over data/eval_set.json and report the
metrics that actually matter for a support-automation system — not just
"% correct," but specifically the false-resolve rate (wrong autonomous
action taken with confidence), since that's the number a reviewer should
ask about first.

Retrieval precision/recall in the textbook sense would need chunk-level
relevance judgments we never hand-labeled; "policy grounding hit rate"
below is a deliberate proxy instead (did retrieval pull from the doc_type
that's actually relevant to this ticket's category), read from the
retrievals table Phase 3/7 already populate — not a substitute for real
precision/recall, just what's honestly measurable from the data we have.

Usage:
    python scripts/evaluate.py                 # full eval_set.json
    python scripts/evaluate.py E-01 E-05        # just these ticket ids
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from voyageai.error import RateLimitError

from app import usage
from app.agent import handle_ticket
from app.config import CATEGORY_DOC_TYPES, ROOT
from app.db import get_conn

RATE_LIMIT_BACKOFF_SECONDS = 21
MAX_RETRIES = 12


def _run_with_retry(ticket: dict) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            return handle_ticket(ticket)
        except RateLimitError:
            print(f"  ({ticket['id']}: Voyage rate limit, waiting {RATE_LIMIT_BACKOFF_SECONDS}s...)")
            time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
    raise RuntimeError(f"Exceeded {MAX_RETRIES} retries for {ticket['id']} — Voyage rate limit not clearing.")


def _top_policy_doc_type(ticket_id: str) -> str | None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT payload->>'doc_type' FROM retrievals
            WHERE ticket_id = %s AND collection_name = 'policy_kb'
            ORDER BY result_rank ASC LIMIT 1
            """,
            (ticket_id,),
        )
        row = cur.fetchone()
        return row[0] if row else None


def run(tickets: list[dict]) -> dict:
    usage.reset()
    results = []

    for t in tickets:
        print(f"  running {t['id']}...")
        outcome = _run_with_retry(t)
        top_doc_type = _top_policy_doc_type(t["id"])
        results.append({"ticket": t, "outcome": outcome, "top_policy_doc_type": top_doc_type})

    return _compute_metrics(results)


def _compute_metrics(results: list[dict]) -> dict:
    n = len(results)
    category_correct = 0
    outcome_correct = 0
    action_correct = 0
    action_checkable = 0
    false_resolves = 0  # expected escalated, got auto_resolved -- the headline risk metric
    over_escalations = 0  # expected auto_resolved, got escalated -- costs efficiency, not safety
    expected_escalated_n = 0
    expected_auto_n = 0
    grounding_hits = 0

    rows = []
    for r in results:
        t, o = r["ticket"], r["outcome"]
        exp = t["expected"]
        got = o["resolution"]

        cat_ok = o["category"] == t["category"]
        category_correct += cat_ok

        out_ok = got["outcome"] == exp["outcome"]
        outcome_correct += out_ok

        if exp["outcome"] == "auto_resolved":
            expected_auto_n += 1
            action_checkable += 1
            act_ok = got["action"] == exp["action"]
            action_correct += act_ok
            if got["outcome"] == "escalated":
                over_escalations += 1
        else:
            expected_escalated_n += 1
            act_ok = out_ok  # no specific action ground truth for escalations
            if got["outcome"] == "auto_resolved":
                false_resolves += 1

        expected_doc_types = CATEGORY_DOC_TYPES.get(t["category"], [])
        grounded = r["top_policy_doc_type"] in expected_doc_types
        grounding_hits += grounded

        rows.append(
            {
                "id": t["id"],
                "category_ok": cat_ok,
                "outcome_ok": out_ok,
                "action_ok": act_ok,
                "grounded": grounded,
                "gate_passed": o["gate_passed"],
            }
        )

    usage_rows, total_cost = usage.summary()

    return {
        "n": n,
        "category_accuracy": category_correct / n,
        "outcome_accuracy": outcome_correct / n,
        "action_accuracy": (action_correct / action_checkable) if action_checkable else None,
        "false_resolve_count": false_resolves,
        "false_resolve_rate": (false_resolves / expected_escalated_n) if expected_escalated_n else None,
        "over_escalation_count": over_escalations,
        "over_escalation_rate": (over_escalations / expected_auto_n) if expected_auto_n else None,
        "policy_grounding_hit_rate": grounding_hits / n,
        "usage_rows": usage_rows,
        "total_cost_usd": total_cost,
        "cost_per_ticket_usd": total_cost / n if n else 0,
        "rows": rows,
    }


def render_report(metrics: dict) -> str:
    def pct(x):
        return f"{x * 100:.1f}%" if x is not None else "n/a"

    lines = []
    lines.append("# Evaluation Report\n")
    lines.append(f"Ran {metrics['n']} tickets from `data/eval_set.json` through the full pipeline ")
    lines.append("(classify -> retrieve -> agent tool-calling loop -> confidence gate).\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Category classification accuracy | {pct(metrics['category_accuracy'])} |")
    lines.append(f"| Outcome accuracy (auto_resolved vs escalated) | {pct(metrics['outcome_accuracy'])} |")
    lines.append(f"| Action accuracy (on correctly auto-resolved tickets) | {pct(metrics['action_accuracy'])} |")
    lines.append(
        f"| **False-resolve rate** (wrongly auto-resolved when it should've escalated) | "
        f"**{pct(metrics['false_resolve_rate'])}** ({metrics['false_resolve_count']} tickets) |"
    )
    lines.append(
        f"| Over-escalation rate (escalated when it could've auto-resolved) | "
        f"{pct(metrics['over_escalation_rate'])} ({metrics['over_escalation_count']} tickets) |"
    )
    lines.append(f"| Policy grounding hit rate (proxy retrieval metric, see script docstring) | {pct(metrics['policy_grounding_hit_rate'])} |")
    lines.append(f"| Estimated cost for this eval run | ${metrics['total_cost_usd']:.5f} (${metrics['cost_per_ticket_usd']:.6f}/ticket) |")
    lines.append("")

    lines.append("## Token usage by model\n")
    lines.append("| Model | Calls | Input tokens | Output tokens | Est. cost |")
    lines.append("|---|---|---|---|---|")
    for u in metrics["usage_rows"]:
        lines.append(
            f"| {u['model']} | {u['calls']} | {u['input_tokens']:,} | {u['output_tokens']:,} | ${u['estimated_cost_usd']:.5f} |"
        )
    lines.append("")

    lines.append("## Per-ticket results\n")
    lines.append("| Ticket | Category | Outcome | Action | Grounded | Gate |")
    lines.append("|---|---|---|---|---|---|")
    for row in metrics["rows"]:
        def mark(b):
            return "OK" if b else "MISS"

        gate = "passed" if row["gate_passed"] else "overridden"
        lines.append(f"| {row['id']} | {mark(row['category_ok'])} | {mark(row['outcome_ok'])} | {mark(row['action_ok'])} | {mark(row['grounded'])} | {gate} |")

    return "\n".join(lines)


if __name__ == "__main__":
    eval_tickets = json.loads((ROOT / "data" / "eval_set.json").read_text())
    ids = sys.argv[1:] or None
    if ids:
        eval_tickets = [t for t in eval_tickets if t["id"] in ids]

    metrics = run(eval_tickets)
    report = render_report(metrics)
    print("\n" + report)

    out_path = ROOT / "EVAL_RESULTS.md"
    out_path.write_text(report + "\n")
    print(f"\nSaved to {out_path}")
