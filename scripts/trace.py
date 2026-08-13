"""Phase 7: pretty-print the full audit trail for a ticket — retrieval,
every tool call, the gate's decision, and the final resolution — pulled
straight from Postgres (tickets, retrievals, tool_calls, resolutions).

This is the "why did the agent do X" reconstruction tool for the whole
project: everything it prints was already being written by Phases 3/5/6,
this just makes it readable instead of ad-hoc SQL.

Usage:
    python scripts/trace.py            # list tickets that have a trace
    python scripts/trace.py E-05       # print the full trace for E-05
"""

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_conn

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"

WIDTH = 88


def _rule(char: str = "-", label: str = "") -> str:
    if label:
        return f"{BOLD}--- {label} {'-' * max(0, WIDTH - len(label) - 5)}{RESET}"
    return char * WIDTH


def _wrap(text: str, indent: str = "      ") -> str:
    return "\n".join(textwrap.wrap(text, width=WIDTH - len(indent), initial_indent=indent, subsequent_indent=indent))


def list_tickets() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT t.ticket_id, t.subject, t.category, t.urgency, r.outcome, r.action
            FROM tickets t
            LEFT JOIN resolutions r ON r.ticket_id = t.ticket_id
            ORDER BY t.created_at DESC
            """
        )
        rows = cur.fetchall()

    if not rows:
        print("No tickets found — run app.agent on some tickets first.")
        return

    print(f"{BOLD}{len(rows)} ticket(s) with a trace:{RESET}\n")
    for ticket_id, subject, category, urgency, outcome, action in rows:
        color = GREEN if outcome == "auto_resolved" else YELLOW if outcome == "escalated" else DIM
        print(f"  {CYAN}{ticket_id:8}{RESET} [{category or '?':8}/{urgency or '?':6}] {color}{outcome or 'no resolution':13}{RESET} {action or '':22} {subject}")
    print(f"\nRun: python scripts/trace.py <ticket_id>")


def trace(ticket_id: str) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT subject, body, customer_email, category, urgency, created_at FROM tickets WHERE ticket_id = %s",
            (ticket_id,),
        )
        ticket_row = cur.fetchone()
        if not ticket_row:
            print(f"No ticket found with id '{ticket_id}'. Run: python scripts/trace.py (no args) to list tickets.")
            return
        subject, body, customer_email, category, urgency, created_at = ticket_row

        cur.execute(
            """
            SELECT collection_name, result_rank, point_id, score, payload
            FROM retrievals WHERE ticket_id = %s
            ORDER BY collection_name, result_rank
            """,
            (ticket_id,),
        )
        retrieval_rows = cur.fetchall()

        cur.execute(
            "SELECT step, tool_name, arguments, result FROM tool_calls WHERE ticket_id = %s ORDER BY step",
            (ticket_id,),
        )
        tool_call_rows = cur.fetchall()

        cur.execute(
            """
            SELECT outcome, action, arguments, result,
                   agent_action, agent_arguments, agent_result, agent_confidence,
                   policy_confidence, precedent_confidence, gate_passed, gate_reason
            FROM resolutions WHERE ticket_id = %s
            """,
            (ticket_id,),
        )
        res_row = cur.fetchone()

    # --- header ---------------------------------------------------------
    print(_rule("="))
    print(f"{BOLD}TICKET {ticket_id}{RESET} — {subject}")
    print(_rule("="))
    print(f"{DIM}Customer:{RESET} {customer_email or 'unknown':30} {DIM}Created:{RESET} {created_at}")
    print(f"{DIM}Category:{RESET} {category:12} {DIM}Urgency:{RESET} {urgency}")
    print(_wrap(f"{DIM}Body:{RESET} {body}", indent="  "))
    print()

    # --- retrieval --------------------------------------------------------
    print(_rule(label="RETRIEVAL"))
    if not retrieval_rows:
        print("  (no retrieval logged)")
    else:
        current_collection = None
        for collection_name, rank, point_id, score, payload in retrieval_rows:
            if collection_name != current_collection:
                current_collection = collection_name
                print(f"\n  {BOLD}{collection_name}{RESET}")
            if collection_name == "policy_kb":
                label = f"{payload['doc_type']} / {payload['section']}"
                snippet = payload["text"].split("\n", 1)[-1][:90]
            else:
                label = f"{payload['id']} \"{payload['subject']}\""
                r = payload["resolution"]
                snippet = f'-> {r["outcome"]} via {r["action"]}'
            print(f"    [{rank}] score={score:.3f}  {label}")
            print(f"        {DIM}{snippet}{RESET}")
    print()

    # --- tool calls ---------------------------------------------------------
    print(_rule(label="AGENT TOOL CALLS"))
    if not tool_call_rows:
        print("  (no tool calls logged)")
    for step, tool_name, arguments, result in tool_call_rows:
        print(f"  [{step}] {CYAN}{tool_name}{RESET}({arguments})")
        print(f"      -> {result}")
    print()

    # --- gate + resolution ---------------------------------------------------
    if not res_row:
        print(_rule(label="RESOLUTION"))
        print("  (no resolution logged)")
        print(_rule("="))
        return

    (
        outcome, action, arguments, result,
        agent_action, agent_arguments, agent_result, agent_confidence,
        policy_confidence, precedent_confidence, gate_passed, gate_reason,
    ) = res_row

    print(_rule(label="GATE"))
    print(f"  agent proposed:      {agent_action}({agent_arguments})  confidence={agent_confidence}")
    print(f"  policy grounding:    {policy_confidence:.2f}   precedent grounding: {precedent_confidence:.2f}")
    gate_color = GREEN if gate_passed else RED
    gate_label = "PASSED" if gate_passed else "OVERRIDDEN"
    print(f"  gate result:         {gate_color}{gate_label}{RESET} — {gate_reason}")
    if not gate_passed:
        print(f"  {DIM}(agent's draft action above was not taken — see final resolution below){RESET}")
    print()

    print(_rule(label="FINAL RESOLUTION"))
    outcome_color = GREEN if outcome == "auto_resolved" else YELLOW
    print(f"  OUTCOME: {outcome_color}{BOLD}{outcome.upper()}{RESET}")
    print(f"  ACTION:  {action}({arguments})")
    print(f"  RESULT:  {result}")
    print(_rule("="))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        list_tickets()
    else:
        trace(sys.argv[1])
