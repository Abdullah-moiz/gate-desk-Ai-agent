"""Phase 7: pretty-print the full audit trail for a ticket — retrieval,
every tool call, the gate's decision, and the final resolution — pulled
straight from Postgres (tickets, retrievals, tool_calls, resolutions) via
the shared query helpers in app/db.py (also used by app/dashboard.py).

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

from app import db

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


def print_ticket_list() -> None:
    rows = db.list_tickets()
    if not rows:
        print("No tickets found — run app.agent on some tickets first.")
        return

    print(f"{BOLD}{len(rows)} ticket(s) with a trace:{RESET}\n")
    for t in rows:
        outcome, action = t["outcome"], t["action"]
        color = GREEN if outcome == "auto_resolved" else YELLOW if outcome == "escalated" else DIM
        print(
            f"  {CYAN}{t['ticket_id']:8}{RESET} [{t['category'] or '?':8}/{t['urgency'] or '?':6}] "
            f"{color}{outcome or 'no resolution':13}{RESET} {action or '':22} {t['subject']}"
        )
    print(f"\nRun: python scripts/trace.py <ticket_id>")


def trace(ticket_id: str) -> None:
    ticket = db.get_ticket(ticket_id)
    if not ticket:
        print(f"No ticket found with id '{ticket_id}'. Run: python scripts/trace.py (no args) to list tickets.")
        return

    retrieval_rows = db.get_retrievals(ticket_id)
    tool_call_rows = db.get_tool_calls(ticket_id)
    res = db.get_resolution(ticket_id)

    # --- header ---------------------------------------------------------
    print(_rule("="))
    print(f"{BOLD}TICKET {ticket_id}{RESET} — {ticket['subject']}")
    print(_rule("="))
    print(f"{DIM}Customer:{RESET} {ticket['customer_email'] or 'unknown':30} {DIM}Created:{RESET} {ticket['created_at']}")
    print(f"{DIM}Category:{RESET} {ticket['category']:12} {DIM}Urgency:{RESET} {ticket['urgency']}")
    print(_wrap(f"{DIM}Body:{RESET} {ticket['body']}", indent="  "))
    print()

    # --- retrieval --------------------------------------------------------
    print(_rule(label="RETRIEVAL"))
    if not retrieval_rows:
        print("  (no retrieval logged)")
    else:
        current_collection = None
        for r in retrieval_rows:
            if r["collection_name"] != current_collection:
                current_collection = r["collection_name"]
                print(f"\n  {BOLD}{current_collection}{RESET}")
            payload = r["payload"]
            if r["collection_name"] == "policy_kb":
                label = f"{payload['doc_type']} / {payload['section']}"
                snippet = payload["text"].split("\n", 1)[-1][:90]
            else:
                label = f"{payload['id']} \"{payload['subject']}\""
                res_payload = payload["resolution"]
                snippet = f'-> {res_payload["outcome"]} via {res_payload["action"]}'
            print(f"    [{r['result_rank']}] score={r['score']:.3f}  {label}")
            print(f"        {DIM}{snippet}{RESET}")
    print()

    # --- tool calls ---------------------------------------------------------
    print(_rule(label="AGENT TOOL CALLS"))
    if not tool_call_rows:
        print("  (no tool calls logged)")
    for tc in tool_call_rows:
        print(f"  [{tc['step']}] {CYAN}{tc['tool_name']}{RESET}({tc['arguments']})")
        print(f"      -> {tc['result']}")
    print()

    # --- gate + resolution ---------------------------------------------------
    if not res:
        print(_rule(label="RESOLUTION"))
        print("  (no resolution logged)")
        print(_rule("="))
        return

    print(_rule(label="GATE"))
    print(f"  agent proposed:      {res['agent_action']}({res['agent_arguments']})  confidence={res['agent_confidence']}")
    print(f"  policy grounding:    {res['policy_confidence']:.2f}   precedent grounding: {res['precedent_confidence']:.2f}")
    gate_color = GREEN if res["gate_passed"] else RED
    gate_label = "PASSED" if res["gate_passed"] else "OVERRIDDEN"
    print(f"  gate result:         {gate_color}{gate_label}{RESET} — {res['gate_reason']}")
    if not res["gate_passed"]:
        print(f"  {DIM}(agent's draft action above was not taken — see final resolution below){RESET}")
    print()

    print(_rule(label="FINAL RESOLUTION"))
    outcome_color = GREEN if res["outcome"] == "auto_resolved" else YELLOW
    print(f"  OUTCOME: {outcome_color}{BOLD}{res['outcome'].upper()}{RESET}")
    print(f"  ACTION:  {res['action']}({res['arguments']})")
    print(f"  RESULT:  {res['result']}")
    print(_rule("="))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_ticket_list()
    else:
        trace(sys.argv[1])
