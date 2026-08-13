"""Phase 5: the agent loop. route -> retrieve -> tool-calling LLM loop ->
Resolution, with the full trace persisted to Postgres.

Uses OpenAI's Responses API (client.responses.create) for the multi-turn
tool-calling loop — the current OpenAI docs demonstrate this pattern rather
than the older Chat Completions tool-call loop for exactly this use case.

No guardrail/confidence gate yet (that's Phase 6) — right now the agent's
own tool choice is the only thing standing between a ticket and an action,
backstopped by issue_refund's hard-coded amount cap and by escalate_to_human
always being in the tool set.
"""

import json

from openai import OpenAI

from app.config import AGENT_MODEL, MAX_AGENT_TURNS, OPENAI_API_KEY
from app.db import init_schema, save_resolution, save_ticket, save_tool_call
from app.retrieval import retrieve
from app.routing import route_ticket
from app.tools import IMPLEMENTATIONS, SCHEMAS, TERMINAL_TOOLS

_client = OpenAI(api_key=OPENAI_API_KEY)
init_schema()

INSTRUCTIONS_TEMPLATE = """You are a support agent for Flowbase, a project-management SaaS.

You must resolve this ticket by calling exactly one of these terminal tools: {terminal_tools}.
You may first call lookup_account / check_subscription_status (if available to you) for account
context — those do not resolve the ticket.

Base your decision strictly on the policy excerpts and similar past-ticket precedent below. Do
not invent a policy that isn't there. If policy and past precedent conflict, follow the policy
excerpts — precedent is there to help you apply the policy consistently, not override it.

If a tool call is rejected (e.g. issue_refund on an amount over the auto-approval limit), do not
retry with different arguments or work around it — call escalate_to_human and explain why.

When you are ready, call exactly one terminal tool. Do not just describe what you would do.

# Policy excerpts
{policy_context}

# Similar past tickets and how they were resolved
{precedent_context}
"""


def _format_policy(chunks: list[dict]) -> str:
    if not chunks:
        return "(none retrieved)"
    return "\n\n".join(
        f"[{c['payload']['doc_type']} / {c['payload']['section']}] {c['payload']['text']}" for c in chunks
    )


def _format_precedent(chunks: list[dict]) -> str:
    if not chunks:
        return "(none retrieved)"
    lines = []
    for c in chunks:
        p = c["payload"]
        r = p["resolution"]
        lines.append(f'- "{p["subject"]}": {p["body"]}\n  -> {r["outcome"]} via {r["action"]} ({r["reason"]})')
    return "\n".join(lines)


def _is_successful_terminal(name: str, result: dict) -> bool:
    if name == "issue_refund":
        return bool(result.get("approved"))
    return True


def _build_resolution(action: str, args: dict, result: dict) -> dict:
    outcome = "escalated" if action == "escalate_to_human" else "auto_resolved"
    return {"outcome": outcome, "action": action, "arguments": args, "result": result}


def handle_ticket(ticket: dict) -> dict:
    ticket_id = ticket["id"]
    subject = ticket["subject"]
    body = ticket["body"]
    customer_email = ticket.get("customer_email")

    routing = route_ticket(subject, body)
    context = retrieve(f"{subject}\n{body}", category=routing.category, top_k=5, ticket_id=ticket_id)

    save_ticket(ticket_id, subject, body, customer_email, routing.category, routing.urgency)

    tools = [SCHEMAS[name] for name in routing.allowed_tools if name in SCHEMAS]
    terminal_allowed = [t for t in routing.allowed_tools if t in TERMINAL_TOOLS]

    instructions = INSTRUCTIONS_TEMPLATE.format(
        terminal_tools=", ".join(terminal_allowed),
        policy_context=_format_policy(context["policy"]),
        precedent_context=_format_precedent(context["precedent"]),
    )

    input_items: list[dict] = [
        {
            "role": "user",
            "content": (
                f"Ticket {ticket_id}\nSubject: {subject}\n"
                f"Customer email: {customer_email or 'unknown'}\n\n{body}"
            ),
        }
    ]

    tool_call_log: list[dict] = []
    resolution: dict | None = None

    for _ in range(MAX_AGENT_TURNS):
        response = _client.responses.create(
            model=AGENT_MODEL,
            instructions=instructions,
            input=input_items,
            tools=tools,
        )
        input_items.extend(item.model_dump(exclude_none=True) for item in response.output)

        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            break

        for call in calls:
            name = call.name
            args = json.loads(call.arguments or "{}")
            fn = IMPLEMENTATIONS.get(name)
            result = fn(**args) if fn else {"error": f"unknown tool {name}"}

            step = len(tool_call_log) + 1
            tool_call_log.append({"step": step, "tool_name": name, "arguments": args, "result": result})
            save_tool_call(ticket_id, step, name, args, result)

            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result),
                }
            )

            if resolution is None and name in TERMINAL_TOOLS and _is_successful_terminal(name, result):
                resolution = _build_resolution(name, args, result)

        if resolution:
            break

    if resolution is None:
        args = {"reason": "Agent did not reach a terminal decision within the turn limit."}
        result = IMPLEMENTATIONS["escalate_to_human"](**args)
        resolution = _build_resolution("escalate_to_human", args, result)
        step = len(tool_call_log) + 1
        tool_call_log.append({"step": step, "tool_name": "escalate_to_human", "arguments": args, "result": result})
        save_tool_call(ticket_id, step, "escalate_to_human", args, result)

    save_resolution(ticket_id, resolution, input_items)

    return {
        "ticket_id": ticket_id,
        "category": routing.category,
        "urgency": routing.urgency,
        "resolution": resolution,
        "tool_calls": tool_call_log,
    }


if __name__ == "__main__":
    import sys

    from app.config import ROOT

    eval_tickets = json.loads((ROOT / "data" / "eval_set.json").read_text())
    ids = sys.argv[1:] or None

    for t in eval_tickets:
        if ids and t["id"] not in ids:
            continue
        outcome = handle_ticket(t)
        exp = t["expected"]
        got = outcome["resolution"]
        outcome_match = got["outcome"] == exp["outcome"]
        action_match = exp["outcome"] == "escalated" or got["action"] == exp["action"]
        match = outcome_match and action_match
        flag = "OK  " if match else "MISS"
        print(
            f"[{flag}] {t['id']:6} expected={exp['outcome']:13}/{str(exp['action']):18} "
            f"got={outcome['resolution']['outcome']:13}/{str(outcome['resolution']['action']):18} "
            f"({len(outcome['tool_calls'])} tool calls)"
        )
