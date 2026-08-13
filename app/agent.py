"""Phase 5+6: the agent loop plus the confidence gate. route -> retrieve ->
tool-calling LLM loop -> gate -> Resolution, with the full trace persisted
to Postgres.

Uses OpenAI's Responses API (client.responses.create) for the multi-turn
tool-calling loop — the current OpenAI docs demonstrate this pattern rather
than the older Chat Completions tool-call loop for exactly this use case.

The agent's tool choice is a *draft*, not a decision — app/gate.py
independently checks it (retrieval grounding + the agent's self-reported
confidence + a couple of hard-coded policy rules) before it's allowed to
stand as an auto-resolution. This is the maker/checker split: nothing here
trusts the LLM's tool choice by itself.
"""

import json

from openai import OpenAI

from app.config import AGENT_MODEL, MAX_AGENT_TURNS, OPENAI_API_KEY
from app.db import clear_ticket_trace, init_schema, save_resolution, save_ticket, save_tool_call
from app.gate import apply_gate
from app.retrieval import retrieve
from app.routing import route_ticket
from app.tools import CONFIDENCE_REQUIRED_TOOLS, IMPLEMENTATIONS, SCHEMAS, TERMINAL_TOOLS

_client = OpenAI(api_key=OPENAI_API_KEY)
init_schema()

INSTRUCTIONS_TEMPLATE = """You are a support agent for Flowbase, a project-management SaaS.

You must resolve this ticket by calling exactly one of these terminal tools: {terminal_tools}.
You may first call lookup_account / check_subscription_status (if available to you) for account
context — those do not resolve the ticket.

Base your decision strictly on the policy excerpts and similar past-ticket precedent below. Do
not invent a policy that isn't there. If policy and past precedent conflict, follow the policy
excerpts — precedent is there to help you apply the policy consistently, not override it.

Every terminal tool (other than escalate_to_human) requires a confidence score from 0.0 to 1.0.
Report it honestly — a downstream check uses it, and an inflated score does not make an
incorrect action correct, it just means the mistake reaches the customer instead of a human
reviewer catching it first.

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


_SUCCESS_KEY = {
    "issue_refund": "approved",
    "send_password_reset": "sent",
    "unlock_account": "unlocked",
}


def _is_successful_terminal(name: str, result: dict) -> bool:
    key = _SUCCESS_KEY.get(name)
    return bool(result.get(key)) if key else True


def _build_draft(action: str, args: dict, result: dict) -> dict:
    outcome = "escalated" if action == "escalate_to_human" else "auto_resolved"
    return {"outcome": outcome, "action": action, "arguments": args, "result": result}


def handle_ticket(ticket: dict) -> dict:
    ticket_id = ticket["id"]
    subject = ticket["subject"]
    body = ticket["body"]
    customer_email = ticket.get("customer_email")

    clear_ticket_trace(ticket_id)

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
    agent_draft: dict | None = None
    agent_confidence: float | None = None

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
            confidence = args.pop("confidence", None) if name in CONFIDENCE_REQUIRED_TOOLS else None
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

            if agent_draft is None and name in TERMINAL_TOOLS and _is_successful_terminal(name, result):
                agent_draft = _build_draft(name, args, result)
                agent_confidence = confidence

        if agent_draft:
            break

    if agent_draft is None:
        args = {"reason": "Agent did not reach a terminal decision within the turn limit."}
        result = IMPLEMENTATIONS["escalate_to_human"](**args)
        agent_draft = _build_draft("escalate_to_human", args, result)
        step = len(tool_call_log) + 1
        tool_call_log.append({"step": step, "tool_name": "escalate_to_human", "arguments": args, "result": result})
        save_tool_call(ticket_id, step, "escalate_to_human", args, result)

    gate = apply_gate(
        action=agent_draft["action"],
        arguments=agent_draft["arguments"],
        result=agent_draft["result"],
        agent_confidence=agent_confidence,
        policy_confidence=context["policy_confidence"],
    )

    final = {
        "outcome": gate.final_outcome,
        "action": gate.final_action,
        "arguments": gate.final_arguments,
        "result": gate.final_result,
    }

    if not gate.passed:
        # The gate overrode the agent's draft — log the resulting escalation
        # as its own tool_call entry so the audit trail matches what actually
        # happened, not just what the agent proposed.
        step = len(tool_call_log) + 1
        tool_call_log.append(
            {"step": step, "tool_name": "escalate_to_human", "arguments": gate.final_arguments, "result": gate.final_result}
        )
        save_tool_call(ticket_id, step, "escalate_to_human", gate.final_arguments, gate.final_result)

    save_resolution(
        ticket_id,
        final=final,
        agent_draft=agent_draft,
        agent_confidence=agent_confidence,
        policy_confidence=context["policy_confidence"],
        precedent_confidence=context["precedent_confidence"],
        gate_passed=gate.passed,
        gate_reason=gate.reason,
        trace=input_items,
    )

    return {
        "ticket_id": ticket_id,
        "category": routing.category,
        "urgency": routing.urgency,
        "resolution": final,
        "agent_draft": agent_draft,
        "agent_confidence": agent_confidence,
        "policy_confidence": context["policy_confidence"],
        "gate_passed": gate.passed,
        "gate_reason": gate.reason,
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
        gated = "" if outcome["gate_passed"] else f"  [GATED: {outcome['gate_reason']}]"
        print(
            f"[{flag}] {t['id']:6} expected={exp['outcome']:13}/{str(exp['action']):18} "
            f"got={got['outcome']:13}/{str(got['action']):18} "
            f"conf={outcome['agent_confidence']} pconf={outcome['policy_confidence']:.2f}"
            f"{gated}"
        )
