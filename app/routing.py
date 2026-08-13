"""Phase 4: classify an incoming ticket (category + urgency) via a cheap
OpenAI call, then route it to the KB doc_types and tool surface that
category is allowed to use — decided here (config.py) rather than left for
the agent to figure out, so a misclassified or off-topic ticket can't reach
tools it has no business calling.
"""

from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

from app.config import CATEGORY_DOC_TYPES, CATEGORY_TOOLS, CLASSIFY_MODEL, OPENAI_API_KEY

_client = OpenAI(api_key=OPENAI_API_KEY)

Category = Literal["billing", "account", "bug", "general"]
Urgency = Literal["low", "medium", "high"]

SYSTEM_PROMPT = """You are a support ticket classifier for Flowbase, a project-management SaaS.

Classify each ticket into exactly one category:
- billing: charges, refunds, subscriptions, payment methods, invoices
- account: passwords, 2FA, account access/ownership, data export/deletion, security
- bug: something broken, an error, a crash, or unexpected product behavior
- general: how-to questions, feature requests, feedback, anything not fitting the above

And an urgency level:
- high: outage or security incident affecting many users, data loss, active
  account compromise, or an explicit chargeback/legal threat
- medium: a broken feature blocking the requester's own work, or a billing
  issue blocking their access
- low: informational questions, cosmetic issues, feedback, requests with no
  urgency signal

When a ticket is genuinely ambiguous between two categories, pick the one
that best matches its most actionable/risky element (e.g. a billing
complaint that also mentions a bug is still billing if the refund is the
actual ask)."""


class TicketClassification(BaseModel):
    category: Category
    urgency: Urgency


class RoutingDecision(BaseModel):
    category: Category
    urgency: Urgency
    doc_types: list[str]
    allowed_tools: list[str]


def classify(subject: str, body: str) -> TicketClassification:
    completion = _client.chat.completions.parse(
        model=CLASSIFY_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Subject: {subject}\n\n{body}"},
        ],
        response_format=TicketClassification,
        reasoning_effort="low",
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError(f"Classification failed: {completion.choices[0].message.refusal}")
    return result


def route_ticket(subject: str, body: str) -> RoutingDecision:
    result = classify(subject, body)
    return RoutingDecision(
        category=result.category,
        urgency=result.urgency,
        doc_types=CATEGORY_DOC_TYPES[result.category],
        allowed_tools=CATEGORY_TOOLS[result.category],
    )


if __name__ == "__main__":
    import json

    from app.config import ROOT

    eval_tickets = json.loads((ROOT / "data" / "eval_set.json").read_text())

    correct = 0
    for t in eval_tickets:
        decision = route_ticket(t["subject"], t["body"])
        match = decision.category == t["category"]
        correct += match
        flag = "OK " if match else "MISS"
        print(f"[{flag}] {t['id']:6} expected={t['category']:8} got={decision.category:8} urgency={decision.urgency:6} | {t['subject']}")

    print(f"\nCategory accuracy: {correct}/{len(eval_tickets)}")
