"""Mock tool implementations + OpenAI Responses-API function schemas for the
support agent.

Every terminal action (refund, password reset, ...) is modeled as an
explicit tool call rather than parsed from the agent's prose — this keeps
the resolution structured, auditable, and makes exactly one row in the
`resolutions` table. lookup_account / check_subscription_status are the
only non-terminal (read-only) tools; every other tool call ends the ticket.
"""

import json

from app.config import ROOT

_ACCOUNTS = {a["account_id"]: a for a in json.loads((ROOT / "data" / "accounts_seed.json").read_text())}
_BY_EMAIL: dict[str, dict] = {}
for _a in _ACCOUNTS.values():
    _BY_EMAIL[_a["owner_email"]] = _a
    for _m in _a.get("team_members", []):
        _BY_EMAIL.setdefault(_m, _a)

REFUND_AUTO_APPROVE_LIMIT = 20  # matches data/policy_docs/refund_policy.md

TERMINAL_TOOLS = {
    "issue_refund",
    "send_password_reset",
    "unlock_account",
    "log_bug",
    "provide_workaround",
    "answer_faq",
    "acknowledge",
    "request_more_info",
    "close_spam",
    "escalate_to_human",
}


def lookup_account(email: str) -> dict:
    account = _BY_EMAIL.get(email)
    if not account:
        return {"found": False}
    return {
        "found": True,
        "account_id": account["account_id"],
        "plan": account["plan"],
        "is_owner": email == account["owner_email"],
        "two_factor_enabled": account["two_factor_enabled"],
        "seats_used": account["seats_used"],
    }


def check_subscription_status(account_id: str) -> dict:
    account = _ACCOUNTS.get(account_id)
    if not account:
        return {"found": False}
    return {
        "found": True,
        "plan": account["plan"],
        "mrr": account["mrr"],
        "created_at": account["created_at"],
    }


def _account_error(account_id: str) -> dict:
    return {
        "error": (
            f"Unknown account_id '{account_id}'. This must be a real account_id, not an email — "
            "call lookup_account first to resolve one."
        )
    }


def issue_refund(account_id: str, amount: float, reason: str) -> dict:
    if account_id not in _ACCOUNTS:
        return {"approved": False, **_account_error(account_id)}
    if amount > REFUND_AUTO_APPROVE_LIMIT:
        return {
            "approved": False,
            "error": (
                f"${amount} exceeds the ${REFUND_AUTO_APPROVE_LIMIT} auto-approval limit; "
                "this must be escalated instead of issued directly."
            ),
        }
    return {"approved": True, "account_id": account_id, "amount": amount, "reason": reason}


def send_password_reset(account_id: str) -> dict:
    if account_id not in _ACCOUNTS:
        return {"sent": False, **_account_error(account_id)}
    return {"sent": True, "account_id": account_id}


def unlock_account(account_id: str) -> dict:
    if account_id not in _ACCOUNTS:
        return {"unlocked": False, **_account_error(account_id)}
    return {"unlocked": True, "account_id": account_id}


def log_bug(severity: str, summary: str) -> dict:
    return {"logged": True, "severity": severity, "summary": summary}


def provide_workaround(workaround: str) -> dict:
    return {"provided": True, "workaround": workaround}


def answer_faq(answer: str) -> dict:
    return {"answered": True, "answer": answer}


def acknowledge(message: str) -> dict:
    return {"acknowledged": True, "message": message}


def request_more_info(question: str) -> dict:
    return {"asked": True, "question": question}


def close_spam() -> dict:
    return {"closed": True}


def escalate_to_human(reason: str) -> dict:
    return {"escalated": True, "reason": reason}


IMPLEMENTATIONS = {
    "lookup_account": lookup_account,
    "check_subscription_status": check_subscription_status,
    "issue_refund": issue_refund,
    "send_password_reset": send_password_reset,
    "unlock_account": unlock_account,
    "log_bug": log_bug,
    "provide_workaround": provide_workaround,
    "answer_faq": answer_faq,
    "acknowledge": acknowledge,
    "request_more_info": request_more_info,
    "close_spam": close_spam,
    "escalate_to_human": escalate_to_human,
}

# Flat schema shape for the Responses API (client.responses.create(tools=...)) —
# no nested "function" wrapper, unlike the older Chat Completions tool format.
SCHEMAS = {
    "lookup_account": {
        "type": "function",
        "name": "lookup_account",
        "description": "Look up the account associated with a customer's email address. Read-only, non-terminal.",
        "parameters": {
            "type": "object",
            "properties": {"email": {"type": "string"}},
            "required": ["email"],
        },
    },
    "check_subscription_status": {
        "type": "function",
        "name": "check_subscription_status",
        "description": "Get the subscription plan/billing status for an account_id. Read-only, non-terminal.",
        "parameters": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
    },
    "issue_refund": {
        "type": "function",
        "name": "issue_refund",
        "description": (
            "Issue a refund for the given account and amount. Only succeeds for amounts at or "
            "under the auto-approval limit — larger amounts are rejected and must be escalated "
            "instead, not retried at a lower amount."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "amount": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["account_id", "amount", "reason"],
        },
    },
    "send_password_reset": {
        "type": "function",
        "name": "send_password_reset",
        "description": "Send a password reset email to the account on file. Terminal action.",
        "parameters": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
    },
    "unlock_account": {
        "type": "function",
        "name": "unlock_account",
        "description": "Unlock an account after a failed-login lockout. Terminal action.",
        "parameters": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
    },
    "log_bug": {
        "type": "function",
        "name": "log_bug",
        "description": (
            "Log a bug report to the engineering backlog with a severity (P0-P3). Terminal action "
            "for P2/P3 bugs only — P0/P1 bugs must be escalated instead of logged."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "severity": {"type": "string", "enum": ["P0", "P1", "P2", "P3"]},
                "summary": {"type": "string"},
            },
            "required": ["severity", "summary"],
        },
    },
    "provide_workaround": {
        "type": "function",
        "name": "provide_workaround",
        "description": "Close the ticket by giving the customer a known workaround. Terminal action.",
        "parameters": {
            "type": "object",
            "properties": {"workaround": {"type": "string"}},
            "required": ["workaround"],
        },
    },
    "answer_faq": {
        "type": "function",
        "name": "answer_faq",
        "description": "Close the ticket by answering a how-to/informational question directly. Terminal action.",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    },
    "acknowledge": {
        "type": "function",
        "name": "acknowledge",
        "description": (
            "Close the ticket by acknowledging feedback, a compliment, or a feature request. "
            "No further action needed. Terminal action."
        ),
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    },
    "request_more_info": {
        "type": "function",
        "name": "request_more_info",
        "description": (
            "Close this turn by asking the customer a clarifying question, when there isn't enough "
            "detail to act safely. Terminal action — does not resolve the ticket, but is the "
            "correct low-risk action when the request is too ambiguous to classify further."
        ),
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
    "close_spam": {
        "type": "function",
        "name": "close_spam",
        "description": "Close the ticket as spam/not a genuine support request. Terminal action.",
        "parameters": {"type": "object", "properties": {}},
    },
    "escalate_to_human": {
        "type": "function",
        "name": "escalate_to_human",
        "description": (
            "Hand the ticket off to a human agent instead of resolving it directly. Always "
            "available regardless of category. Terminal action — use this whenever policy "
            "requires human review, a tool call was rejected, or you are not confident."
        ),
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
}

# Phase 6: every terminal tool except escalate_to_human must self-report a
# confidence score — escalating is already the safe default, so gating it
# on confidence would be pointless. Injected here rather than repeated in
# every schema above so the description has one source of truth.
CONFIDENCE_REQUIRED_TOOLS = TERMINAL_TOOLS - {"escalate_to_human"}

_CONFIDENCE_PROPERTY = {
    "type": "number",
    "minimum": 0,
    "maximum": 1,
    "description": (
        "Your confidence (0.0-1.0) that this is the correct action, based strictly on the "
        "policy excerpts and precedent provided. Use a lower value when policy is ambiguous, "
        "precedent conflicts, or you are inferring beyond what's explicitly stated."
    ),
}

for _name in CONFIDENCE_REQUIRED_TOOLS:
    _params = SCHEMAS[_name]["parameters"]
    _params["properties"]["confidence"] = _CONFIDENCE_PROPERTY
    _params.setdefault("required", []).append("confidence")
