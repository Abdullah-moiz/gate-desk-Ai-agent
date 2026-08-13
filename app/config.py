import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

VOYAGE_API_KEY = os.environ["VOYAGE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DATABASE_URL = os.environ["DATABASE_URL"]

DENSE_MODEL = "voyage-4-lite"
DENSE_DIM = 1024
SPARSE_MODEL = "Qdrant/bm25"

# gpt-5-nano: cheapest current OpenAI model ($0.05/1M input, $0.40/1M output) —
# classification/routing is a simple enough task that it doesn't need a bigger model.
CLASSIFY_MODEL = "gpt-5-nano"

# Same model for the agent's tool-calling loop. A 16-ticket eval run showed
# nano is occasionally unreliable here (1/16: skipped calling its own
# lookup_account tool and wrote escalation instructions addressed to a human
# instead of acting) — classification is simpler and scored 16/16 twice, but
# multi-step tool orchestration is a harder task. Deliberately kept on nano
# anyway for this project (zero-cost learning project, not production) —
# Phase 6's guardrail layer is the intended backstop for this kind of miss,
# not a bigger model. Worth A/B-testing gpt-5-mini if this were going to prod.
AGENT_MODEL = "gpt-5-nano"
MAX_AGENT_TURNS = 6

POLICY_COLLECTION = "policy_kb"
TICKETS_COLLECTION = "resolved_tickets"

# Routes a ticket's classified category to the policy doc_types worth retrieving from.
CATEGORY_DOC_TYPES = {
    "billing": ["refund_policy", "subscription_tiers", "sla"],
    "account": ["account_security_policy"],
    "bug": ["bug_report_handling", "sla"],
    "general": ["faq"],
}

# Routes a ticket's classified category to the tool surface exposed to the agent.
# Kept deliberately narrow per category (e.g. billing tickets never see
# send_password_reset) so a misclassification can't hand the agent an
# irrelevant, higher-risk action. escalate_to_human is always available.
CATEGORY_TOOLS = {
    "billing": ["lookup_account", "check_subscription_status", "issue_refund", "answer_faq", "escalate_to_human"],
    "account": ["lookup_account", "send_password_reset", "unlock_account", "answer_faq", "escalate_to_human"],
    "bug": ["lookup_account", "log_bug", "provide_workaround", "request_more_info", "escalate_to_human"],
    "general": ["answer_faq", "acknowledge", "request_more_info", "close_spam", "escalate_to_human"],
}
