import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

VOYAGE_API_KEY = os.environ["VOYAGE_API_KEY"]
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DATABASE_URL = os.environ["DATABASE_URL"]

DENSE_MODEL = "voyage-4-lite"
DENSE_DIM = 1024
SPARSE_MODEL = "Qdrant/bm25"

POLICY_COLLECTION = "policy_kb"
TICKETS_COLLECTION = "resolved_tickets"

# Routes a ticket's classified category to the policy doc_types worth retrieving from.
CATEGORY_DOC_TYPES = {
    "billing": ["refund_policy", "subscription_tiers"],
    "account": ["account_security_policy"],
    "bug": ["bug_report_handling", "sla"],
    "general": ["faq"],
}
